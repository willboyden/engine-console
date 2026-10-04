"""The pure builders: node messages + link state -> HostSummary / HostDetail, labels, and storage shared between hosts."""
from __future__ import annotations

from typing import Any

import pytest
from ecnode import proto

from engine_console.domain.fleet import HostCounts, ImageInfo
from engine_console.services import fleet_status as fs
from engine_console.services.fleet_status import HostData

from .builders import GIB, data, disks, hello, mount, root, sample


# ── summary ─────────────────────────────────────────────────────────────────────────────────────────────
def test_a_live_summary_converts_bytes_to_gib_and_orders_gpus_by_index() -> None:
    s = fs.build_summary(data(h=hello(), s=sample(), skew_s=2.0))
    assert s.id == "gpu-box-2" and s.label == "Gpu-Box-2" and s.remote is True and s.link.state == "up"
    a, b = s.gpus
    assert (a.index, a.uuid, a.name) == (0, "GPU-A", "RTX A") and (b.index, b.uuid) == (1, "GPU-B")
    assert a.total_gib == 48.0 and a.used_gib == 40.0 and a.free_gib == 8.0
    assert (a.util_pct, a.temp_c, a.power_w, a.power_limit_w, a.fan_pct) == (80.0, 61.0, 250.0, 300.0, 44.0)
    assert a.compute_capability == "8.9" and a.unified_memory is False
    assert b.total_gib == 96.0 and b.used_gib == 1.0 and b.free_gib == 95.0 and b.power_w is None   # absent stays None
    assert s.memory is not None
    assert (s.memory.total_gib, s.memory.available_gib, s.memory.used_gib, s.memory.free_gib) == (256.0, 200.0, 56.0, 150.0)
    assert (s.memory.cached_gib, s.memory.shmem_gib, s.memory.swap_total_gib, s.memory.swap_used_gib) == (40.0, 1.0, 8.0, 0.0)
    assert s.memory.updated_at == 998.0                     # the node's clock minus its measured skew: the hub's clock
    assert s.cpu is not None and (s.cpu.load1, s.cpu.util_pct) == (1.5, 12.0)
    assert s.alerts == []


def test_gpu_processes_name_the_instance_behind_each_container() -> None:
    calls: list[tuple[str, str]] = []

    def resolver(host_id: str, container: str) -> str | None:
        calls.append((host_id, container))
        return "inst_abc" if container == "ec-engine-1" else None

    s = fs.build_summary(data(h=hello(), s=sample()), resolver=resolver)
    procs = {p.pid: p for p in s.gpus[0].procs}
    assert procs[11].instance_id == "inst_abc" and procs[11].container == "ec-engine-1" and procs[11].used_gib == 30.0
    assert procs[12].instance_id is None and procs[13].container is None and procs[13].used_gib is None
    assert procs[11].name == "python3"
    assert ("gpu-box-2", "ec-engine-1") in calls and all(c[1] for c in calls)    # never asked about "no container"


def test_a_broken_resolver_or_counts_provider_does_not_take_the_summary_down() -> None:
    def boom(*_: Any) -> Any:
        raise RuntimeError("bug in another service")

    s = fs.build_summary(data(h=hello(), s=sample()), resolver=boom, counts=boom)
    assert s.gpus[0].procs[0].instance_id is None and s.counts == HostCounts()


def test_counts_come_from_the_provider() -> None:
    s = fs.build_summary(data(h=hello()), counts=lambda hid: HostCounts(engines_running=2, engines_total=3, models=7, models_bytes=9))
    assert (s.counts.engines_running, s.counts.engines_total, s.counts.models, s.counts.models_bytes) == (2, 3, 7, 9)


def test_node_docker_and_policy_facts() -> None:
    s = fs.build_summary(data(h=hello(mode="manage", allow_pull=True, external_control="stop_start"), s=sample()))
    assert s.node is not None
    assert (s.node.version, s.node.hostname, s.node.arch, s.node.os, s.node.kernel) == ("0.4.0", "gpu-box-2", "aarch64", "Ubuntu 24.04", "6.8")
    assert (s.node.cpu_model, s.node.cpu_cores, s.node.python, s.node.driver, s.node.cuda) == ("Grace", 20, "3.12.3", "570.1", "12.8")
    assert s.node.gpu_source == "nvml" and s.node.forced_cmd is True and s.node.via_ssh is True
    assert s.docker is not None and (s.docker.available, s.docker.context, s.docker.rootless, s.docker.nvidia_runtime) == (True, "rootless", True, True)
    assert s.policy is not None
    assert s.policy.mode == "manage" and s.policy.allow_pull is True and s.policy.external_control == "stop_start"
    assert s.policy.port_range == [18000, 18099] and s.policy.engine_images == ["vllm/vllm-openai:*"] and s.policy.max_shm_gib == 64.0


@pytest.mark.parametrize("mode,ext,ipc", [("admin", "everything", None), (None, None, None), ("MANAGE", "STOP_START", False)])
def test_policy_values_the_hub_does_not_recognise_fail_safe(mode: Any, ext: Any, ipc: Any) -> None:
    s = fs.build_summary(data(h=hello(mode=mode, external_control=ext, allow_ipc_host=ipc)))
    assert s.policy is not None
    assert s.policy.mode == "monitor" and s.policy.external_control == "none"          # nothing may change on that machine
    assert s.policy.allow_ipc_host is (True if ipc is None else ipc)


def test_a_policy_error_and_a_bad_port_range_are_passed_on_safely() -> None:
    s = fs.build_summary(data(h=hello(error="policy.json is writable by group", port_range=[18000])))
    assert s.policy is not None and s.policy.error == "policy.json is writable by group" and s.policy.port_range == []


def test_a_down_host_keeps_its_static_facts_and_shows_nothing_live() -> None:
    s = fs.build_summary(data(h=hello(), s=sample(), state="down", error="the node exited (3)"))
    assert [g.uuid for g in s.gpus] == ["GPU-A", "GPU-B"] and s.gpus[0].total_gib == 48.0
    assert s.gpus[0].used_gib is None and s.gpus[0].util_pct is None and s.gpus[0].procs == []
    assert s.memory is None and s.cpu is None and s.alerts == []
    assert s.node is not None and s.node.version == "0.4.0" and s.link.error == "the node exited (3)"


def test_a_stale_host_still_shows_its_last_sample() -> None:
    s = fs.build_summary(data(h=hello(), s=sample(), state="stale"))
    assert s.gpus[0].used_gib == 40.0 and s.memory is not None


def test_a_host_that_never_connected_has_nothing_but_its_name_and_link() -> None:
    s = fs.build_summary(data(state="connecting"))
    assert s.node is None and s.docker is None and s.policy is None and s.gpus == [] and s.memory is None
    assert s.id == "gpu-box-2" and s.link.state == "connecting" and s.counts == HostCounts()


def test_the_local_host_is_not_remote() -> None:
    assert fs.build_summary(data("local")).remote is False


def test_unified_memory_gpus_report_no_per_gpu_usage() -> None:
    h = hello()
    next(g for g in h["gpus"] if g["uuid"] == "GPU-A")["unified_memory"] = True
    s = fs.build_summary(data(h=h, s=sample(gpus={"GPU-A": {"util_pct": 5.0}, "GPU-B": {}})))
    gpu = next(g for g in s.gpus if g.uuid == "GPU-A")
    assert gpu.unified_memory is True and gpu.used_gib is None and gpu.free_gib is None and gpu.util_pct == 5.0


def test_free_memory_is_never_negative_even_when_a_driver_reports_more_used_than_total() -> None:
    s = fs.build_summary(data(h=hello(), s=sample(gpus={"GPU-A": {"mem_used_bytes": 50 * GIB}, "GPU-B": {}})))
    gpu = next(g for g in s.gpus if g.uuid == "GPU-A")
    assert gpu.total_gib == 48.0 and gpu.used_gib == 50.0 and gpu.free_gib == 0.0


def test_a_gpu_without_a_uuid_or_unknown_to_the_hello_is_not_listed() -> None:
    h = hello()
    h["gpus"].append({"index": 5, "name": "no uuid"})
    s = fs.build_summary(data(h=h, s=sample(gpus={"GPU-A": {"util_pct": 1.0}, "GPU-ROGUE": {"util_pct": 99.0}})))
    assert [g.uuid for g in s.gpus] == ["GPU-A", "GPU-B"]


@pytest.mark.parametrize("avail,total,code,level", [(10, 256, "host_ram_low", "crit"), (20, 256, "host_ram_low", "warn")])
def test_memory_alerts_use_the_same_rules_as_the_local_report(avail: int, total: int, code: str, level: str) -> None:
    s = fs.build_summary(data(h=hello(), s=sample(mem={"total": total * GIB, "available": avail * GIB})))
    assert [(a.code, a.level) for a in s.alerts] == [(code, level)]


def test_shared_memory_and_swap_alerts() -> None:
    mem = {"total": 100 * GIB, "available": 60 * GIB, "shmem": 40 * GIB, "swap_total": 10 * GIB, "swap_free": 2 * GIB}
    s = fs.build_summary(data(h=hello(), s=sample(mem=mem)))
    assert sorted(a.code for a in s.alerts) == ["shmem_high", "swap_heavy"]


def test_a_sample_without_memory_numbers_gives_no_memory_and_no_alerts() -> None:
    s = fs.build_summary(data(h=hello(), s=sample(mem={"total": None, "available": 5})))
    assert s.memory is None and s.alerts == []


def test_hostile_garbage_never_raises() -> None:
    nasty_hello = proto.sanitize_push({"type": "hello", "proto": 1, "host": "text", "gpu": [], "gpus": [None, 5, {"uuid": 3}, {"uuid": "g"}],
                                       "docker": None, "addresses": [None, {"addr": 5}], "policy": {"model_roots": [{"id": None}, 7],
                                       "scripts": [{"name": None}, {"name": "ok", "actions": [None, {"name": 5}]}]}})
    nasty_sample = proto.sanitize_push({"type": "sample", "gpus": {"g": {"procs": [None, {"pid": "x"}, {"pid": 1, "used_bytes": "big"}]}},
                                        "mem": "lots", "cpu": [1]})
    assert nasty_hello is not None and nasty_sample is not None
    d = data(h=nasty_hello, s=nasty_sample, d={"mounts": [None, {"mount": 5}], "t": None}, inv={"scripts": [None, 1]})
    s = fs.build_summary(d)
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": d})
    assert s.gpus and s.gpus[0].uuid == "g" and detail.scripts[0].name == "ok" and detail.scripts[0].actions == []


# ── labels ──────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("args,expected", [
    (dict(mount="/data/fast", fstype="zfs", source="tank/models"), "tank (zfs)"),
    (dict(mount="/data/fast", fstype="zfs", source="tank"), "tank (zfs)"),
    (dict(mount="/data/fast", fstype="zfs", source="tank@snap"), "tank (zfs)"),
    (dict(mount="/data/fast", fstype="zfs", source=None), "/data/fast (zfs)"),
    (dict(mount="/", fstype="ext4", model="Samsung 990"), "/ (ext4, Samsung 990)"),
    (dict(mount="/data", fstype="xfs", model=None), "/data (xfs)"),
    (dict(mount="/data", fstype=None, model="Some Disk"), "/data (Some Disk)"),
    (dict(mount="/data", fstype=None), "/data"),
    (dict(mount="/mnt/m", fstype="nfs4", remote=True, server="10.0.0.9"), "NFS from 10.0.0.9"),
    (dict(mount="/mnt/m", fstype="nfs", remote=True, server="10.0.0.9", server_label="GPU box 2"), "NFS from GPU box 2"),
    (dict(mount="/mnt/m", fstype="cifs", remote=True, server="nas"), "CIFS from nas"),
    (dict(mount="/mnt/m", fstype="fuse.sshfs", remote=True, server="box"), "FUSE.SSHFS from box"),
    (dict(mount="/mnt/m", fstype=None, remote=True, source="nas:/export"), "Network from nas:/export"),
])
def test_mount_labels(args: dict[str, Any], expected: str) -> None:
    base: dict[str, Any] = dict(mount="/", fstype=None, source=None, model=None, remote=False, server=None, server_label=None)
    assert fs.mount_label(**{**base, **args}) == expected


# ── detail: disks, roots, shared storage ────────────────────────────────────────────────────────────────

def test_disks_become_mountinfo_with_human_labels() -> None:
    h = hello(roots=[root("hf", "/data/hf")])
    d = disks([mount("/", model="Samsung 990", device="nvme0n1", source="/dev/nvme0n1p2"),
               mount("/data", fstype="zfs", source="tank/data", readonly=True),
               mount("/mnt/n", fstype="nfs4", source="10.0.0.9:/export", remote=True, remote_host="10.0.0.9", remote_path="/export")])
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h, d=d)})
    root_m, data_m, nfs_m = detail.disks
    assert root_m.label == "/ (ext4, Samsung 990)" and root_m.model == "Samsung 990" and root_m.device == "nvme0n1"
    assert (root_m.size_bytes, root_m.used_bytes, root_m.free_bytes) == (100 * GIB, 40 * GIB, 60 * GIB) and root_m.readonly is False
    assert data_m.label == "tank (zfs)" and data_m.readonly is True
    assert nfs_m.label == "NFS from 10.0.0.9" and nfs_m.remote is True and nfs_m.remote_path == "/export" and nfs_m.remote_host_id is None
    assert [m.mount for m in detail.disks] == ["/", "/data", "/mnt/n"]            # the node's order is kept


def test_a_root_lives_on_the_mount_with_the_longest_matching_path() -> None:
    h = hello(roots=[root("a", "/data/models/hf"), root("b", "/data2/x"), root("c", "/other"), root("d", "/data/models")])
    d = disks([mount("/"), mount("/data"), mount("/data/models")])
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h, d=d)})
    by = {r.id: r.mount.mount if r.mount else None for r in detail.roots}
    assert by == {"a": "/data/models", "b": "/", "c": "/", "d": "/data/models"}          # "/data2/x" is NOT under "/data"
    assert {m.mount: m.roots for m in detail.disks} == {"/": ["b", "c"], "/data": [], "/data/models": ["a", "d"]}
    assert all(r.visible_on == ["gpu-box-2"] and r.shared_from is None for r in detail.roots)


def test_the_mount_the_node_names_wins_over_path_matching() -> None:
    h = hello(roots=[root("hf", "/data/hf")])
    d = disks([mount("/"), mount("/data", roots=[]), mount("/bind/elsewhere", roots=["hf"])])
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h, d=d)})
    assert detail.roots[0].mount is not None and detail.roots[0].mount.mount == "/bind/elsewhere"


def test_roots_keep_their_policy_fields_and_survive_missing_disks() -> None:
    h = hello(roots=[root("hf", "/data/hf", kind="gguf", writable=False, label=None), {"id": "bad", "path": "relative"}, {"path": "/x"}])
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h)})
    assert len(detail.roots) == 1                                                 # relative and id-less roots are not usable
    r = detail.roots[0]
    assert (r.id, r.path, r.kind, r.writable, r.label, r.mount) == ("hf", "/data/hf", "gguf", False, None, None)


def two_hosts(b_root_path: str = "/mnt/a-hf", export: str = "/data/hf", a_root: str = "/data/hf", server: str = "10.0.0.5",
              fstype: str = "nfs4") -> dict[str, HostData]:
    a = data("gpu-box-2", label="GPU box 2",
             h=hello(addresses=[("10.0.0.5", "eth0"), ("172.17.0.1", "docker0")], roots=[root("hf", a_root)]),
             d=disks([mount("/"), mount("/data", fstype="zfs", source="pool/data")]))
    b = data("local", label="Workstation", h=hello(addresses=[("10.0.0.7", "eth0"), ("172.17.0.1", "docker0")], hostname="workstation",
                                                    roots=[root("shared", b_root_path)]),
             d=disks([mount("/"), mount("/mnt/a-hf", fstype=fstype, source=f"{server}:{export}", remote=True,
                                        remote_host=server, remote_path=export)]))
    return {"local": b, "gpu-box-2": a}


def test_an_nfs_root_is_the_other_hosts_storage() -> None:
    fleet = two_hosts()
    b = fs.build_detail("local", fleet)
    a = fs.build_detail("gpu-box-2", fleet)
    (b_root,) = b.roots
    assert b_root.shared_from == "gpu-box-2" and b_root.visible_on == ["local", "gpu-box-2"]
    assert b_root.mount is not None and b_root.mount.label == "NFS from GPU box 2" and b_root.mount.remote_host_id == "gpu-box-2"
    (a_root,) = a.roots
    assert a_root.shared_from is None and a_root.visible_on == ["gpu-box-2", "local"]             # this host first
    assert a_root.mount is not None and a_root.mount.label == "pool (zfs)"                        # /data/hf lives on the zfs mount


@pytest.mark.parametrize("b_root,export,a_root", [
    ("/mnt/a-hf", "/data/hf", "/data/hf"),                 # the export is the root
    ("/mnt/a-hf/hf", "/data", "/data/hf"),                 # the export is a parent of the root, mounted somewhere else
    ("/mnt/a-hf/hub", "/data/hf", "/data/hf"),             # the root is a folder inside the exported root
    ("/mnt/a-hf", "/data", "/data/hf"),                    # the whole export holds the other host's root
])
def test_shared_storage_is_found_whichever_way_the_paths_overlap(b_root: str, export: str, a_root: str) -> None:
    fleet = two_hosts(b_root_path=b_root, export=export, a_root=a_root)
    b = fs.build_detail("local", fleet)
    assert b.roots[0].shared_from == "gpu-box-2" and b.roots[0].visible_on == ["local", "gpu-box-2"]
    assert fs.build_detail("gpu-box-2", fleet).roots[0].visible_on == ["gpu-box-2", "local"]


def test_overlap_is_by_whole_path_components() -> None:
    fleet = two_hosts(export="/data/hf2", a_root="/data/hf")                  # "/data/hf2" is not inside "/data/hf"
    b = fs.build_detail("local", fleet)
    assert b.roots[0].shared_from is None and b.roots[0].visible_on == ["local"]
    assert fs.build_detail("gpu-box-2", fleet).roots[0].visible_on == ["gpu-box-2"]


def test_a_mount_of_an_export_that_holds_none_of_the_other_hosts_roots_is_not_shared() -> None:
    fleet = two_hosts(export="/srv/backups", a_root="/data/hf")
    b = fs.build_detail("local", fleet)
    assert b.roots[0].shared_from is None
    assert b.roots[0].mount is not None and b.roots[0].mount.remote_host_id == "gpu-box-2"      # still said to be from that machine
    assert b.roots[0].mount.label == "NFS from GPU box 2"


def test_a_server_that_is_not_a_configured_host_is_just_an_address() -> None:
    fleet = two_hosts(server="10.0.0.99")
    b = fs.build_detail("local", fleet)
    r = b.roots[0]
    assert r.shared_from is None and r.visible_on == ["local"]
    assert r.mount is not None and r.mount.label == "NFS from 10.0.0.99" and r.mount.remote_host_id is None


def test_only_nfs_is_matched_to_a_host_by_path() -> None:
    fleet = two_hosts(fstype="cifs")
    assert fs.build_detail("local", fleet).roots[0].shared_from is None


def test_a_name_instead_of_an_address_matches_by_hostname_case_insensitively() -> None:
    fleet = two_hosts(server="GPU-Box-2")
    fleet["gpu-box-2"] = data("gpu-box-2", label="GPU box 2", h=hello(addresses=[("10.0.0.5", "eth0")], hostname="gpu-box-2.lab.example",
                                                                       roots=[root("hf", "/data/hf")]),
                              d=disks([mount("/")]))
    b = fs.build_detail("local", fleet)
    assert b.roots[0].shared_from == "gpu-box-2" and b.roots[0].mount is not None and b.roots[0].mount.remote_host_id == "gpu-box-2"


def test_an_address_two_hosts_both_report_identifies_nobody() -> None:
    fleet = two_hosts(server="172.17.0.1")                                      # a docker bridge address, present on both
    assert fs.build_detail("local", fleet).roots[0].shared_from is None
    assert fs.build_detail("local", fleet).roots[0].mount is not None
    assert fs.build_detail("local", fleet).roots[0].mount.remote_host_id is None   # type: ignore[union-attr]


def test_loopback_servers_and_a_host_mounting_itself_are_never_matched() -> None:
    fleet = two_hosts(server="127.0.0.1")
    assert fs.build_detail("local", fleet).roots[0].shared_from is None
    own = two_hosts(server="10.0.0.7")                                          # the workstation's own address
    assert fs.build_detail("local", own).roots[0].shared_from is None
    own_mount = fs.build_detail("local", own).roots[0].mount
    assert own_mount is not None and own_mount.remote_host_id is None and own_mount.label == "NFS from 10.0.0.7"


def test_three_hosts_sharing_one_disk_all_see_each_other() -> None:
    fleet = two_hosts()
    fleet["edge"] = data("edge", ssh="ops@10.0.0.8", label="Edge",
                         h=hello(addresses=[("10.0.0.8", "eth0")], hostname="edge", roots=[root("models", "/mnt/models")]),
                         d=disks([mount("/"), mount("/mnt/models", fstype="nfs4", remote=True, remote_host="10.0.0.5",
                                                    remote_path="/data/hf", source="10.0.0.5:/data/hf")]))
    assert fs.build_detail("gpu-box-2", fleet).roots[0].visible_on == ["gpu-box-2", "local", "edge"]
    assert fs.build_detail("local", fleet).roots[0].visible_on == ["local", "gpu-box-2", "edge"]
    assert fs.build_detail("edge", fleet).roots[0].visible_on == ["edge", "gpu-box-2", "local"]
    assert fs.build_detail("edge", fleet).roots[0].shared_from == "gpu-box-2"


def test_a_down_host_still_takes_part_in_the_correlation_from_its_last_known_data() -> None:
    fleet = two_hosts()
    fleet["gpu-box-2"] = data("gpu-box-2", label="GPU box 2", state="down", h=fleet["gpu-box-2"].hello, d=fleet["gpu-box-2"].disks)
    assert fs.build_detail("local", fleet).roots[0].shared_from == "gpu-box-2"


# ── addresses, scripts, images ──────────────────────────────────────────────────────────────────────────
def test_addresses_are_listed_without_loopback() -> None:
    h = hello()
    h["addresses"] = [{"iface": "lo", "addr": "127.0.0.1", "prefix": 8}, {"iface": "eth0", "addr": "10.0.0.5", "prefix": 24},
                      {"iface": "eth0", "addr": "fd00::5", "prefix": None}, {"addr": "::1", "iface": "lo"}, {"addr": "10.0.0.6"}, {"iface": "x"}]
    assert fs.addresses_of(h) == ["10.0.0.5/24 (eth0)", "fd00::5 (eth0)", "10.0.0.6"]
    assert fs.addresses_of(None) == []


def test_scripts_combine_the_policy_with_the_last_reported_status() -> None:
    scripts = [{"name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang", "description": "Starts before the small one.",
                "containers": ["proj-sglang-*"], "status_action": "status",
                "actions": [{"name": "up", "label": "Start", "confirm": True}, {"name": "down", "label": "Stop", "confirm": True, "danger": True},
                            {"name": "switch", "label": "Switch", "args": ["model-a", "model-b"]}]},
               {"name": "other", "actions": []}]
    inv = proto.sanitize_push({"type": "inventory", "t": 1, "scripts": [{"name": "big-model", "status": "up", "detail": "running", "checked_at": 5.5},
                                                                       {"name": "other", "status": "weird"}]})
    h = hello(scripts=scripts)
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h, inv=inv)})
    big, other = detail.scripts
    assert (big.host_id, big.name, big.label, big.engine, big.status, big.detail, big.checked_at) == (
        "gpu-box-2", "big-model", "Big model (2 GPUs)", "sglang", "up", "running", 5.5)
    assert big.containers == ["proj-sglang-*"] and big.description == "Starts before the small one."
    up, down, switch = big.actions
    assert (up.name, up.confirm, up.danger) == ("up", True, False) and (down.confirm, down.danger) == (True, True)
    assert switch.args == ["model-a", "model-b"] and switch.confirm is False
    assert other.label == "other" and other.status == "unknown" and other.description == ""      # unknown status words are not trusted
    assert fs.build_detail("gpu-box-2", {"gpu-box-2": data(h=h)}).scripts[0].status == "unknown"   # nothing reported yet


def test_images_from_a_docker_images_result() -> None:
    result = {"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * GIB, "created": 1.7e9, "arch": "amd64"},
                         {"ref": None, "id": "x"}, {"ref": "bare", "size": None}]}
    assert fs.images_from_result(result) == [
        ImageInfo(ref="vllm/vllm-openai:v0.1", id="sha256:abc", size_bytes=10 * GIB, created=1.7e9, arch="amd64"), ImageInfo(ref="bare")]
    assert fs.images_from_result({}) == []


def test_the_detail_carries_the_whole_summary_and_the_images() -> None:
    d = data(h=hello(), s=sample())
    detail = fs.build_detail("gpu-box-2", {"gpu-box-2": d}, images=[ImageInfo(ref="x:1")],
                             counts=lambda hid: HostCounts(models=3))
    summary = fs.build_summary(d, counts=lambda hid: HostCounts(models=3))
    assert detail.model_dump(include=set(type(summary).model_fields)) == summary.model_dump()
    assert [i.ref for i in detail.images] == ["x:1"] and detail.addresses == ["10.0.0.5/24 (eth0)"]
