"""Machine facts from fake /proc, /sys and /etc trees, fake nvidia-smi and fake ip."""
from __future__ import annotations

import hashlib

import pytest

from ecnode import hostinfo
from ecnode.disks import Disks, mount_of, parse_mountinfo, split_remote
from ecnode.gpus import Gpus
from ecnode.memory import Mem
from ecnode.policy import Root

from .conftest import CID_ENGINE, CID_EXT, GPU_A, GPU_B


def test_host_facts_and_node_id(machine):
    h = hostinfo.host_facts(machine.paths)
    assert h["os"] == "Test Linux 1.0" and h["cpu_model"] == "Test CPU 9000" and h["cpu_cores"] == 2
    assert h["ram_total_bytes"] == 263000000 * 1024 and h["boot_time"] == 1700000000.0
    assert hostinfo.node_id(machine.paths) == hashlib.sha256(b"0123456789abcdef0123456789abcdef").hexdigest()[:16]


def test_addresses_skip_loopback_link_local_and_bridges(fakebin):
    addrs = hostinfo.addresses()
    assert [(a["iface"], a["addr"], a["prefix"]) for a in addrs] == [("eth0", "10.0.0.5", 24),
                                                                     ("eth0", "fd00::5", 64)]


def test_meminfo_load_and_cpu(machine):
    m = Mem(machine.paths)
    info = m.meminfo()
    assert info["total"] == 263000000 * 1024 and info["available"] == 200000000 * 1024
    assert m.loadavg() == (0.5, 0.4, 0.3)
    assert m.cpu_util() is None                          # first call: no delta yet
    (machine.paths.proc / "stat").write_text("cpu  200 0 200 900 0 0 0 0 0 0\nbtime 1700000000\n")
    assert m.cpu_util() == 66.7                          # 200 busy of 300 jiffies


def test_container_memory_from_cgroup_v2_and_rss_fallback(machine):
    m = Mem(machine.paths)
    assert m.container_id_of(4242) == CID_ENGINE and m.container_id_of(4343) == CID_EXT
    assert m.comm(4242) == "vllm"
    cg = m.container_mem(4242)
    assert cg == {"current": 3000000000, "anon": 1000000000, "file": 1500000000, "shmem": 500000000,
                  "kernel": 100000000, "source": "cgroup"}
    assert m.container_cpu_usec(4242) == 5000000
    rss = m.container_mem(4343)                           # no cgroup dir: VmRSS of the tree (4343 + child 4344)
    assert rss is not None and rss["source"] == "rss" and rss["current"] == 3072 * 1024
    assert m.container_mem(0) is None and m.container_mem(77777) is None


@pytest.mark.parametrize("line,expected", [
    (f"0::/user.slice/user-1000.slice/user@1000.service/user.slice/docker-{'1' * 64}.scope", "1" * 64),  # rootless
    (f"0::/system.slice/docker-{'2' * 64}.scope", "2" * 64),                                            # rootful
    (f"0::/docker/{'3' * 64}", "3" * 64),                                                               # cgroupfs
    (f"0::/user.slice/user-1000.slice/user@1000.service/docker/{'4' * 64}", "4" * 64),                  # rootless fs
    (f"12:memory:/docker/{'5' * 64}\n0::/", "5" * 64),                                                  # cgroup v1
    (f"0::/system.slice/docker-{'6' * 64}.scope/init.scope", "6" * 64),                                 # nested unit
    (f"0::/docker/{'7' * 64}/docker/{'8' * 64}", "7" * 64),                                             # docker in docker
    (f"0::/machine.slice/libpod-{'9' * 64}.scope", None),                                               # not docker
    (f"0::/kubepods/besteffort/pod1/{'a' * 64}", None),
    ("0::/user.slice/user-1000.slice/session-2.scope", None),
    (f"0::/system.slice/docker-{'b' * 63}.scope", None),
])
def test_container_id_from_every_cgroup_path_form(machine, line, expected):
    d = machine.paths.proc / "6000"
    d.mkdir()
    (d / "cgroup").write_text(line + "\n")
    assert Mem(machine.paths).container_id_of(6000) == expected


def test_hostile_cgroup_paths_are_ignored(machine):
    proc = machine.paths.proc
    (proc / "5000").mkdir()
    (proc / "5000" / "cgroup").write_text("0::/../../../etc\n")
    assert Mem(machine.paths).cgroup_dir(5000) is None


def test_mountinfo_parsing_and_remote_split():
    mi = parse_mountinfo("36 1 0:5 / /mnt/my\\040models rw - nfs4 [fd00::9]:/export/m rw\n"
                         "37 1 0:6 / /mnt/smb rw - cifs //nas/share/models rw\nbroken line\n")
    assert mi[0].mount == "/mnt/my models" and split_remote(mi[0].fstype, mi[0].source) == ("fd00::9", "/export/m")
    assert split_remote("cifs", "//nas/share/models") == ("nas", "/share/models")
    assert split_remote("fuse.sshfs", "user@box:/srv/x") == ("box", "/srv/x")
    assert mount_of("/mnt/my models/a", mi).mount == "/mnt/my models"


def test_disks_report_roots_drives_and_network_mounts(machine):
    roots = [Root("hf", str(machine.data / "hf")), Root("shared", str(machine.nfs / "sub")),
             Root("gone", "/nonexistent-for-test/x")]
    rep = Disks(machine.paths, statvfs=machine.statvfs).report(roots)
    by = {m["mount"]: m for m in rep["mounts"]}
    assert set(by) == {"/", str(machine.data), str(machine.nfs)}         # no /proc, no squashfs
    root = by["/"]
    assert root["device"] == "nvme0n1p2" and root["model"] == "Test SSD 4TB" and root["roots"] == ["gone"]
    data = by[str(machine.data)]
    assert data["model"] == "Data Drive 8TB" and data["roots"] == ["hf"] and data["size"] == (1 << 30) * 4096
    assert data["free"] == (1 << 28) * 4096 and data["used"] == ((1 << 30) - (1 << 28)) * 4096
    nfs = by[str(machine.nfs)]
    assert nfs["remote"] and nfs["remote_host"] == "10.0.0.9" and nfs["remote_path"] == "/export/models"
    assert nfs["readonly"] and nfs["roots"] == ["shared"] and nfs["model"] is None


def test_gpus_from_nvidia_smi(fakebin):
    g = Gpus(use_nvml=False)
    static, info = g.static()
    assert [x["uuid"] for x in static] == [GPU_A, GPU_B] and info["source"] == "nvidia-smi"
    assert info["driver"] == "595.0" and info["cuda"] == "13.0"
    assert static[0]["total_bytes"] == 97887 * 1024 * 1024 and static[0]["compute_capability"] == "12.0"
    stats, procs = g.sample()
    assert stats[GPU_A]["util_pct"] == 7.0 and stats[GPU_A]["power_w"] == 71.5 and stats[GPU_A]["fan_pct"] == 30
    assert (GPU_A, 4242, 2048 * 1024 * 1024) in procs


def test_unified_memory_gpu_reports_no_memory_numbers(fakebin):
    fakebin.smi(gpus=[{"index": 0, "uuid": GPU_A, "name": "NVIDIA GB10", "memory.total": "[N/A]",
                       "compute_cap": "12.1", "pci.bus_id": "0000000F:01:00.0", "driver_version": "580.0",
                       "utilization.gpu": "3", "memory.used": "[N/A]", "temperature.gpu": "50",
                       "power.draw": "[N/A]", "power.limit": "[N/A]", "fan.speed": "[N/A]"}],
               apps=[{"pid": 4242, "used_memory": "[N/A]", "gpu_uuid": GPU_A}])
    g = Gpus(use_nvml=False)
    static, _ = g.static()
    assert static[0]["unified_memory"] is True and static[0]["total_bytes"] is None
    stats, procs = g.sample()
    assert stats[GPU_A]["mem_used_bytes"] is None and stats[GPU_A]["power_w"] is None
    assert procs == [(GPU_A, 4242, None)]


def test_zero_gpus_and_no_driver(fakebin, monkeypatch, tmp_path):
    fakebin.smi(gpus=[])
    static, info = Gpus(use_nvml=False).static()
    assert static == [] and info["source"] == "none" and "No devices" in info["error"]
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    static, info = Gpus(use_nvml=False).static()
    assert static == [] and "not on PATH" in info["error"]
    assert Gpus(use_nvml=False).sample() == ({}, [])
