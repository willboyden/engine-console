"""Shared storage (docs/ARCHITECTURE.md 10.5): a copy on host B's NFS mount of host A's disk is A's copy, seen by B."""
from __future__ import annotations

from pathlib import Path

import pytest

from engine_console.domain.errors import Conflict
from engine_console.domain.refs import ModelRef

from .conftest import World
from .fakes import FakeNode, RootCfg, entry, gguf_entry, hf_entry, mount_dict
from .helpers import nfs_alpha as alpha
from .helpers import nfs_beta as beta
from .helpers import nfs_mount

HF = ModelRef(format="hf", repo_id="org/m")


async def test_a_copy_on_an_nfs_mount_of_a_configured_host_is_that_hosts_copy(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [hf_entry("hf", "org/m", size=500)]
    b.entries = [hf_entry("shared", "org/m", size=500), hf_entry("own", "org/own", home=False)]
    w = await World(tmp_path, [a, b]).start()
    lib = w.library
    model = next(m for m in lib.models() if m.name == "org/m")
    assert len(model.copies) == 1 and model.copies[0].host_id == "alpha"          # listed once, on the disk that holds it
    assert model.copies[0].visible_on == ["alpha", "beta"] and model.copies[0].path == "/data/hf/hub/models--org--m"
    assert [c.rel for c in lib.copies("beta")] == ["models--org--own"]            # beta's own disk only
    # what beta can load, each in beta's own terms: the shared copy by beta's root and beta's path through the mount
    seen = {c.rel: c for c in lib.visible_copies("beta")}
    assert set(seen) == {"models--org--own", "hub/models--org--m"} and {c.host_id for c in seen.values()} == {"beta"}
    shared = seen["hub/models--org--m"]
    assert (shared.root_id, shared.path, shared.id) == ("shared", "/mnt/alpha/hf/hub/models--org--m", model.copies[0].id)
    assert shared.visible_on == ["alpha", "beta"] and shared.writable is False and shared.host_label == "Beta"
    assert shared.drive is not None and shared.drive.label == "NFS from Alpha" and shared.drive.remote_host_id == "alpha"
    found = lib.find("beta", HF)
    assert found == shared                                                        # what an engine on beta mounts
    owner_view = lib.find("alpha", HF)
    assert owner_view is not None and (owner_view.host_id, owner_view.root_id, owner_view.path) == ("alpha", "hf", "/data/hf/hub/models--org--m")
    assert owner_view.id == shared.id and owner_view.drive.remote is False        # type: ignore[union-attr]
    assert lib.find("beta", HF, own=True) is None and lib.find("alpha", HF, own=True) == owner_view    # a delete only reaches own disks
    assert [m.name for m in lib.models(host="beta")] == ["org/m", "org/own"]       # beta can load it
    assert model.copies[0].host_id == "alpha"                                     # the library page still shows the real copy
    await w.stop()


async def test_use_on_either_machine_marks_the_one_copy_in_use(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    w.busy[("beta", "/mnt/alpha/hf/hub/models--org--m")] = ["inst_on_beta"]      # an engine on beta mounts the share
    c = w.library.models()[0].copies[0]
    assert c.in_use_by == ["inst_on_beta"] and c.deletable is False
    with pytest.raises(Conflict) as ei:
        await w.library.delete_copy(c.id)                                          # alpha's node cannot see beta's containers
    assert ei.value.code == "model_in_use" and "inst_on_beta" in ei.value.detail
    w.busy[("alpha", "/data/hf/hub/models--org--m")] = ["inst_on_alpha"]
    assert w.library.models()[0].copies[0].in_use_by == ["inst_on_beta", "inst_on_alpha"] or \
        set(w.library.models()[0].copies[0].in_use_by) == {"inst_on_beta", "inst_on_alpha"}
    await w.stop()


async def test_gguf_copies_are_found_by_either_machines_coordinates(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [gguf_entry("gguf", "a/x-Q4_K_M.gguf", name="x")]
    b.entries = [gguf_entry("shared_g", "a/x-Q4_K_M.gguf", name="x")]
    w = await World(tmp_path, [a, b]).start()
    assert len(w.library.models()) == 1 and w.library.models()[0].copies[0].visible_on == ["alpha", "beta"]
    by_beta = w.library.find("beta", ModelRef(format="gguf", root_id="shared_g", rel="a/x-Q4_K_M.gguf"))
    by_owner = w.library.find("beta", ModelRef(format="gguf", root_id="gguf", rel="a/x-Q4_K_M.gguf"))     # the UI sends the owner's
    assert by_beta is not None and by_owner is not None and by_owner == by_beta
    assert (by_beta.host_id, by_beta.root_id, by_beta.rel, by_beta.path) == ("beta", "shared_g", "a/x-Q4_K_M.gguf",
                                                                           "/mnt/alpha/gguf/a/x-Q4_K_M.gguf")
    assert by_beta.id == w.library.models()[0].copies[0].id                       # still the one real copy
    on_alpha = w.library.find("alpha", ModelRef(format="gguf", root_id="gguf", rel="a/x-Q4_K_M.gguf"))
    assert on_alpha is not None and (on_alpha.host_id, on_alpha.root_id, on_alpha.path) == ("alpha", "gguf", "/data/gguf/a/x-Q4_K_M.gguf")
    assert w.library.find("alpha", ModelRef(format="gguf", root_id="shared_g", rel="a/x-Q4_K_M.gguf")) is None  # not alpha's root
    assert w.library.find("beta", ModelRef(format="gguf", root_id="shared_g", rel="a/nope.gguf")) is None
    await w.stop()


async def test_a_machine_with_two_roots_on_the_same_storage_gets_the_one_that_naturally_holds_the_model(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = FakeNode(tmp_path, "beta", addresses=["10.0.0.2"], hostname="beta", scan_disk=False, roots=[
        RootCfg("aaa_auto", "auto", writable=False, path=Path("/mnt/alpha/hf/hub")),
        RootCfg("zzz_hub", "hf-hub", writable=False, path=Path("/mnt/alpha/hf/hub"))],
        mounts=[mount_dict("/"), nfs_mount(roots=["aaa_auto", "zzz_hub"])])
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("aaa_auto", "org/m", home=False), hf_entry("zzz_hub", "org/m", home=False)]
    w = await World(tmp_path, [a, b]).start()
    assert len(w.library.models()[0].copies) == 1
    found = w.library.find("beta", HF)
    assert found is not None and found.root_id == "zzz_hub" and found.rel == "models--org--m"      # the hf-hub root, not the auto one
    assert [c.root_id for c in w.library.visible_copies("beta")] == ["zzz_hub"]
    src = w.library.source(found.id)
    assert src is not None and sorted(src.locations) == [("alpha", "hf"), ("beta", "aaa_auto"), ("beta", "zzz_hub")]
    await w.stop()


async def test_an_export_of_a_subdirectory_maps_the_same_way(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = FakeNode(tmp_path, "beta", roots=[RootCfg("shared", "hf-home", writable=False, path=Path("/mnt/hfshare"))],
                 addresses=["10.0.0.2"], hostname="beta", scan_disk=False,
                 mounts=[mount_dict("/"), nfs_mount(remote_path="/data/hf", mount="/mnt/hfshare", roots=["shared"])])
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    c = w.library.models()[0].copies
    assert len(c) == 1 and c[0].visible_on == ["alpha", "beta"]
    await w.stop()


async def test_when_the_owner_is_not_a_configured_host_the_copy_stays_with_the_address(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = beta(tmp_path, nfs_mount(remote_host="10.9.9.9", roots=["shared", "shared_g"]))
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    copies = w.library.models()[0].copies
    assert [c.host_id for c in copies] == ["alpha", "beta"] and all(c.visible_on == [c.host_id] for c in copies)
    drive = copies[1].drive
    assert drive is not None and drive.remote and drive.remote_host == "10.9.9.9" and drive.remote_host_id is None
    assert drive.label == "NFS from 10.9.9.9" and drive.remote_path == "/data"
    await w.stop()


async def test_a_configured_owner_that_does_not_list_the_copy_keeps_the_borrowers_copy(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = []                                                       # the export is outside alpha's model roots
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    copies = w.library.models()[0].copies
    assert [c.host_id for c in copies] == ["beta"] and copies[0].visible_on == ["beta"]
    assert copies[0].drive is not None and copies[0].drive.remote_host_id == "alpha"
    assert copies[0].drive.label == "NFS from Alpha"                      # the owner is known even when it does not list it
    await w.stop()


async def test_an_export_that_does_not_cover_the_copy_is_not_merged(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = beta(tmp_path, nfs_mount(remote_path="/other", roots=["shared"]))
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]            # /other/hf/hub/... is not where alpha's copy is
    w = await World(tmp_path, [a, b]).start()
    assert [c.host_id for c in w.library.models()[0].copies] == ["alpha", "beta"]
    await w.stop()


async def test_an_address_two_machines_report_identifies_nobody(tmp_path: Path) -> None:
    # every machine with docker has 172.17.0.1: an NFS server of that address is not alpha, and not gamma
    a = alpha(tmp_path, addresses=["10.0.0.1", "172.17.0.1"])
    g = FakeNode(tmp_path, "gamma", roots=[RootCfg("hf", "hf-home")], addresses=["10.0.0.3", "172.17.0.1"], scan_disk=False)
    b = beta(tmp_path, nfs_mount(remote_host="172.17.0.1", roots=["shared"]))
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, g, b]).start()
    assert [c.host_id for c in w.library.models()[0].copies] == ["alpha", "beta"]
    assert w.library.models()[0].copies[1].drive.remote_host_id is None  # type: ignore[union-attr]
    await w.stop()


async def test_loopback_nfs_and_non_nfs_remotes_are_never_merged(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = beta(tmp_path, nfs_mount(remote_host="127.0.0.1", roots=["shared"]))
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    assert len(w.library.models()[0].copies) == 2
    await w.stop()
    cifs = mount_dict("/mnt/alpha", fstype="cifs", source="//10.0.0.1/data", model=None, remote=True, remote_host="10.0.0.1",
                      remote_path="/data", roots=["shared"])
    b2 = beta(tmp_path / "x", cifs)
    a2 = alpha(tmp_path / "x")
    a2.entries = [hf_entry("hf", "org/m")]
    b2.entries = [hf_entry("shared", "org/m")]
    w2 = await World(tmp_path / "x", [a2, b2]).start()
    copies = w2.library.models()[0].copies
    assert len(copies) == 2 and copies[1].drive.label == "CIFS from Alpha"          # type: ignore[union-attr]
    await w2.stop()


async def test_the_owner_may_be_named_by_hostname(tmp_path: Path) -> None:
    for server in ("alpha.lan", "ALPHA", "alpha"):     # fully qualified, short, any case
        a = alpha(tmp_path / server)
        b = beta(tmp_path / server, nfs_mount(remote_host=server, roots=["shared"]))
        a.entries = [hf_entry("hf", "org/m")]
        b.entries = [hf_entry("shared", "org/m")]
        w = await World(tmp_path / server, [a, b]).start()
        assert len(w.library.models()[0].copies) == 1, server
        await w.stop()


async def test_the_merge_survives_a_restart_with_the_owner_down(tmp_path: Path) -> None:
    from engine_console.services.store import Store
    db = tmp_path / "console.db"
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w1 = await World(tmp_path, [a, b], store=Store(db)).start()
    await w1.stop()
    a2, b2 = alpha(tmp_path), beta(tmp_path)
    b2.entries = [hf_entry("shared", "org/m")]            # beta is up and lists what it sees; alpha is off
    a2.state = "down"
    a2.forget()
    w2 = await World(tmp_path, [a2, b2], store=Store(db)).start()
    c = w2.library.models()[0].copies
    assert len(c) == 1 and c[0].visible_on == ["alpha", "beta"] and c[0].deletable is False
    await w2.stop()


async def test_a_change_of_mounts_rebuilds_the_merge_without_a_new_scan(tmp_path: Path) -> None:
    a = alpha(tmp_path)
    b = beta(tmp_path, nfs_mount(remote_host="10.9.9.9", roots=["shared"]))      # not yet recognisable
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("shared", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    assert len(w.library.models()[0].copies) == 2
    b._mounts = [mount_dict("/", roots=["own"]), nfs_mount(roots=["shared"])]    # the right server, a minute later
    w.bus.publish("node.disks", {"host_id": "beta", "msg": b.disks()})
    import asyncio
    await asyncio.sleep(0.01)
    assert len(w.library.models()[0].copies) == 1
    await w.stop()


# ---- roots ---------------------------------------------------------------------------------------------------------------------
async def test_roots_report_the_mount_the_owner_and_who_can_read_them(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    w = await World(tmp_path, [a, b]).start()
    shared = w.library.root("beta", "shared")
    assert shared is not None and shared.shared_from == "alpha" and shared.visible_on == ["beta", "alpha"]
    assert shared.mount is not None and shared.mount.label == "NFS from Alpha" and shared.mount.remote_host_id == "alpha"
    own = w.library.root("beta", "own")
    assert own is not None and own.shared_from is None and own.visible_on == ["beta"] and own.mount.mount == "/"  # type: ignore[union-attr]
    owner_root = w.library.root("alpha", "hf")
    assert owner_root is not None and owner_root.shared_from is None and owner_root.visible_on == ["alpha", "beta"]
    assert w.library.root("alpha", "nope") is None and w.library.roots("nobody") == []
    await w.stop()


async def test_hf_root_is_the_first_hf_root_and_auto_roots_count_when_their_scan_found_one(tmp_path: Path) -> None:
    a = FakeNode(tmp_path, "alpha", scan_disk=False, roots=[RootCfg("g", "gguf"), RootCfg("auto1", "auto"),
                                                           RootCfg("h", "hf-hub"), RootCfg("ol", "ollama")])
    b = FakeNode(tmp_path, "beta", scan_disk=False, roots=[RootCfg("auto2", "auto"), RootCfg("g2", "gguf")])
    a.entries = [entry("auto1", "hub/models--o--n", "hf", "o/n")]
    b.entries = [gguf_entry("auto2", "x.gguf")]
    w = await World(tmp_path, [a, b]).start()
    assert w.library.hf_root("alpha").id == "auto1"                        # type: ignore[union-attr]  # first hf root, in policy order
    assert w.library.hf_root("beta") is None                                # an auto root with only gguf files is not HF
    assert w.library.ollama_root("alpha").id == "ol" and w.library.ollama_root("beta") is None  # type: ignore[union-attr]
    assert w.library.hf_root("nobody") is None
    usage = w.library.root_usage("alpha", "auto1")
    assert usage.hf_home and not usage.hf_hub and usage.entries == 1
    await w.stop()


async def test_a_mount_the_node_claims_for_a_root_wins_over_the_longest_prefix(tmp_path: Path) -> None:
    b = FakeNode(tmp_path, "beta", scan_disk=False, roots=[RootCfg("r", "hf-home", path=Path("/data/hf"))],
                 mounts=[mount_dict("/", roots=[]), mount_dict("/data", fstype="zfs", source="tank/models", model=None, roots=["r"]),
                         mount_dict("/data/hf", fstype="ext4", roots=[])])
    w = await World(tmp_path, [b]).start()
    r = w.library.root("beta", "r")
    assert r is not None and r.mount is not None and r.mount.mount == "/data" and r.mount.label == "tank (zfs)"
    await w.stop()


@pytest.mark.parametrize(("mount", "label"), [
    (mount_dict("/", fstype="ext4", model="Samsung 990"), "/ (ext4, Samsung 990)"),
    (mount_dict("/", fstype="ext4", model=None), "/ (ext4)"),
    (mount_dict("/data", fstype="zfs", source="tank/models", model=None), "tank (zfs)"),
    (mount_dict("/data", fstype="zfs", source="tank", model=None), "tank (zfs)"),
    (mount_dict("/mnt/x", fstype="nfs4", model=None, remote=True, remote_host="nas.lan", remote_path="/x"), "NFS from nas.lan"),
    (mount_dict("/mnt/x", fstype="cifs", model=None, remote=True, remote_host="nas.lan", remote_path="share"), "CIFS from nas.lan"),
    (mount_dict("/mnt/x", fstype=None, model=None), "/mnt/x"),
])
async def test_drive_labels(tmp_path: Path, mount: dict[str, object], label: str) -> None:
    a = FakeNode(tmp_path, "alpha", scan_disk=False, roots=[RootCfg("r", "hf-home", path=Path(str(mount["mount"]) + "/hf"))],
                 mounts=[mount])
    a.entries = [hf_entry("r", "org/m")]
    w = await World(tmp_path, [a]).start()
    drive = w.library.models()[0].copies[0].drive
    assert drive is not None and drive.label == label
    await w.stop()


async def test_other_entries_do_not_break_the_merge(tmp_path: Path) -> None:
    a, b = alpha(tmp_path), beta(tmp_path)
    a.entries = [entry("hf", "weird", "other", "weird")]
    b.entries = [entry("shared", "weird", "other", "weird")]
    w = await World(tmp_path, [a, b]).start()
    assert w.library.models()[0].copies[0].visible_on == ["alpha", "beta"]
    await w.stop()
