"""The library: fetching and paging, the stored snapshot of a host that is down, grouping, the live overlay (in use,
deletable), find(), rescan and delete."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from engine_console.domain.errors import Conflict, NotFound, ProblemError
from engine_console.domain.refs import ModelRef
from engine_console.services.library import copy_id
from engine_console.services.store import Store

from .conftest import World
from .fakes import COMMIT, FakeNode, RootCfg, entry, gguf_entry, hf_entry


def node(tmp_path: Path, host_id: str = "alpha", **kw: object) -> FakeNode:
    roots = kw.pop("roots", None) or [RootCfg("hf", "hf-home"), RootCfg("gguf", "gguf"), RootCfg("ro", "plain", writable=False)]
    kw.setdefault("scan_disk", False)
    return FakeNode(tmp_path, host_id, roots=roots, **kw)  # type: ignore[arg-type]


async def test_fetch_pages_through_models_list_until_the_cursor_ends(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", f"org/m{i}") for i in range(5)]
    w = await World(tmp_path, [a], page_limit=2).start()
    pages = [c for op, c in a.calls if op == "models.list"]
    assert pages == [{"limit": 2}, {"limit": 2, "cursor": 2}, {"limit": 2, "cursor": 4}]
    assert [m.name for m in w.library.models()] == [f"org/m{i}" for i in range(5)]
    await w.stop()


async def test_a_scan_finishing_between_two_pages_restarts_the_pass(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", f"org/m{i}") for i in range(3)]
    a.list_generation_bump_once = True
    w = await World(tmp_path, [a], page_limit=2).start()
    lists = [c for op, c in a.calls if op == "models.list"]
    assert len(lists) == 4 and lists[2] == {"limit": 2}        # first pass abandoned at page two, then a full second pass
    assert len(w.library.models()) == 3
    await w.stop()


async def test_a_cursor_that_does_not_advance_is_not_followed_forever(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m0")]
    real = a._op_models_list

    def stuck(args: dict) -> dict:  # type: ignore[type-arg]
        out = real(args)
        out["next_cursor"] = 1
        return out

    a._op_models_list = stuck  # type: ignore[method-assign]
    w = await World(tmp_path, [a]).start()
    assert len([1 for op, _ in a.calls if op == "models.list"]) == 2
    assert [m.name for m in w.library.models()] == ["org/m0"]
    await w.stop()


async def test_copy_id_is_the_first_16_hex_of_sha1_of_host_root_rel(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    rel = "hub/models--org--m"
    want = hashlib.sha1(f"alpha|hf|{rel}".encode()).hexdigest()[:16]  # noqa: S324
    assert w.library.models()[0].copies[0].id == want == copy_id("alpha", "hf", rel)
    await w.stop()


async def test_copy_fields_path_drive_and_policy_flags(tmp_path: Path) -> None:
    a = node(tmp_path, label="Alpha box")
    a.entries = [hf_entry("hf", "org/m", size=1234, files=7, revision=COMMIT)]
    w = await World(tmp_path, [a]).start()
    c = w.library.models()[0].copies[0]
    root = a.roots["hf"].path
    assert c.host_id == "alpha" and c.host_label == "Alpha box" and c.root_id == "hf" and c.root_label == "hf"
    assert c.path == f"{root}/hub/models--org--m" and c.size_bytes == 1234 and c.files == 7
    assert c.mtime == 10.0 and c.atime == 20.0 and c.writable and c.deletable and c.in_use_by == []
    assert c.visible_on == ["alpha"]
    assert c.drive is not None and c.drive.mount == "/" and c.drive.label == "/ (ext4, Test SSD)"
    await w.stop()


async def test_grouping_keys_per_format_and_engines(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [
        hf_entry("hf", "org/m1"),
        gguf_entry("gguf", "a/Qwen3-8B-Q4_K_M.gguf", name="Qwen3-8B", quant="Q4_K_M"),
        gguf_entry("gguf", "a/Qwen3-8B-Q8_0.gguf", name="Qwen3-8B", quant="Q8_0"),
        entry("ro", "llama-dir", "plain", "llama-dir"),
        entry("ro", "manifests/x", "ollama", "llama3:8b"),
        entry("ro", "stuff", "other", "stuff"),
        entry("ro", "weird", "bogus-format", "x"),                        # a format this hub does not know is "other"
        entry("gguf", "loose/Phi-4-mini.Q6_K.gguf", "gguf", None, quant=None),     # no name, no quant: both derived
    ]
    b.entries = [hf_entry("hf", "org/m1", size=999)]
    w = await World(tmp_path, [a, b]).start()
    by_key = {m.key: m for m in w.library.models()}
    assert set(by_key) == {"hf:org/m1", "gguf:Qwen3-8B:Q4_K_M", "gguf:Qwen3-8B:Q8_0", "plain:llama-dir", "ollama:llama3:8b",
                           "other:stuff", "other:x", "gguf:Phi-4-mini.Q6_K:Q6_K"}
    assert [c.host_id for c in by_key["hf:org/m1"].copies] == ["alpha", "beta"]       # one model, two machines
    assert by_key["hf:org/m1"].size_bytes == 1000                                    # the largest copy
    assert by_key["gguf:Qwen3-8B:Q4_K_M"].quant == "Q4_K_M" and by_key["gguf:Qwen3-8B:Q4_K_M"].format == "gguf"
    assert by_key["hf:org/m1"].engines == ["vllm"] and by_key["gguf:Qwen3-8B:Q8_0"].engines == ["vllm", "llamacpp"]
    assert by_key["plain:llama-dir"].engines == ["llamacpp"] and by_key["ollama:llama3:8b"].engines == ["ollama"]
    assert by_key["other:stuff"].engines == [] and by_key["other:x"].format == "other"
    await w.stop()


async def test_a_quant_already_in_the_name_is_not_added_twice(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [gguf_entry("gguf", "x/m.gguf", name="m:Q4_K_M", quant="Q4_K_M")]
    w = await World(tmp_path, [a]).start()
    assert [m.key for m in w.library.models()] == ["gguf:m:Q4_K_M"]
    await w.stop()


async def test_model_fields_come_from_the_first_copy_that_has_them(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "org/m")]
    b.entries = [hf_entry("hf", "org/m", params=8_000_000_000, arch="Qwen3ForCausalLM", context=40960, shards=4)]
    w = await World(tmp_path, [a, b]).start()
    m = w.library.models()[0]
    assert (m.params, m.arch, m.context, m.shards, m.revision) == (8_000_000_000, "Qwen3ForCausalLM", 40960, 4, COMMIT)
    await w.stop()


async def test_copies_at_different_revisions_are_flagged(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "org/m", revision="a" * 40)]
    b.entries = [hf_entry("hf", "org/m", revision="b" * 40)]
    w = await World(tmp_path, [a, b]).start()
    assert any("different revisions" in x for x in w.library.models()[0].warnings)
    await w.stop()


async def test_entries_that_break_the_nodes_own_rules_are_not_listed(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/ok"), entry("hf", "../etc", "plain", "evil"), entry("nope", "x", "plain", "ghost"),
                 entry("hf", "/abs", "plain", "abs"), entry("hf", "a//b", "plain", "empty-seg")]
    w = await World(tmp_path, [a]).start()
    assert [m.name for m in w.library.models()] == ["org/ok"]
    await w.stop()


async def test_incomplete_scans_and_node_warnings_surface_on_the_model(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m", warnings=["no config.json"])]
    real = a._op_models_list

    def with_scan_error(args: dict) -> dict:  # type: ignore[type-arg]
        out = real(args)
        out["roots"] = [{"root": "hf", "scanned_at": 1.0, "complete": False, "entries": 1, "error": "too many entries"}]
        return out

    a._op_models_list = with_scan_error  # type: ignore[method-assign]
    w = await World(tmp_path, [a]).start()
    warnings = w.library.models()[0].warnings
    assert "no config.json" in warnings and any("did not finish: too many entries" in x for x in warnings)
    await w.stop()


# ---- stored snapshots: a machine that is down still shows its models ----------------------------------------------------------------
async def test_a_host_that_is_down_shows_its_last_known_library_stale(tmp_path: Path) -> None:
    db = tmp_path / "console.db"
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m"), gguf_entry("gguf", "x/q.gguf")]
    w1 = await World(tmp_path, [a], store=Store(db)).start()
    first = {m.key for m in w1.library.models()}
    await w1.stop()

    # the hub restarts while the machine is off: no hello, no disks, link down
    a2 = node(tmp_path)
    a2.entries = []
    a2.state = "down"
    a2.forget()
    w2 = await World(tmp_path, [a2], store=Store(db)).start()
    models = w2.library.models()
    assert {m.key for m in models} == first
    for m in models:
        assert any("Alpha is down" in x and "last-known" in x for x in m.warnings)
        for c in m.copies:
            assert c.deletable is False and c.host_id == "alpha" and c.path.endswith(("models--org--m", "x/q.gguf"))
            assert c.drive is not None and c.drive.mount == "/"        # drives come from the stored mounts too
    assert w2.library.status()["alpha"]["stale"] is True
    assert [op for op, _ in a2.calls if op == "models.list"] == []        # nothing was asked of a machine that is down
    with pytest.raises(ProblemError) as ei:
        await w2.library.delete_copy(models[0].copies[0].id)
    assert ei.value.status == 503
    await w2.stop()


async def test_stored_snapshots_of_hosts_that_are_no_longer_configured_are_dropped(tmp_path: Path) -> None:
    db = tmp_path / "console.db"
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries, b.entries = [hf_entry("hf", "org/a")], [hf_entry("hf", "org/b")]
    w1 = await World(tmp_path, [a, b], store=Store(db)).start()
    await w1.stop()
    w2 = await World(tmp_path, [node(tmp_path)], store=Store(db)).start()
    assert w2.store.one("SELECT COUNT(*) AS n FROM library_snapshots")["n"] == 1  # type: ignore[index]
    await w2.stop()


async def test_an_unreadable_stored_snapshot_is_dropped_not_fatal(tmp_path: Path) -> None:
    store = Store(":memory:")
    store.execute("INSERT INTO library_snapshots(host_id,generation,roots_json,entries_json,updated_at) VALUES('alpha',1,'[]','{not json',1)")
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a], store=store).start()
    assert [m.name for m in w.library.models()] == ["org/m"]
    await w.stop()


async def test_the_snapshot_is_replaced_on_every_fetch(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    a.set_entries([hf_entry("hf", "org/m"), hf_entry("hf", "org/n")])
    await w.library.refresh("alpha")
    row = w.store.one("SELECT generation, entries_json FROM library_snapshots WHERE host_id='alpha'")
    assert row is not None and row["generation"] == a.generation and row["entries_json"].count('"org/n"') == 1
    await w.stop()


# ---- following the nodes -----------------------------------------------------------------------------------------------------------
async def test_models_changed_triggers_a_refetch(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    a.set_entries([hf_entry("hf", "org/m"), hf_entry("hf", "org/new")])
    a.announce_models_changed()          # right after start(): the subscription must already be in place
    await asyncio.sleep(0.01)
    await w.settle()
    assert {m.name for m in w.library.models()} == {"org/m", "org/new"}
    await w.stop()


async def test_a_burst_of_models_changed_events_collapses_into_few_fetches(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    before = len([1 for op, _ in a.calls if op == "models.list"])
    for _ in range(20):
        a.announce_models_changed()
    await asyncio.sleep(0.05)
    await w.settle()
    assert len([1 for op, _ in a.calls if op == "models.list"]) - before <= 3
    await w.stop()


async def test_a_host_coming_up_is_fetched_and_a_host_that_is_down_at_start_is_not(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta", state="down")
    a.entries, b.entries = [hf_entry("hf", "org/a")], [hf_entry("hf", "org/b")]
    w = await World(tmp_path, [a, b]).start()
    assert {m.name for m in w.library.models()} == {"org/a"} and not [1 for op, _ in b.calls if op == "models.list"]
    w.fleet.set_state("beta", "up")
    await asyncio.sleep(0.01)
    await w.settle()
    assert {m.name for m in w.library.models()} == {"org/a", "org/b"}
    await w.stop()


async def test_a_failed_fetch_keeps_the_last_snapshot_and_retries_while_the_host_is_up(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    real = a._op_models_list
    fails = {"n": 2}

    def flaky(args: dict) -> dict:  # type: ignore[type-arg]
        if fails["n"]:
            fails["n"] -= 1
            raise a._err("io_error", "the scan is broken")
        return real(args)

    a._op_models_list = flaky  # type: ignore[method-assign]
    a.set_entries([hf_entry("hf", "org/m"), hf_entry("hf", "org/n")])
    a.announce_models_changed()
    await asyncio.sleep(0.01)
    assert {m.name for m in w.library.models()} == {"org/m"} and w.library.status()["alpha"]["error"]
    await asyncio.sleep(0.1)
    await w.settle()
    assert {m.name for m in w.library.models()} == {"org/m", "org/n"} and w.library.status()["alpha"]["error"] is None
    await w.stop()


async def test_a_library_event_is_published_when_a_snapshot_changes(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = World(tmp_path, [a])
    seen: list[object] = []

    async def listen() -> None:
        async for ev in w.bus.subscribe("library"):
            seen.append(ev)

    t = asyncio.create_task(listen())
    await asyncio.sleep(0)
    await w.start()
    await asyncio.sleep(0)
    t.cancel()
    assert seen and seen[0]["host_id"] == "alpha" and seen[0]["models"] == 1  # type: ignore[index]
    await w.stop()


# ---- the live overlay ----------------------------------------------------------------------------------------------------------------
async def test_in_use_by_comes_from_the_callback_and_blocks_delete(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m"), hf_entry("hf", "org/free")]
    w = await World(tmp_path, [a]).start()
    path = w.library.find("alpha", ModelRef(format="hf", repo_id="org/m")).path  # type: ignore[union-attr]
    w.busy[("alpha", path)] = ["inst_1"]
    c = w.library.find("alpha", ModelRef(format="hf", repo_id="org/m"))
    assert c is not None and c.in_use_by == ["inst_1"] and c.deletable is False
    free = w.library.find("alpha", ModelRef(format="hf", repo_id="org/free"))
    assert free is not None and free.deletable is True
    w.busy.clear()                                   # the instance stopped: no event needed, the next read is current
    assert w.library.find("alpha", ModelRef(format="hf", repo_id="org/m")).deletable is True  # type: ignore[union-attr]
    await w.stop()


async def test_a_broken_in_use_callback_does_not_blank_the_library(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]

    def boom(host: str, path: str) -> list[str]:
        raise RuntimeError("bug")

    w = await World(tmp_path, [a], in_use=boom).start()
    assert [m.name for m in w.library.models()] == ["org/m"]
    await w.stop()


@pytest.mark.parametrize(("mode", "allow_delete", "writable", "state", "deletable"), [
    ("manage", True, True, "up", True),
    ("manage", False, True, "up", False),      # the node's policy forbids deletes
    ("monitor", True, True, "up", False),      # monitor mode changes nothing
    ("manage", True, False, "up", False),      # a read-only root
    ("manage", True, True, "down", False),     # a machine that is not connected
])
async def test_deletable_needs_policy_a_writable_root_a_live_link_and_no_user(
        tmp_path: Path, mode: str, allow_delete: bool, writable: bool, state: str, deletable: bool) -> None:
    a = node(tmp_path, mode=mode, allow_delete=allow_delete, roots=[RootCfg("hf", "hf-home", writable=writable)])
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    a.state = state
    c = w.library.models()[0].copies[0]
    assert c.deletable is deletable and c.writable is writable
    await w.stop()


async def test_a_policy_change_shows_without_a_new_scan(tmp_path: Path) -> None:
    a = node(tmp_path, allow_delete=False)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    assert w.library.models()[0].copies[0].deletable is False
    a.allow_delete = True
    assert w.library.models()[0].copies[0].deletable is True
    await w.stop()


# ---- find ---------------------------------------------------------------------------------------------------------------------------
async def test_find_by_format(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "org/m"), gguf_entry("gguf", "x/q-Q4_K_M.gguf"), entry("ro", "pdir", "plain", "pdir"),
                 entry("ro", "manifests/l", "ollama", "llama3:8b")]
    b.entries = [hf_entry("hf", "org/m"), gguf_entry("gguf", "x/q-Q4_K_M.gguf")]
    w = await World(tmp_path, [a, b]).start()
    lib = w.library
    assert lib.find("alpha", ModelRef(format="hf", repo_id="org/m")).host_id == "alpha"        # type: ignore[union-attr]
    assert lib.find("beta", ModelRef(format="hf", repo_id="org/m")).host_id == "beta"          # its own copy first
    assert lib.find("alpha", ModelRef(format="hf", repo_id="org/none")) is None
    g = lib.find("beta", ModelRef(format="gguf", root_id="gguf", rel="x/q-Q4_K_M.gguf"))
    assert g is not None and g.host_id == "beta" and g.root_id == "gguf"
    assert lib.find("alpha", ModelRef(format="gguf", root_id="gguf", rel="x/other.gguf")) is None
    assert lib.find("alpha", ModelRef(format="plain", root_id="ro", rel="pdir")) is not None
    assert lib.find("alpha", ModelRef(format="gguf", root_id="ro", rel="pdir")) is None          # right path, wrong format
    assert lib.find("alpha", ModelRef(format="ollama", name="llama3:8b")) is not None
    assert lib.find("beta", ModelRef(format="ollama", name="llama3:8b")) is None                # alpha's store, not visible to beta
    await w.stop()


async def test_an_hf_directory_with_no_snapshot_is_listed_but_is_not_a_model_an_engine_can_load(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    cut_off = hf_entry("hf", "org/m", revision=None, files=0, size=500, warnings=["no snapshot: the download has not finished"])
    a.entries = [cut_off]
    b.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a, b]).start()
    m = w.library.models()[0]
    assert [c.host_id for c in m.copies] == ["alpha", "beta"] and "no snapshot: the download has not finished" in m.warnings
    ref = ModelRef(format="hf", repo_id="org/m")
    assert w.library.find("alpha", ref) is None                                        # not loadable there ...
    assert w.library.find("beta", ref).host_id == "beta"                              # type: ignore[union-attr]  # ... but fine on beta
    assert [c.rel for c in w.library.copies("alpha")] == ["hub/models--org--m"]        # still shown as what is on that disk
    await w.stop()


async def test_copies_lists_what_is_physically_on_a_host(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "org/a1"), hf_entry("hf", "org/a2")]
    b.entries = [hf_entry("hf", "org/b1")]
    w = await World(tmp_path, [a, b]).start()
    assert [c.host_id for c in w.library.copies("alpha")] == ["alpha", "alpha"] and len(w.library.copies("beta")) == 1
    assert w.library.copies("nobody") == [] and w.library.copy("0" * 16) is None and w.library.copy("not-an-id") is None
    await w.stop()


# ---- the list API ----------------------------------------------------------------------------------------------------------------------
async def test_models_filters_by_host_format_and_query_words(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "Qwen/Qwen3-8B"), gguf_entry("gguf", "x/Llama-3-Q4_K_M.gguf", name="Llama-3")]
    b.entries = [hf_entry("hf", "Qwen/Qwen3-8B"), hf_entry("hf", "org/Gemma-3")]
    w = await World(tmp_path, [a, b]).start()
    lib = w.library
    assert [m.name for m in lib.models(host="beta")] == ["org/Gemma-3", "Qwen/Qwen3-8B"]
    assert [m.name for m in lib.models(format="gguf")] == ["Llama-3"]
    assert [m.name for m in lib.models(q="qwen 8b")] == ["Qwen/Qwen3-8B"]
    assert [m.name for m in lib.models(q="Q4_K_M")] == ["Llama-3"]                       # the quantisation is searchable
    assert lib.models(q="nothing like this") == []
    assert [c.host_id for c in lib.models(host="beta", q="qwen")[0].copies] == ["alpha", "beta"]   # all copies stay listed
    assert lib.model("hf:Qwen/Qwen3-8B") is not None and lib.model("hf:nope") is None
    await w.stop()


async def test_local_models_is_the_v1_shape_for_every_host(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    a.entries = [hf_entry("hf", "org/m", size=10), gguf_entry("gguf", "x/q.gguf")]
    b.entries = [hf_entry("hf", "org/m", size=20, revision="b" * 40)]
    w = await World(tmp_path, [a, b]).start()
    rows = w.library.local_models()
    assert [(m.repo_id, m.host_id, m.size_bytes, m.revisions) for m in rows] == [
        ("org/m", "alpha", 10, [COMMIT]), ("org/m", "beta", 20, ["b" * 40])]
    assert rows[0].last_used == 20.0 and rows[0].path.endswith("hub/models--org--m")
    await w.stop()


# ---- commands --------------------------------------------------------------------------------------------------------------------------
async def test_rescan_asks_the_connected_nodes_and_refetches(tmp_path: Path) -> None:
    a, b, c = node(tmp_path), node(tmp_path, "beta"), node(tmp_path, "gamma", state="down")
    w = await World(tmp_path, [a, b, c]).start()
    assert await w.library.rescan() == ["alpha", "beta"]
    assert a.scans == [{"force": True}] and b.scans == [{"force": True}] and c.scans == []
    assert await w.library.rescan("beta") == ["beta"]
    with pytest.raises(NotFound) as ei:
        await w.library.rescan("nobody")
    assert ei.value.code == "unknown_host"
    with pytest.raises(ProblemError) as pi:
        await w.library.rescan("gamma")                                        # one named host that is down is an error ...
    assert pi.value.status == 503 and pi.value.code == "node_unavailable" and pi.value.detail.startswith("Gamma: ")
    await w.stop()


async def test_rescan_of_all_hosts_skips_one_that_refuses(tmp_path: Path) -> None:
    a, b = node(tmp_path), node(tmp_path, "beta")
    w = await World(tmp_path, [a, b]).start()
    a._op_models_scan = lambda args: (_ for _ in ()).throw(a._err("busy", "scanning already"))  # type: ignore[method-assign]
    assert await w.library.rescan() == ["beta"]
    with pytest.raises(ProblemError) as ei:
        await w.library.rescan("alpha")
    assert ei.value.status == 503 and ei.value.code == "busy"
    await w.stop()


async def test_delete_removes_the_files_and_the_entry_at_once(tmp_path: Path) -> None:
    a = node(tmp_path)
    repo = a.roots["hf"].path / "hub" / "models--org--m"          # type: ignore[operator]
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "x").write_bytes(b"12345")
    a.entries = [hf_entry("hf", "org/m"), hf_entry("hf", "org/keep")]
    w = await World(tmp_path, [a]).start()
    cid = w.library.models(q="org/m")[0].copies[0].id
    await w.library.delete_copy(cid)
    assert a.ops("fs.delete") == [{"root": "hf", "rel": "hub/models--org--m"}] and not repo.exists()
    assert [m.name for m in w.library.models()] == ["org/keep"]                  # gone from the list without a rescan
    assert a.scans == [{"force": True}]                                         # and the node was asked to look again
    assert w.store.one("SELECT entries_json FROM library_snapshots")["entries_json"].count("org/m") == 0  # type: ignore[index]
    a.announce_models_changed()                                                 # the node's scan agrees: nothing comes back
    await asyncio.sleep(0.01)
    await w.settle()
    assert [m.name for m in w.library.models()] == ["org/keep"]
    await w.stop()


async def test_delete_refusals(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m"), entry("ro", "pdir", "plain", "pdir"), entry("hf", ".", "plain", "the-whole-root")]
    w = await World(tmp_path, [a]).start()
    lib = w.library
    cid = lib.find("alpha", ModelRef(format="hf", repo_id="org/m")).id            # type: ignore[union-attr]

    with pytest.raises(NotFound) as e0:
        await lib.delete_copy("f" * 16)
    assert e0.value.code == "copy_not_found"
    with pytest.raises(NotFound):
        await lib.delete_copy("not an id")

    w.busy[("alpha", lib.find("alpha", ModelRef(format="hf", repo_id="org/m")).path)] = ["inst_9"]  # type: ignore[union-attr]
    with pytest.raises(Conflict) as e1:
        await lib.delete_copy(cid)                                                # 409: in use
    assert e1.value.status == 409 and e1.value.code == "model_in_use" and "inst_9" in e1.value.detail
    w.busy.clear()

    ro = lib.find("alpha", ModelRef(format="plain", root_id="ro", rel="pdir")).id  # type: ignore[union-attr]
    with pytest.raises(ProblemError) as e2:
        await lib.delete_copy(ro)                                                 # 403: read-only root
    assert e2.value.status == 403 and e2.value.code == "policy_denied" and e2.value.detail.startswith("Alpha: ")

    root_entry = next(c for c in lib.copies("alpha") if c.rel == "")
    with pytest.raises(Conflict) as e3:
        await lib.delete_copy(root_entry.id)
    assert e3.value.code == "root_not_deletable"

    a.allow_delete = False
    with pytest.raises(ProblemError) as e4:
        await lib.delete_copy(cid)                                                # 403: the node's policy forbids deletes
    assert e4.value.status == 403 and e4.value.code == "policy_denied"
    a.allow_delete = True

    a.mode = "monitor"
    with pytest.raises(ProblemError) as e5:
        await lib.delete_copy(cid)
    assert e5.value.status == 409 and e5.value.code == "monitor_only"
    a.mode = "manage"

    a.in_use_paths.add(lib.find("alpha", ModelRef(format="hf", repo_id="org/m")).path)  # type: ignore[union-attr]
    with pytest.raises(ProblemError) as e6:
        await lib.delete_copy(cid)                                                # the node itself says a container mounts it
    assert e6.value.status == 409 and e6.value.code == "in_use"
    assert [m.name for m in lib.models(q="org/m")] == ["org/m"]                  # a refusal removes nothing
    await w.stop()


async def test_a_stale_host_cannot_delete(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    cid = w.library.models()[0].copies[0].id
    a.state = "connecting"
    with pytest.raises(ProblemError) as ei:
        await w.library.delete_copy(cid)
    assert ei.value.status == 503 and ei.value.code == "node_unavailable"
    assert not a.ops("fs.delete")
    await w.stop()


async def test_start_is_idempotent_and_stop_ends_the_subscriptions(tmp_path: Path) -> None:
    a = node(tmp_path)
    a.entries = [hf_entry("hf", "org/m")]
    w = await World(tmp_path, [a]).start()
    await w.library.start()
    assert len([1 for op, _ in a.calls if op == "models.list"]) == 1
    await w.library.stop()
    a.set_entries([])
    a.announce_models_changed()
    await asyncio.sleep(0.01)
    assert [m.name for m in w.library.models()] == ["org/m"]            # nobody is listening any more
    await w.stop()
