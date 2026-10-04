"""NodeLink against the fake node over a real subprocess: handshake, sanitizing, requests, timeouts and limits."""
from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest
from ecnode import proto

from engine_console.domain.fleet import NodeError, NodeUnavailable

from .harness import FAST, REMOTE_HOST, Rig, alive, wait_until

MakeRig = Callable[..., Awaitable[Rig]]


# ── handshake ───────────────────────────────────────────────────────────────────────────────────────────
async def test_handshake_brings_the_link_up(make_rig: MakeRig) -> None:
    rig = await make_rig()
    assert rig.link.link().state == "connecting"
    await rig.up()
    st = rig.link.link()
    assert st.error is None and st.retry_in_s is None and st.notes == []
    assert st.connected_since is not None and st.last_seen is not None
    hello = rig.link.latest("hello")
    assert hello is not None and hello["version"] == "0.4.0-test" and hello["proto"] == proto.PROTO
    assert [g["uuid"] for g in hello["gpus"]] == ["GPU-0000", "GPU-0001"]
    sample = await wait_until(lambda: rig.link.latest("sample"), what="a sample")
    assert sample["mem"]["total"] > 0 and sample["type"] == "sample"
    assert rig.link.latest("disks") is None and rig.link.latest("inventory") is None


async def test_ping_pong_measures_rtt_and_the_clock_skew(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    await wait_until(lambda: rig.link.link().rtt_ms is not None, what="a pong")
    st = rig.link.link()
    assert st.rtt_ms is not None and 0 <= st.rtt_ms < 1000
    assert st.skew_s is not None and abs(st.skew_s) < 2.0     # same machine, same clock
    assert rig.events("ping")


async def test_pushed_messages_are_republished_on_the_bus(make_rig: MakeRig) -> None:
    rig = await make_rig({"push_disks": {"mounts": [{"mount": "/", "fstype": "ext4"}]},
                          "push_inventory": {"containers": [], "endpoints": [], "scripts": []},
                          "pushes": [{"after_s": 0.05, "msg": {"type": "models_changed", "generation": 4, "t": 1.0}},
                                     {"after_s": 0.06, "msg": {"type": "metrics", "t": 1.0, "port": 18000,
                                                              "container": "ec-a", "engine": "vllm", "status": 200,
                                                              "text": "vllm:num_requests_running 1\n"}}]})
    seen: dict[str, dict[str, Any]] = {}

    async def collect(topic: str) -> None:
        async for ev in rig.bus.subscribe(topic):
            seen.setdefault(topic, ev)

    tasks = [asyncio.create_task(collect(f"node.{t}")) for t in ("hello", "sample", "disks", "inventory", "metrics", "models_changed")]
    try:
        await wait_until(lambda: len(seen) == 6, what=f"all six topics (have {sorted(seen)})")
    finally:
        for t in tasks:
            t.cancel()
    assert all(ev["host_id"] == "local" for ev in seen.values())
    assert seen["node.models_changed"]["msg"]["generation"] == 4
    assert seen["node.metrics"]["msg"]["port"] == 18000
    assert rig.link.latest("disks") is not None and rig.link.latest("inventory") is not None
    # metrics and models_changed are events only: the hub keeps no copy of them
    assert rig.link.latest("metrics") is None  # type: ignore[arg-type]


async def test_pong_and_credit_and_log_are_not_republished(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [{"after_s": 0.05, "msg": {"type": "log", "level": "info", "msg": "hello"}}]})
    got: list[str] = []

    async def collect(topic: str) -> None:
        async for _ in rig.bus.subscribe(topic):
            got.append(topic)

    tasks = [asyncio.create_task(collect(f"node.{t}")) for t in ("pong", "credit", "log", "fatal")]
    await rig.up()
    await asyncio.sleep(0.4)
    for t in tasks:
        t.cancel()
    assert got == []


async def test_junk_before_the_magic_line_is_skipped_and_noted(make_rig: MakeRig) -> None:
    junk = "Welcome to gpu-box-2\nLast login: yesterday\n"
    rig = await make_rig({"junk": junk})
    await rig.up()
    (note,) = rig.link.link().notes
    assert f"ignored {len(junk)} byte(s)" in note and "shell startup file" in note


async def test_too_much_junk_is_refused(make_rig: MakeRig) -> None:
    rig = await make_rig({"junk_lines": 80})                  # 80 KiB of lines and never the protocol
    await rig.state("down")
    assert "something other than the ecnode protocol" in (rig.link.link().error or "")


async def test_the_unrestricted_key_note_is_for_remote_hosts_only(make_rig: MakeRig) -> None:
    remote = await make_rig({"forced_cmd": False}, host=REMOTE_HOST)
    local = await make_rig({"forced_cmd": False})
    restricted = await make_rig({"forced_cmd": True}, host=REMOTE_HOST)
    for r in (remote, local, restricted):
        await r.up()
    (note,) = remote.link.link().notes
    assert "UNRESTRICTED ssh key" in note and "hosts init-key" in note
    assert local.link.link().notes == [] and restricted.link.link().notes == []


async def test_a_hello_without_the_forced_command_flag_counts_as_unrestricted(make_rig: MakeRig) -> None:
    rig = await make_rig({"hello_extra": {"forced_cmd": "yes"}}, host=REMOTE_HOST)   # not a bool: cleaned to None
    await rig.up()
    assert any("UNRESTRICTED" in n for n in rig.link.link().notes)


@pytest.mark.parametrize("theirs,update", [(2, "Update the Engine Console hub"), (0, "Update ecnode on This machine")])
async def test_a_protocol_mismatch_is_incompatible_and_says_which_side_to_update(make_rig: MakeRig, theirs: int, update: str) -> None:
    rig = await make_rig({"proto": theirs})
    await rig.state("incompatible")
    st = rig.link.link()
    assert update in (st.error or "") and f"protocol {theirs}" in (st.error or "") and "0.4.0-test" in (st.error or "")
    assert rig.link.last_wait_s == FAST.fix_retry_s and st.retry_in_s == 9
    assert rig.link.latest("hello") is not None            # the page can still show which version it is
    with pytest.raises(NodeUnavailable):
        await rig.link.call("docker.images")


async def test_no_hello_within_the_timeout_drops_and_kills_the_node(make_rig: MakeRig) -> None:
    rig = await make_rig({"hello_delay_s": 30}, timing=replace(FAST, hello_timeout_s=0.3))
    await rig.state("down")
    assert "no answer from the node within 0.3 s" in (rig.link.link().error or "")
    (pid,) = [e["pid"] for e in rig.events("start")][:1]
    await wait_until(lambda: not alive(pid), what="the silent node to be killed")


async def test_a_fatal_before_the_hello_is_reported(make_rig: MakeRig) -> None:
    rig = await make_rig({"first_frame": "fatal"})
    await rig.state("down")
    err = rig.link.link().error or ""
    assert err.startswith("the node gave up: scripted fatal: no GPU support") and "\n" not in err


async def test_a_first_message_that_is_not_a_hello_is_a_protocol_error(make_rig: MakeRig) -> None:
    rig = await make_rig({"first_frame": "sample"})
    await rig.state("down")
    assert "first message was 'sample', not hello" in (rig.link.link().error or "")


async def test_a_hello_without_a_protocol_version_is_refused(make_rig: MakeRig) -> None:
    rig = await make_rig({"proto": None})
    await rig.state("down")
    assert "protocol" in (rig.link.link().error or "")


async def test_the_node_exiting_before_hello_is_explained_by_its_stderr(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "Traceback (most recent call last):\nValueError: boom", "exit_before_hello": 1})
    await rig.state("down")
    assert rig.link.link().error == "the node exited (1): ValueError: boom"


async def test_a_missing_command_is_a_clear_error_and_retries(make_rig: MakeRig) -> None:
    rig = await make_rig(argv=["/nonexistent/path/to/ecnode", "--stdio"])
    await rig.state("down")
    assert rig.link.link().error == "could not start ecnode: No such file or directory"
    assert rig.link.last_wait_s == FAST.backoff_min_s


# ── sanitizing ──────────────────────────────────────────────────────────────────────────────────────────
async def test_hostile_fields_are_dropped_and_values_are_capped(make_rig: MakeRig) -> None:
    rig = await make_rig({"hello_extra": {"evil": "x", "shell": ["rm", "-rf"]},
                          "hello_patch": {"host": {"hostname": "box\x00name" + "h" * 400, "ssh_agent": "/tmp/agent"},
                                          "gpu": {"driver": "570\x00.00"}}})
    await rig.up()
    hello = rig.link.latest("hello")
    assert hello is not None
    assert "evil" not in hello and "shell" not in hello and "ssh_agent" not in hello["host"]
    name = hello["host"]["hostname"]
    assert name.startswith("boxname") and "\x00" not in name and 200 < len(name) <= 253   # capped (253), NUL removed
    assert hello["gpu"]["driver"] == "570.00"


async def test_a_sample_cannot_add_gpus_the_hello_did_not_announce(make_rig: MakeRig) -> None:
    rig = await make_rig()
    sample = await wait_until(lambda: rig.link.latest("sample"), what="a sample")
    assert set(sample["gpus"]) == {"GPU-0000", "GPU-0001"}          # the node also sent "GPU-ROGUE"


async def test_unknown_message_types_and_unknown_ids_are_ignored(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [
        {"after_s": 0.05, "msg": {"type": "weird", "payload": {"deep": [1, 2, 3]}}},
        {"after_s": 0.06, "msg": {"type": "res", "id": 999, "ok": True, "result": {}}},
        {"after_s": 0.07, "msg": {"type": "event", "id": 999, "event": {"line": "x"}}},
        {"after_s": 0.08, "msg": {"type": "res", "id": "7", "ok": True}},
        {"after_s": 0.09, "msg": {"type": "credit", "id": 424242, "bytes": 5}},
        {"after_s": 0.10, "msg": {"type": "credit", "id": 1, "bytes": -5}},
        {"after_s": 0.11, "msg": {"type": "pong", "id": 5555}},
        {"after_s": 0.12, "data_stream": 999, "bytes": 100},
        {"after_s": 0.13, "msg": {"type": 7}},
    ]})
    await rig.up()
    await asyncio.sleep(0.6)
    assert rig.link.link().state == "up" and rig.link.inflight == 0
    assert (await rig.link.call("docker.images"))["images"]       # and the link still works


async def test_a_non_finite_number_drops_the_link(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [{"after_s": 0.1, "raw": '{"type":"sample","seq":1,"t":NaN}'}]})
    await rig.up()
    await rig.state("down")
    assert "protocol error from the node" in (rig.link.link().error or "") and "non-finite" in (rig.link.link().error or "")


async def test_invalid_json_and_a_broken_frame_drop_the_link(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [{"after_s": 0.1, "raw": "{not json at all"}]})
    await rig.up()
    await rig.state("down")
    assert "protocol error from the node: not JSON" in (rig.link.link().error or "")


async def test_wrongly_typed_fields_become_none_not_errors(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [{"after_s": 0.05, "msg": {"type": "disks", "t": "yesterday", "mounts": [
        {"mount": 5, "size": "big", "roots": "all", "remote": "yes"}, {"mount": "/data", "size": 10}]}}]})
    await rig.up()
    disks = await wait_until(lambda: rig.link.latest("disks"), what="the disks message")
    assert disks["t"] is None
    assert disks["mounts"][1]["mount"] == "/data" and disks["mounts"][1]["size"] == 10
    assert disks["mounts"][0]["mount"] is None and disks["mounts"][0]["size"] is None and disks["mounts"][0]["remote"] is None


# ── requests ────────────────────────────────────────────────────────────────────────────────────────────
async def test_call_returns_the_sanitized_result(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    res = await rig.link.call("docker.images")
    assert res == {"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * 1024**3,
                               "created": 1.7e9, "arch": "amd64"}]}
    sent = [e for e in rig.events("req") if e["op"] == "docker.images"]
    assert len(sent) == 1 and sent[0]["args"] == {}


async def test_results_are_rebuilt_from_the_whitelist(make_rig: MakeRig) -> None:
    rig = await make_rig({"hostile_images": True})
    await rig.up()
    res = await rig.link.call("docker.images")
    assert set(res) == {"images"}                                      # "evil_top" is gone
    (img,) = res["images"]
    assert len(img["ref"]) == 256 and img["id"] == "id" and "evil" not in img


async def test_args_reach_the_node_unchanged(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    await rig.link.call("docker.image_present", {"image": "vllm/vllm-openai:v0.1"})
    (req,) = [e for e in rig.events("req") if e["op"] == "docker.image_present"]
    assert req["args"] == {"image": "vllm/vllm-openai:v0.1"}


@pytest.mark.parametrize("code", ["not_found", "policy_denied", "monitor_only", "no_space", "timeout", "future_code_x"])
async def test_an_error_res_becomes_a_node_error_with_the_nodes_code(make_rig: MakeRig, code: str) -> None:
    rig = await make_rig()
    await rig.up()
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.inspect", {"name": f"err-{code}"})
    assert e.value.code == code and e.value.message == "scripted refusal" and e.value.host_id == "local"
    assert not isinstance(e.value, NodeUnavailable)


@pytest.mark.parametrize("code", ["BAD CODE!", "../../etc", "x" * 60, "Not_Snake", "9lives"])
async def test_a_malformed_error_code_becomes_internal(make_rig: MakeRig, code: str) -> None:
    rig = await make_rig()
    await rig.up()
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.inspect", {"name": f"err-{code}"})
    assert e.value.code == "internal"


async def test_an_error_res_without_an_error_object_is_internal_with_a_message(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.inspect", {"name": "no-error-object"})
    assert e.value.code == "internal" and "without saying why" in e.value.message


async def test_node_error_text_is_made_safe(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    # a node's message can carry anything: control characters, newlines and token-looking text must not reach
    # logs and problem documents as they are
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.inspect", {"name": "ctl"})
    assert e.value.code == "io_error"
    assert e.value.message == "disk full second line [redacted]"


async def test_calls_are_refused_until_the_link_is_up(make_rig: MakeRig) -> None:
    rig = await make_rig(start=False)
    with pytest.raises(NodeUnavailable) as e:
        await rig.link.call("docker.images")
    assert e.value.code == "node_unavailable" and "not connected" in e.value.message
    stream = rig.link.open("docker.images")
    assert [i async for i in stream] == []
    with pytest.raises(NodeUnavailable):
        await stream.result()
    with pytest.raises(NodeUnavailable):
        await stream.write(b"x")
    with pytest.raises(NodeUnavailable):
        await stream.end()
    await stream.cancel()                                               # harmless


async def test_an_unknown_op_is_refused_locally_and_nothing_is_sent(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.rm_rf", {})
    assert e.value.code == "unknown_op"
    assert not [r for r in rig.events("req") if r["op"] == "docker.rm_rf"]
    stream = rig.link.open("nope")
    with pytest.raises(NodeError, match="unknown_op"):
        await stream.result()


async def test_arguments_json_cannot_carry_are_bad_args_and_nothing_is_sent(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    for bad in ({"name": float("nan")}, {"name": b"bytes"}, {"name": "x" * (2 * 1024 * 1024)}):
        with pytest.raises(NodeError) as e:
            await rig.link.call("docker.stop", bad)
        assert e.value.code == "bad_args"
    assert not rig.events("req")
    assert rig.link.inflight == 0


async def test_node_info_refreshes_the_stored_hello(make_rig: MakeRig) -> None:
    rig = await make_rig({"info_mode": "manage"})
    await rig.up()
    got: list[dict[str, Any]] = []

    async def collect() -> None:
        async for ev in rig.bus.subscribe("node.hello"):
            got.append(ev)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.05)
    before = rig.link.latest("hello")
    assert before is not None and before["policy"]["mode"] == "monitor"
    await rig.link.call("node.info")
    after = rig.link.latest("hello")
    assert after is not None and after["policy"]["mode"] == "manage" and after["type"] == "hello"
    assert [g["uuid"] for g in after["gpus"]] == ["GPU-0000", "GPU-0001"]
    await wait_until(lambda: got, what="a node.hello event")
    task.cancel()


# ── timeouts, cancel, limits ────────────────────────────────────────────────────────────────────────────
async def test_a_timeout_says_so_and_sends_cancel(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    with pytest.raises(NodeError) as e:
        await rig.link.call("models.scan", timeout=0.2)
    assert e.value.code == "timeout" and "may or may not have taken effect" in e.value.message
    (req,) = [r for r in rig.events("req") if r["op"] == "models.scan"]
    await wait_until(lambda: [c for c in rig.events("cancel") if c["id"] == req["id"]], what="the cancel frame")
    # the node answers the cancel with a res, which settles the request: nothing is left in flight
    await wait_until(lambda: rig.link.inflight == 0, what="the cancelled request to settle")


async def test_a_request_the_node_never_answers_after_cancel_keeps_its_slot(make_rig: MakeRig) -> None:
    rig = await make_rig({"ignore_cancel": True})
    await rig.up()
    with pytest.raises(NodeError, match="timeout"):
        await rig.link.call("models.scan", timeout=0.1)
    await asyncio.sleep(0.2)
    assert rig.link.inflight == 1                   # still counts: the node still counts it too


async def test_cancelling_the_caller_cancels_the_request_on_the_node(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    task = asyncio.create_task(rig.link.call("models.scan", timeout=30))
    await wait_until(lambda: rig.events("req"), what="the request to reach the node")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_until(lambda: rig.events("cancel"), what="the cancel frame")
    await wait_until(lambda: rig.link.inflight == 0, what="the cancelled request to settle")


async def test_at_most_64_requests_in_flight(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    tasks = [asyncio.create_task(rig.link.call("models.scan", timeout=30)) for _ in range(proto.MAX_INFLIGHT)]
    await wait_until(lambda: rig.link.inflight == proto.MAX_INFLIGHT, what="64 requests in flight")
    with pytest.raises(NodeError) as e:
        await rig.link.call("docker.images")
    assert e.value.code == "busy" and not isinstance(e.value, NodeUnavailable)
    stream = rig.link.open("docker.images")
    with pytest.raises(NodeError, match="busy"):
        await stream.result()
    await wait_until(lambda: len(rig.events("req")) == proto.MAX_INFLIGHT, what="the node to read all 64 requests")
    ids = [r["id"] for r in rig.events("req")]
    assert len(ids) == len(set(ids)) == proto.MAX_INFLIGHT
    for t in tasks[:3]:                              # free a few slots: new requests work again
        t.cancel()
    await asyncio.gather(*tasks[:3], return_exceptions=True)
    await wait_until(lambda: rig.link.inflight <= proto.MAX_INFLIGHT - 3, what="slots to free up")
    assert (await rig.link.call("docker.images"))["images"]
    for t in tasks[3:]:
        t.cancel()
    await asyncio.gather(*tasks[3:], return_exceptions=True)


async def test_request_ids_wrap_and_never_collide_with_one_in_flight(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    rig.link._next_id = proto.MAX_ID - 1
    hang = asyncio.create_task(rig.link.call("models.scan", timeout=30))       # takes MAX_ID - 1
    await wait_until(lambda: rig.link.inflight == 1, what="the hanging request")
    for _ in range(4):                                                         # MAX_ID, then 1, 2, 3
        await rig.link.call("docker.images")
    rig.link._next_id = proto.MAX_ID - 1                                      # wrap onto the id still in flight
    await rig.link.call("docker.images")
    ids = [r["id"] for r in rig.events("req")]
    assert ids[:5] == [proto.MAX_ID - 1, proto.MAX_ID, 1, 2, 3]
    assert ids[5] not in (proto.MAX_ID - 1,) and ids[5] == proto.MAX_ID                # skipped the in-flight id
    hang.cancel()
    await asyncio.gather(hang, return_exceptions=True)


# ── liveness ────────────────────────────────────────────────────────────────────────────────────────────
async def test_a_silent_node_is_dropped_and_pending_calls_fail(make_rig: MakeRig) -> None:
    rig = await make_rig({"silent_after_hello": True}, timing=replace(FAST, silent_s=0.5))
    await rig.up()
    call = asyncio.create_task(rig.link.call("docker.images", timeout=30))
    await rig.state("down")
    assert "no message from the node for 0.5 s" in (rig.link.link().error or "")
    with pytest.raises(NodeUnavailable) as e:
        await call
    assert "no message from the node" in e.value.message
    (pid,) = [e["pid"] for e in rig.events("start")][:1]
    await wait_until(lambda: not alive(pid), what="the silent node to be killed")


async def test_a_node_that_stops_sending_samples_turns_stale_but_stays_usable(make_rig: MakeRig) -> None:
    rig = await make_rig({"samples_stop_after_s": 0.3}, timing=replace(FAST, stale_s=0.5))
    await rig.up()
    assert rig.link.link().state == "up"
    await rig.state("stale")
    assert rig.link.link().error is None and rig.link.link().connected_since is not None
    assert (await rig.link.call("docker.images"))["images"]       # stale is not unavailable: pings still prove it is alive
    assert rig.link.latest("sample") is not None


async def test_no_samples_at_all_is_stale_measured_from_the_hello(make_rig: MakeRig) -> None:
    rig = await make_rig({"samples": False}, timing=replace(FAST, stale_s=0.4))
    await rig.up()
    assert rig.link.link().state == "up"
    await rig.state("stale")


async def test_a_flood_of_json_messages_drops_the_link(make_rig: MakeRig) -> None:
    rig = await make_rig({"flood": {"after_s": 0.1, "frame_kb": 64, "count": 400}},
                         timing=replace(FAST, flood_bytes=2 * 1024 * 1024))
    await rig.up()
    await rig.state("down")
    err = rig.link.link().error or ""
    assert "the node sent over 2 MB" in err and "dropped it" in err


async def test_a_flood_of_tiny_messages_drops_the_link(make_rig: MakeRig) -> None:
    rig = await make_rig({"flood": {"after_s": 0.1, "frame_kb": 0, "count": 5000}},
                         timing=replace(FAST, flood_frames=300))
    await rig.up()
    await rig.state("down")
    assert "messages in" in (rig.link.link().error or "")


async def test_a_busy_but_honest_node_is_not_a_flood(make_rig: MakeRig) -> None:
    rig = await make_rig({"sample_interval_s": 0.005}, timing=replace(FAST, flood_bytes=10 * 1024 * 1024, flood_frames=100_000))
    await rig.up()
    await asyncio.sleep(1.0)
    assert rig.link.link().state == "up"


# ── the child process ───────────────────────────────────────────────────────────────────────────────────
async def test_the_node_gets_only_the_environment_it_was_given(make_rig: MakeRig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_not_reach_the_node")
    rig = await make_rig(env={"PATH": os.environ["PATH"], "LANG": "C.UTF-8"})
    await rig.up()
    (start,) = rig.events("start")
    assert "HF_TOKEN" not in start["env"]
    assert {"PATH", "LANG"} <= set(start["env"]) and len(start["env"]) <= 6


async def test_stop_closes_stdin_so_a_node_exits_cleanly_and_is_fast(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    t0 = asyncio.get_running_loop().time()
    await rig.link.stop()
    assert asyncio.get_running_loop().time() - t0 < 1.0
    assert rig.events("eof")                                        # it saw the pipe close and left by itself
    assert rig.link.link().state == "down" and rig.link.link().error == "stopped"
    assert rig.link.latest("sample") is None


async def test_stop_kills_a_node_that_ignores_eof_and_fails_pending_calls(make_rig: MakeRig) -> None:
    rig = await make_rig({"silent_after_hello": True}, timing=replace(FAST, exit_grace_s=0.3, silent_s=60))
    await rig.up()
    call = asyncio.create_task(rig.link.call("docker.images", timeout=30))
    await asyncio.sleep(0.1)
    await rig.link.stop()
    with pytest.raises(NodeUnavailable):
        await call
    (pid,) = [e["pid"] for e in rig.events("start")][:1]
    assert not alive(pid)


async def test_stop_before_the_first_connection_finishes(make_rig: MakeRig) -> None:
    rig = await make_rig({"hello_delay_s": 30})
    await asyncio.sleep(0.1)
    await asyncio.wait_for(rig.link.stop(), 5)
    (pid,) = [e["pid"] for e in rig.events("start")][:1]
    assert not alive(pid)


async def test_stop_on_a_link_that_never_started(make_rig: MakeRig) -> None:
    rig = await make_rig(start=False)
    await rig.link.stop()
    assert rig.link.link().state == "down"
