"""Streaming requests over a real subprocess: events, DATA both ways, credit (flow control), cancel, and limits."""
from __future__ import annotations

import asyncio
import gc
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import replace

import pytest
from ecnode import proto

from engine_console.domain.fleet import NodeError, NodeUnavailable, StreamItem
from engine_console.services import fleet as fleet_mod

from .fake_node import pattern
from .harness import FAST, Rig, wait_until

MakeRig = Callable[..., Awaitable[Rig]]
MIB = 1024 * 1024


def sha_of_pattern(length: int, offset: int = 0) -> str:
    h = hashlib.sha256()
    for off in range(0, length, MIB):
        h.update(pattern(offset + off, min(MIB, length - off)))
    return h.hexdigest()


# ── events ──────────────────────────────────────────────────────────────────────────────────────────────
async def test_events_arrive_in_order_and_the_stream_ends_with_its_result(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("docker.logs_follow", {"name": "ec-a", "tail": 5})
    items = [i async for i in stream]
    assert [i.kind for i in items] == ["event"] * 5
    assert [i.event for i in items] == [{"line": f"line {n}"} for n in range(1, 6)]
    assert await stream.result() == {"lines": 5}
    assert await stream.result() == {"lines": 5}                   # idempotent
    assert [i async for i in stream] == []                         # a second iteration ends at once
    assert rig.link.inflight == 0


async def test_a_refused_stream_ends_and_result_raises_the_nodes_reason(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("docker.pull", {"image": "vllm/vllm-openai:v9"})
    lines = [i.event["line"] async for i in stream if i.event]    # the event with only unknown fields was dropped
    assert lines == ["layer 0", "layer 1", "layer 2"]
    with pytest.raises(NodeError) as e:
        await stream.result()
    assert e.value.code == "docker_error" and e.value.message == "pull failed (scripted)"


async def test_script_output_events_carry_stream_and_line(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("scripts.run", {"name": "big-model", "action": "up"})
    events = [i.event async for i in stream]
    assert events == [{"stream": "out", "line": "step 0"}, {"stream": "out", "line": "step 1"}]
    assert await stream.result() == {"rc": 0}


async def test_events_a_node_sends_for_an_op_that_has_none_are_dropped(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": []})
    await rig.up()
    stream = rig.link.open("docker.stop", {"name": "ec-a"})
    assert [i async for i in stream] == []
    assert await stream.result() == {}


async def test_the_event_queue_of_a_stream_nobody_reads_is_capped(make_rig: MakeRig, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fleet_mod, "MAX_STREAM_EVENTS", 50)
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("docker.logs_follow", {"name": "ec-a", "tail": 400})
    await wait_until(lambda: rig.link.inflight == 0, what="the node to finish")      # it answered; nobody has read a thing
    items = [i async for i in stream]
    assert len(items) == 50 and items[0].event == {"line": "line 1"}               # the rest were dropped, not buffered
    assert await stream.result() == {"lines": 400}
    assert rig.link.link().state == "up"


# ── cancel ──────────────────────────────────────────────────────────────────────────────────────────────
async def test_cancel_stops_a_stream_and_the_node_is_told(make_rig: MakeRig) -> None:
    rig = await make_rig({"follow_delay_s": 0.01})
    await rig.up()
    stream = rig.link.open("docker.logs_follow", {"name": "ec-a", "tail": 0})        # follows until cancelled
    seen = 0
    async for _item in stream:
        seen += 1
        if seen == 3:
            await stream.cancel()
    assert seen >= 3                                                                 # events up to the cancel were delivered
    with pytest.raises(NodeError) as e:
        await stream.result()
    assert e.value.code == "cancelled"
    assert rig.events("cancel") and rig.link.inflight == 0


async def test_abandoning_an_iteration_cancels_the_request(make_rig: MakeRig) -> None:
    rig = await make_rig({"follow_delay_s": 0.01})
    await rig.up()
    stream = rig.link.open("docker.logs_follow", {"name": "ec-a", "tail": 0})
    async for _item in stream:
        break                                                                        # stop early, no cancel() call
    await wait_until(lambda: rig.events("cancel"), what="the node to be told to stop")
    await wait_until(lambda: rig.link.inflight == 0, what="the request to settle")


async def test_events_after_a_cancel_are_not_delivered(make_rig: MakeRig) -> None:
    rig = await make_rig({"follow_delay_s": 0.005, "ignore_cancel": True})
    await rig.up()
    stream = rig.link.open("docker.logs_follow", {"name": "ec-a", "tail": 0})
    first = await anext(aiter(stream))
    assert first.kind == "event"
    await stream.cancel()
    q = stream._req.queue                        # type: ignore[attr-defined]
    assert q is not None
    queued_at_cancel = q.qsize()
    await wait_until(lambda: rig.events("cancel"), what="the cancel to reach the node")
    sent_at_cancel = max(e["n"] for e in rig.events("follow"))
    await asyncio.sleep(0.3)
    # the node ignored the cancel and kept talking ...
    assert max(e["n"] for e in rig.events("follow")) > sent_at_cancel + 20
    # ... and the hub kept none of it
    assert q.qsize() == queued_at_cancel
    assert rig.link.link().state == "up"


# ── DATA and flow control: node -> hub ──────────────────────────────────────────────────────────────────
async def test_a_download_larger_than_the_window_is_flow_controlled(make_rig: MakeRig) -> None:
    rig = await make_rig({"data_chunk": 512 * 1024})
    await rig.up()
    length = 24 * MIB
    stream = rig.link.open("fs.read", {"root": "hf", "rel": "model.safetensors", "length": length})

    # Nobody reads: the node uses its initial credit and then stalls at exactly that boundary.
    stalled = await wait_until(lambda: rig.events("stalled"), what="the node to run out of credit")
    assert stalled[0]["sent"] == proto.WINDOW and stalled[0]["credit"] < 512 * 1024
    sent_at_stall = rig.events("sent")[-1]["sent"]
    assert sent_at_stall == proto.WINDOW
    await asyncio.sleep(0.3)
    assert rig.events("sent")[-1]["sent"] == proto.WINDOW          # still stalled: no credit has been granted

    # The stalled stream blocks nothing else: pings are answered, other requests complete, samples keep coming.
    pings_before = len(rig.events("ping"))
    seq_before = (rig.link.latest("sample") or {}).get("seq", 0)
    t0 = asyncio.get_running_loop().time()
    assert (await rig.link.call("docker.images"))["images"]
    assert asyncio.get_running_loop().time() - t0 < 1.0
    await asyncio.sleep(0.3)
    assert len(rig.events("ping")) > pings_before
    assert (rig.link.latest("sample") or {}).get("seq", 0) > seq_before
    assert rig.link.link().state == "up" and rig.link.link().rtt_ms is not None

    # A slow consumer: every byte arrives, in order, intact, and credit goes back in batches as it reads.
    got = hashlib.sha256()
    total = 0
    async for item in stream:
        assert item.kind == "data"
        got.update(item.data)
        total += len(item.data)
        if total % (4 * MIB) == 0:
            await asyncio.sleep(0.02)
    assert total == length and got.hexdigest() == sha_of_pattern(length)
    assert await stream.result() == {"size": length}
    credits = [c["bytes"] for c in rig.events("credit")]
    assert credits and all(c >= fleet_mod.CREDIT_BATCH for c in credits) and sum(credits) <= length
    assert rig.link.inflight == 0


async def test_two_streams_flow_independently(make_rig: MakeRig) -> None:
    rig = await make_rig({"data_chunk": 512 * 1024})
    await rig.up()
    slow = rig.link.open("fs.read", {"root": "hf", "rel": "a", "length": 16 * MIB})
    await wait_until(lambda: rig.events("stalled"), what="the first stream to stall")
    fast = rig.link.open("fs.read", {"root": "hf", "rel": "b", "length": 6 * MIB, "offset": 4096})
    got = bytearray()
    async for item in fast:                                         # completes while `slow` is stuck
        got += item.data
    assert bytes(got) == pattern(4096, 6 * MIB)
    assert await fast.result() == {"size": 6 * MIB}
    total = 0
    async for item in slow:
        total += len(item.data)
    assert total == 16 * MIB


async def test_download_data_is_not_a_flood(make_rig: MakeRig) -> None:
    """A model copy moves far more than 20 MB in 10 s; only JSON messages count toward the flood limit."""
    rig = await make_rig({"data_chunk": MIB}, timing=replace(FAST, flood_bytes=1 * MIB, flood_frames=500))
    await rig.up()
    stream = rig.link.open("fs.read", {"root": "hf", "rel": "big", "length": 40 * MIB})
    total = 0
    async for item in stream:
        total += len(item.data)
    assert total == 40 * MIB and rig.link.link().state == "up"


async def test_a_node_that_sends_more_than_its_credit_is_dropped(make_rig: MakeRig) -> None:
    rig = await make_rig({"ignore_credit": True, "data_chunk": MIB})
    await rig.up()
    stream = rig.link.open("fs.read", {"root": "hf", "rel": "x", "length": 12 * MIB})
    await rig.state("down")                                          # nobody read, so no credit was ever granted
    err = rig.link.link().error or ""
    assert "more data" in err and "than the credit this hub granted" in err
    got = 0
    async for item in stream:
        got += len(item.data)
    assert got <= proto.WINDOW
    with pytest.raises(NodeUnavailable):
        await stream.result()


async def test_a_node_dying_mid_stream_fails_the_stream_and_the_link_recovers(make_rig: MakeRig) -> None:
    rig = await make_rig({"crash_after_bytes": MIB, "data_chunk": 512 * 1024})
    await rig.up()
    stream = rig.link.open("fs.read", {"root": "hf", "rel": "x", "length": 4 * MIB})
    got = bytearray()
    async for item in stream:
        got += item.data
    assert bytes(got) == pattern(0, len(got)) and len(got) <= 4 * MIB   # what arrived is a clean prefix
    with pytest.raises(NodeUnavailable) as e:
        await stream.result()
    assert e.value.code == "node_unavailable"
    await rig.state("down")
    assert "the node exited (7)" in (rig.link.link().error or "")
    await rig.up()                                                    # a new node, same host
    assert (await rig.link.call("docker.images"))["images"]


# ── DATA and flow control: hub -> node ──────────────────────────────────────────────────────────────────
async def test_an_upload_larger_than_the_window_waits_for_credit(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 0.8, "credit_batch": MIB})
    await rig.up()
    size = 20 * MIB
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "model.safetensors", "size": size})
    writer = asyncio.create_task(stream.write(pattern(0, size)))

    # The node does not read or grant anything yet: the hub stops at exactly the window it was given.
    await wait_until(lambda: (rig.events("data") or [{"total": 0}])[-1]["total"] == proto.WINDOW, what="the window to fill")
    await asyncio.sleep(0.3)
    assert not writer.done()
    assert rig.events("data")[-1]["total"] == proto.WINDOW
    # exactly eight full frames: a hub that kept "sending" while out of credit would show up as a flood of empty ones
    assert [d["total"] for d in rig.events("data")] == [i * proto.MAX_DATA for i in range(1, proto.WINDOW // proto.MAX_DATA + 1)]

    # Blocked on credit, the hub still answers: pings flow, other requests complete.
    pings_before = len(rig.events("ping"))
    assert (await rig.link.call("docker.images"))["images"]
    await asyncio.sleep(0.2)
    assert len(rig.events("ping")) > pings_before and rig.link.link().state == "up"

    await asyncio.wait_for(writer, 15)                                # the node starts consuming and granting credit
    await stream.end()
    assert await stream.result() == {"size": size, "sha256": sha_of_pattern(size)}
    assert not rig.events("violation")
    assert max(d["total"] for d in rig.events("data")) == size


async def test_small_writes_are_sent_as_they_are_and_in_order(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 3000})
    for i in range(3):
        await stream.write(bytes([i]) * 1000)
    await stream.write(b"")                                           # an empty write sends nothing
    await stream.end()
    await stream.end()                                                # and end() twice is harmless
    expected = hashlib.sha256(b"\x00" * 1000 + b"\x01" * 1000 + b"\x02" * 1000).hexdigest()
    assert await stream.result() == {"size": 3000, "sha256": expected}
    assert len(rig.events("end")) == 1


async def test_a_single_huge_write_is_split_into_frames_of_at_most_one_mib(make_rig: MakeRig) -> None:
    rig = await make_rig({"credit_batch": 256 * 1024})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 3 * MIB + 5})
    await stream.write(b"z" * (3 * MIB + 5))
    await stream.end()
    assert (await stream.result())["size"] == 3 * MIB + 5
    totals = [d["total"] for d in rig.events("data")]
    steps = [b - a for a, b in zip([0, *totals], totals, strict=False)]
    assert all(0 < s <= proto.MAX_DATA for s in steps) and sum(steps) == 3 * MIB + 5


async def test_cancelling_an_upload_that_waits_for_credit_unblocks_the_writer(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 30})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 20 * MIB})
    writer = asyncio.create_task(stream.write(b"q" * (20 * MIB)))
    await wait_until(lambda: (rig.events("data") or [{"total": 0}])[-1]["total"] == proto.WINDOW, what="the window to fill")
    await stream.cancel()
    with pytest.raises(NodeError) as e:
        await asyncio.wait_for(writer, 5)
    assert e.value.code == "cancelled"
    await wait_until(lambda: rig.link.inflight == 0, what="the request to settle")


async def test_a_node_that_refuses_an_upload_early_fails_the_writer_with_its_reason(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("docker.stop", {"name": "ec-a"})            # answers at once, never wants DATA
    await stream.result()
    with pytest.raises(NodeError) as e:
        await stream.write(b"too late")
    assert e.value.code == "conflict"


async def test_write_after_end_is_a_bad_args_error(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 5})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 1})
    await stream.write(b"x")
    await stream.end()
    with pytest.raises(NodeError) as e:
        await stream.write(b"y")
    assert e.value.code == "bad_args"
    await stream.cancel()


async def test_a_link_drop_fails_a_writer_waiting_for_credit(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 30, "exit_after_s": 0.6})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 20 * MIB})
    with pytest.raises(NodeUnavailable):
        await asyncio.wait_for(stream.write(b"q" * (20 * MIB)), 10)


async def test_dropping_an_unfinished_upload_cancels_it_on_the_node(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 30})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 100})
    await stream.write(b"abc")
    assert rig.link.inflight == 1
    del stream
    gc.collect()
    await wait_until(lambda: rig.events("cancel"), what="the abandoned upload to be cancelled")


async def test_a_finished_upload_that_is_dropped_before_its_answer_is_left_alone(make_rig: MakeRig) -> None:
    rig = await make_rig({"upload_hold_s": 0.5})
    await rig.up()
    stream = rig.link.open("fs.write", {"root": "hf", "rel": "x", "size": 3})
    await stream.write(b"abc")
    await stream.end()
    del stream
    gc.collect()
    await asyncio.sleep(0.2)
    assert not rig.events("cancel")                                    # end() was sent: the request is still wanted
    await wait_until(lambda: rig.link.inflight == 0, what="the node to answer")


# ── http.request: events, then DATA, then the result ────────────────────────────────────────────────────
async def test_http_request_with_a_body_streams_the_response(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("http.request", {"port": 18000, "method": "POST", "path": "/v1/chat/completions",
                                            "body_len": 11, "headers": {"content-type": "application/json"}})
    await stream.write(b"hello ")
    await stream.write(b"world")
    await stream.end()
    items: list[StreamItem] = [i async for i in stream]
    assert items[0].kind == "event" and items[0].event == {"status": 200, "headers": {"content-type": "text/plain"}}
    body = b"".join(i.data for i in items[1:] if i.kind == "data")
    assert body == b"engine answer, request body was 11 bytes"
    assert await stream.result() == {"status": 200, "bytes": len(body)}


async def test_http_request_without_a_body(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    stream = rig.link.open("http.request", {"port": 18000, "method": "GET", "path": "/health"})
    items = [i async for i in stream]
    assert items[0].event is not None and items[0].event["status"] == 200
    assert b"".join(i.data for i in items[1:]) == b"engine answer, request body was 0 bytes"
