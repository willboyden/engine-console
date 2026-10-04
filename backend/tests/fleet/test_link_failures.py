"""How a link fails and recovers: backoff, reconnect, ssh failures in words, crashes and cleanup."""
from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import pytest

from engine_console.domain.fleet import HostConfig, NodeUnavailable
from engine_console.services.fleet import explain_exit
from engine_console.services.hosts import HostsError

from .harness import FAST, REMOTE_HOST, Rig, alive, node_argv, wait_until

MakeRig = Callable[..., Awaitable[Rig]]


# ── explain_exit, case by case ──────────────────────────────────────────────────────────────────────────
REMOTE = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5")
LOCAL = HostConfig(id="local", label="This machine")
CMD = "/usr/local/lib/ecnode/ecnode --stdio"


@pytest.mark.parametrize("host,rc,stderr,needle,fix", [
    (REMOTE, 255, "ops@10.0.0.5: Permission denied (publickey).", "hosts init-key", True),
    (REMOTE, 255, "Warning: x\nops@10.0.0.5: Permission denied (publickey,password).", "authorized_keys", True),
    (REMOTE, 255, "No ED25519 host key is known for 10.0.0.5 and you have requested strict checking.\nHost key verification failed.",
     "connect once by hand", True),
    (REMOTE, 255, "@@@ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @@@", "ssh ops@10.0.0.5 true", True),
    (REMOTE, 255, "@ WARNING: UNPROTECTED PRIVATE KEY FILE! @\nPermissions 0644 are too open.", "chmod 600", True),
    (REMOTE, 255, "ssh: Could not resolve hostname gpu-box-9: Name or service not known", "cannot resolve", False),
    (REMOTE, 255, "ssh: connect to host 10.0.0.5 port 22: No route to host", "cannot reach ops@10.0.0.5", False),
    (REMOTE, 255, "ssh: connect to host 10.0.0.5 port 22: Connection refused", "cannot reach", False),
    (REMOTE, 255, "ssh: connect to host 10.0.0.5 port 22: Connection timed out", "cannot reach", False),
    (REMOTE, 255, "something odd", "ssh to ops@10.0.0.5 failed: something odd", False),
    (REMOTE, 255, "", "ssh to ops@10.0.0.5 failed: exit 255", False),
    (REMOTE, 127, "bash: line 1: /usr/local/lib/ecnode/ecnode: No such file or directory", "not installed on GPU box 2 (expected /usr/local/lib/ecnode/ecnode)", True),
    (REMOTE, 1, "FileNotFoundError: No such file or directory: '/etc/x'", "the node exited (1)", False),   # not "not installed"
    (LOCAL, 255, "Permission denied", "the node exited (255): Permission denied", False),                # no ssh, no ssh advice
    (LOCAL, 1, "/usr/bin/python3: No module named ecnode", "missing from the console's Python environment", False),
    (LOCAL, 2, "", "the node exited (2)", False),
    (LOCAL, -15, "", "stopped by signal 15", False),
    (LOCAL, -9, "", "the connection closed", False),
    (LOCAL, 0, "", "the connection closed", False),
    (LOCAL, None, "last words", "last words", False),
])
def test_explain_exit(host: HostConfig, rc: int | None, stderr: str, needle: str, fix: bool) -> None:
    msg, wait = explain_exit(host, rc, stderr, CMD, 300.0)
    assert needle in msg
    assert (wait == 300.0) is fix and (wait is None) is (not fix)
    assert "\n" not in msg


def test_explain_exit_masks_the_hub_key_path() -> None:
    key = "/home/ops/.config/engine-console/ssh/hub-key"
    msg, wait = explain_exit(REMOTE, 255, f'Load key "{key}": error in libcrypto', CMD, 300.0, mask=[key])
    assert msg == 'ssh to ops@10.0.0.5 failed: Load key "<hub key>": error in libcrypto' and wait is None
    assert key not in msg
    again, _ = explain_exit(REMOTE, 255, f'Load key "{key}": bad\nand again {key}', CMD, 300.0, mask=["", key])
    assert key not in again and "<hub key>" in again


def test_explain_exit_makes_untrusted_text_safe() -> None:
    msg, _ = explain_exit(LOCAL, 1, "boom hf_abcdefghijklmnop\x1b[31m red\r\n", CMD, 300.0)
    assert "hf_abcdefghijklmnop" not in msg and "[redacted]" in msg
    assert "\x1b" not in msg and "\r" not in msg and "\n" not in msg
    long_msg, _ = explain_exit(LOCAL, 1, "x" * 5000, CMD, 300.0)
    assert len(long_msg) < 400


# ── backoff ─────────────────────────────────────────────────────────────────────────────────────────────
def waits(rig: Rig) -> list[float]:
    return [w for state, w in rig.changes if state == "down" and w is not None]


async def test_backoff_doubles_and_is_capped(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_after_s": 0.05}, timing=replace(FAST, backoff_min_s=0.02, backoff_max_s=0.16))
    await wait_until(lambda: len(waits(rig)) >= 6, what=f"six failures (have {waits(rig)})")
    assert waits(rig)[:6] == [0.02, 0.04, 0.08, 0.16, 0.16, 0.16]
    assert "the node exited (3): fake node crashing on purpose" in (rig.link.link().error or "")


async def test_backoff_starts_over_after_a_long_enough_connection(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_after_s": 0.3}, timing=replace(FAST, backoff_min_s=0.02, backoff_max_s=0.5, stable_s=0.15))
    await wait_until(lambda: len(waits(rig)) >= 3, what="three failures")
    assert waits(rig)[:3] == [0.02, 0.02, 0.02]            # each session outlived stable_s, so the wait never grew


async def test_a_session_that_never_got_up_backs_off_too(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_before_hello": 1}, timing=replace(FAST, backoff_min_s=0.02, backoff_max_s=0.16))
    await wait_until(lambda: len(waits(rig)) >= 5, what="five failures")
    assert waits(rig)[:5] == [0.02, 0.04, 0.08, 0.16, 0.16]


async def test_the_link_shows_the_retry_countdown_while_down(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_before_hello": 2}, timing=replace(FAST, backoff_min_s=5.0, backoff_max_s=5.0))
    await rig.state("down")
    st = rig.link.link()
    assert st.retry_in_s in (4, 5) and st.connected_since is None and st.rtt_ms is None


# ── ssh-style failures end to end (the fake exits like ssh does) ────────────────────────────────────────
async def test_login_refused_names_the_fix_and_retries_rarely(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "ops@10.0.0.5: Permission denied (publickey).", "exit_before_hello": 255}, host=REMOTE_HOST)
    await rig.state("down")
    assert "hosts init-key" in (rig.link.link().error or "")
    assert rig.link.last_wait_s == FAST.fix_retry_s and rig.link.link().retry_in_s == 9


async def test_host_key_trouble_says_to_connect_once_by_hand(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "Host key verification failed.", "exit_before_hello": 255}, host=REMOTE_HOST)
    await rig.state("down")
    assert "connect once by hand" in (rig.link.link().error or "") and rig.link.last_wait_s == FAST.fix_retry_s


async def test_node_not_installed_says_where_it_expected_it(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "sh: 1: /usr/local/lib/ecnode/ecnode: not found", "exit_before_hello": 127},
                         host=REMOTE_HOST)
    rig.link._remote_cmd = CMD
    await rig.state("down")
    assert "not installed on GPU box 2" in (rig.link.link().error or "") and rig.link.last_wait_s == FAST.fix_retry_s


async def test_no_route_is_an_ordinary_failure_with_ordinary_backoff(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "ssh: connect to host 10.0.0.5 port 22: No route to host", "exit_before_hello": 255}, host=REMOTE_HOST)
    await rig.state("down")
    assert "cannot reach ops@10.0.0.5" in (rig.link.link().error or "") and rig.link.last_wait_s == FAST.backoff_min_s


async def test_reconnect_skips_a_long_wait_and_forgets_the_backoff(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "Permission denied", "exit_before_hello": 255}, host=REMOTE_HOST)
    await rig.state("down")
    assert len(rig.events("start")) == 1 and rig.link.last_wait_s == FAST.fix_retry_s
    rig.link.reconnect()
    await wait_until(lambda: len(rig.events("start")) >= 2, timeout=3, what="an immediate second attempt")


async def test_incompatible_waits_are_cut_short_by_reconnect_too(make_rig: MakeRig) -> None:
    rig = await make_rig({"proto": 2})
    await rig.state("incompatible")
    rig.link.reconnect()
    await wait_until(lambda: len(rig.events("start")) >= 2, timeout=3, what="a new attempt")


async def test_a_command_that_cannot_be_built_is_shown_and_retried_rarely_then_fixed(make_rig: MakeRig, tmp_path: Any) -> None:
    attempts: list[int] = []

    def argv() -> list[str]:
        attempts.append(1)
        if len(attempts) == 1:
            raise HostsError("the console has no ssh key yet: run `engine-console hosts init-key`")
        return node_argv(log=str(tmp_path / "late.jsonl"))

    rig = await make_rig(argv=argv)
    await rig.state("down")
    assert rig.link.link().error == "the console has no ssh key yet: run `engine-console hosts init-key`"
    assert rig.link.last_wait_s == FAST.fix_retry_s
    rig.link.reconnect()                               # the person did what it said
    await rig.up()
    assert len(attempts) == 2


# ── crashes and recovery ────────────────────────────────────────────────────────────────────────────────
async def test_the_link_recovers_after_the_node_dies_and_keeps_what_it_knew(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_after_s": 0.5}, timing=replace(FAST, backoff_min_s=0.05))
    await rig.up()
    first = rig.link.link().connected_since
    await wait_until(lambda: rig.link.latest("sample"), what="a sample")
    await rig.state("down")
    assert rig.link.latest("hello") is not None                     # the last hello is kept: the page can still show the machine
    assert rig.link.latest("sample") is None                         # a live sample from a dead link is gone
    assert rig.link.link().connected_since is None and rig.link.link().rtt_ms is None
    await rig.up()
    assert rig.link.link().connected_since is not None and rig.link.link().connected_since != first
    assert len(rig.events("start")) >= 2


async def test_pending_calls_fail_at_once_when_the_node_dies(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_after_s": 0.4})
    await rig.up()
    t0 = asyncio.get_running_loop().time()
    with pytest.raises(NodeUnavailable) as e:
        await rig.link.call("models.scan", timeout=30)
    assert asyncio.get_running_loop().time() - t0 < 3.0
    assert "closed the connection" in e.value.message
    assert rig.link.inflight == 0


async def test_a_fatal_after_the_hello_ends_the_link_with_the_nodes_words(make_rig: MakeRig) -> None:
    rig = await make_rig({"pushes": [{"after_s": 0.2, "msg": {"type": "fatal", "error": "docker socket vanished"}}]})
    await rig.up()
    await rig.state("down")
    assert (rig.link.link().error or "").startswith("the node gave up: docker socket vanished")


async def test_reconnect_on_a_live_link_drops_and_re_establishes_it(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    first = rig.link.link().connected_since
    call = asyncio.create_task(rig.link.call("models.scan", timeout=30))
    await wait_until(lambda: rig.link.inflight == 1, what="a pending call")
    await asyncio.sleep(0.01)
    rig.link.reconnect()
    with pytest.raises(NodeUnavailable) as e:
        await call
    assert "reconnecting at your request" in e.value.message
    await wait_until(lambda: len(rig.events("start")) == 2 and rig.link.link().state == "up", timeout=5, what="a new connection")
    assert rig.link.link().connected_since != first
    assert (await rig.link.call("docker.images"))["images"]
    assert waits(rig) == []                                          # a requested reconnect is not a failure to back off from


async def test_reconnect_while_connecting_does_not_leave_a_stale_request_behind(make_rig: MakeRig) -> None:
    rig = await make_rig({"hello_delay_s": 1.5})
    await wait_until(lambda: rig.events("start"), what="the first attempt to be running (and not yet past its hello)")
    rig.link.reconnect()
    await rig.up()
    await asyncio.sleep(0.05)
    assert rig.link.link().state == "up"
    # the slow first attempt was abandoned at once (no hello yet, nothing graceful to do) and replaced; the link that
    # came up has a working writer, and nothing was counted as a failure to back off from
    assert len(rig.events("start")) == 2 and rig.link.last_wait_s is None
    assert (await rig.link.call("docker.images"))["images"]


async def test_a_listener_that_raises_does_not_break_the_link(make_rig: MakeRig) -> None:
    rig = await make_rig(start=False)

    def bad() -> None:
        raise RuntimeError("listener bug")

    rig.link._on_change = bad
    await rig.link.start()
    await rig.up()
    assert (await rig.link.call("docker.images"))["images"]


# ── cleanup ─────────────────────────────────────────────────────────────────────────────────────────────
async def test_every_node_process_is_gone_after_many_restarts_and_stop(make_rig: MakeRig) -> None:
    rig = await make_rig({"exit_after_s": 0.05}, timing=replace(FAST, backoff_min_s=0.01, backoff_max_s=0.02))
    await wait_until(lambda: len(rig.events("start")) >= 5, what="five restarts")
    await rig.link.stop()
    pids = [e["pid"] for e in rig.events("start")]
    assert len(pids) >= 5 and not [p for p in pids if alive(p)]


async def test_the_whole_process_group_is_killed_not_just_the_child(make_rig: MakeRig) -> None:
    """ssh may spawn helpers (a ProxyCommand); a node may spawn children. Each node is its own session, and the hub
    kills the group, so nothing a dropped node started is left running."""
    rig = await make_rig({"spawn_child": True, "silent_after_hello": True}, timing=replace(FAST, silent_s=0.4, exit_grace_s=0.3))
    child = (await wait_until(lambda: rig.events("child"), what="the grandchild"))[0]["pid"]
    assert alive(child)
    await rig.state("down")                                          # the silent node was dropped and killed
    await wait_until(lambda: not alive(child), what="the grandchild to be killed with its parent")
    await rig.link.stop()


async def test_stop_kills_the_whole_group_too(make_rig: MakeRig) -> None:
    rig = await make_rig({"spawn_child": True, "silent_after_hello": True}, timing=replace(FAST, silent_s=60, exit_grace_s=0.3))
    child = (await wait_until(lambda: rig.events("child"), what="the grandchild"))[0]["pid"]
    await rig.up()
    await rig.link.stop()
    await wait_until(lambda: not alive(child), what="the grandchild to die with its parent's group")


async def test_the_node_is_its_own_session_so_a_signal_to_the_hub_group_misses_it(make_rig: MakeRig) -> None:
    rig = await make_rig()
    await rig.up()
    (pid,) = [e["pid"] for e in rig.events("start")]
    assert os.getpgid(pid) == pid != os.getpgid(0)


async def test_huge_stderr_is_bounded_and_the_last_line_still_explains(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "the actual reason", "stderr_noise": 20000, "exit_before_hello": 4})
    await rig.state("down")
    assert rig.link.link().error == "the node exited (4): the actual reason"


async def test_signal_exit_is_reported(make_rig: MakeRig) -> None:
    rig = await make_rig({"stderr": "about to be killed", "silent_after_hello": True}, timing=replace(FAST, silent_s=60))
    await rig.up()
    (pid,) = [e["pid"] for e in rig.events("start")]
    os.kill(pid, signal.SIGTERM)
    await rig.state("down")
    assert "stopped by signal 15" in (rig.link.link().error or "")
