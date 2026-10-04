"""scripts.run: an operator's own lifecycle script, registered in the node policy (docs/FLEET-PROTOCOL.md §6.3).

argv comes from the policy only (the hub names a script and an action, and may pick one argument from that
action's allowlist), never a shell. The environment is a minimal one built from the node's own (PATH, HOME, USER,
LOGNAME, LANG, XDG_RUNTIME_DIR, DOCKER_HOST, DOCKER_CONTEXT): nothing from the hub. One run per script at a time;
a run that outlives its timeout_s, or whose request is cancelled (or whose hub link drops), is stopped with its
whole process group.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Mapping
from typing import IO, Any

from . import proto
from .errors import Cancelled, OpError, Stream
from .policy import Policy
from .util import iter_lines, kill_group, scrub, spawn

ENV_KEEP = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "XDG_RUNTIME_DIR", "DOCKER_HOST", "DOCKER_CONTEXT")
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
OUTPUT_GRACE_S = 2.0


def script_env(base: Mapping[str, str] | None = None) -> dict[str, str]:
    src = os.environ if base is None else base
    env = {k: src[k] for k in ENV_KEEP if k in src}
    env.setdefault("PATH", DEFAULT_PATH)
    return env


class Scripts:
    def __init__(self) -> None:
        self._running: set[str] = set()
        self._lock = threading.Lock()

    def run(self, policy: Policy, args: dict[str, Any], stream: Stream) -> dict[str, Any]:
        s = policy.script(args["name"])
        if s is None:
            raise OpError(proto.NOT_FOUND, f"no script named {str(args['name'])[:32]!r} in this node's policy")
        a = s.action(args["action"])
        if a is None:
            raise OpError(proto.NOT_FOUND, f"script {s.name!r} has no action {str(args['action'])[:32]!r}")
        arg = args.get("arg") or None
        if a.args:
            if arg is None:
                raise OpError(proto.BAD_ARGS, f"{s.name}/{a.name} needs an argument: one of {', '.join(a.args)}")
            if arg not in a.args:
                raise OpError(proto.POLICY_DENIED, f"{arg[:64]!r} is not an allowed argument of {s.name}/{a.name} "
                                                   f"(allowed: {', '.join(a.args)})")
        elif arg is not None:
            raise OpError(proto.BAD_ARGS, f"{s.name}/{a.name} takes no argument")
        argv = [*a.argv, arg] if arg is not None else list(a.argv)
        with self._lock:
            if s.name in self._running:
                raise OpError(proto.BUSY, f"script {s.name!r} is already running; wait for it to finish")
            self._running.add(s.name)
        try:
            return self._run(argv, s.timeout_s, stream)
        finally:
            with self._lock:
                self._running.discard(s.name)

    def _run(self, argv: list[str], timeout_s: int, stream: Stream) -> dict[str, Any]:
        env = script_env()
        try:
            proc = spawn(argv, env=env, merge_stderr=False, cwd=env.get("HOME") or "/")
        except FileNotFoundError:
            raise OpError(proto.NOT_FOUND, f"the script program {argv[0]!r} does not exist on this machine") from None
        except OSError as e:
            raise OpError(proto.IO_ERROR, f"cannot start {argv[0]!r}: {e.strerror}") from None
        timed_out = threading.Event()

        def on_timeout() -> None:
            if proc.poll() is None:          # an exited script waiting out its output grace is not late
                timed_out.set()
                kill_group(proc)

        timer = threading.Timer(timeout_s, on_timeout)
        timer.daemon = True
        timer.start()
        stream.on_cancel(lambda: kill_group(proc))
        lines: queue.Queue[tuple[str, str] | None] = queue.Queue(maxsize=4096)
        finished = threading.Event()

        def pump(f: IO[bytes] | None, name: str) -> None:
            # After the run is over, keep draining (and dropping) output: a background child the script left
            # running may still hold the pipe, and must not block on a full one.
            try:
                if f is not None:
                    for line in iter_lines(f, 8192):
                        if not finished.is_set():
                            lines.put((name, line))
            finally:
                if not finished.is_set():
                    lines.put(None)

        readers = [threading.Thread(target=pump, args=(proc.stdout, "out"), daemon=True),
                   threading.Thread(target=pump, args=(proc.stderr, "err"), daemon=True)]
        for t in readers:
            t.start()
        try:
            done = 0
            exited_at: float | None = None
            while done < len(readers):
                try:
                    item = lines.get(timeout=0.5)
                except queue.Empty:
                    if proc.poll() is not None:
                        exited_at = exited_at or time.monotonic()
                        if time.monotonic() - exited_at > OUTPUT_GRACE_S:
                            break     # the script exited; something it started in the background holds the pipe
                    continue
                if item is None:
                    done += 1
                    continue
                if not stream.cancelled:
                    stream.event({"stream": item[0], "line": scrub(item[1])})
            rc = proc.wait()
        finally:
            finished.set()
            timer.cancel()
            if proc.poll() is None:       # only a run that is still going is stopped (timeout, cancel, link loss)
                kill_group(proc, 2.0)
        if timed_out.is_set():
            raise OpError(proto.TIMEOUT, f"the script ran longer than its timeout_s ({timeout_s} s) and was stopped")
        if stream.cancelled:
            raise Cancelled
        return {"rc": rc}
