"""scripts.run: policy argv only, one allowlisted argument, minimal environment, streaming, busy, timeouts."""
from __future__ import annotations

import contextlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

from ecnode import proto
from ecnode.errors import Cancelled, OpError
from ecnode.policy import parse
from ecnode.scripts import ENV_KEEP, Scripts

from .test_fsops import FakeStream

SCRIPT = """#!{py}
import json, os, sys, time
action = sys.argv[1] if len(sys.argv) > 1 else ""
if action == "env":
    print(json.dumps(sorted(os.environ)))
elif action == "talk":
    print("out line 1"); print("api_key=sk-test-not-a-real-key-0000", flush=True)
    print("err line", file=sys.stderr, flush=True)
    sys.exit(3)
elif action == "sleep":
    time.sleep(float(sys.argv[2]))
elif action == "background":
    import subprocess
    subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"])    # inherits our stdout
    print("started")
print("args:", sys.argv[1:])
"""


@pytest.fixture
def policy(tmp_path, monkeypatch):
    prog = tmp_path / "lab.py"
    prog.write_text(SCRIPT.format(py=sys.executable))
    prog.chmod(0o755)
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("CANARY_SECRET", "must-not-reach-scripts")
    monkeypatch.setenv("HOME", str(tmp_path))
    return parse({"mode": "manage", "scripts": [{
        "name": "lab", "containers": ["lab-*"], "timeout_s": 2, "status_action": "env",
        "actions": [{"name": "env", "argv": [str(prog), "env"]},
                    {"name": "talk", "argv": [str(prog), "talk"]},
                    {"name": "sleep", "argv": [str(prog), "sleep"], "args": ["0.1", "10"]},
                    {"name": "bg", "argv": [str(prog), "background"]},
                    {"name": "missing", "argv": [str(tmp_path / "nope.sh")]}]}]}, "/x")


def run(pol, action, arg=None, stream=None, runner=None):
    s = stream or FakeStream()
    res = (runner or Scripts()).run(pol, proto.clean_args("scripts.run", {"name": "lab", "action": action,
                                                                          "arg": arg}), s)
    return res, s


def test_environment_is_minimal(policy):
    res, s = run(policy, "env")
    keys = json.loads(s.events[0]["line"])
    assert set(keys) <= set(ENV_KEEP) and "CANARY_SECRET" not in keys and res == {"rc": 0}


def test_lines_are_streamed_scrubbed_and_rc_returned(policy):
    res, s = run(policy, "talk")
    assert res == {"rc": 3}
    lines = [(e["stream"], e["line"]) for e in s.events]
    assert ("out", "out line 1") in lines and ("err", "err line") in lines
    assert all("sk-test-not-a-real-key" not in line for _st, line in lines)


def test_arguments_come_from_the_allowlist(policy):
    res, s = run(policy, "sleep", "0.1")
    assert res == {"rc": 0} and any("'0.1'" in e["line"] for e in s.events)
    for action, arg, code in (("sleep", "rm -rf /", proto.POLICY_DENIED), ("sleep", None, proto.BAD_ARGS),
                              ("env", "x", proto.BAD_ARGS), ("nope", None, proto.NOT_FOUND),
                              ("missing", None, proto.NOT_FOUND)):
        with pytest.raises(OpError) as e:
            run(policy, action, arg)
        assert e.value.code == code, (action, arg)
    with pytest.raises(OpError) as e:
        Scripts().run(policy, {"name": "other", "action": "env", "arg": None}, FakeStream())
    assert e.value.code == proto.NOT_FOUND


def test_one_run_per_script(policy):
    runner = Scripts()
    s = FakeStream()
    t2_err: list[OpError] = []

    def second() -> None:
        try:
            run(policy, "env", runner=runner)
        except OpError as e:
            t2_err.append(e)

    long = FakeStream()
    t = threading.Thread(target=lambda: _swallow(run, policy, "sleep", "10", long, runner))
    t.start()
    time.sleep(0.3)
    second()
    assert t2_err and t2_err[0].code == proto.BUSY
    long._cancelled = True
    for h in long.hooks:
        h()
    t.join(10)
    assert s.events == []


def _swallow(fn, *a):
    with contextlib.suppress(OpError, Cancelled):
        fn(*a)


def test_timeout_stops_the_run(policy):
    t0 = time.monotonic()
    with pytest.raises(OpError) as e:
        run(policy, "sleep", "10")
    assert e.value.code == proto.TIMEOUT and time.monotonic() - t0 < 8


def test_cancel_stops_the_run(policy):
    s = FakeStream()
    threading.Timer(0.3, lambda: (setattr(s, "_cancelled", True), [h() for h in s.hooks])).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        run(policy, "sleep", "10", s)
    assert time.monotonic() - t0 < 5


def test_a_background_child_holding_the_pipe_does_not_hang_the_run(policy):
    t0 = time.monotonic()
    res, s = run(policy, "bg")
    assert res == {"rc": 0} and any(e["line"] == "started" for e in s.events)
    assert time.monotonic() - t0 < 6


def test_status_patterns(policy):
    from ecnode.discovery import script_status

    assert script_status(policy, ["lab-sglang-1"], 1.0)[0]["status"] == "up"
    assert script_status(policy, ["other"], 1.0)[0]["status"] == "down"
    assert Path(sys.executable).exists()
