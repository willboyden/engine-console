"""`engine-console hosts ...`: list, check (against the fake node, real subprocess) and init-key (real ssh-keygen)."""
from __future__ import annotations

import io
import json
import shutil
import stat
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from engine_console.cli import hosts_cli
from engine_console.config import Settings
from engine_console.services.hosts import key_line

from .harness import alive, node_argv

needs_keygen = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen is not installed")
Make = Callable[..., Settings]


def run(argv: list[str], cfg: Settings) -> tuple[int, str]:
    out = io.StringIO()
    code = hosts_cli(argv, cfg=cfg, out=out)
    return code, out.getvalue()


def write_hosts(path: Path, hosts: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps({"hosts": hosts}))
    path.chmod(0o600)
    return path


# ── list ────────────────────────────────────────────────────────────────────────────────────────────────
def test_list_without_a_file_shows_this_machine(make_settings: Make) -> None:
    code, out = run(["list"], make_settings())
    assert code == 0 and "not present: only this machine is managed" in out
    rows = out.splitlines()
    assert rows[1].split() == ["ID", "LABEL", "REACHED", "BY", "ENABLED"] and rows[2].startswith("local")
    assert "this machine" in rows[2] and rows[2].rstrip().endswith("yes")


def test_list_shows_how_each_host_is_reached(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [
        {"id": "local", "label": "Workstation"},
        {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5", "port": 2222},
        {"id": "edge", "ssh": "edge-alias", "ssh_config": True, "enabled": False}])
    code, out = run(["list"], make_settings(hosts_file=hosts_file))
    assert code == 0 and "not present" not in out
    lines = out.splitlines()
    assert any(ln.startswith("gpu-box-2") and "ssh ops@10.0.0.5 -p 2222" in ln and ln.rstrip().endswith("yes") for ln in lines)
    assert any(ln.startswith("edge") and "(ssh_config)" in ln and ln.rstrip().endswith("no") for ln in lines)


def test_a_bad_hosts_file_is_an_error_on_stderr_and_a_failing_exit_code(make_settings: Make, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = write_hosts(tmp_path / "hosts.json", [])
    code, out = run(["list"], make_settings(hosts_file=bad))
    err = capsys.readouterr().err
    assert code == 1 and out == "" and err.startswith("engine-console: ") and str(bad) in err and "1-32" in err


def test_usage_errors_return_two_and_help_returns_zero(make_settings: Make, capsys: pytest.CaptureFixture[str]) -> None:
    assert hosts_cli([], cfg=make_settings()) == 2
    assert hosts_cli(["frobnicate"], cfg=make_settings()) == 2
    assert hosts_cli(["init-key"], cfg=make_settings()) == 2                       # --from is required
    assert hosts_cli(["--help"], cfg=make_settings()) == 0
    assert "init-key" in capsys.readouterr().out


# ── init-key ────────────────────────────────────────────────────────────────────────────────────────────
@needs_keygen
def test_init_key_creates_the_key_and_prints_the_one_line(make_settings: Make, tmp_path: Path) -> None:
    cfg = make_settings()
    code, out = run(["init-key", "--from", "10.0.0.5"], cfg)
    assert code == 0 and f"created {cfg.hub_key_file} (private half 0600, never printed)" in out
    pub = cfg.hub_key_file.with_name(cfg.hub_key_file.name + ".pub").read_text()
    line = key_line(pub, "10.0.0.5", cfg.remote_node_command)
    assert line in out.splitlines()
    assert 'restrict,from="10.0.0.5",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 ' in line
    assert "PRIVATE KEY" not in out and cfg.hub_key_file.read_text().splitlines()[1] not in out
    assert stat.S_IMODE(cfg.hub_key_file.stat().st_mode) == 0o600
    assert f"Then list the machines in {cfg.hosts_file}" in out and "WARNING" not in out


@needs_keygen
def test_init_key_twice_reuses_the_key(make_settings: Make) -> None:
    cfg = make_settings()
    _, first = run(["init-key", "--from", "10.0.0.5"], cfg)
    before = cfg.hub_key_file.read_bytes()
    code, second = run(["init-key", "--from", "10.0.0.5"], cfg)
    assert code == 0 and f"using the existing key {cfg.hub_key_file}" in second and cfg.hub_key_file.read_bytes() == before
    assert [ln for ln in first.splitlines() if ln.startswith("restrict")] == [ln for ln in second.splitlines() if ln.startswith("restrict")]


@needs_keygen
def test_init_key_honours_the_key_file_option_and_the_configured_remote_command(make_settings: Make, tmp_path: Path) -> None:
    cfg = make_settings(remote_node_command="/opt/ecnode/bin/ecnode --stdio")
    other = tmp_path / "elsewhere" / "key"
    code, out = run(["init-key", "--from", "10.0.0.0/24,10.1.0.5", "--key-file", str(other)], cfg)
    assert code == 0 and other.exists() and not cfg.hub_key_file.exists()
    assert 'from="10.0.0.0/24,10.1.0.5",command="/opt/ecnode/bin/ecnode --stdio"' in out


@needs_keygen
def test_a_wildcard_from_is_allowed_but_warned_about(make_settings: Make) -> None:
    code, out = run(["init-key", "--from", "*"], make_settings())
    assert code == 0 and "WARNING" in out and 'from="*"' in out


def test_init_key_refuses_a_from_that_could_break_out_of_its_quotes(make_settings: Make, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = make_settings()
    code, out = run(["init-key", "--from", 'x",command="evil'], cfg)
    assert code == 1 and out == "" and "--from" in capsys.readouterr().err
    assert not cfg.hub_key_file.exists() and not cfg.hub_key_file.parent.exists()      # nothing was created


# ── check ───────────────────────────────────────────────────────────────────────────────────────────────
def test_check_reports_a_working_node(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [{"id": "local", "label": "Workstation"},
                                                       {"id": "off", "ssh": "ops@10.0.0.9", "enabled": False}])
    cfg = make_settings(hosts_file=hosts_file, node_command=node_argv(hello_patch={"policy": {"mode": "manage"}}))
    t0 = time.monotonic()
    code, out = run(["check"], cfg)
    assert code == 0 and time.monotonic() - t0 < 15
    rows = [ln.split() for ln in out.splitlines()]
    assert rows[0] == ["HOST", "STATE", "NODE", "ARCH", "GPUS", "POLICY"]
    assert rows[1] == ["local", "up", "0.4.0-test", "x86_64", "2", "manage"]
    assert rows[2][:2] == ["off", "disabled"]                                          # a disabled host is not a failure
    assert "off: disabled in hosts.json" in out


def test_check_reports_a_node_that_fails_and_exits_nonzero(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [{"id": "local"}])
    cfg = make_settings(hosts_file=hosts_file, node_command=node_argv(stderr="Traceback...\nValueError: boom", exit_before_hello=1))
    code, out = run(["check"], cfg)
    assert code == 1
    assert [ln.split()[:2] for ln in out.splitlines()][1] == ["local", "down"]
    assert "local: the node exited (1): ValueError: boom" in out


def test_check_reports_a_version_mismatch_in_words(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [{"id": "local", "label": "Workstation"}])
    code, out = run(["check"], make_settings(hosts_file=hosts_file, node_command=node_argv(proto=2)))
    assert code == 1 and "incompatible" in out and "Update the Engine Console hub" in out


def test_check_prints_a_policy_problem_and_notes(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [{"id": "local"}])
    cfg = make_settings(hosts_file=hosts_file, node_command=node_argv(
        junk="Welcome\n", hello_patch={"policy": {"error": "policy.json is group-writable"}}))
    code, out = run(["check"], cfg)
    assert code == 0 and "node policy: policy.json is group-writable" in out
    assert "local: note: ignored 8 byte(s)" in out and [ln.split() for ln in out.splitlines()][1][-1] == "monitor"


def test_check_gives_up_waiting_after_the_timeout_and_cleans_up(make_settings: Make, tmp_path: Path) -> None:
    hosts_file = write_hosts(tmp_path / "hosts.json", [{"id": "local"}])
    log = tmp_path / "slow.jsonl"
    cfg = make_settings(hosts_file=hosts_file, node_command=node_argv(hello_delay_s=30, log=str(log)))
    t0 = time.monotonic()
    code, out = run(["check", "--timeout", "1"], cfg)
    assert code == 1 and time.monotonic() - t0 < 12
    assert [ln.split()[:2] for ln in out.splitlines()][1] == ["local", "connecting"] and "local: no answer yet" in out
    (pid,) = [json.loads(line)["pid"] for line in log.read_text().splitlines() if '"start"' in line][:1]
    assert not alive(pid)                                                              # the silent node was killed on the way out


def test_check_with_a_bad_hosts_file_fails_cleanly(make_settings: Make, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = write_hosts(tmp_path / "hosts.json", [])
    code, _ = run(["check"], make_settings(hosts_file=bad))
    assert code == 1 and "engine-console: " in capsys.readouterr().err


def test_check_with_the_fleet_disabled_reports_every_host_disabled(make_settings: Make) -> None:
    code, out = run(["check"], make_settings(fleet_enabled=False))
    assert code == 0 and [ln.split()[:2] for ln in out.splitlines()][1] == ["local", "disabled"]
