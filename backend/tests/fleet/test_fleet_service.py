"""FleetService: its hosts and links, summaries and details, the "fleet" topic, and the default link factories."""
from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from engine_console.config import Settings
from engine_console.domain.errors import Conflict, NotFound
from engine_console.domain.fleet import (
    HostConfig,
    HostCounts,
    LinkStatus,
    NodeError,
    NodeKind,
    NodeStream,
    NodeUnavailable,
)
from engine_console.services import fleet as fleet_mod
from engine_console.services.common import EventBus
from engine_console.services.fleet import DisabledLink, FailedStream, FleetService
from engine_console.services.hosts import HostsError, ssh_argv

from .builders import disks, hello, mount, root, sample
from .fake_node import read_log
from .harness import FAKE, alive, node_argv, wait_until


class FakeLink:
    """A link that never starts a process: state and data are set by the test."""

    def __init__(self, host: HostConfig, *, state: str = "up", **latest: dict[str, Any]) -> None:
        self.host = host
        self.state = state
        self.error: str | None = None
        self.data: dict[str, dict[str, Any]] = dict(latest)
        self.calls: list[tuple[str, dict[str, Any] | None, float]] = []
        self.results: dict[str, Any] = {}
        self.delay = 0.0
        self.started = self.stopped = False
        self.reconnects = 0

    def link(self) -> LinkStatus:
        return LinkStatus(state=self.state, error=self.error)  # type: ignore[arg-type]

    def latest(self, kind: NodeKind) -> dict[str, Any] | None:
        return self.data.get(kind)

    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        self.calls.append((op, args, timeout))
        if self.delay:
            await asyncio.sleep(self.delay)
        r = self.results.get(op, {})
        if isinstance(r, Exception):
            raise r
        assert isinstance(r, dict)
        return r

    def open(self, op: str, args: dict[str, Any] | None = None) -> NodeStream:
        return FailedStream(NodeUnavailable(self.host.id, "fake"))

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def reconnect(self) -> None:
        self.reconnects += 1


A = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5")
L = HostConfig(id="local", label="Workstation")
IMAGES = {"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 5, "created": 1.0, "arch": "amd64"}]}


def make(cfg: Settings, hosts: list[HostConfig] | None = None, **states: str) -> tuple[FleetService, dict[str, FakeLink], EventBus]:
    bus = EventBus()
    links: dict[str, FakeLink] = {}

    def factory(h: HostConfig) -> FakeLink:
        links[h.id] = FakeLink(h, state=states.get(h.id, "up"), hello=hello(), sample=sample(), disks=disks([mount("/")]))
        links[h.id].results["docker.images"] = IMAGES
        return links[h.id]

    return FleetService(cfg, bus, hosts=hosts or [L, A], link_factory=factory), links, bus


# ── hosts and nodes ─────────────────────────────────────────────────────────────────────────────────────
def test_hosts_and_nodes(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings())
    got = fleet.hosts()
    assert [h.id for h in got] == ["local", "gpu-box-2"]
    got.clear()
    assert len(fleet.hosts()) == 2                                  # a copy: callers cannot edit the fleet
    assert fleet.node("gpu-box-2") is links["gpu-box-2"]


def test_an_unknown_host_is_not_found_with_a_stable_code(make_settings: Callable[..., Settings]) -> None:
    fleet, _, _ = make(make_settings())
    with pytest.raises(NotFound) as e:
        fleet.node("nope")
    assert e.value.code == "unknown_host" and e.value.status == 404
    with pytest.raises(NotFound) as long:
        fleet.node("x" * 500)
    assert len(long.value.detail) < 100                             # an attacker-sized id is not echoed back whole
    for call in (lambda: fleet.summary("nope"), lambda: fleet.reconnect("nope")):
        with pytest.raises(NotFound):
            call()


def test_two_hosts_with_one_id_are_refused(make_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValueError, match="share an id"):
        make(make_settings(), hosts=[A, A])


def test_summaries_follow_the_configuration_order(make_settings: Callable[..., Settings]) -> None:
    fleet, _, _ = make(make_settings(), hosts=[A, L])
    assert [s.id for s in fleet.summaries()] == ["gpu-box-2", "local"]
    s = fleet.summary("gpu-box-2")
    assert s.remote is True and s.link.state == "up" and s.gpus and s.node is not None and s.node.version == "0.4.0"


def test_other_services_plug_in_a_resolver_and_a_counts_provider(make_settings: Callable[..., Settings]) -> None:
    fleet, _, _ = make(make_settings())
    asked: list[tuple[str, str]] = []
    fleet.set_instance_resolver(lambda hid, container: asked.append((hid, container)) or f"inst_{hid}")
    fleet.set_counts_provider(lambda hid: HostCounts(engines_running=1, models=4))
    s = fleet.summary("local")
    assert s.gpus[0].procs[0].instance_id == "inst_local" and s.counts.models == 4 and ("local", "ec-engine-1") in asked
    fleet.set_instance_resolver(None)
    fleet.set_counts_provider(None)
    s = fleet.summary("local")
    assert s.gpus[0].procs[0].instance_id is None and s.counts == HostCounts()


async def test_start_and_stop_reach_every_link(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings())
    await fleet.start()
    assert all(link.started for link in links.values())
    await fleet.start()                                              # idempotent for the watcher
    await fleet.stop()
    assert all(link.stopped for link in links.values())
    await fleet.stop()                                               # and stopping twice is harmless


# ── detail and images ───────────────────────────────────────────────────────────────────────────────────
async def test_detail_adds_disks_roots_scripts_addresses_and_images(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings())
    links["gpu-box-2"].data["hello"] = hello(roots=[root("hf", "/data/hf")])
    links["gpu-box-2"].data["disks"] = disks([mount("/"), mount("/data", fstype="zfs", source="pool/data")])
    d = await fleet.detail("gpu-box-2")
    assert [m.label for m in d.disks] == ["/ (ext4)", "pool (zfs)"] and d.roots[0].mount is not None
    assert d.roots[0].mount.mount == "/data" and d.addresses == ["10.0.0.5/24 (eth0)"]
    assert [i.ref for i in d.images] == ["vllm/vllm-openai:v0.1"] and d.id == "gpu-box-2" and d.gpus
    with pytest.raises(NotFound):
        await fleet.detail("nope")


async def test_images_are_asked_for_once_a_minute_not_on_every_detail(make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(fleet_mod, "time", types.SimpleNamespace(monotonic=lambda: now[0], time=lambda: 0.0))
    fleet, links, _ = make(make_settings())
    link = links["local"]
    for _ in range(3):
        assert [i.ref for i in (await fleet.detail("local")).images] == ["vllm/vllm-openai:v0.1"]
    assert [c[0] for c in link.calls] == ["docker.images"]
    assert link.calls[0][2] <= 5.0                                  # a slow node cannot hold the page for long
    now[0] += 59
    await fleet.detail("local")
    assert len(link.calls) == 1
    now[0] += 2
    link.results["docker.images"] = {"images": []}
    assert (await fleet.detail("local")).images == []
    assert len(link.calls) == 2


async def test_a_failing_image_call_keeps_the_old_list_and_is_not_retried_at_once(make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch) -> None:
    now = [1000.0]
    monkeypatch.setattr(fleet_mod, "time", types.SimpleNamespace(monotonic=lambda: now[0], time=lambda: 0.0))
    fleet, links, _ = make(make_settings())
    link = links["local"]
    await fleet.detail("local")
    now[0] += 61
    link.results["docker.images"] = NodeError("local", "timeout", "slow")
    assert [i.ref for i in (await fleet.detail("local")).images] == ["vllm/vllm-openai:v0.1"]    # last good list
    assert [i.ref for i in (await fleet.detail("local")).images] == ["vllm/vllm-openai:v0.1"]
    assert len(link.calls) == 2                                      # the failure was remembered for a few seconds
    now[0] += 11
    link.results["docker.images"] = IMAGES
    await fleet.detail("local")
    assert len(link.calls) == 3


async def test_no_image_call_for_a_host_that_is_down_or_has_no_docker(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings(), local="down")
    assert (await fleet.detail("local")).images == [] and links["local"].calls == []
    links["gpu-box-2"].data["hello"] = hello()
    links["gpu-box-2"].data["hello"]["docker"]["available"] = False
    assert (await fleet.detail("gpu-box-2")).images == [] and links["gpu-box-2"].calls == []
    links["gpu-box-2"].data.pop("hello")
    assert (await fleet.detail("gpu-box-2")).images == [] and links["gpu-box-2"].calls == []


async def test_concurrent_details_share_one_image_call(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings())
    links["local"].delay = 0.1
    await asyncio.gather(*(fleet.detail("local") for _ in range(8)))
    assert len(links["local"].calls) == 1


async def test_detail_of_a_stale_host_still_asks_for_images(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings(), local="stale")
    assert (await fleet.detail("local")).images and len(links["local"].calls) == 1


# ── reconnect ───────────────────────────────────────────────────────────────────────────────────────────
def test_reconnect_kicks_the_link_and_returns_the_summary(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings(), **{"gpu-box-2": "down"})
    s = fleet.reconnect("gpu-box-2")
    assert links["gpu-box-2"].reconnects == 1 and s.id == "gpu-box-2" and s.link.state == "down"


def test_reconnecting_a_disabled_host_is_a_conflict(make_settings: Callable[..., Settings]) -> None:
    fleet, links, _ = make(make_settings(), local="disabled")
    with pytest.raises(Conflict) as e:
        fleet.reconnect("local")
    assert e.value.code == "host_disabled" and links["local"].reconnects == 0


# ── the "fleet" topic ───────────────────────────────────────────────────────────────────────────────────
async def test_state_changes_are_published_as_host_summaries(make_settings: Callable[..., Settings], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(FleetService, "WATCH_S", 0.05)
    fleet, links, bus = make(make_settings())
    got: list[dict[str, Any]] = []

    async def collect() -> None:
        async for ev in bus.subscribe("fleet"):
            got.append(ev)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.02)
    await fleet.start()
    await wait_until(lambda: {e["id"] for e in got} == {"local", "gpu-box-2"}, what="the first snapshot of both hosts")
    n = len(got)
    links["gpu-box-2"].state = "down"
    links["gpu-box-2"].error = "the node exited (3)"
    await wait_until(lambda: any(e["id"] == "gpu-box-2" and e["link"]["state"] == "down" for e in got), what="the down event")
    down = [e for e in got if e["link"]["state"] == "down"][0]
    assert down["link"]["error"] == "the node exited (3)" and down["label"] == "GPU box 2" and "gpus" in down
    assert len(got) >= n + 1
    quiet = len(got)
    await asyncio.sleep(0.3)
    assert len(got) == quiet                                         # nothing changed, nothing published
    await fleet.stop()
    task.cancel()


# ── switched off ────────────────────────────────────────────────────────────────────────────────────────
async def test_with_the_fleet_disabled_nothing_connects_and_the_hosts_file_is_not_read(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    bad = tmp_path / "hosts.json"
    bad.write_text("{not even json")
    bad.chmod(0o666)                                                  # would be refused if it were read
    fleet = FleetService(make_settings(fleet_enabled=False, hosts_file=bad), EventBus())
    await fleet.start()
    (host,) = fleet.hosts()
    assert host.id == "local"
    s = fleet.summary("local")
    assert s.link.state == "disabled" and "FLEET_ENABLED" in (s.link.error or "")
    with pytest.raises(NodeUnavailable):
        await fleet.node("local").call("docker.images")
    stream = fleet.node("local").open("docker.images")
    assert [i async for i in stream] == []
    with pytest.raises(NodeUnavailable):
        await stream.result()
    assert fleet.node("local").latest("hello") is None
    await fleet.stop()


async def test_a_host_disabled_in_the_file_gets_a_disabled_link_and_the_rest_connect(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    hosts_file = tmp_path / "hosts.json"
    hosts_file.write_text(json.dumps({"hosts": [{"id": "local"}, {"id": "off", "ssh": "ops@10.0.0.9", "enabled": False}]}))
    hosts_file.chmod(0o600)
    cfg = make_settings(hosts_file=hosts_file, node_command=node_argv())
    fleet = FleetService(cfg, EventBus())
    assert isinstance(fleet.node("off"), DisabledLink) and not isinstance(fleet.node("local"), DisabledLink)
    await fleet.start()
    try:
        await wait_until(lambda: fleet.summary("local").link.state == "up", what="the local node")
        off = fleet.summary("off")
        assert off.link.state == "disabled" and off.link.error == "disabled in hosts.json" and off.remote is True
    finally:
        await fleet.stop()


def test_a_bad_hosts_file_stops_the_console_at_construction(make_settings: Callable[..., Settings], tmp_path: Path) -> None:
    hosts_file = tmp_path / "hosts.json"
    hosts_file.write_text('{"hosts": []}')
    hosts_file.chmod(0o600)
    with pytest.raises(HostsError, match="1-32"):
        FleetService(make_settings(hosts_file=hosts_file), EventBus())


def test_no_hosts_file_means_the_local_machine(make_settings: Callable[..., Settings]) -> None:
    fleet = FleetService(make_settings(), EventBus())
    assert [h.id for h in fleet.hosts()] == ["local"]


# ── the default local link: a real subprocess ───────────────────────────────────────────────────────────
async def test_the_default_local_link_runs_the_configured_command_with_a_scrubbed_environment(
        make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_this_must_stay_in_the_console")
    monkeypatch.setenv("VLLM_API_KEY", "also-secret")
    log = tmp_path / "local.jsonl"
    fleet = FleetService(make_settings(node_command=node_argv(log=str(log))), EventBus(), hosts=[L])
    bus_events: list[dict[str, Any]] = []

    async def collect() -> None:
        async for ev in fleet._bus.subscribe("fleet"):
            bus_events.append(ev)

    task = asyncio.create_task(collect())
    await asyncio.sleep(0.02)
    await fleet.start()
    try:
        await wait_until(lambda: any(e["link"]["state"] == "up" for e in bus_events), what="an 'up' event on the fleet topic")
        up = next(e for e in bus_events if e["link"]["state"] == "up")
        assert up["id"] == "local" and up["node"]["version"] == "0.4.0-test" and len(up["gpus"]) == 2
        s = fleet.summary("local")
        assert s.link.state == "up" and s.docker is not None and s.docker.available
        d = await fleet.detail("local")
        assert [i.ref for i in d.images] == ["vllm/vllm-openai:v0.1"]
        await fleet.detail("local")
        assert len([e for e in read_log(log) if e["ev"] == "req" and e["op"] == "docker.images"]) == 1   # cached
        (start,) = [e for e in read_log(log) if e["ev"] == "start"]
        assert "HF_TOKEN" not in start["env"] and "VLLM_API_KEY" not in start["env"]
        assert fleet.node("local").latest("hello") is not None
    finally:
        await fleet.stop()
        task.cancel()
    (pid,) = [e["pid"] for e in read_log(log) if e["ev"] == "start"]
    assert not alive(pid)


# ── the default remote link: a fake `ssh` on PATH ───────────────────────────────────────────────────────
def install_fake_ssh(bindir: Path, log: Path, mode: str, scenario: dict[str, Any] | None = None, key: Path | None = None) -> None:
    """A stand-in `ssh`: records its argv and the names of its environment variables, then either behaves like a node
    (exec of the fake node, so the pipes are the same) or fails like ssh does."""
    bindir.mkdir(exist_ok=True)
    script = bindir / "ssh"
    script.write_text(f"""#!{sys.executable}
import json, os, sys
with open({str(log)!r}, "a") as f:
    f.write(json.dumps({{"argv": sys.argv[1:], "env": sorted(os.environ)}}) + "\\n")
if {mode!r} == "denied":
    sys.stderr.write("ops@10.0.0.5: Permission denied (publickey).\\n")
    sys.exit(255)
if {mode!r} == "badkey":
    sys.stderr.write('Load key "' + {str(key)!r} + '": error in libcrypto\\n')
    sys.exit(255)
os.execv({sys.executable!r}, [{sys.executable!r}, {str(FAKE)!r}, {json.dumps(json.dumps(scenario or {}))}])
""")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    try:
        subprocess.run([str(script)], capture_output=True, timeout=10, check=False, stdin=subprocess.DEVNULL,
                       env={"PATH": os.environ["PATH"]})
    except PermissionError:
        pytest.skip("the temp dir does not allow executing scripts")
    log.unlink(missing_ok=True)


def key_file(tmp_path: Path) -> Path:
    key = tmp_path / "keys" / "hub-key"
    key.parent.mkdir()
    key.write_text("stand-in, not a real key\n")
    return key


async def test_the_default_remote_link_runs_the_documented_ssh_command(
        make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SSH_AUTH_SOCK", "/run/some-agent")
    monkeypatch.setenv("HF_TOKEN", "hf_must_not_reach_ssh")
    ssh_log = tmp_path / "ssh.jsonl"
    install_fake_ssh(tmp_path / "bin", ssh_log, "node", {"forced_cmd": True})
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    cfg = make_settings(hub_key_file=key_file(tmp_path))
    host = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5", port=2222)
    fleet = FleetService(cfg, EventBus(), hosts=[host])
    await fleet.start()
    try:
        await wait_until(lambda: fleet.summary("gpu-box-2").link.state == "up", what="the remote node")
        (call,) = [json.loads(line) for line in ssh_log.read_text().splitlines()]
        assert ["ssh", *call["argv"]] == ssh_argv(host, cfg.hub_key_file, cfg.remote_node_command)
        assert set(call["env"]) <= {"HOME", "PATH", "LANG", "USER"}          # nothing of the console's environment
        s = fleet.summary("gpu-box-2")
        assert s.remote is True and s.link.notes == [] and s.node is not None and s.node.version == "0.4.0-test"
    finally:
        await fleet.stop()


async def test_a_login_the_key_cannot_make_names_the_fix_and_waits_five_minutes(
        make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_ssh(tmp_path / "bin", tmp_path / "ssh.jsonl", "denied")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    fleet = FleetService(make_settings(hub_key_file=key_file(tmp_path)), EventBus(), hosts=[A])
    await fleet.start()
    try:
        await wait_until(lambda: fleet.summary("gpu-box-2").link.state == "down" and fleet.summary("gpu-box-2").link.retry_in_s is not None,
                         what="the refused login")
        link = fleet.summary("gpu-box-2").link
        assert "hosts init-key" in (link.error or "") and "GPU box 2" in (link.error or "")
        assert link.retry_in_s is not None and 295 <= link.retry_in_s <= 300
    finally:
        await fleet.stop()


async def test_the_hub_key_path_never_reaches_the_link_error(
        make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    key = key_file(tmp_path)
    install_fake_ssh(tmp_path / "bin", tmp_path / "ssh.jsonl", "badkey", key=key)
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    fleet = FleetService(make_settings(hub_key_file=key), EventBus(), hosts=[A])
    await fleet.start()
    try:
        await wait_until(lambda: fleet.summary("gpu-box-2").link.state == "down" and fleet.summary("gpu-box-2").link.retry_in_s is not None,
                         what="the failed login")
        error = fleet.summary("gpu-box-2").link.error or ""
        assert error == 'ssh to ops@10.0.0.5 failed: Load key "<hub key>": error in libcrypto' and str(key) not in error
    finally:
        await fleet.stop()


async def test_without_a_hub_key_ssh_is_never_started(
        make_settings: Callable[..., Settings], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ssh would otherwise try the user's own default keys when the key named with -i is missing."""
    ssh_log = tmp_path / "ssh.jsonl"
    install_fake_ssh(tmp_path / "bin", ssh_log, "node")
    monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}")
    cfg = make_settings(hub_key_file=tmp_path / "keys" / "does-not-exist")
    fleet = FleetService(cfg, EventBus(), hosts=[A])
    await fleet.start()
    try:
        await wait_until(lambda: fleet.summary("gpu-box-2").link.error, what="the missing-key error")
        link = fleet.summary("gpu-box-2").link
        assert link.state in ("down", "connecting") and "no ssh key yet" in (link.error or "") and "hosts init-key" in (link.error or "")
        await asyncio.sleep(0.3)
        assert not ssh_log.exists()                                       # ssh was never run
        cfg.hub_key_file.parent.mkdir(exist_ok=True)                      # create the key where the settings point...
        cfg.hub_key_file.write_text("stand-in, not a real key\n")
        fleet.reconnect("gpu-box-2")                                      # ...and the next attempt uses it
        await wait_until(lambda: ssh_log.exists(), timeout=10, what="ssh to be started now that the key exists")
    finally:
        await fleet.stop()
