"""Shared fixtures. Nothing here touches the real docker, GPUs, /proc, /sys or network: PATH holds only fake
`docker`, `nvidia-smi` and `ip` programs, machine trees live under tmp_path, and endpoint probes are refused
for any port that is not a fake engine's. The node gives its children an environment built from scratch, so each
fake's settings travel inside its wrapper, not in the test's environment."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from ecnode import discovery
from ecnode.agent import Agent
from ecnode.disks import Disks
from ecnode.gpus import Gpus
from ecnode.hostinfo import Paths
from ecnode.policy import PolicyStore

FAKES = Path(__file__).parent / "fakes"
GPU_A = "GPU-aaaaaaaa-1111-2222-3333-444444444444"
GPU_B = "GPU-bbbbbbbb-1111-2222-3333-444444444444"
CID_ENGINE = "e" * 64
CID_EXT = "f" * 64
GATEWAY_IMAGE = "nginx@sha256:" + "0" * 64
ENV_VALUE = "NEVER-LEAK-THIS-ENV-VALUE"          # in every fake container's Config.Env: must never be reported
ARGV_SECRET = "sk-test-not-a-real-key-0000"      # in a fake container's argv: must be redacted


def _wrapper(bin_dir: Path, name: str, module: str, settings: dict[str, str]) -> None:
    # CHILD_ENV: the environment the node gave this child, taken before the wrapper adds the fake's settings.
    p = bin_dir / name
    p.write_text(f"#!{sys.executable}\nimport os, sys\nCHILD_ENV = dict(os.environ)\nos.environ.update({settings!r})\n"
                 f"sys.path.insert(0, {str(FAKES)!r})\nimport {module}\n{module}.CHILD_ENV = CHILD_ENV\n"
                 f"sys.exit({module}.main())\n")
    p.chmod(0o755)


@dataclass
class FakeBin:
    dir: Path
    docker_state: Path
    docker_log: Path
    smi_state: Path

    def state(self) -> dict[str, Any]:
        return json.loads(self.docker_state.read_text())

    def set_state(self, state: dict[str, Any]) -> None:
        self.docker_state.write_text(json.dumps(state))

    def update(self, **kw: Any) -> None:
        s = self.state()
        s.update(kw)
        self.set_state(s)

    def calls(self) -> list[dict[str, Any]]:
        if not self.docker_log.exists():
            return []
        return [json.loads(line) for line in self.docker_log.read_text().splitlines()]

    def smi(self, **kw: Any) -> None:
        s = json.loads(self.smi_state.read_text())
        s.update(kw)
        self.smi_state.write_text(json.dumps(s))


def container(name: str, *, cid: str, image: str, args: list[str], labels: dict[str, str] | None = None,
              running: bool = True, pid: int = 4242, ports: dict[str, Any] | None = None,
              mounts: list[dict[str, Any]] | None = None, network: str = "bridge",
              device_requests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"Id": cid, "Name": "/" + name, "Image": "sha256:" + "ab" * 32, "Created": "2026-09-01T10:00:00.123456789Z",
            "Path": args[0], "Args": args[1:],
            "State": {"Status": "running" if running else "exited", "Running": running, "Pid": pid if running else 0,
                      "ExitCode": 0, "StartedAt": "2026-09-01T10:00:01.5Z"},
            "Config": {"Image": image, "Labels": labels or {},
                       "Env": [f"API_TOKEN={ENV_VALUE}", "PATH=/usr/bin"]},
            "HostConfig": {"NetworkMode": network, "DeviceRequests": device_requests or []},
            "NetworkSettings": {"Ports": ports or {}}, "Mounts": mounts or []}


def default_docker_state() -> dict[str, Any]:
    engine_labels = {"engine-console": "1", "engine-console.role": "engine", "engine-console.instance": "inst_1",
                     "engine-console.engine": "vllm"}
    return {
        "context": "testctx",
        "info": {"ServerVersion": "29.0.0", "SecurityOptions": ["name=seccomp,profile=builtin", "name=rootless"],
                 "Runtimes": {"runc": {}, "nvidia": {}}, "HttpProxy": "http://user:pw@proxy.invalid:3128"},
        "containers": [
            container("ec-owned-1", cid=CID_ENGINE, image="vllm/vllm-openai:v0.11.0",
                      args=["vllm", "serve", "org/model-a", "--api-key", ARGV_SECRET,
                            "--port", "8000"], labels=engine_labels),
            container("other-engine", cid=CID_EXT, image="lmsysorg/sglang:v0.5.0", pid=4343,
                      args=["python3", "-m", "sglang.launch_server", "--model-path", "org/model-b",
                            "--api-key=topsecretvalue", "--port", "30000"],
                      labels={"com.docker.compose.project": "lab", "com.docker.compose.service": "sglang"}),
            container("grafana", cid="a" * 64, image="grafana/grafana:11", pid=4444, args=["/run.sh"]),
        ],
        "images": [
            {"Id": "sha256:" + "11" * 32, "RepoTags": ["vllm/vllm-openai:v0.11.0"], "Size": 123456789,
             "Created": "2026-08-01T00:00:00Z", "Architecture": "amd64"},
            {"Id": "sha256:" + "22" * 32, "RepoTags": [], "RepoDigests": [GATEWAY_IMAGE], "Size": 1234,
             "Created": "2026-08-02T00:00:00Z", "Architecture": "amd64"},
        ],
        "networks": {},
        "logs": {"ec-owned-1": ["INFO starting", f"non-default args: {{'api_key': ['{ARGV_SECRET}']}}", "INFO ready"]},
    }


def default_smi_state() -> dict[str, Any]:
    gpu = {"memory.total": "97887", "compute_cap": "12.0", "driver_version": "595.0", "utilization.gpu": "7",
           "memory.used": "1024", "temperature.gpu": "41", "power.draw": "71.50", "power.limit": "300.00",
           "fan.speed": "30"}
    return {"driver": "595.0", "cuda": "13.0",
            "gpus": [{**gpu, "index": 0, "uuid": GPU_A, "name": "NVIDIA Test GPU", "pci.bus_id": "00000000:01:00.0"},
                     {**gpu, "index": 1, "uuid": GPU_B, "name": "NVIDIA Test GPU", "pci.bus_id": "00000000:02:00.0"}],
            "apps": [{"pid": 4242, "used_memory": "2048", "gpu_uuid": GPU_A},
                     {"pid": 999999, "used_memory": "10", "gpu_uuid": GPU_B}]}


@pytest.fixture
def fakebin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBin:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fb = FakeBin(bin_dir, tmp_path / "docker.json", tmp_path / "docker.log", tmp_path / "smi.json")
    settings = {"FAKE_DOCKER_STATE": str(fb.docker_state), "FAKE_DOCKER_LOG": str(fb.docker_log),
                "FAKE_SMI_STATE": str(fb.smi_state)}
    for name, module in (("docker", "fake_docker"), ("nvidia-smi", "fake_smi"), ("ip", "fake_ip")):
        _wrapper(bin_dir, name, module, settings)
    fb.set_state(default_docker_state())
    fb.smi_state.write_text(json.dumps(default_smi_state()))
    monkeypatch.setenv("PATH", str(bin_dir))
    for var in ("SSH_ORIGINAL_COMMAND", "SSH_CONNECTION", "DOCKER_HOST", "DOCKER_CONTEXT"):
        monkeypatch.delenv(var, raising=False)
    return fb


# ── a fake machine: /proc, /sys, /etc and the mounts it reports ────────────────────────────────────────────
@dataclass
class Machine:
    root: Path
    paths: Paths
    data: Path          # an ext4 "disk" holding model roots
    nfs: Path           # an NFS mount

    def statvfs(self, path: str) -> os.statvfs_result:
        # 4 TiB disk, 1 TiB free, in 4 KiB blocks (f_bsize, f_frsize, f_blocks, f_bfree, f_bavail, ...)
        return os.statvfs_result((4096, 4096, 1 << 30, 1 << 28, 1 << 28, 1000, 500, 500, 0, 255))


def _w(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


@pytest.fixture
def machine(tmp_path: Path) -> Machine:
    root = tmp_path / "machine"
    proc, sysd, etc = root / "proc", root / "sys", root / "etc"
    data, nfs = tmp_path / "data", tmp_path / "nfs"
    data.mkdir()
    nfs.mkdir()
    _w(etc / "machine-id", "0123456789abcdef0123456789abcdef\n")
    _w(etc / "os-release", 'NAME="Test Linux"\nPRETTY_NAME="Test Linux 1.0"\n')
    _w(proc / "cpuinfo", "".join(f"processor\t: {i}\nmodel name\t: Test CPU 9000\nphysical id\t: 0\n"
                                 f"core id\t\t: {i // 2}\n\n" for i in range(4)))
    _w(proc / "meminfo", "MemTotal:       263000000 kB\nMemFree:         1000000 kB\nMemAvailable:   200000000 kB\n"
                         "Cached:          5000000 kB\nShmem:           2000000 kB\nSwapTotal:       8000000 kB\n"
                         "SwapFree:        8000000 kB\n")
    _w(proc / "stat", "cpu  100 0 100 800 0 0 0 0 0 0\ncpu0 1 0 1 8 0 0 0 0 0 0\nbtime 1700000000\n")
    _w(proc / "loadavg", "0.50 0.40 0.30 1/100 1234\n")
    _w(proc / "self" / "mountinfo",
       "1 0 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
       f"2 1 259:3 / {data} rw,noatime shared:2 - ext4 /dev/nvme1n1 rw\n"
       f"3 1 0:50 / {nfs} ro,relatime shared:3 - nfs4 10.0.0.9:/export/models ro,vers=4.2\n"
       "4 1 0:5 / /proc rw - proc proc rw\n"
       "5 1 7:1 / /snap/core/1 ro - squashfs /dev/loop1 ro\n")
    # a container's init process (pid 4242) in a rootless-docker cgroup, with memory and CPU counters
    cg = f"/user.slice/user-1000.slice/user@1000.service/user.slice/docker-{CID_ENGINE}.scope"
    _w(proc / "4242" / "cgroup", f"0::{cg}\n")
    _w(proc / "4242" / "comm", "vllm\n")
    _w(proc / "4242" / "stat", "4242 (vllm) S 1 4242 4242 0 -1 0\n")
    _w(proc / "4242" / "status", "Name:\tvllm\nVmRSS:\t  1024 kB\n")
    cgdir = sysd / "fs" / "cgroup" / cg.lstrip("/")
    _w(cgdir / "memory.current", "3000000000\n")
    _w(cgdir / "memory.stat", "anon 1000000000\nfile 1500000000\nshmem 500000000\nkernel 100000000\n")
    _w(cgdir / "cpu.stat", "usage_usec 5000000\nuser_usec 4000000\n")
    # pid 4343: no cgroup v2 dir -> RSS fallback over its process tree
    _w(proc / "4343" / "cgroup", f"0::/docker/{CID_EXT}\n")
    _w(proc / "4343" / "comm", "python3\n")
    _w(proc / "4343" / "stat", "4343 (python3) S 1 4343 4343 0 -1 0\n")
    _w(proc / "4343" / "status", "VmRSS:\t  2048 kB\n")
    _w(proc / "4344" / "stat", "4344 (worker) S 4343 4343 4343 0 -1 0\n")
    _w(proc / "4344" / "status", "VmRSS:\t  1024 kB\n")
    # block devices: a partition of nvme0n1, and a whole disk nvme1n1
    dev0 = sysd / "devices" / "pci0" / "nvme" / "nvme0" / "nvme0n1"
    _w(dev0 / "nvme0n1p2" / "partition", "2\n")
    _w(sysd / "block" / "nvme0n1" / "device" / "model", "Test SSD 4TB          \n")
    dev1 = sysd / "devices" / "pci0" / "nvme" / "nvme1" / "nvme1n1"
    dev1.mkdir(parents=True)
    _w(sysd / "block" / "nvme1n1" / "device" / "model", "Data Drive 8TB\n")
    (sysd / "dev" / "block").mkdir(parents=True)
    (sysd / "dev" / "block" / "259:2").symlink_to(dev0 / "nvme0n1p2")
    (sysd / "dev" / "block" / "259:3").symlink_to(dev1)
    paths = Paths(proc=proc, sys=sysd, etc=etc, state=tmp_path / "state")
    return Machine(root, paths, data, nfs)


# ── policies ─────────────────────────────────────────────────────────────────────────────────────────────
def write_policy(path: Path, data: dict[str, Any], mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(data))
    path.chmod(mode)
    return path


def manage_policy(machine: Machine, **over: Any) -> dict[str, Any]:
    for d in ("hf", "gguf", "plain", "ro"):
        (machine.data / d).mkdir(exist_ok=True)
    data = {
        "version": 1, "mode": "manage", "docker": {"binary": "docker", "context": "testctx"},
        "engine_images": ["vllm/vllm-openai:*", "lmsysorg/sglang@sha256:*", "ghcr.io/ggml-org/llama.cpp:server-cuda"],
        "gateway_image": GATEWAY_IMAGE, "engine_network": "engine-console-engines", "port_range": [18000, 18099],
        "discovery_ports": [],
        "model_roots": [
            {"id": "hf", "path": str(machine.data / "hf"), "kind": "hf-home", "writable": True, "label": "HF"},
            {"id": "gguf", "path": str(machine.data / "gguf"), "kind": "gguf", "writable": True},
            {"id": "plain", "path": str(machine.data / "plain"), "kind": "plain", "writable": True},
            {"id": "ro", "path": str(machine.data / "ro"), "kind": "auto", "writable": False},
            {"id": "shared", "path": str(machine.nfs), "kind": "auto", "writable": False},
        ],
        "scripts": [],
        "allow_pull": True, "allow_delete": True, "allow_ipc_host": True, "external_control": "stop_start",
        "max_shm_gib": 64,
    }
    data.update(over)
    return data


def store_for(tmp_path: Path, data: dict[str, Any] | None) -> PolicyStore:
    """A PolicyStore over a policy file in tmp_path (or no file at all), never looking at the real /etc."""
    path = None
    if data is not None:
        path = str(write_policy(tmp_path / "pol" / "policy.json", data))
    return PolicyStore(path, home=str(tmp_path / "home"), etc_path=str(tmp_path / "no-etc" / "policy.json"),
                       recheck_s=0.0)


# ── probes and a fake engine ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SafeProber:
    """Probes only the ports a test allows (fake engines); anything else is refused without a connection."""
    allowed: set[int] = field(default_factory=set)
    tried: list[int] = field(default_factory=list)

    def probe(self, port: int) -> discovery.PortProbe:
        self.tried.append(port)
        if port not in self.allowed:
            empty = discovery.Probe()
            return discovery.PortProbe(port, empty, empty, empty)
        return discovery.probe_port(port)

    def get(self, port: int, path: str, **kw: Any) -> discovery.Probe:
        if port not in self.allowed:
            return discovery.Probe()
        return discovery.http_get(port, path, **kw)


@pytest.fixture
def prober() -> SafeProber:
    return SafeProber()


METRICS_TEXT = ("# HELP vllm:num_requests_running Requests running.\n# TYPE vllm:num_requests_running gauge\n"
                'vllm:num_requests_running{model_name="org/model-a"} 1.0\n'
                "# HELP process_cpu_seconds_total CPU.\n# TYPE process_cpu_seconds_total counter\n"
                "process_cpu_seconds_total 5.0\n")


class _EngineHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    seen: list[dict[str, Any]] = []

    def log_message(self, *a: Any) -> None:
        pass

    def _send(self, status: int, body: bytes, ctype: str = "application/json", extra: dict[str, str] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        type(self).seen.append({"method": "GET", "path": self.path, "headers": dict(self.headers)})
        if self.path == "/v1/models":
            self._send(200, json.dumps({"object": "list", "data": [{"id": "org/model-a", "owned_by": "vllm"}]})
                       .encode())
        elif self.path == "/health":
            self._send(200, b"")
        elif self.path == "/metrics":
            self._send(200, METRICS_TEXT.encode(), "text/plain")
        elif self.path == "/v1/redirect":
            self._send(302, b"", extra={"Location": "http://example.invalid/"})
        elif self.path == "/v1/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for i in range(3):
                chunk = f"data: {i}\n\n".encode()
                self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                self.wfile.flush()
                time.sleep(0.02)
            self.wfile.write(b"0\r\n\r\n")
        elif self.path == "/v1/slow":
            time.sleep(3)
            self._send(200, b"late")
        else:
            self._send(404, b"{}")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n)
        type(self).seen.append({"method": "POST", "path": self.path, "headers": dict(self.headers), "body": body})
        self._send(200, b"echo:" + body)


@dataclass
class FakeEngine:
    port: int
    seen: list[dict[str, Any]]


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass                    # a client that hung up mid-response (the cancel tests) is expected here


@pytest.fixture
def engine():
    handler = type("Handler", (_EngineHandler,), {"seen": []})
    srv = _QuietServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    try:
        yield FakeEngine(srv.server_address[1], handler.seen)
    finally:
        srv.shutdown()
        srv.server_close()


@pytest.fixture
def agent_for(fakebin: FakeBin, machine: Machine, tmp_path: Path, prober: SafeProber):
    """agent_for(policy_dict_or_None) -> an Agent over the fake machine, fake binaries and safe probes."""
    def make(data: dict[str, Any] | None) -> Agent:
        a = Agent(store_for(tmp_path, data), paths=machine.paths, gpus=Gpus(use_nvml=False),
                  prober=prober.probe, metrics_get=prober.get)
        a.disk = Disks(machine.paths, statvfs=machine.statvfs)
        return a
    return make
