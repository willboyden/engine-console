"""Docker access through the `docker` CLI only: argv lists, timeouts, explicit --context, no shell.
The console only ever touches containers carrying the engine-console=1 label."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shlex
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import yaml

from engine_console.domain.errors import Forbidden, ProblemError
from engine_console.domain.ports import CmdResult, CommandRunner

LABEL = "engine-console"
ROLE_LABEL = "engine-console.role"
INSTANCE_LABEL = "engine-console.instance"
_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
GATEWAY_LISTEN = 8080
# constant shell text: the only variable input (the nginx config) travels in an env var, never in this string
GATEWAY_SH = 'printf "%s\\n" "$NGINX_CONF" > /tmp/nginx.conf && exec nginx -c /tmp/nginx.conf'


class SubprocessRunner:
    async def run(self, argv: Sequence[str], *, timeout: float, env_extra: Mapping[str, str] | None = None,
                  merge_stderr: bool = False) -> CmdResult:
        env = {**os.environ, **(env_extra or {})}
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT if merge_stderr else asyncio.subprocess.PIPE, env=env)
        except FileNotFoundError:
            return CmdResult(127, "", f"{argv[0]}: command not found")
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return CmdResult(124, "", f"timed out after {timeout}s: {' '.join(argv[:4])} ...")
        return CmdResult(proc.returncode or 0, out.decode(errors="replace"), (err or b"").decode(errors="replace"))

    async def stream(self, argv: Sequence[str]) -> AsyncIterator[str]:
        proc = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        assert proc.stdout is not None
        try:
            async for line in proc.stdout:
                yield line.decode(errors="replace").rstrip("\n")
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
            await proc.wait()


@dataclass
class ContainerSpec:
    name: str
    image: str
    argv: list[str]
    host_port: int | None    # None: not published (engines live on an internal network; the gateway publishes)
    container_port: int
    gpu_uuids: list[str]
    network: str
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    env_passthrough: list[str] = field(default_factory=list)   # names only; values come from the child env
    mounts: list[tuple[str, str, bool]] = field(default_factory=list)   # (host, container, read_only)
    shm_size: str = "16g"
    ipc_host: bool = True
    network_alias: str | None = None


@dataclass
class ContainerInfo:
    name: str
    running: bool
    status: str
    exit_code: int | None
    labels: dict[str, str]
    pid: int = 0          # host PID of the container's init process (0 when not running)


def gpus_arg(uuids: list[str]) -> str:
    # docker parses --gpus as CSV: several devices need the value wrapped in literal double quotes
    return f'"device={",".join(uuids)}"' if len(uuids) > 1 else f"device={uuids[0]}"


def build_run_args(spec: ContainerSpec) -> list[str]:
    """Everything after `docker --context X`. Pure, so it doubles as the 'copy as command' source."""
    # Hardening: drop every Linux capability, no setuid escalation, bounded process count. Single-node NCCL/TP
    # over NVLink/PCIe/shm needs none of them, but this is NOT verified on real multi-GPU TP here; RDMA/IB
    # transports would need IPC_LOCK added back.
    a = ["run", "-d", "--name", spec.name, "--network", spec.network, "--restart", "no",
         "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "4096"]
    if spec.gpu_uuids:
        a += ["--gpus", gpus_arg(spec.gpu_uuids)]
    if spec.network_alias:
        a += ["--network-alias", spec.network_alias]
    if spec.host_port is not None:
        a += ["-p", f"127.0.0.1:{spec.host_port}:{spec.container_port}"]
    a += ["--shm-size", spec.shm_size]
    a += ["--ipc", "host" if spec.ipc_host else "private"]
    for k, v in {LABEL: "1", **spec.labels}.items():
        a += ["--label", f"{k}={v}"]
    for host, cont, ro in spec.mounts:
        a += ["-v", f"{host}:{cont}{':ro' if ro else ''}"]
    for k, v in spec.env.items():
        a += ["-e", f"{k}={v}"]
    for k in spec.env_passthrough:
        a += ["-e", k]
    return [*a, spec.image, *spec.argv]


class DockerCli:
    def __init__(self, runner: CommandRunner, *, binary: str = "docker", context: str = "rootless") -> None:
        self._r = runner
        self._bin = binary
        self._ctx = context

    def argv(self, *args: str) -> list[str]:
        return [self._bin, "--context", self._ctx, *args]

    async def _ok(self, *args: str, timeout: float = 30, env: Mapping[str, str] | None = None) -> CmdResult:
        res = await self._r.run(self.argv(*args), timeout=timeout, env_extra=env)
        if res.rc != 0:
            raise ProblemError(f"docker {args[0]} failed: {res.stderr.strip()[:400]}", code="docker_error", status=502)
        return res

    async def run_container(self, spec: ContainerSpec, env: Mapping[str, str] | None = None) -> str:
        res = await self._ok(*build_run_args(spec), timeout=120, env=env)
        return res.stdout.strip()[:64]

    async def inspect(self, name: str) -> ContainerInfo | None:
        res = await self._r.run(self.argv("inspect", name), timeout=15)
        if res.rc != 0:
            return None
        try:
            d = json.loads(res.stdout)[0]
        except (ValueError, IndexError):
            return None
        st = d.get("State", {})
        return ContainerInfo(name=name, running=bool(st.get("Running")), status=str(st.get("Status", "")),
                             exit_code=st.get("ExitCode"), labels=(d.get("Config", {}) or {}).get("Labels") or {},
                             pid=st["Pid"] if isinstance(st.get("Pid"), int) and st["Pid"] > 0 else 0)

    async def _owned(self, name: str, *, role: str | None = None, instance: str | None = None) -> ContainerInfo | None:
        """Inspect and insist on our label (and, for gateways, role + owning instance): the console never touches
        containers it didn't create, and never another instance's gateway."""
        info = await self.inspect(name)
        if info is None:
            return None
        bad = info.labels.get(LABEL) != "1" or (role is not None and info.labels.get(ROLE_LABEL) != role) \
            or (instance is not None and info.labels.get(INSTANCE_LABEL) != instance)
        if bad:
            raise Forbidden(f"refusing to touch container {name!r}: it is not one of this console's own",
                            code="not_owned")
        return info

    async def stop(self, name: str, timeout_s: int = 30, *, role: str | None = None, instance: str | None = None) -> None:
        if await self._owned(name, role=role, instance=instance) is None:
            return
        res = await self._r.run(self.argv("stop", "-t", str(timeout_s), name), timeout=timeout_s + 30)
        if res.rc != 0 and "No such container" not in res.stderr:
            raise ProblemError(f"docker stop failed: {res.stderr.strip()[:400]}", code="docker_error", status=502)

    async def remove(self, name: str, *, role: str | None = None, instance: str | None = None) -> None:
        if await self._owned(name, role=role, instance=instance) is None:
            return
        res = await self._r.run(self.argv("rm", "-f", name), timeout=60)
        if res.rc != 0 and "No such container" not in res.stderr:
            raise ProblemError(f"docker rm failed: {res.stderr.strip()[:400]}", code="docker_error", status=502)

    async def logs(self, name: str, tail: int = 200) -> str:
        if await self._owned(name) is None:
            return ""
        res = await self._r.run(self.argv("logs", "--tail", str(tail), name), timeout=20, merge_stderr=True)
        return res.stdout if res.rc == 0 else ""

    async def follow_logs(self, name: str, tail: int = 200) -> AsyncIterator[str]:
        if await self._owned(name) is None:
            return
        async for line in self._r.stream(self.argv("logs", "-f", "--tail", str(tail), name)):
            yield line

    async def list_console_containers(self) -> list[str]:
        res = await self._r.run(self.argv("ps", "-a", "--filter", f"label={LABEL}=1", "--format", "{{.Names}}"),
                                timeout=20)
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()] if res.rc == 0 else []

    async def list_running(self) -> list[str]:
        """Names of ALL running containers (labelled or not). Read-only; used by discovery."""
        res = await self._r.run(self.argv("ps", "--format", "{{.Names}}"), timeout=20)
        return [ln.strip() for ln in res.stdout.splitlines() if ln.strip()] if res.rc == 0 else []

    async def inspect_doc(self, name: str) -> dict[str, Any] | None:
        """Raw `docker inspect` document. Callers must treat it as untrusted and must never keep Config.Env."""
        res = await self._r.run(self.argv("inspect", name), timeout=15)
        if res.rc != 0:
            return None
        try:
            doc = json.loads(res.stdout)[0]
        except (ValueError, IndexError):
            return None
        return doc if isinstance(doc, dict) else None

    async def image_present(self, image: str) -> bool:
        res = await self._r.run(self.argv("image", "inspect", "--format", "{{.Id}}", image), timeout=20)
        return res.rc == 0


def command_snippets(spec: ContainerSpec, context: str = "rootless") -> dict[str, Any]:
    """Human-copyable equivalents. Secrets are never included: HF_TOKEN appears as a bare `-e HF_TOKEN`."""
    docker_run = shlex.join(["docker", "run", *build_run_args(spec)[1:]])
    compose: dict[str, Any] = {
        "services": {spec.name: {
            "image": spec.image, "command": spec.argv, "restart": "no", "ipc": "host" if spec.ipc_host else "private",
            "shm_size": spec.shm_size,
            "ports": [f"127.0.0.1:{spec.host_port}:{spec.container_port}"] if spec.host_port is not None else [],
            "environment": {**spec.env, **{k: "${" + k + "}" for k in spec.env_passthrough}},
            "volumes": [f"{h}:{c}{':ro' if ro else ''}" for h, c, ro in spec.mounts],
            "labels": {LABEL: "1", **spec.labels}, "networks": [spec.network],
            "security_opt": ["no-new-privileges:true"], "cap_drop": ["ALL"], "pids_limit": 4096,
            "deploy": {"resources": {"reservations": {"devices": [
                {"driver": "nvidia", "device_ids": spec.gpu_uuids, "capabilities": ["gpu"]}]}}},
        }},
        "networks": {spec.network: {"external": True}},
    }
    return {"docker_run": docker_run, "compose_yaml": yaml.safe_dump(compose, sort_keys=False),
            "engine_cli": shlex.join(spec.argv)}


# ---- per-instance gateway (nginx stream TCP forward) ------------------------------------------------------------
def gateway_config(engine_host: str, engine_port: int) -> str:
    """nginx.conf for a pure TCP forward. Only a slugged name and an int can reach the template, so a hostile
    instance name or port cannot inject directives. A TCP stream never buffers, so SSE flows unchanged."""
    if not isinstance(engine_host, str) or not _NAME.fullmatch(engine_host):
        raise ValueError("unsafe engine host name for gateway config")
    if isinstance(engine_port, bool) or not isinstance(engine_port, int) or not 1 <= engine_port <= 65535:
        raise ValueError("engine port must be an int in 1..65535")
    return (
        # daemon off: nginx forks by default, which ends the container's PID 1 (found by live test, not mocks).
        "daemon off;\nworker_processes 1;\npid /tmp/nginx.pid;\nerror_log /dev/stderr warn;\n"
        "events { worker_connections 1024; }\n"
        f"stream {{\n  server {{\n    listen {GATEWAY_LISTEN};\n    proxy_pass {engine_host}:{engine_port};\n"
        "    proxy_connect_timeout 5s;\n    proxy_timeout 1h;\n  }\n}\n")


@dataclass
class GatewaySpec:
    name: str
    image: str
    host_port: int
    engine_host: str
    engine_port: int
    internal_network: str
    instance_id: str
    bridge_network: str = "bridge"


def build_gateway_create_args(g: GatewaySpec) -> list[str]:
    """`docker create` args. Created on the default bridge (so it can publish on loopback); the internal network
    is attached afterwards with `docker network connect`, since `run` accepts only one --network."""
    if not isinstance(g.host_port, int) or isinstance(g.host_port, bool) or not 1 <= g.host_port <= 65535:
        raise ValueError("host port must be an int")
    conf = gateway_config(g.engine_host, g.engine_port)
    return ["create", "--name", g.name, "--network", g.bridge_network, "--restart", "no", "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "256",
            "--user", "101:101",
            "--tmpfs", "/var/cache/nginx:rw,mode=1777,size=4m", "--tmpfs", "/var/run:rw,mode=1777,size=1m",
            "--tmpfs", "/tmp:rw,mode=1777,size=1m",  # noqa: S108 - container-private tmpfs
            "-p", f"127.0.0.1:{g.host_port}:{GATEWAY_LISTEN}",
            "--label", f"{LABEL}=1", "--label", f"{ROLE_LABEL}=gateway", "--label", f"{INSTANCE_LABEL}={g.instance_id}",
            "-e", f"NGINX_CONF={conf}", "--entrypoint", "/bin/sh", g.image, "-c", GATEWAY_SH]


async def ensure_internal_network(docker: DockerCli, name: str) -> None:
    """Create the engine network with --internal, or verify an existing one really is internal."""
    res = await docker._r.run(docker.argv("network", "inspect", "--format", "{{.Internal}}", name), timeout=15)  # noqa: SLF001
    if res.rc == 0:
        if res.stdout.strip().lower() != "true":
            raise ProblemError(f"docker network {name!r} exists but is not --internal; refusing to attach engines to it",
                               code="engine_network_not_internal", status=409)
        return
    await docker._ok("network", "create", "--internal", "--label", f"{LABEL}=1", name)  # noqa: SLF001


async def start_gateway(docker: DockerCli, g: GatewaySpec) -> None:
    await docker._ok(*build_gateway_create_args(g), timeout=60)  # noqa: SLF001
    try:
        await docker._ok("network", "connect", g.internal_network, g.name)  # noqa: SLF001
        await docker._ok("start", g.name)  # noqa: SLF001
    except ProblemError:
        await docker._r.run(docker.argv("rm", "-f", g.name), timeout=30)  # noqa: SLF001
        raise
