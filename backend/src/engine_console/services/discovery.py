"""Discovery of engines the console did NOT create: running containers and loopback ports.

External instances are MONITOR-ONLY (never started/stopped/removed here). Everything read from a container or
an endpoint is untrusted data: it is length-capped, control-character-stripped and never executed or used to
build shell/SQL/paths. `Config.Env` of a container is never read into our data structures at all."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from engine_console.config import Settings
from engine_console.domain.models import Instance, InstanceState
from engine_console.services.common import scrub
from engine_console.services.docker import LABEL, DockerCli

log = logging.getLogger(__name__)

PROBE_TIMEOUT = httpx.Timeout(connect=1.0, read=2.0, write=2.0, pool=2.0)
MAX_BODY = 1024 * 1024
MAX_CONTAINERS = 200
MAX_CMDLINE = 8192
MAX_PARAMS = 60
EXT = "ext-"


# ---- data-driven engine signatures ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Signature:
    engine: str
    image: tuple[str, ...] = ()       # regexes on the lower-cased image reference
    cmd: tuple[str, ...] = ()         # regexes on the (redacted) command line
    ports: tuple[int, ...] = ()       # default container ports, used to pick among several published ports


SIGNATURES: tuple[Signature, ...] = (
    Signature("vllm", image=(r"(^|/)vllm([-/:@]|$)", r"(^|/)vllm-openai"), cmd=(r"\bvllm\s+serve\b", r"vllm\.entrypoints"),
              ports=(8000,)),
    Signature("sglang", image=(r"(^|/)sglang([-/:@]|$)",), cmd=(r"sglang\.launch_server", r"\bsglang\s+serve\b"),
              ports=(30000,)),
    Signature("ollama", image=(r"(^|/)ollama([-/:@]|$)",), cmd=(r"\bollama\s+serve\b",), ports=(11434,)),
    Signature("llamacpp", image=(r"llama[.-]?cpp", r"(^|/)llama-server"), cmd=(r"(^|/|\s)llama-server\b",), ports=(8080,)),
    Signature("trtllm", image=(r"tensorrt[-_]?llm", r"trtllm"), cmd=(r"\btrtllm-serve\b", r"tensorrt_llm"), ports=(8000,)),
    Signature("dynamo", image=(r"(^|/)dynamo([-/:@]|$)", r"ai-dynamo"), cmd=(r"dynamo\.frontend", r"\bdynamo-run\b"),
              ports=(8000,)),
)
# Images that are never engines (routers, UIs, proxies, databases, telemetry). Checked BEFORE signatures, so e.g.
# a "vllm-router" image is not misclassified as an engine.
NON_ENGINE_IMAGE = re.compile(
    r"litellm|open-?webui|mcpo|mitmproxy|grafana|prometheus|loki|promtail|otel|opentelemetry|langfuse|phoenix|"
    r"clickhouse|postgres|redis|searxng|nginx|traefik|caddy|haproxy|envoy|router|relay|switchyard|gateway|"
    r"cadvisor|exporter|minio|mysql|mongo|rabbitmq|kafka|elasticsearch|jaeger|tempo|nats|etcd", re.I)
# Metrics-body markers, used to identify an engine behind a bare port.
METRIC_MARKERS: tuple[tuple[str, str], ...] = (
    ("vllm:", "vllm"), ("sglang:", "sglang"), ("llamacpp:", "llamacpp"), ("trtllm", "trtllm"),
    ("tensorrt_llm", "trtllm"), ("dynamo_", "dynamo"),
)
# generic Prometheus -> canonical keys, for engines without an adapter (missing keys are simply omitted)
GENERIC_METRICS: dict[str, dict[str, str]] = {
    "llamacpp": {"llamacpp:requests_processing": "requests_running", "llamacpp:requests_deferred": "requests_waiting",
                 "llamacpp:prompt_tokens_total": "prompt_tokens_total",
                 "llamacpp:tokens_predicted_total": "generation_tokens_total"},
}
_SECRET_FLAG = re.compile(r"(api[-_]?key|secret|passw(or)?d|credential|access[-_]?key|auth[-_]?key|"
                          r"(?<![a-z])token(?![a-z])|bearer)", re.I)
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def classify(image: str, cmdline: str) -> str | None:
    """Engine id for a container by image/command signature, or None (unknown or explicitly not an engine)."""
    img = image.lower()[:300]
    cmd = cmdline[:MAX_CMDLINE]
    if NON_ENGINE_IMAGE.search(img):
        return None
    for sig in SIGNATURES:
        if any(re.search(p, img) for p in sig.image) or any(re.search(p, cmd) for p in sig.cmd):
            return sig.engine
    return None


def clean_text(s: object, limit: int = 200) -> str:
    return _CTRL.sub("", str(s))[:limit]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "unknown"


def display_name(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9 _.:\-]", "", _CTRL.sub("", s)).strip(" .-")[:64]
    return out or "external"


# ---- argv redaction + display-only parsing --------------------------------------------------------------------------
def redact_argv(argv: Sequence[str]) -> list[str]:
    """Mask secret flag values BEFORE anything else looks at the argv (`--api-key X`, `--token=X`, `NAME=X`)."""
    out: list[str] = []
    skip = False
    for raw in argv[:400]:
        tok = _CTRL.sub(" ", str(raw))[:1000]
        if skip:
            out.append("[redacted]")
            skip = False
            continue
        head, eq, _ = tok.partition("=")
        if _SECRET_FLAG.search(head) and (head.startswith("-") or eq):
            if eq:
                out.append(f"{head}=[redacted]")
            else:
                out.append(tok)
                skip = True
            continue
        out.append(scrub(tok))
    return out


def parse_params(argv: Sequence[str]) -> dict[str, Any]:
    """Best-effort `--flag value` parse of an ALREADY redacted argv. Display only; never used to launch anything."""
    params: dict[str, Any] = {}
    i = 0
    while i < len(argv) and len(params) < MAX_PARAMS:
        tok = argv[i]
        if tok in ("serve",) and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
            params["model"] = clean_text(argv[i + 1])
            i += 2
            continue
        if tok.startswith("--") and len(tok) > 2:
            name, eq, val = tok[2:].partition("=")
            key = re.sub(r"[^a-z0-9_]", "_", name.lower())[:64]
            if not eq and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                val, i = argv[i + 1], i + 1
                eq = "="
            if key:
                params[key] = clean_text(val) if eq else True
        i += 1
    return params


def _pid_of(doc: dict[str, Any]) -> int:
    pid = (doc.get("State") or {}).get("Pid")
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else 0


# ---- docker inspect view --------------------------------------------------------------------------------------------------
@dataclass
class ContainerView:
    name: str
    image: str
    argv: list[str]                   # redacted
    ports: list[int]                  # host ports reachable on 127.0.0.1
    host_network: bool
    labels: dict[str, str] = field(default_factory=dict)
    pid: int = 0


def parse_inspect(doc: dict[str, Any]) -> ContainerView | None:
    """Sanitised view of an inspect document. Reads NO environment values."""
    cfg = doc.get("Config") or {}
    name = clean_text(str(doc.get("Name", "")).lstrip("/"), 128)
    if not name:
        return None
    raw_argv: list[str] = []
    if isinstance(doc.get("Path"), str):
        raw_argv.append(doc["Path"])
    if isinstance(doc.get("Args"), list):
        raw_argv += [str(a) for a in doc["Args"]]
    if not raw_argv:
        for k in ("Entrypoint", "Cmd"):
            v = cfg.get(k)
            if isinstance(v, list):
                raw_argv += [str(a) for a in v]
            elif isinstance(v, str):
                raw_argv.append(v)
    argv = redact_argv(raw_argv)
    host_net = str((doc.get("HostConfig") or {}).get("NetworkMode", "")) == "host"
    ports: list[int] = []
    bindings = (doc.get("NetworkSettings") or {}).get("Ports") or {}
    if isinstance(bindings, dict):
        for binds in bindings.values():
            for b in binds or []:
                if not isinstance(b, dict):
                    continue
                if b.get("HostIp", "") not in ("", "0.0.0.0", "127.0.0.1"):  # noqa: S104 - matching, not binding
                    continue          # bound to another interface: not reachable as 127.0.0.1
                try:
                    p = int(b.get("HostPort", ""))
                except (TypeError, ValueError):
                    continue
                if 1 <= p <= 65535 and p not in ports:
                    ports.append(p)
    if host_net and not ports:
        for i, tok in enumerate(argv):
            if tok in ("--port", "-p") and i + 1 < len(argv) and argv[i + 1].isdigit() and 1 <= int(argv[i + 1]) <= 65535:
                ports.append(int(argv[i + 1]))
    labels = {clean_text(k, 100): clean_text(v, 200) for k, v in list((cfg.get("Labels") or {}).items())[:50]}
    return ContainerView(name=name, image=clean_text(cfg.get("Image", ""), 300), argv=argv, ports=ports[:20],
                         host_network=host_net, labels=labels, pid=_pid_of(doc))


# ---- probing ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Probe:
    status: int | None = None
    body: bytes = b""

    @property
    def ok(self) -> bool:
        return self.status is not None and 200 <= self.status < 300

    @property
    def auth(self) -> bool:
        return self.status in (401, 403)

    @property
    def answered(self) -> bool:
        return self.ok or self.auth


@dataclass
class PortProbe:
    port: int
    models: Probe
    health: Probe
    metrics: Probe

    @property
    def any_answer(self) -> bool:
        return self.models.answered or self.health.answered or self.metrics.answered

    def model_names(self) -> list[str]:
        try:
            data = json.loads(self.models.body)["data"]
            return [clean_text(m["id"]) for m in data[:20] if isinstance(m, dict) and isinstance(m.get("id"), str)]
        except (ValueError, KeyError, TypeError):
            return []

    def hint(self) -> str | None:
        text = self.metrics.body[:65536].decode("utf-8", "replace")
        for marker, engine in METRIC_MARKERS:
            if marker in text:
                return engine
        try:
            data = json.loads(self.models.body).get("data") or []
            if any(isinstance(m, dict) and m.get("owned_by") == "library" for m in data[:20]):
                return "ollama"       # ollama's OpenAI-compat listing
        except (ValueError, AttributeError):
            pass
        return "ollama" if self.port == 11434 and self.models.ok else None


def _qualifies_generic(p: PortProbe) -> bool:
    """A bare endpoint counts as an engine only if it looks like one: /v1/models AND (/metrics or /health)."""
    return p.models.answered and (p.metrics.answered or p.health.answered)


@dataclass
class _Meta:
    first_seen: float
    metrics_path: str | None = None


@dataclass(frozen=True)
class ScrapeTarget:
    iid: str
    engine: str
    port: int
    path: str | None       # None: no metrics endpoint, host-RAM sampling only


class DiscoveryService:
    def __init__(self, docker: DockerCli, http: httpx.AsyncClient, cfg: Settings, *,
                 owned_ports: Callable[[], set[int]] = lambda: set(), clock: Callable[[], float] = time.time) -> None:
        self._docker = docker
        self._http = http
        self._cfg = cfg
        self._owned = owned_ports
        self._clock = clock
        self._current: dict[str, Instance] = {}
        self._gone: dict[str, Instance] = {}
        self._meta: dict[str, _Meta] = {}
        self._pids: dict[str, int] = {}     # external instance id -> container init PID (for cgroup memory)
        self._last = 0.0
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(16)

    # -- queries ---------------------------------------------------------------------------------------------------
    def snapshot(self) -> list[Instance]:
        return [*self._current.values(), *self._gone.values()]

    def pid(self, iid: str) -> int:
        return self._pids.get(iid, 0)

    def get(self, iid: str) -> Instance | None:
        return self._current.get(iid) or self._gone.get(iid)

    def targets(self) -> list[ScrapeTarget]:
        out = []
        for iid, inst in self._current.items():
            path = self._meta.get(iid, _Meta(0)).metrics_path
            if inst.state in ("ready", "auth_required") and inst.port and (path or self._pids.get(iid)):
                out.append(ScrapeTarget(iid, inst.engine, inst.port, path))
        return out

    # -- probing ---------------------------------------------------------------------------------------------------
    async def _get(self, port: int, path: str) -> Probe:
        """GET only, loopback only, no redirects, 1 MB cap. URL is built from an int port and a constant path."""
        url = f"http://127.0.0.1:{int(port)}{path}"
        async with self._sem:
            try:
                async with self._http.stream("GET", url, timeout=PROBE_TIMEOUT, follow_redirects=False) as r:
                    body = b""
                    async for chunk in r.aiter_bytes():
                        body += chunk
                        if len(body) >= MAX_BODY:
                            body = body[:MAX_BODY]
                            break
                    return Probe(r.status_code, body)
            except (httpx.HTTPError, OSError):
                return Probe()

    async def probe_port(self, port: int) -> PortProbe:
        m, h, x = await asyncio.gather(self._get(port, "/v1/models"), self._get(port, "/health"), self._get(port, "/metrics"))
        return PortProbe(port, m, h, x)

    # -- refresh ----------------------------------------------------------------------------------------------------
    async def maybe_refresh(self) -> None:
        if self._cfg.discovery_enabled and self._clock() - self._last >= self._cfg.discovery_interval_s:
            await self.refresh()

    async def refresh(self) -> list[Instance]:
        async with self._lock:
            self._last = self._clock()
            if not self._cfg.discovery_enabled:
                self._current, self._gone = {}, {}
                return []
            try:
                found = await self._discover()
            except Exception:  # noqa: BLE001 - discovery must never take the console down
                log.exception("discovery failed")
                return self.snapshot()
            previous = self._current
            self._current = found
            self._gone = {i: self._as_stopped(inst) for i, inst in previous.items() if i not in found and inst.state != "stopped"}
            self._meta = {i: m for i, m in self._meta.items() if i in found or i in self._gone}
            return self.snapshot()

    @staticmethod
    def _as_stopped(inst: Instance) -> Instance:
        return inst.model_copy(update={"state": "stopped", "state_reason": "container is no longer running"})

    async def _discover(self) -> dict[str, Instance]:
        found: dict[str, Instance] = {}
        claimed: set[int] = set()
        entries: list[tuple[ContainerView, str | None]] = []
        names = (await self._docker.list_running())[:MAX_CONTAINERS]
        for name in sorted(names):
            doc = await self._docker.inspect_doc(name)
            view = parse_inspect(doc) if doc else None
            if view is None:
                continue
            claimed.update(view.ports)     # ports of ANY container are never re-probed as bare endpoints
            if view.labels.get(LABEL) == "1":
                continue          # ours (managed instance or gateway)
            engine = classify(view.image, " ".join(view.argv))
            entries.append((view, engine))
        for view, engine in entries:
            inst = await self._from_container(view, engine, found)
            if inst is not None:
                found[inst.id] = inst
        skip = claimed | self._owned() | {p for i in found.values() if (p := i.port)}
        ports = [p for p in dict.fromkeys(self._cfg.discovery_ports) if 1 <= p <= 65535 and p not in skip]
        for pr in await asyncio.gather(*(self.probe_port(p) for p in ports)):
            if not pr.any_answer:
                continue
            hint = pr.hint()
            if not (hint or _qualifies_generic(pr)):
                continue
            inst = self._build(f"127.0.0.1:{pr.port}", f"127.0.0.1:{pr.port}", hint or "openai_compatible", None, None,
                               {}, pr, found)
            found[inst.id] = inst
        return found

    async def _from_container(self, view: ContainerView, engine: str | None, taken: dict[str, Instance]) -> Instance | None:
        probes = list(await asyncio.gather(*(self.probe_port(p) for p in view.ports)))
        if engine is None:
            if NON_ENGINE_IMAGE.search(view.image.lower()):
                return None
            good = next((p for p in probes if _qualifies_generic(p)), None)
            if good is None:
                return None       # not an engine we can recognise: never listed
            engine = good.hint() or "openai_compatible"
        best = self._pick(view, engine, probes)
        return self._build(view.name, view.name, engine, view.image, view, parse_params(view.argv), best, taken,
                           no_ports=not view.ports, tried=[p.port for p in probes])

    @staticmethod
    def _pick(view: ContainerView, engine: str, probes: list[PortProbe]) -> PortProbe | None:
        if not probes:
            return None
        # prefer a port that reports healthy, then any that answers at all, then the first published one
        return min(probes, key=lambda p: (not (p.models.ok or p.health.ok), not p.any_answer))

    def _build(self, key: str, name: str, engine: str, image: str | None, view: ContainerView | None,
               params: dict[str, Any], probe: PortProbe | None, taken: dict[str, Instance], *,
               no_ports: bool = False, tried: list[int] | None = None) -> Instance:
        iid = EXT + slug(key)
        if iid in taken:                       # slug collision: disambiguate deterministically from the raw name
            iid += "-" + hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:6]
        meta = self._meta.setdefault(iid, _Meta(self._clock()))
        self._pids[iid] = view.pid if view else 0
        state: InstanceState
        reason: str | None = None
        models: list[str] = []
        port: int | None = None
        if no_ports:
            state, reason = "unreachable", ("no published host port: the console runs on the host and cannot reach a "
                                            "container's private IP (publish a port on 127.0.0.1 to monitor it)")
        elif probe is None or not probe.any_answer:
            state = "unreachable"
            reason = f"published port(s) {tried or []} did not answer /v1/models, /health or /metrics"
            port = tried[0] if tried else None
        else:
            port = probe.port
            models = probe.model_names()
            meta.metrics_path = "/metrics" if probe.metrics.ok else None
            if probe.health.ok or probe.models.ok:
                state = "ready"
                if probe.models.auth:
                    reason = "/v1 endpoints need an API key (HTTP 401/403): metrics work, chat does not"
            else:
                state = "auth_required"
                reason = "the engine answered HTTP 401/403: an API key is required, so metrics and chat are unavailable"
                meta.metrics_path = None
        return Instance(
            id=iid, name=display_name(name), engine=engine, repo_id=models[0] if models else display_name(name),   # exact served name: chat sends it as `model`
            params=params, gpu_ids=[], gpu_uuids=[], port=port, container_name=view.name if view else None,
            image=image or None, state=state, created_at=meta.first_seen, managed=False, source="external",
            endpoint=f"http://127.0.0.1:{port}" if port else None, served_models=models, state_reason=reason,
            history_since=meta.first_seen)


def parse_generic_metrics(engine: str, text: str) -> dict[str, float]:
    """Map whatever canonical keys an adapter-less engine exposes; everything else is ignored."""
    mapping = GENERIC_METRICS.get(engine)
    if not mapping:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        name = name.split("{", 1)[0]
        if name in mapping:
            try:
                out[mapping[name]] = float(rest.split()[0])
            except (ValueError, IndexError):
                continue
    return out
