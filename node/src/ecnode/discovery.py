"""What runs on this machine: containers (sanitized `docker inspect`), engine classification and endpoints.

Everything read from a container or an endpoint is untrusted data: capped, control-character-stripped, never
executed or used to build a shell command or a path. A container's argv is REDACTED here, on the node, before it
goes anywhere; `Config.Env` is never read at all. Endpoint probes are GETs to 127.0.0.1 only, with short timeouts,
a 1 MB cap and no redirects.
"""
from __future__ import annotations

import http.client
import json
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from . import __version__
from .docker import ENGINE_LABEL, INSTANCE_LABEL, LABEL, ROLE_LABEL
from .policy import Policy, Root
from .util import clean_text, parse_time, redact_argv, scrub

MAX_BODY = 1024 * 1024
MAX_CMDLINE = 8192
MAX_PARAMS = 60
PROBE_CONNECT_S = 1.0
PROBE_READ_S = 2.0
PROBE_TOTAL_S = 5.0


# ── data-driven engine signatures ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Signature:
    engine: str
    image: tuple[str, ...] = ()       # regexes on the lower-cased image reference
    cmd: tuple[str, ...] = ()         # regexes on the (redacted) command line
    ports: tuple[int, ...] = ()       # default container ports


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
# Images that are never engines (routers, UIs, proxies, databases, telemetry), checked BEFORE the signatures, so
# e.g. a "vllm-router" image is not taken for an engine.
NON_ENGINE_IMAGE = re.compile(
    r"litellm|open-?webui|mcpo|mitmproxy|grafana|prometheus|loki|promtail|otel|opentelemetry|langfuse|phoenix|"
    r"clickhouse|postgres|redis|searxng|nginx|traefik|caddy|haproxy|envoy|router|relay|switchyard|gateway|"
    r"cadvisor|exporter|minio|mysql|mongo|rabbitmq|kafka|elasticsearch|jaeger|tempo|nats|etcd", re.I)
METRIC_MARKERS: tuple[tuple[str, str], ...] = (
    ("vllm:", "vllm"), ("sglang:", "sglang"), ("llamacpp:", "llamacpp"), ("trtllm", "trtllm"),
    ("tensorrt_llm", "trtllm"), ("dynamo_", "dynamo"),
)
_ENGINE_ID = re.compile(r"[a-z0-9_]{1,32}")


def classify(image: str, cmdline: str) -> str | None:
    """Engine id of a container by image/command signature; None if unknown or explicitly not an engine."""
    img = image.lower()[:300]
    cmd = cmdline[:MAX_CMDLINE]
    if NON_ENGINE_IMAGE.search(img):
        return None
    for sig in SIGNATURES:
        if any(re.search(p, img) for p in sig.image) or any(re.search(p, cmd) for p in sig.cmd):
            return sig.engine
    return None


def default_ports(engine: str | None) -> tuple[int, ...]:
    return next((s.ports for s in SIGNATURES if s.engine == engine), ())


def parse_params(argv: Sequence[str]) -> dict[str, str]:
    """Best-effort `--flag value` parse of an ALREADY redacted argv. Display only; never used to launch."""
    params: dict[str, str] = {}
    i = 0
    while i < len(argv) and len(params) < MAX_PARAMS:
        tok = argv[i]
        if tok == "serve" and i + 1 < len(argv) and not argv[i + 1].startswith("-"):
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
                params[key] = clean_text(val) if eq else "true"
        i += 1
    return params


# ── one container ──────────────────────────────────────────────────────────────────────────────────────────
@dataclass
class View:
    """A sanitized container plus the node-internal facts the hub never sees."""
    info: dict[str, Any]                  # proto.CONTAINER
    pid: int = 0
    running: bool = False
    host_network: bool = False
    loopback_ports: list[int] = field(default_factory=list)       # host ports reachable on 127.0.0.1
    published: dict[int, int] = field(default_factory=dict)       # container port -> host port


def _map_root(source: str, roots: Sequence[tuple[Root, str]]) -> tuple[str | None, str | None]:
    """(root id, rel) of a host path that lies inside a policy model root ("" = the root itself)."""
    src = os.path.normpath(source)
    best: tuple[str | None, str | None, int] = (None, None, -1)
    for root, real in roots:
        for base in {os.path.normpath(root.path), real}:
            if (src == base or src.startswith(base.rstrip("/") + "/")) and len(base) > best[2]:
                best = (root.id, src[len(base):].lstrip("/"), len(base))
    return best[0], best[1]


def _gpus(doc: Mapping[str, Any], gpu_by_index: Mapping[str, str]) -> tuple[list[str], bool]:
    hc = doc.get("HostConfig") or {}
    uuids: list[str] = []
    all_gpus = False

    def add(dev: str) -> None:
        nonlocal all_gpus
        dev = dev.strip()
        if dev.startswith("nvidia.com/gpu="):
            dev = dev.split("=", 1)[1]
        if dev == "all":
            all_gpus = True
        elif dev in gpu_by_index:
            uuids.append(gpu_by_index[dev])
        elif dev.startswith(("GPU-", "MIG-")):
            uuids.append(dev[:96])

    for req in (hc.get("DeviceRequests") or []) if isinstance(hc, dict) else []:
        if not isinstance(req, dict):
            continue
        ids = req.get("DeviceIDs") or []
        if req.get("Count") == -1 and not ids:
            all_gpus = True
        for dev in ids if isinstance(ids, list) else []:
            add(str(dev))
    return list(dict.fromkeys(uuids))[:16], all_gpus


def container_view(doc: Mapping[str, Any], *, roots: Sequence[tuple[Root, str]] = (),
                   gpu_by_index: Mapping[str, str] | None = None, status: str | None = None) -> View | None:
    """proto.CONTAINER from an inspect document. Reads NO environment values; argv is redacted first."""
    cfg = doc.get("Config") or {}
    cfg = cfg if isinstance(cfg, dict) else {}
    name = clean_text(str(doc.get("Name", "")).lstrip("/"), 128)
    if not name:
        return None
    raw: list[str] = []
    if isinstance(doc.get("Path"), str):
        raw.append(doc["Path"])
    if isinstance(doc.get("Args"), list):
        raw += [str(a) for a in doc["Args"]]
    if not raw:
        for k in ("Entrypoint", "Cmd"):
            v = cfg.get(k)
            raw += [str(a) for a in v] if isinstance(v, list) else ([v] if isinstance(v, str) else [])
    argv = redact_argv(raw, max_items=400, max_len=1024)
    raw_labels = cfg.get("Labels")
    raw_labels = raw_labels if isinstance(raw_labels, dict) else {}
    labels = {clean_text(k, 256): clean_text(scrub(str(v)), 256) for k, v in list(raw_labels.items())[:48]}
    st = doc.get("State") or {}
    st = st if isinstance(st, dict) else {}
    hc = doc.get("HostConfig") or {}
    hc = hc if isinstance(hc, dict) else {}
    net_mode = clean_text(hc.get("NetworkMode") or "", 64)
    image = clean_text(cfg.get("Image") or "", 256)
    managed = labels.get(LABEL) == "1"
    role = labels.get(ROLE_LABEL) if managed else None
    engine: str | None = None
    if managed and role == "engine" and _ENGINE_ID.fullmatch(labels.get(ENGINE_LABEL, "")):
        engine = labels[ENGINE_LABEL]
    elif role != "gateway":
        engine = classify(image, " ".join(argv))

    ports: list[dict[str, Any]] = []
    loopback: list[int] = []
    published: dict[int, int] = {}
    bindings = (doc.get("NetworkSettings") or {}).get("Ports") if isinstance(doc.get("NetworkSettings"), dict) else None
    if not bindings:
        bindings = hc.get("PortBindings")
    for cport, binds in (bindings.items() if isinstance(bindings, dict) else []):
        try:
            cp = int(str(cport).split("/")[0])
        except ValueError:
            continue
        for b in binds or []:
            if not isinstance(b, dict):
                continue
            try:
                hp = int(b.get("HostPort") or "")
            except (TypeError, ValueError):
                continue
            hip = clean_text(b.get("HostIp") or "", 64)
            if not 1 <= hp <= 65535 or len(ports) >= 20:
                continue
            ports.append({"host_ip": hip, "host_port": hp, "container_port": cp})
            published.setdefault(cp, hp)
            if hip in ("", "0.0.0.0", "127.0.0.1", "::", "::1") and hp not in loopback:  # noqa: S104 - matching
                loopback.append(hp)
    mounts: list[dict[str, Any]] = []
    for m in (doc.get("Mounts") or [])[:32]:
        if not isinstance(m, dict):
            continue
        src = str(m.get("Source") or "")
        rid, rel = _map_root(src, roots) if src.startswith("/") else (None, None)
        mounts.append({"source": clean_text(src, 512), "dest": clean_text(m.get("Destination") or "", 512),
                       "ro": not m.get("RW", True), "root": rid, "rel": rel})
    gpus, all_gpus = _gpus(doc, gpu_by_index or {})
    pid = st.get("Pid")
    pid = pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else 0
    state = clean_text(st.get("Status") or "", 16)
    exit_code = st.get("ExitCode")
    info = {
        "name": name, "id": clean_text(doc.get("Id") or "", 64), "image": image,
        "image_id": clean_text(doc.get("Image") or "", 80), "state": state,
        "status": clean_text(status or (f"{state} (exit {exit_code})" if state == "exited" else state), 128),
        "created": parse_time(doc.get("Created")), "started_at": parse_time(st.get("StartedAt")),
        "pid": pid or None, "exit_code": exit_code if isinstance(exit_code, int) and not isinstance(exit_code, bool)
        else None, "labels": labels, "engine": engine, "managed": managed, "role": role,
        "instance": labels.get(INSTANCE_LABEL) if managed else None, "ports": ports, "argv": argv,
        "params": parse_params(argv), "gpus": gpus, "all_gpus": all_gpus, "mounts": mounts,
        "network_mode": net_mode, "compose_project": labels.get("com.docker.compose.project"),
        "compose_service": labels.get("com.docker.compose.service")}
    return View(info=info, pid=pid, running=st.get("Running") is True, host_network=net_mode == "host",
                loopback_ports=loopback, published=published)


# ── probing ──────────────────────────────────────────────────────────────────────────────────────────────
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


def http_get(port: int, path: str, *, max_body: int = MAX_BODY, connect_s: float = PROBE_CONNECT_S,
             read_s: float = PROBE_READ_S, total_s: float = PROBE_TOTAL_S) -> Probe:
    """GET http://127.0.0.1:<port><path>: loopback only, no redirects, body capped, bounded in time."""
    conn = http.client.HTTPConnection("127.0.0.1", int(port), timeout=connect_s)
    deadline = time.monotonic() + total_s
    try:
        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(read_s)
        conn.request("GET", path, headers={"Accept": "*/*", "User-Agent": f"ecnode/{__version__}"})
        r = conn.getresponse()
        body = b""
        while len(body) < max_body and time.monotonic() < deadline:
            chunk = r.read1(min(65536, max_body - len(body)))
            if not chunk:
                break
            body += chunk
        return Probe(r.status, body[:max_body])
    except (OSError, http.client.HTTPException, ValueError):
        return Probe()
    finally:
        conn.close()


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
        except (ValueError, AttributeError):
            data = []
        if isinstance(data, list) and any(isinstance(m, dict) and m.get("owned_by") == "library" for m in data[:20]):
            return "ollama"           # ollama's OpenAI-compatible listing
        return "ollama" if self.port == 11434 and self.models.ok else None

    def models_listing(self) -> bool:
        """/v1/models answers like an OpenAI-style server: a JSON object with a `data` list, or 401/403 (an engine
        with an API key). A web app's catch-all page (200 with HTML) does not count."""
        if self.models.auth:
            return True
        if not self.models.ok:
            return False
        try:
            data = json.loads(self.models.body)
        except ValueError:
            return False
        return isinstance(data, dict) and isinstance(data.get("data"), list)

    def qualifies(self) -> bool:
        """A bare endpoint counts as an engine only if it looks like one: a /v1/models listing AND (/metrics or
        /health)."""
        return self.models_listing() and (self.metrics.answered or self.health.answered)


def probe_port(port: int, get: Callable[[int, str], Probe] = http_get) -> PortProbe:
    return PortProbe(port, get(port, "/v1/models"), get(port, "/health"), get(port, "/metrics"))


# Engines whose /metrics answers slowly and whose every read drains the engine's statistics queue (TensorRT-LLM's
# iteration stats: one read takes at least 2 s). Discovery never probes their /metrics, so only the metrics scraper
# reads it, once per interval and with a longer timeout.
SLOW_METRICS = frozenset({"trtllm"})
SLOW_METRICS_READ_S = 10.0
SLOW_METRICS_TOTAL_S = 15.0


def probe_port_quiet(port: int, get: Callable[[int, str], Probe] = http_get) -> PortProbe:
    """probe_port without the /metrics GET, for engines in SLOW_METRICS."""
    return PortProbe(port, get(port, "/v1/models"), get(port, "/health"), Probe())


def _endpoint(p: PortProbe, container: str | None, engine: str | None) -> dict[str, Any]:
    return {"port": p.port, "container": container, "engine": engine, "models": p.model_names(),
            "health_status": p.health.status, "models_status": p.models.status, "metrics_status": p.metrics.status,
            "auth_required": p.models.auth or p.health.auth}


def build_endpoints(views: Sequence[View], discovery_ports: Sequence[int],
                    prober: Callable[[int], PortProbe] = probe_port, workers: int = 8,
                    quiet_prober: Callable[[int], PortProbe] = probe_port_quiet) -> list[dict[str, Any]]:
    """Endpoints of running engine containers, of console gateways and of bare loopback ports."""
    running = [v for v in views if v.running]
    claimed = {p for v in running for p in v.loopback_ports}
    plans: list[tuple[str | None, str | None, list[int], bool]] = []    # (container, engine, ports, bare)
    engines_by_instance = {v.info["instance"]: v for v in running
                           if v.info["managed"] and v.info["role"] == "engine" and v.info["instance"]}
    for v in running:
        info = v.info
        if info["managed"] and info["role"] == "gateway":
            eng = engines_by_instance.get(info["instance"])
            port = v.published.get(8080) or (v.loopback_ports[0] if v.loopback_ports else None)
            if port:
                plans.append(((eng or v).info["name"], eng.info["engine"] if eng else None, [port], False))
            continue
        if not info["engine"]:
            continue
        ports = list(v.loopback_ports)
        if v.host_network and not ports:
            argv = info["argv"]
            ports = [int(argv[i + 1]) for i, tok in enumerate(argv[:-1])
                     if tok in ("--port", "-p") and argv[i + 1].isdigit() and 1 <= int(argv[i + 1]) <= 65535]
            ports = ports or list(default_ports(info["engine"]))
        if ports:
            plans.append((info["name"], info["engine"], ports[:5], False))
    taken = {p for _c, _e, ports, _b in plans for p in ports}
    bare = [p for p in dict.fromkeys(discovery_ports) if p not in claimed and p not in taken and 1 <= p <= 65535]
    plans += [(None, None, [p], True) for p in bare]
    all_ports = list(dict.fromkeys(p for _c, _e, ports, _b in plans for p in ports))
    quiet = {p for _c, e, ports, _b in plans if e in SLOW_METRICS for p in ports}
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(all_ports) or 1))) as ex:
        probes = dict(zip(all_ports, ex.map(lambda p: (quiet_prober if p in quiet else prober)(p), all_ports),
                          strict=True))
    out: list[dict[str, Any]] = []
    for container, engine, ports, is_bare in plans:
        candidates = [probes[p] for p in ports]
        if is_bare:
            pr = candidates[0]
            hint = pr.hint()
            if pr.any_answer and (hint or pr.qualifies()):
                out.append(_endpoint(pr, None, hint or "openai_compatible"))
            continue
        # prefer a port that reports healthy, then any that answers at all, then the first one
        best = min(candidates, key=lambda p: (not (p.models.ok or p.health.ok), not p.any_answer))
        out.append(_endpoint(best, container, engine or best.hint()))
    return out[:64]


def script_status(policy: Policy, running_names: Sequence[str], now: float) -> list[dict[str, Any]]:
    """`up` when a running container matches one of a script's patterns, else `down` (§6.3)."""
    out: list[dict[str, Any]] = []
    for s in policy.scripts:
        if not s.containers:
            out.append({"name": s.name, "status": "unknown", "checked_at": now,
                        "detail": "this script lists no container patterns; use its status action"})
            continue
        hits = [n for n in running_names if s.matches(n)]
        out.append({"name": s.name, "status": "up" if hits else "down", "checked_at": now,
                    "detail": clean_text(f"running: {', '.join(hits[:5])}" if hits else
                                         f"no running container matches {', '.join(s.containers[:4])}", 500)})
    return out
