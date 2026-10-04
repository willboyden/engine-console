"""Fleet: the hub's view of every machine ("host") and the port services use to reach one.

LOCKED CONTRACT v2 (docs/ARCHITECTURE.md §10). Additive changes only; tell every agent the same minute.

Every host, including the machine the console runs on, is reached the same way: a NodeClient speaking the
ecnode wire protocol (`ecnode.proto`) to an `ecnode --stdio` process: a local subprocess for host "local",
ssh through a key restricted to that one command for the others. Services never run docker, read /proc or
open a model directory themselves; they ask the host's node, and they treat everything a node says as
untrusted data (it has already been through `ecnode.proto.sanitize_*`).

Pushed node messages are re-published on the hub's EventBus as topic "node.<type>" (sample, disks, inventory,
metrics, models_changed) with payload {"host_id": str, "msg": <sanitized message>}; link changes go to topic
"fleet" as a HostSummary dump.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from engine_console.domain.models import HostAlert, HostMemInfo
from engine_console.domain.refs import InstanceControls, InstanceVram, Management, ModelLocation, ModelRef

LOCAL = "local"
__all__ = ["InstanceControls", "InstanceVram", "Management", "ModelLocation", "ModelRef"]  # re-exported
HOST_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,31}$"
LinkState = Literal["connecting", "up", "stale", "down", "incompatible", "disabled"]
LibraryFormat = Literal["hf", "gguf", "ollama", "plain", "other"]
NodeKind = Literal["hello", "sample", "disks", "inventory"]


# ── reaching a node ──────────────────────────────────────────────────────────────────────────────────────
class HostConfig(BaseModel):
    """One machine from ~/.config/engine-console/hosts.json (validated: owner-only file, strict schema)."""
    id: str = Field(pattern=HOST_ID_PATTERN)     # "local" (and only "local") is the console's own machine
    label: str = Field(min_length=1, max_length=64)
    ssh: str | None = None       # user@address, or an ssh_config alias when ssh_config is true; None for local
    port: int | None = Field(default=None, ge=1, le=65535)
    ssh_config: bool = False     # False: `ssh -F none`, so only the hub's own key is ever offered
    enabled: bool = True


class NodeError(Exception):
    """A node refused or failed a request. `code` is an ecnode.proto error code or "node_unavailable"."""

    def __init__(self, host_id: str, code: str, message: str) -> None:
        super().__init__(f"{host_id}: {code}: {message}")
        self.host_id, self.code, self.message = host_id, code, message


class NodeUnavailable(NodeError):
    """The link to that host is not up (connecting, down, incompatible or disabled)."""

    def __init__(self, host_id: str, message: str) -> None:
        super().__init__(host_id, "node_unavailable", message)


@dataclass(frozen=True)
class StreamItem:
    kind: Literal["event", "data"]
    event: dict[str, Any] | None = None   # sanitized with ecnode.proto.EVENTS[op]
    data: bytes = b""


class NodeStream(Protocol):
    """One streaming request: http.request, docker.logs_follow, docker.pull, fs.read, fs.write, scripts.run."""

    def __aiter__(self) -> AsyncIterator[StreamItem]: ...   # events and DATA from the node, in order, until its res
    async def write(self, data: bytes) -> None: ...         # upload DATA; waits for credit (fs.write, http body)
    async def end(self) -> None: ...                         # no more upload DATA
    async def result(self) -> dict[str, Any]: ...            # the sanitized res result; NodeError if ok=false
    async def cancel(self) -> None: ...


class LinkStatus(BaseModel):
    state: LinkState
    error: str | None = None          # why it is not up, in words a person can act on
    retry_in_s: int | None = None
    notes: list[str] = []             # e.g. "reached with an UNRESTRICTED ssh key"
    last_seen: float | None = None    # epoch seconds of the last message from the node
    connected_since: float | None = None
    rtt_ms: int | None = None
    skew_s: float | None = None       # node clock minus hub clock


class NodeClient(Protocol):
    @property
    def host(self) -> HostConfig: ...
    def link(self) -> LinkStatus: ...
    def latest(self, kind: NodeKind) -> dict[str, Any] | None: ...   # last sanitized pushed message of that type
    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        """One request/response. Raises NodeUnavailable, or NodeError(code=...) when the node answers ok=false
        (code "timeout" when no res arrives in time: the request may or may not have taken effect)."""
        ...
    def open(self, op: str, args: dict[str, Any] | None = None) -> NodeStream: ...


class FleetPort(Protocol):
    def hosts(self) -> list[HostConfig]: ...
    def node(self, host_id: str) -> NodeClient: ...   # domain.errors.NotFound(code="unknown_host") if not configured


# ── API shapes (GET /hosts, /hosts/{id}) ─────────────────────────────────────────────────────────────────────
class GpuProc(BaseModel):
    pid: int
    name: str | None = None              # process name (comm), never its command line
    container: str | None = None
    instance_id: str | None = None       # the console instance running in that container, when known
    used_gib: float | None = None        # None where the driver cannot say (unified-memory GPUs)


class GpuLive(BaseModel):
    index: int
    uuid: str
    name: str
    total_gib: float
    used_gib: float | None = None
    free_gib: float | None = None
    util_pct: float | None = None
    temp_c: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    fan_pct: float | None = None
    compute_capability: str | None = None
    unified_memory: bool = False         # GPU memory is the host's RAM (e.g. DGX Spark / GB10)
    procs: list[GpuProc] = []


class NodeFacts(BaseModel):
    version: str | None = None
    hostname: str | None = None
    arch: str | None = None
    os: str | None = None
    kernel: str | None = None
    cpu_model: str | None = None
    cpu_cores: int | None = None
    cpu_threads: int | None = None
    python: str | None = None
    driver: str | None = None
    cuda: str | None = None
    gpu_source: str | None = None        # "nvml" | "nvidia-smi" | "none"
    forced_cmd: bool | None = None       # False on a remote = reached with an unrestricted key (warn)
    via_ssh: bool | None = None


class CpuFacts(BaseModel):
    load1: float | None = None
    load5: float | None = None
    load15: float | None = None
    util_pct: float | None = None


class DockerFacts(BaseModel):
    available: bool = False
    context: str | None = None
    rootless: bool | None = None         # False = the docker group is root-equivalent on that machine
    version: str | None = None
    nvidia_runtime: bool | None = None
    error: str | None = None


class PolicyFacts(BaseModel):
    """The node's own local policy, as it reported it. The hub can read it, never change it."""
    path: str | None = None
    mode: str = "monitor"                # "monitor" (read-only) | "manage"
    error: str | None = None             # policy file problem: the node then runs in monitor mode
    engine_images: list[str] = []        # allowed image references / "repo:*" patterns
    gateway_image: str | None = None
    engine_network: str | None = None
    port_range: list[int] = []
    allow_pull: bool = False
    allow_delete: bool = False
    allow_ipc_host: bool = True
    external_control: str = "none"       # "none" | "stop_start": may the console stop/start engines it did not create
    max_shm_gib: float | None = None


class MountInfo(BaseModel):
    mount: str
    label: str                           # human: "tank (zfs)", "NFS from gpu-box-2", "/ (ext4, Samsung 990)"
    fstype: str | None = None
    source: str | None = None
    device: str | None = None
    model: str | None = None             # drive model when it is a plain block device
    size_bytes: int | None = None
    used_bytes: int | None = None
    free_bytes: int | None = None
    readonly: bool = False
    remote: bool = False                 # network filesystem (NFS, CIFS...)
    remote_host: str | None = None       # address the export comes from
    remote_host_id: str | None = None    # ... matched to a configured host when the hub can tell
    remote_path: str | None = None
    roots: list[str] = []                # model roots on this mount


class RootInfo(BaseModel):
    """A model root from the node policy, plus where it physically lives."""
    id: str
    path: str
    kind: str                            # auto | hf-home | hf-hub | plain | gguf | ollama | other
    writable: bool = False
    label: str | None = None
    mount: MountInfo | None = None
    shared_from: str | None = None       # host id whose storage this is, when it is an NFS mount of another host
    visible_on: list[str] = []           # host ids that can read the same storage (this host first)


class ImageInfo(BaseModel):
    ref: str
    id: str | None = None
    size_bytes: int | None = None
    created: float | None = None
    arch: str | None = None


class ScriptAction(BaseModel):
    name: str
    label: str
    args: list[str] = []                 # the only values `arg` may take (empty: no argument)
    confirm: bool = False                # the UI asks before running it
    danger: bool = False                 # styled as destructive (stops engines, frees GPUs)


class ScriptInfo(BaseModel):
    """A lifecycle script registered in a node's policy (e.g. a lab's own start/stop script for an engine)."""
    host_id: str
    name: str
    label: str
    engine: str | None = None
    description: str = ""
    status: str = "unknown"              # "up" | "down" | "unknown" | "error"
    detail: str | None = None
    checked_at: float | None = None
    actions: list[ScriptAction] = []
    containers: list[str] = []           # container name patterns this script owns (they show as management=script)


class HostCounts(BaseModel):
    engines_running: int = 0
    engines_total: int = 0
    models: int = 0
    models_bytes: int = 0


class HostSummary(BaseModel):
    id: str
    label: str
    remote: bool
    link: LinkStatus
    node: NodeFacts | None = None
    gpus: list[GpuLive] = []
    memory: HostMemInfo | None = None
    cpu: CpuFacts | None = None
    docker: DockerFacts | None = None
    policy: PolicyFacts | None = None
    counts: HostCounts = HostCounts()
    alerts: list[HostAlert] = []


class HostDetail(HostSummary):
    disks: list[MountInfo] = []
    roots: list[RootInfo] = []
    images: list[ImageInfo] = []
    scripts: list[ScriptInfo] = []
    addresses: list[str] = []            # "10.0.0.5/24 (eth0)"


# ── the library ─────────────────────────────────────────────────────────────────────────────────────────
class LibraryCopy(BaseModel):
    id: str                              # stable: first 16 hex of sha1("host_id|root_id|rel")
    host_id: str
    host_label: str
    root_id: str
    root_label: str | None = None
    rel: str
    path: str
    size_bytes: int
    files: int | None = None
    mtime: float | None = None
    atime: float | None = None
    drive: MountInfo | None = None
    visible_on: list[str] = []           # host ids that can load this copy (its host + hosts mounting that storage)
    writable: bool = False
    deletable: bool = False
    in_use_by: list[str] = []            # instance ids whose container mounts this copy


class LibraryModel(BaseModel):
    key: str                             # "<format>:<name>[@<revision>]": copies of one model group under one key
    name: str
    format: LibraryFormat
    revision: str | None = None
    quant: str | None = None
    params: int | None = None
    arch: str | None = None
    context: int | None = None
    shards: int | None = None
    size_bytes: int                      # one copy
    engines: list[str] = []              # adapter ids that can serve this format
    copies: list[LibraryCopy] = []
    warnings: list[str] = []


class CopyRequest(BaseModel):
    host_id: str = Field(pattern=HOST_ID_PATTERN)
    root_id: str | None = None           # default: the target host's first writable root of a matching kind


class ScriptRunRequest(BaseModel):
    arg: str | None = None


class ScriptJob(BaseModel):
    id: str
    host_id: str
    name: str
    action: str
    arg: str | None = None
    state: Literal["running", "completed", "failed"]
    rc: int | None = None
    started_at: float
    finished_at: float | None = None
    output_tail: str = ""                # last ~20 kB, secrets masked
