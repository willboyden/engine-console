"""API/domain data shapes shared by services and routers."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints

from engine_console.adapters.base import Compat
from engine_console.domain.repo_id import RepoId, Revision

# names end up in container names/labels and are re-read at adoption, so keep them boring
InstanceName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}$")]
Verdict = Literal["fits", "tight", "wont_fit", "unknown"]
Confidence = Literal["high", "medium", "low"]


class Page[T](BaseModel):
    items: list[T]
    next_cursor: str | None = None


# ---- hardware -------------------------------------------------------------------------------
class GpuStat(BaseModel):
    index: int
    uuid: str
    name: str
    total_gib: float
    free_gib: float
    used_gib: float
    util_pct: float | None = None
    temp_c: float | None = None
    power_w: float | None = None
    fan_pct: float | None = None
    compute_capability: str | None = None


class HardwareReport(BaseModel):
    gpus: list[GpuStat]
    host_ram_gib: float
    host_ram_free_gib: float
    source: str  # "nvml" | "nvidia-smi" | "none"


# ---- fit ------------------------------------------------------------------------------------
class GpuFit(BaseModel):
    gpu_id: int
    uuid: str
    name: str
    weights_gib: float
    kv_cache_gib: float
    activations_gib: float
    cuda_graphs_gib: float
    overhead_gib: float
    total_gib: float
    budget_gib: float
    vram_total_gib: float
    free_gib: float
    utilization_pct: float  # total / budget * 100
    verdict: Verdict


class FitReport(BaseModel):
    verdict: Verdict
    confidence: Confidence
    tp: int
    tp_required: int | None
    concurrency: int
    max_len: int | None
    kv_bytes_per_token: float | None = None  # per GPU, full-attention layers only counted once
    per_gpu: list[GpuFit]
    max_context_at_current_concurrency: int | None
    max_concurrency_at_current_context: int | None
    fits_if_stop: list[str] = []
    notes: list[str] = []
    compat: list[Compat] = []


class ResidentUse(BaseModel):
    """A running instance's VRAM on given GPUs, so the estimator can say 'fits if you stop X'."""
    name: str
    gpu_ids: list[int]
    gib_per_gpu: float


class FitRequest(BaseModel):
    engine: str
    repo_id: RepoId
    revision: Revision | None = None
    params: dict[str, Any] = {}
    gpu_ids: list[int] = []
    concurrency: int = Field(default=1, ge=1, le=4096)  # additive to the contract, defaults to 1


# ---- downloads ------------------------------------------------------------------------------
DownloadState = Literal["queued", "running", "paused", "completed", "failed", "cancelled"]


class Download(BaseModel):
    id: str
    repo_id: str
    revision: str | None = None
    commit_sha: str | None = None
    allow_patterns: list[str] | None = None
    state: DownloadState
    total_bytes: int = 0
    done_bytes: int = 0
    speed_bps: float = 0.0
    eta_s: float | None = None
    files_total: int = 0
    files_done: int = 0
    current_file: str | None = None
    error: str | None = None
    error_code: str | None = None
    created_at: float
    updated_at: float


class DownloadRequest(BaseModel):
    repo_id: RepoId
    revision: Revision | None = None
    allow_patterns: list[str] | None = None


class LocalModel(BaseModel):
    repo_id: str
    size_bytes: int
    revisions: list[str]
    last_used: float | None
    path: str
    engines_that_fit: list[str] = []


# ---- instances ------------------------------------------------------------------------------
InstanceState = Literal["stopped", "starting", "loading", "ready", "stopping", "failed", "auth_required", "unreachable"]


class InstanceCreate(BaseModel):
    engine: str
    repo_id: RepoId
    params: dict[str, Any] = {}
    gpu_ids: list[int] = []
    profile_id: str | None = None
    name: InstanceName | None = None
    ttl_idle_s: int | None = None


class Instance(BaseModel):
    id: str
    name: str
    engine: str
    repo_id: str
    params: dict[str, Any]
    gpu_ids: list[int]
    gpu_uuids: list[str]
    port: int | None
    container_name: str | None
    container_port: int | None = None
    internal_endpoint: str | None = None   # http://<alias>:<port> on the engine network (opt-in router access)
    image: str | None
    state: InstanceState
    phase: str | None = None
    progress_pct: float | None = None
    pinned: bool = False
    ttl_idle_s: int | None = None
    profile_id: str | None = None
    error: str | None = None
    last_logs: str | None = None
    fit: dict[str, Any] | None = None
    created_at: float
    started_at: float | None = None
    last_request_at: float | None = None
    uptime_s: float | None = None
    # --- external (discovered, monitor-only) engines; console-owned instances keep the defaults ---
    managed: bool = True
    source: Literal["console", "external"] = "console"
    endpoint: str | None = None                # http://127.0.0.1:<published port>
    served_models: list[str] = []              # from GET /v1/models (untrusted text, sanitised)
    state_reason: str | None = None            # human reason for auth_required / unreachable / stopped
    history_since: float | None = None         # epoch seconds: metrics history starts here (discovery time)


class InstancePatch(BaseModel):
    pinned: bool | None = None
    ttl_idle_s: int | None = None
    name: InstanceName | None = None


class PreflightCheck(BaseModel):
    code: str
    level: Literal["ok", "warn", "block"]
    message: str


class PreflightReport(BaseModel):
    ok: bool
    gpu_ids: list[int] = []
    checks: list[PreflightCheck]
    fit: FitReport | None = None


class CommandSnippets(BaseModel):
    docker_run: str
    compose_yaml: str
    engine_cli: str


# ---- profiles -------------------------------------------------------------------------------
class ProfileIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    engine: str
    repo_id: RepoId | None = None
    preset: str | None = None
    params: dict[str, Any] = {}
    description: str = ""


class Profile(BaseModel):
    id: str
    name: str
    engine: str
    repo_id: str | None
    params: dict[str, Any]
    description: str
    created_at: float
    updated_at: float


class ProfileDiff(BaseModel):
    a: str
    b: str
    changed: dict[str, dict[str, Any]]
    only_a: dict[str, Any]
    only_b: dict[str, Any]


# ---- metrics / bench ------------------------------------------------------------------------
class MetricPoint(BaseModel):
    t: float
    values: dict[str, float]


class BenchRequest(BaseModel):
    instance_id: str
    suite: str | dict[str, Any] = "quick"
    confirm_external: bool = False   # required to send load to an engine the console does not own


class BenchRun(BaseModel):
    id: str
    instance_id: str
    engine: str | None = None
    repo_id: str | None = None
    profile_id: str | None = None
    params: dict[str, Any] = {}
    suite: dict[str, Any]
    state: Literal["running", "completed", "failed"]
    results: dict[str, Any] = {}
    error: str | None = None
    created_at: float
    finished_at: float | None = None


# ---- chat / arena / usage -------------------------------------------------------------------
class ConversationIn(BaseModel):
    title: str = "New chat"
    system_prompt: str | None = None
    instance_id: str | None = None


class Conversation(BaseModel):
    id: str
    title: str
    system_prompt: str | None
    instance_id: str | None
    created_at: float
    updated_at: float
    messages: list[dict[str, Any]] | None = None


class PromptIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str
    tags: list[str] = []


class Prompt(PromptIn):
    id: str
    created_at: float
    updated_at: float


class ArenaMatchIn(BaseModel):
    prompt: str
    instance_a: str
    instance_b: str
    system_prompt: str | None = None
    blind: bool = True
    max_tokens: int = Field(default=512, ge=1, le=32768)


class ArenaVote(BaseModel):
    winner: Literal["a", "b", "tie"]


class UsageRow(BaseModel):
    key: str
    requests: int
    prompt_tokens: int
    completion_tokens: int
    avg_latency_ms: float


# ---- keys / settings ------------------------------------------------------------------------
class KeyIn(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    role: Literal["admin", "viewer"] = "viewer"


class KeyInfo(BaseModel):
    id: str
    name: str
    role: str
    prefix: str
    created_at: float
    last_used_at: float | None
    secret: str | None = None  # only ever populated once, in the create response


class SettingsPatch(BaseModel):
    default_gpu_ids: list[int] | None = None
    idle_ttl_s: int | None = Field(default=None, ge=0)
    image_pins: dict[str, str] | None = None
