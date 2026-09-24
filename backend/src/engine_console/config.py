"""Process configuration: environment variables (no prefix, so HF_CACHE_DIR / DOCKER_CONTEXT work as documented),
with an optional per-user env file so settings survive restarts. Real environment variables win over the file."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Fixed locations rather than XDG_*: XDG vars are unreliable under snap-packaged terminals and IDEs.
_HOME = Path.home()
# Optional KEY=VALUE file (dotenv syntax), e.g. HF_CACHE_DIR=/data/hf. Keep it chmod 600 if it holds a token.
ENV_FILE = Path(os.environ.get("ENGINE_CONSOLE_ENV_FILE") or _HOME / ".config" / "engine-console" / "env")
GATEWAY_IMAGE = "nginx@sha256:62ff2089abf5a9ed33bd232895bef5e22f7bb4b200675cec49a5ebc48e3d4ac8"


def _default_cache_dir() -> Path:
    """~/.cache/huggingface. If that is a symlink (common when the cache lives on another disk) use its target,
    because the startup validator refuses symlinked cache dirs."""
    p = Path.home() / ".cache" / "huggingface"
    return p.resolve() if p.is_symlink() else p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", case_sensitive=False)

    host: str = "127.0.0.1"          # ADR 2: never bind wider without a reverse proxy + keys
    port: int = 8791
    data_dir: Path = _HOME / ".local" / "share" / "engine-console"
    hf_cache_dir: Path = _default_cache_dir()  # HF_HOME layout: hub/ lives beneath it (override with HF_CACHE_DIR)
    hf_cache_container_path: str = "/root/.cache/huggingface"
    hf_endpoint: str = "https://huggingface.co"
    hf_token: str | None = Field(default=None, repr=False)  # ADR 7: never logged or returned
    hf_token_file: Path = _HOME / ".config" / "engine-console" / "hf_token"
    hf_allowed_hosts: list[str] = [
        "huggingface.co", "cdn-lfs*.huggingface.co", "cas-bridge.xethub.hf.co", "*.hf.co",
    ]
    docker_bin: str = "docker"
    docker_context: str = "rootless"  # repo hook requires the rootless context
    # Engines live on an --internal network (no external routing, no host ports). Only the per-instance
    # gateway sidecar is on the default bridge and publishes 127.0.0.1:<port>.
    engine_network: str = "engine-console-engines"
    # nginx:alpine, pinned by digest, must already be present locally (the console never pulls).
    gateway_image: str = GATEWAY_IMAGE
    # Optional egress chokepoint: an operator-provided HTTP(S) proxy running on the host, e.g. http://127.0.0.1:8082
    egress_proxy: str | None = None
    # PEM certificate the proxy re-signs with. Unset: the system trust store is used. Set but missing: fail closed.
    egress_ca_bundle: Path | None = None
    require_egress_proxy: bool = False
    # Discovery of engines the console did not create (monitor-only). Loopback ports only.
    discovery_enabled: bool = True
    discovery_interval_s: float = 10.0
    discovery_ports: list[int] = [8000, 30000, 11434, 8001, *range(18000, 18100)]
    port_range_start: int = 18000
    port_range_end: int = 18099
    # Engines run with HF_HUB_OFFLINE=1 and only read the cache; the console process does all writing.
    hf_cache_readonly: bool = True
    # Extra engine images permitted as settings pins, matched EXACTLY (full reference: repo:tag or repo@sha256:...).
    # Each adapter's own pinned default_image is always allowed on top of this list; there are no prefix matches.
    engine_image_allowlist: list[str] = []
    # The gateway sidecar has its own exact-match list (it must never be usable as an engine image).
    gateway_image_allowlist: list[str] = [GATEWAY_IMAGE]
    enable_api_docs: bool = False    # reserved: the Swagger page is not served (it needs CDN assets; CSP forbids them)
    allowed_hosts: list[str] = []     # extra Host header values accepted (DNS-rebinding guard)
    body_cap_bytes: int = 1024 * 1024
    chat_body_cap_bytes: int = 32 * 1024 * 1024   # images ride inside chat bodies
    import_body_cap_bytes: int = 256 * 1024
    max_sse_connections: int = 8
    sse_idle_timeout_s: float = 300.0
    max_download_queue: int = 50
    startup_timeout_s: int = 1800
    engine_hf_offline: bool = True    # engines read the local cache only: no surprise egress
    otlp_endpoint: str = "localhost:4317"
    otlp_enabled: bool = True
    frontend_dir: Path | None = None
    background_tasks: bool = True
    scrape_interval_s: float = 2.0
    supervisor_interval_s: float = 2.0
    download_concurrency: int = 2
    trust_loopback: bool = True

    @field_validator("hf_cache_dir")
    @classmethod
    def _check_cache_dir(cls, v: Path) -> Path:
        s = str(v)
        if not v.is_absolute():
            raise ValueError("HF_CACHE_DIR must be an absolute path")
        if any(ch in s for ch in (":", ",", "\n", "\r", "\x00")):
            raise ValueError("HF_CACHE_DIR must not contain ':' ',' or newlines (docker -v syntax injection)")
        if v.is_symlink():
            raise ValueError("HF_CACHE_DIR must not be a symlink")
        return v

    @property
    def db_path(self) -> Path:
        return self.data_dir / "console.db"

    @property
    def hub_dir(self) -> Path:
        return self.hf_cache_dir / "hub"
