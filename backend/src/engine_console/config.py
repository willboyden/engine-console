"""Process configuration: environment variables, with an optional per-user env file so settings survive restarts. Real
environment variables win over the file. Most names carry no prefix (HF_TOKEN, EGRESS_PROXY, ... as documented); the
generic ones in PREFIXED are read only as ENGINE_CONSOLE_<NAME>."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Fixed locations rather than XDG_*: XDG vars are unreliable under snap-packaged terminals and IDEs.
_HOME = Path.home()
# Optional KEY=VALUE file (dotenv syntax), e.g. HF_CACHE_DIR=/data/hf. Keep it chmod 600 if it holds a token.
ENV_FILE = Path(os.environ.get("ENGINE_CONSOLE_ENV_FILE") or _HOME / ".config" / "engine-console" / "env")
GATEWAY_IMAGE = "nginx@sha256:62ff2089abf5a9ed33bd232895bef5e22f7bb4b200675cec49a5ebc48e3d4ac8"
# Review finding L5: names this generic are often exported for something else (a shell or another server's HOST or PORT),
# and an unprefixed HOST=0.0.0.0 used to widen the bind. These settings are read ONLY with the prefix, from the
# environment and from the env file alike; the bare names are ignored (main.run() warns when one is set).
PREFIX = "ENGINE_CONSOLE_"
PREFIXED = ("host", "port", "allowed_hosts", "trust_loopback", "data_dir")


def _prefix_only(env: Mapping[str, str | None]) -> dict[str, str | None]:
    """A source's variables (lower-case names: the settings are case-insensitive) with each bare PREFIXED name dropped
    and ENGINE_CONSOLE_<NAME> moved to its place, so pydantic-settings finds only the prefixed value."""
    out = {k: v for k, v in env.items() if k not in PREFIXED}
    for name in PREFIXED:
        value = out.pop(f"{PREFIX.lower()}{name}", None)
        if value is not None:
            out[name] = value
    return out


def legacy_env_names(environ: Mapping[str, str], env_file: Path | None) -> list[str]:
    """The bare PREFIXED names (HOST, PORT, ...) that are set in `environ` or in the env file and are therefore ignored.
    Names only: their values are never returned."""
    found = {k.lower() for k in environ}
    if env_file is not None:
        found |= set(DotEnvSettingsSource(Settings, env_file=env_file).env_vars)   # parsed as Settings parses it
    return [name.upper() for name in PREFIXED if name in found]


def _default_cache_dir() -> Path:
    """~/.cache/huggingface. If that is a symlink (common when the cache lives on another disk) use its target,
    because the startup validator refuses symlinked cache dirs."""
    p = Path.home() / ".cache" / "huggingface"
    return p.resolve() if p.is_symlink() else p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", case_sensitive=False)

    host: str = "127.0.0.1"          # ENGINE_CONSOLE_HOST. ADR 2: never bind wider without a reverse proxy + keys
    port: int = 8791                 # ENGINE_CONSOLE_PORT
    data_dir: Path = _HOME / ".local" / "share" / "engine-console"   # ENGINE_CONSOLE_DATA_DIR
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
    # Extra engine images permitted as settings pins, matched EXACTLY (full reference: repo:tag or repo@sha256:...).
    # Each adapter's own pinned default_image is always allowed on top of this list; there are no prefix matches.
    engine_image_allowlist: list[str] = []
    # The gateway sidecar has its own exact-match list (it must never be usable as an engine image).
    gateway_image_allowlist: list[str] = [GATEWAY_IMAGE]
    enable_api_docs: bool = False    # reserved: the Swagger page is not served (it needs CDN assets; CSP forbids them)
    allowed_hosts: list[str] = []     # ENGINE_CONSOLE_ALLOWED_HOSTS: extra Host header values accepted (DNS-rebinding guard)
    body_cap_bytes: int = 1024 * 1024
    chat_body_cap_bytes: int = 32 * 1024 * 1024   # images ride inside chat bodies
    import_body_cap_bytes: int = 256 * 1024
    max_sse_connections: int = 8
    max_sse_per_key: int = 4          # L4: one key (or the no-key loopback caller) cannot hold every stream slot
    max_chat_streams: int = 16        # streamed chat replies at once; each holds an engine request open
    max_chat_streams_per_key: int = 4
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
    # Decision 21 (docs/ARCHITECTURE.md §10.1): loopback is NOT trusted by default. The console can start containers on
    # every machine of the fleet, so a local process (or a page in a local browser) must present an API key like anyone else.
    trust_loopback: bool = False      # ENGINE_CONSOLE_TRUST_LOOPBACK
    # --- contract v2: the fleet (docs/ARCHITECTURE.md §10.3) ---
    hosts_file: Path = _HOME / ".config" / "engine-console" / "hosts.json"
    hub_key_file: Path = _HOME / ".config" / "engine-console" / "ssh" / "id_ed25519"
    node_policy: Path | None = None              # policy file for the LOCAL node (default: the node's own lookup)
    node_command: list[str] | None = None        # override the local node argv (development and tests only)
    remote_node_command: str = "/usr/local/lib/ecnode/ecnode --stdio"   # what the restricted key runs remotely
    fleet_enabled: bool = True                   # false: no node links at all (unit tests of unrelated services)
    # A restart stops an engine and starts it again; the fit check reads the host's last pushed sample (every 2 s), which may
    # still show the VRAM the engine just freed. Waiting two sample periods avoids a restart being blocked by itself.
    restart_settle_s: float = 4.0

    @classmethod
    def settings_customise_sources(cls, settings_cls: type[BaseSettings], init_settings: PydanticBaseSettingsSource,
                                   env_settings: PydanticBaseSettingsSource, dotenv_settings: PydanticBaseSettingsSource,
                                   file_secret_settings: PydanticBaseSettingsSource,
                                   ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Both read their variables once, at construction: rewrite them before anything is looked up. Keyword arguments
        # (tests, create_app callers) still use the field names.
        for src in (env_settings, dotenv_settings):
            if isinstance(src, EnvSettingsSource):     # the env-file source is one too
                src.env_vars = _prefix_only(src.env_vars)
        return init_settings, env_settings, dotenv_settings, file_secret_settings

    @field_validator("host")
    @classmethod
    def _check_host(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ENGINE_CONSOLE_HOST must not be empty: an empty address listens on every interface")
        return v

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
