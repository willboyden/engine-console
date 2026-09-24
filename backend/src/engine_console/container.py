"""Composition root: builds every service from ports. Tests inject fakes through the keyword arguments."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx

from engine_console.adapters import AdapterRegistry, load_default_adapters
from engine_console.config import Settings
from engine_console.domain.ports import CommandRunner, GpuProbe, HubClient
from engine_console.services.audit import AuditService
from engine_console.services.bench import BenchService
from engine_console.services.chat import ChatService
from engine_console.services.common import EventBus, SseLimiter
from engine_console.services.docker import DockerCli, SubprocessRunner
from engine_console.services.downloads import DownloadService
from engine_console.services.fitting import FitService
from engine_console.services.hardware import HardwareService
from engine_console.services.hf import HfService
from engine_console.services.hf_http import EgressConfig, HfHttpClient
from engine_console.services.lifecycle import LifecycleService, _bind_free
from engine_console.services.metrics import MetricsService
from engine_console.services.profiles import ProfileService
from engine_console.services.secrets_store import SecretStore
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store
from engine_console.services.telemetry import Telemetry
from engine_console.services.usage import UsageService


def egress_config(cfg: Settings) -> EgressConfig:
    return EgressConfig(proxy=cfg.egress_proxy, ca_bundle=cfg.egress_ca_bundle, require_proxy=cfg.require_egress_proxy)


@dataclass
class Container:
    cfg: Settings
    store: Store
    bus: EventBus
    adapters: AdapterRegistry
    http: httpx.AsyncClient
    settings: SettingsService
    hardware: HardwareService
    hf: HfService
    docker: DockerCli
    downloads: DownloadService
    fit: FitService
    lifecycle: LifecycleService
    profiles: ProfileService
    metrics: MetricsService
    bench: BenchService
    usage: UsageService
    chat: ChatService
    audit: AuditService
    telemetry: Telemetry
    sse: SseLimiter


def build_container(cfg: Settings, *, adapters: AdapterRegistry | None = None, probes: list[GpuProbe] | None = None,
                    hub: HubClient | None = None, runner: CommandRunner | None = None,
                    http: httpx.AsyncClient | None = None, store: Store | None = None,
                    port_free: Callable[[int], bool] = _bind_free) -> Container:
    store = store or Store(cfg.db_path)
    bus = EventBus()
    reg = adapters if adapters is not None else load_default_adapters()
    # engine + local traffic only (loopback gateways). trust_env=False: an ambient HTTP(S)_PROXY must never capture
    # it; HF has its own allowlisted, proxy-aware client.
    http = http or httpx.AsyncClient(trust_env=False)
    settings = SettingsService(store, cfg, lambda: {a.default_image for a in reg.list()})
    hardware = HardwareService(probes)
    hub = hub or HfHttpClient(cfg.hf_endpoint, cfg.hf_allowed_hosts, settings.hf_token, egress=egress_config(cfg))
    hf = HfService(hub)
    docker = DockerCli(runner or SubprocessRunner(), binary=cfg.docker_bin, context=cfg.docker_context)
    fit = FitService(reg, hf, hardware, settings)
    usage = UsageService(store)
    # downloads needs to know which repos are in use; lifecycle needs downloads: break the cycle with a late binding
    holder: dict[str, LifecycleService] = {}
    downloads = DownloadService(store, hub, hf, settings, bus, concurrency=cfg.download_concurrency, max_queue=cfg.max_download_queue,
                                in_use=lambda r: holder["l"].uses_repo(r) if "l" in holder else False)
    lifecycle = LifecycleService(store, reg, docker, hardware, settings, downloads, fit, http, cfg, bus, port_free,
                                 SecretStore(cfg.data_dir / "secrets"))
    holder["l"] = lifecycle
    return Container(
        cfg=cfg, store=store, bus=bus, adapters=reg, http=http, settings=settings, hardware=hardware, hf=hf,
        docker=docker, downloads=downloads, fit=fit, lifecycle=lifecycle, profiles=ProfileService(store, reg),
        metrics=MetricsService(store, lifecycle, reg, http, bus, interval_s=cfg.scrape_interval_s),
        bench=BenchService(store, lifecycle, http, bus), usage=usage,
        chat=ChatService(store, lifecycle, http, usage), audit=AuditService(store), telemetry=Telemetry(),
        sse=SseLimiter(cfg.max_sse_connections))
