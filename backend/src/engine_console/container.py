"""Composition root: builds every service from ports. Tests inject fakes through the keyword arguments.

Contract v2: the hub never runs docker, reads /proc or opens a model directory itself. Every machine (the hub's own,
`local`, included) is reached through its node, and every service that needs one holds the `FleetPort`."""
from __future__ import annotations

from dataclasses import dataclass

from engine_console.adapters import AdapterRegistry, load_default_adapters
from engine_console.config import Settings
from engine_console.domain.ports import HubClient
from engine_console.services.audit import AuditService
from engine_console.services.bench import BenchService
from engine_console.services.chat import ChatService
from engine_console.services.common import EventBus, SseLimiter
from engine_console.services.downloads import DownloadService
from engine_console.services.fitting import FitService
from engine_console.services.fleet import FleetService
from engine_console.services.hardware import HardwareService
from engine_console.services.hf import HfService
from engine_console.services.hf_http import EgressConfig, HfHttpClient
from engine_console.services.library import LibraryService
from engine_console.services.lifecycle import LifecycleService
from engine_console.services.metrics import MetricsService
from engine_console.services.nodehttp import NodeHttp
from engine_console.services.profiles import ProfileService
from engine_console.services.scripts import ScriptService
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
    settings: SettingsService
    fleet: FleetService            # any FleetPort with start()/stop() in tests (see tests/core/fakefleet.py)
    hardware: HardwareService
    hf: HfService
    library: LibraryService
    downloads: DownloadService
    fit: FitService
    scripts: ScriptService
    nodehttp: NodeHttp
    lifecycle: LifecycleService
    profiles: ProfileService
    metrics: MetricsService
    bench: BenchService
    usage: UsageService
    chat: ChatService
    audit: AuditService
    telemetry: Telemetry
    sse: SseLimiter
    chat_streams: SseLimiter


def build_container(cfg: Settings, *, adapters: AdapterRegistry | None = None, fleet: FleetService | None = None,
                    library: LibraryService | None = None, hub: HubClient | None = None,
                    store: Store | None = None) -> Container:
    store = store or Store(cfg.db_path)
    bus = EventBus()
    reg = adapters if adapters is not None else load_default_adapters()
    settings = SettingsService(store, cfg, lambda: {a.default_image for a in reg.list()})
    hub = hub or HfHttpClient(cfg.hf_endpoint, cfg.hf_allowed_hosts, settings.hf_token, egress=egress_config(cfg))
    hf = HfService(hub)
    fleet = fleet or FleetService(cfg, bus)
    hardware = HardwareService(fleet)
    nodehttp = NodeHttp(fleet)
    scripts = ScriptService(fleet)
    usage = UsageService(store)
    fit = FitService(reg, hf, hardware, settings)
    # the library needs to know which instances use a model, and the lifecycle needs the library: bind the first late
    holder: dict[str, LifecycleService] = {}
    library = library or LibraryService(store, fleet, reg, bus, in_use=lambda host, path: (
        holder["l"].instances_using(host, path) if "l" in holder else []))
    lifecycle = LifecycleService(store, reg, fleet, hardware, settings, fit, nodehttp, cfg, bus, library=library,
                                 scripts=scripts, secrets=SecretStore(cfg.data_dir / "secrets"))
    holder["l"] = lifecycle
    downloads = DownloadService(store, hub, hf, settings, bus, fleet, library, concurrency=cfg.download_concurrency,
                                max_queue=cfg.max_download_queue)
    # what the fleet shows next to each host: which instance a GPU process belongs to, and how many engines/models
    fleet.set_instance_resolver(lifecycle.instance_for_container)
    fleet.set_counts_provider(lifecycle.counts)
    return Container(
        cfg=cfg, store=store, bus=bus, adapters=reg, settings=settings, fleet=fleet, hardware=hardware, hf=hf,
        library=library, downloads=downloads, fit=fit, scripts=scripts, nodehttp=nodehttp, lifecycle=lifecycle,
        profiles=ProfileService(store, reg),
        metrics=MetricsService(store, lifecycle, reg, fleet, nodehttp, bus, interval_s=cfg.scrape_interval_s),
        bench=BenchService(store, lifecycle, nodehttp, bus), usage=usage,
        chat=ChatService(store, lifecycle, nodehttp, usage), audit=AuditService(store), telemetry=Telemetry(),
        sse=SseLimiter(cfg.max_sse_connections, cfg.max_sse_per_key),
        chat_streams=SseLimiter(cfg.max_chat_streams, cfg.max_chat_streams_per_key))
