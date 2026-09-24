"""Observability for the console itself (ADR 10): JSON logs with secret scrubbing, OTLP traces, Prometheus."""
from __future__ import annotations

import json
import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from engine_console.config import Settings
from engine_console.domain.models import HostMemInfo
from engine_console.services.common import redact


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"), "level": record.levelname,
                               "logger": record.name, "msg": redact(record.getMessage())}
        if record.exc_info:
            doc["exc"] = record.exc_info[0].__name__ if record.exc_info[0] else "error"   # no traceback: may hold secrets
        return json.dumps(doc)


def configure_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root.handlers = [h]
    root.setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # httpx logs full URLs at INFO


def setup_tracing(cfg: Settings) -> TracerProvider | None:
    """OTLP/gRPC to the local collector. Export failures are non-fatal by design (BatchSpanProcessor drops)."""
    if not cfg.otlp_enabled:
        return None
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

    provider = TracerProvider(resource=Resource.create({"service.name": "engine-console"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=True)))
    trace.set_tracer_provider(provider)
    return provider


class Telemetry:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("engine_console_http_requests_total", "HTTP requests", ["method", "route", "status"],
                                registry=self.registry)
        self.latency = Histogram("engine_console_http_request_seconds", "HTTP request latency", ["method", "route"],
                                 registry=self.registry)
        self.bench_runs = Counter("engine_console_bench_runs_total", "Benchmarks started", registry=self.registry)
        self.instance_gauge = Gauge("engine_console_instances", "Instances by state", ["state"], registry=self.registry)
        self.tracer = trace.get_tracer("engine_console")
        self.host_gauges = {k: Gauge(f"engine_console_host_{k}_gib", f"Host {k.replace('_', ' ')} in GiB", registry=self.registry)
                            for k in ("ram_used", "ram_available", "shmem", "swap_used")}
        self.instance_ram = Gauge("engine_console_instance_host_ram_gib", "Host RAM held by an instance's container (cgroup)",
                                  ["instance"], registry=self.registry)

    def set_host_memory(self, m: HostMemInfo | None, per_instance: dict[str, float]) -> None:
        if m is not None:
            for k, v in (("ram_used", m.used_gib), ("ram_available", m.available_gib), ("shmem", m.shmem_gib),
                         ("swap_used", m.swap_used_gib)):
                self.host_gauges[k].set(v)
        self.instance_ram.clear()          # drop label sets of instances that are gone: cardinality stays bounded
        for iid, gib in per_instance.items():
            self.instance_ram.labels(instance=iid).set(gib)

    def set_instance_states(self, counts: dict[str, int]) -> None:
        for s in ("stopped", "starting", "loading", "ready", "stopping", "failed", "auth_required", "unreachable"):
            self.instance_gauge.labels(state=s).set(counts.get(s, 0))

    def render(self) -> bytes:
        return generate_latest(self.registry)


