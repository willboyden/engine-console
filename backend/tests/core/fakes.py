"""Offline fakes for the hub ports (Hugging Face) and the engine adapters used across the core tests.

The machine side (GPUs, docker, /proc, model roots, engine HTTP) is faked by `fakefleet.py`: a node speaking the
node protocol's data shapes. Nothing here touches docker, NVML or the network."""
from __future__ import annotations

import hashlib
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from engine_console.adapters.base import (
    Compat,
    EngineAdapter,
    Hardware,
    LaunchContext,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from engine_console.domain.errors import GatedModel, NotFound
from engine_console.domain.ports import HubModel, RepoFile


class FakeAdapter(EngineAdapter):
    id = "fake"
    display_name = "Fake Engine"
    default_image = "fake/engine:1.0"

    def param_catalog(self) -> list[ParamSpec]:
        return [
            ParamSpec(key="max_model_len", flag="--max-model-len", label="Max len", help="ctx", type="int",
                      group="context", affects_memory=True),
            ParamSpec(key="tp", flag="--tp", label="TP", help="tensor parallel", type="int", group="parallelism",
                      affects_memory=True),
            ParamSpec(key="mem_fraction", flag="--mem", label="Mem", help="fraction", type="float", group="memory",
                      affects_memory=True),
            ParamSpec(key="kv_dtype", flag="--kv", label="KV dtype", help="kv", type="enum", group="memory",
                      choices=["auto", "fp8"], affects_memory=True),
            ParamSpec(key="api_key", flag="env:FAKE_API_KEY", label="API key", help="key", type="string", group="network"),
            ParamSpec(key="leaky_key", flag="--leaky-key", label="Leaky", help="argv secret", type="string",
                      group="network"),
            ParamSpec(key="trust_remote_code", flag="--trust-remote-code", label="Trust", help="t", type="bool",
                      group="advanced"),
            ParamSpec(key="image", flag="@image", label="Image", help="override", type="string", group="advanced"),
        ]

    def presets(self) -> dict[str, dict[str, Any]]:
        return {"balanced": {"mem_fraction": 0.9}, "long-context": {"mem_fraction": 0.9, "max_model_len": 8192}}

    def validate(self, params: dict[str, Any]) -> list[str]:
        known = {p.key for p in self.param_catalog()}
        return [f"unknown parameter '{k}'" for k in params if k not in known]

    def build_launch(self, model: str, params: dict[str, Any], hw: Hardware, *, served_name: str,
                     hf_cache_container_path: str) -> LaunchSpec:
        argv = ["--model", model, "--served-model-name", served_name, "--port", "8000"]
        for k, v in params.items():
            if k not in ("api_key", "image"):
                argv += [f"--{k.replace('_', '-')}", str(v)]
        return LaunchSpec(image=self.default_image, argv=argv,
                          env={"HF_HOME": hf_cache_container_path, "FAKE_API_KEY": params.get("api_key", "s3cr3t-key-value")},
                          container_port=8000, health_path="/health", metrics_path="/metrics")

    def compatibility(self, model: ModelInfo, params: dict[str, Any], hw: Hardware) -> list[Compat]:
        if model.quantization == "nvfp4" and model.is_moe:
            return [Compat(level="block", code="nvfp4_moe_sm120", message="use FP8 or SGLang")]
        return []

    def memory_model(self, model: ModelInfo, params: dict[str, Any]) -> dict[str, float]:
        out = {"mem_fraction": float(params.get("mem_fraction", 0.9)), "tp": float(params.get("tp", 1)),
               "kv_bytes_per_elem": 1.0 if params.get("kv_dtype") == "fp8" else 2.0}
        if params.get("max_model_len"):
            out["max_len"] = float(params["max_model_len"])
        return out

    def parse_metrics(self, prometheus_text: str) -> dict[str, float]:
        m = {"running": "requests_running", "waiting": "requests_waiting", "prompt_total": "prompt_tokens_total",
             "gen_total": "generation_tokens_total", "kv": "kv_cache_usage_pct"}
        out: dict[str, float] = {}
        for line in prometheus_text.splitlines():
            name, _, val = line.partition(" ")
            if name in m:
                out[m[name]] = float(val)
        return out

    def parse_startup_log(self, line: str) -> dict[str, Any] | None:
        if (mt := re.search(r"loading weights (\d+)%", line)):
            return {"phase": "loading_weights", "pct": float(mt.group(1))}
        if "server ready" in line:
            return {"phase": "ready"}
        return None


@dataclass
class FakeStream:
    data: bytes
    offset: int = 0
    total: int | None = None

    async def chunks(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.data), 4):   # tiny chunks so pause/resume tests have somewhere to stop
            yield self.data[i:i + 4]

    async def aclose(self) -> None:
        return None


class FakeHub:
    """In-memory hub: repo_id -> (HubModel, {path: bytes})."""

    def __init__(self) -> None:
        self.repos: dict[str, tuple[HubModel, dict[str, bytes]]] = {}
        self.gated: set[str] = set()
        self.calls: list[str] = []
        self.corrupt: set[str] = set()
        self.ignore_range = False

    def add(self, repo_id: str, files: dict[str, bytes], config: dict[str, Any] | None = None, *, sha: str = "c0ffee00000000000000000000000000000000ff",
            gated: bool = False, tags: list[str] | None = None, params: dict[str, int] | None = None) -> None:
        rf = [RepoFile(path=p, size=len(b), sha256=hashlib.sha256(b).hexdigest() if p.endswith(".safetensors") else None)
              for p, b in files.items()]
        hm = HubModel(repo_id=repo_id, sha=sha, gated=gated, license="apache-2.0", pipeline_tag="text-generation",
                      library_name="transformers", tags=tags or [], files=rf, config=config,
                      safetensors_params=params or {}, safetensors_total=sum((params or {}).values()) or None,
                      downloads=100, likes=5)
        self.repos[repo_id] = (hm, files)

    async def search(self, *, q: str, task: str | None, library: str | None, quant: str | None, sort: str, limit: int,
                     token_ok: bool = True) -> list[HubModel]:
        self.calls.append(f"search:{q}")
        hits = [m for m, _ in self.repos.values() if q.lower() in m.repo_id.lower()]
        return hits[:limit]

    async def model(self, repo_id: str, revision: str | None) -> HubModel:
        self.calls.append(f"model:{repo_id}")
        if repo_id in self.gated:
            raise GatedModel(f"gated: {repo_id}")
        if repo_id not in self.repos:
            raise NotFound(f"not found on Hugging Face: {repo_id}")
        return self.repos[repo_id][0]

    async def card(self, repo_id: str, revision: str | None) -> str:
        return f"# {repo_id}\nA model card."

    async def open_file(self, repo_id: str, revision: str, path: str, start: int) -> FakeStream:
        data = self.repos[repo_id][1][path]
        if path in self.corrupt:
            data = b"X" * len(data)
        if self.ignore_range and start:
            return FakeStream(data, 0, len(data))
        return FakeStream(data[start:], start, len(data))


class FakeMultiAdapter(FakeAdapter):
    """A llama.cpp-like engine: loads gguf, plain and ollama models as well as HF ones, needs no HF cache for them,
    ships a second image for arm64 hosts and declares its own metric prefixes."""
    id = "fakecpp"
    display_name = "Fake C++ Engine"
    default_image = "fake/engine:1.0"
    model_formats = frozenset({"hf", "gguf", "plain", "ollama"})
    images_by_arch = {"aarch64": "fake/engine:1.0-arm64"}
    multi_model = False
    metric_prefixes = ["fakecpp:"]

    def build_launch_ctx(self, ctx: LaunchContext) -> LaunchSpec:
        argv = ["--model", ctx.model_path or ctx.model, "--alias", ctx.served_name, "--port", "8000"]
        for k, v in ctx.params.items():
            if k not in ("api_key", "image"):
                argv += [f"--{k.replace('_', '-')}", str(v)]
        return LaunchSpec(image=self.image_for(ctx.hw.arch), argv=argv, env={"FAKE_FORMAT": ctx.model_format},
                          container_port=8000, health_path="/health", metrics_path="/metrics",
                          needs_hf_cache=ctx.model_format == "hf")
