"""Offline fakes for every port: NVML/smoke probe, Hugging Face hub, docker runner, and a FakeAdapter."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from engine_console.adapters.base import (
    Compat,
    EngineAdapter,
    Hardware,
    LaunchSpec,
    ModelInfo,
    ParamSpec,
)
from engine_console.domain.errors import GatedModel, NotFound
from engine_console.domain.models import GpuStat
from engine_console.domain.ports import CmdResult, HubModel, RepoFile


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


class FakeProbe:
    name = "fake"

    def __init__(self, gpus: int = 2, total: float = 96.0, free: float | None = None) -> None:
        self.gpus, self.total, self.free = gpus, total, total if free is None else free

    def query(self) -> list[GpuStat]:
        return [GpuStat(index=i, uuid=f"GPU-uuid-{i}", name="RTX PRO 6000 Blackwell", total_gib=self.total,
                        free_gib=self.free, used_gib=self.total - self.free, util_pct=5.0, temp_c=40.0, power_w=90.0,
                        fan_pct=30.0, compute_capability="12.0") for i in range(self.gpus)]


class BrokenProbe:
    name = "broken"

    def query(self) -> list[GpuStat]:
        raise RuntimeError("no driver")


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


@dataclass
class FakeRunner:
    """Pretends to be the docker CLI. Records every argv so tests can assert on exact commands."""
    calls: list[list[str]] = field(default_factory=list)
    envs: list[Mapping[str, str] | None] = field(default_factory=list)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    logs: dict[str, str] = field(default_factory=dict)
    missing_images: set[str] = field(default_factory=set)
    fail_run: str | None = None
    networks: dict[str, bool] = field(default_factory=dict)   # name -> internal?
    fail_connect: bool = False

    async def run(self, argv: Sequence[str], *, timeout: float, env_extra: Mapping[str, str] | None = None,
                  merge_stderr: bool = False) -> CmdResult:
        a = list(argv)
        self.calls.append(a)
        self.envs.append(env_extra)
        assert a[:3] == ["docker", "--context", "rootless"], a   # every call must pin the rootless context
        sub = a[3]
        if sub == "run":
            if self.fail_run:
                return CmdResult(125, "", self.fail_run)
            name = a[a.index("--name") + 1]
            labels = dict(x.split("=", 1) for i, x in enumerate(a) if i and a[i - 1] == "--label")
            self.containers[name] = {"running": True, "labels": labels, "exit": 0}
            return CmdResult(0, "abcdef1234567890\n", "")
        if sub == "network":
            op = a[4]
            if op == "inspect":
                name = a[-1]
                if name not in self.networks:
                    return CmdResult(1, "", "Error: No such network")
                return CmdResult(0, "true\n" if self.networks[name] else "false\n", "")
            if op == "create":
                self.networks[a[-1]] = "--internal" in a
                return CmdResult(0, "netid\n", "")
            if op == "connect":
                if self.fail_connect:
                    return CmdResult(1, "", "connect failed")
                self.containers[a[-1]].setdefault("networks", []).append(a[-2])
                return CmdResult(0, "", "")
        if sub == "create":
            name = a[a.index("--name") + 1]
            labels = dict(x.split("=", 1) for i, x in enumerate(a) if i and a[i - 1] == "--label")
            self.containers[name] = {"running": False, "labels": labels, "exit": 0, "networks": [a[a.index("--network") + 1]]}
            return CmdResult(0, "gwid\n", "")
        if sub == "start":
            self.containers[a[-1]]["running"] = True
            return CmdResult(0, a[-1], "")
        if sub == "inspect":
            c = self.containers.get(a[4])
            if c is None:
                return CmdResult(1, "[]", "Error: No such container")
            argv = c.get("argv", [])
            doc = [{"Name": "/" + a[4], "Path": argv[0] if argv else "", "Args": argv[1:],
                    "State": {"Running": c["running"], "Status": "running" if c["running"] else "exited",
                              "ExitCode": c["exit"]},
                    "Config": {"Labels": c["labels"], "Image": c.get("image", ""), "Env": c.get("env", [])},
                    "HostConfig": {"NetworkMode": "host" if c.get("host_network") else "bridge"},
                    "NetworkSettings": {"Ports": c.get("ports", {})}}]
            return CmdResult(0, json.dumps(doc), "")
        if sub == "image":
            return CmdResult(1 if a[-1] in self.missing_images else 0, "sha256:x", "")
        if sub == "logs":
            return CmdResult(0, self.logs.get(a[-1], ""), "")
        if sub == "stop":
            if a[-1] in self.containers:
                self.containers[a[-1]]["running"] = False
            return CmdResult(0, "", "")
        if sub == "rm":
            self.containers.pop(a[-1], None)
            return CmdResult(0, "", "")
        if sub == "ps":
            if "--filter" in a:
                return CmdResult(0, "\n".join(self.containers), "")
            return CmdResult(0, "\n".join(n for n, c in self.containers.items() if c["running"]), "")
        return CmdResult(1, "", f"unhandled {sub}")

    async def stream(self, argv: Sequence[str]) -> AsyncIterator[str]:
        self.calls.append(list(argv))
        for line in self.logs.get(list(argv)[-1], "").splitlines():
            yield line

    def crash(self, name: str, code: int = 137) -> None:
        self.containers[name]["running"] = False
        self.containers[name]["exit"] = code
