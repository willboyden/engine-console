from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from engine_console.adapters import AdapterRegistry
from engine_console.config import Settings
from engine_console.container import Container, build_container
from engine_console.main import create_app

from .fakes import FakeAdapter, FakeHub, FakeProbe, FakeRunner

QWEN_CFG: dict[str, Any] = {
    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 64,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}


class Env:
    """Everything a test needs: the app container, the fakes behind it, and a programmable fake engine."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.hub, self.runner = FakeHub(), FakeRunner()
        self.engine_routes: dict[str, Callable[[httpx.Request], httpx.Response]] = {}
        self.engine_requests: list[httpx.Request] = []
        self.port_routes: dict[tuple[int, str], Callable[[httpx.Request], httpx.Response]] = {}
        self.cfg = Settings(data_dir=tmp / "data", hf_cache_dir=tmp / "hf", otlp_enabled=False, background_tasks=False,
                            frontend_dir=tmp / "fe", hf_token=None, hf_token_file=tmp / "no-token",
                            allowed_hosts=["testserver"], discovery_ports=[], engine_image_allowlist=["fake/engine:2", "fake/engine:2.0"], gateway_image="fake/gateway:1",
                            gateway_image_allowlist=["fake/gateway:1"])
        (tmp / "fe").mkdir()
        (tmp / "fe" / "index.html").write_text("<html>console</html>")
        (tmp / "fe" / "js").mkdir()
        (tmp / "fe" / "js" / "app.js").write_text("console.log(1)")
        self.probe = FakeProbe()
        self.runner.networks[self.cfg.engine_network] = True   # pre-existing internal network
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(self._engine))
        self.container: Container = build_container(
            self.cfg, adapters=AdapterRegistry([FakeAdapter()]), probes=[self.probe], hub=self.hub, runner=self.runner,
            http=self.http, port_free=lambda p: True)

    def _engine(self, req: httpx.Request) -> httpx.Response:
        self.engine_requests.append(req)
        h = self.port_routes.get((req.url.port or 0, req.url.path)) or self.engine_routes.get(req.url.path)
        return h(req) if h else httpx.Response(503, text="not up")

    def seed_local_model(self, repo_id: str = "Qwen/Qwen3-32B", cfg: dict[str, Any] | None = None,
                         size: int = 65_530_000_000) -> None:
        """Write a tiny HF-cache snapshot; the fake safetensors file's stat size is what counts, so use a sparse file."""
        snap = self.cfg.hf_cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "c0ffee00000000000000000000000000000000ff"
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(cfg or QWEN_CFG))
        with (snap / "model.safetensors").open("wb") as f:
            f.truncate(size)   # sparse: no real disk used
        refs = snap.parent.parent / "refs"
        refs.mkdir()
        (refs / "main").write_text("c0ffee00000000000000000000000000000000ff")

    def engine_up(self) -> None:
        self.engine_routes["/health"] = lambda r: httpx.Response(200, text="ok")


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def client(env: Env) -> Iterator[TestClient]:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("127.0.0.1", 50000), headers={"X-Engine-Console": "1"}) as c:
        yield c
