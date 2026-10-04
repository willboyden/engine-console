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
from engine_console.domain.fleet import HostConfig
from engine_console.main import create_app

from .fakefleet import FakeFleet, FakeLibrary, FakeNode
from .fakes import FakeAdapter, FakeHub

QWEN_CFG: dict[str, Any] = {
    "model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 5120, "num_hidden_layers": 64,
    "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
    "torch_dtype": "bfloat16",
}


class Routes:
    """`env.engine_routes["/health"] = handler` / `env.port_routes[(port, "/path")] = handler`: what the fake engines answer,
    registered on the fake node (the hub reaches every engine through `http.request`)."""

    def __init__(self, node: FakeNode, any_port: bool) -> None:
        self._node, self._any = node, any_port

    def __setitem__(self, key: Any, handler: Callable[[httpx.Request], httpx.Response]) -> None:
        port, path = (None, key) if self._any else key
        self._node.serve(port, path, handler)


class Env:
    """Everything a test needs: the app container, the fake fleet behind it (one machine, `local`, by default), the fake
    model library and hub, and programmable fake engines."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp
        self.hub = FakeHub()
        self.cfg = Settings(data_dir=tmp / "data", hf_cache_dir=tmp / "hf", otlp_enabled=False, background_tasks=False,
                            frontend_dir=tmp / "fe", hf_token=None, hf_token_file=tmp / "no-token",
                            allowed_hosts=["testserver"], trust_loopback=True, scrape_interval_s=0.01, restart_settle_s=0.0,
                            engine_image_allowlist=["fake/engine:2", "fake/engine:2.0"], hosts_file=tmp / "hosts.json")
        (tmp / "fe").mkdir()
        (tmp / "fe" / "index.html").write_text("<html>console</html>")
        (tmp / "fe" / "js").mkdir()
        (tmp / "fe" / "js" / "app.js").write_text("console.log(1)")
        self.node = FakeNode(HostConfig(id="local", label="testbox"))
        self.fleet = FakeFleet([self.node])
        self.library = FakeLibrary(self.fleet)
        self.engine_routes = Routes(self.node, any_port=True)
        self.port_routes = Routes(self.node, any_port=False)
        self.container: Container = build_container(
            self.cfg, adapters=AdapterRegistry([FakeAdapter()]), fleet=self.fleet,  # type: ignore[arg-type]
            library=self.library, hub=self.hub)  # type: ignore[arg-type]

    @property
    def engine_requests(self) -> list[httpx.Request]:
        return self.node.http_requests

    def add_host(self, host_id: str = "spark", label: str = "Spark", **hello_kw: Any) -> FakeNode:
        """A second machine in the fleet (a remote one: ssh is irrelevant to the fake)."""
        node = FakeNode(HostConfig(id=host_id, label=label, ssh=f"u@{host_id}.example"), **hello_kw)
        self.fleet.add(node)
        return node

    def seed_local_model(self, repo_id: str = "Qwen/Qwen3-32B", cfg: dict[str, Any] | None = None,
                         size: int = 65_530_000_000, host_id: str = "local") -> None:
        """Put a Hugging Face model in a host's library (nothing is written to disk: the library is a fake)."""
        self.library.add_hf(host_id, repo_id, cfg or QWEN_CFG, size)

    def engine_up(self) -> None:
        self.engine_routes["/health"] = lambda r: httpx.Response(200, text="ok")

    def seed_hub_model(self, repo_id: str, cfg: dict[str, Any] | None = None) -> None:
        """A repo on the (fake) Hugging Face hub, for the routes that describe models that are not downloaded yet."""
        self.hub.add(repo_id, {"config.json": json.dumps(cfg or QWEN_CFG).encode(), "model.safetensors": b"x" * 100},
                     cfg or QWEN_CFG, params={"BF16": 32_762_000_000})


@pytest.fixture
def env(tmp_path: Path) -> Env:
    return Env(tmp_path)


@pytest.fixture
def client(env: Env) -> Iterator[TestClient]:
    app = create_app(env.cfg, env.container)
    with TestClient(app, client=("127.0.0.1", 50000), headers={"X-Engine-Console": "1"}) as c:
        yield c


@pytest.fixture(autouse=True)
def _ignore_user_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests must not depend on the developer's ~/.config/engine-console/env (or real EGRESS_*/HF_* variables)."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
