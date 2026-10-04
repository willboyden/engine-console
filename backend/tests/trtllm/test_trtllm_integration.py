"""How the adapter meets the rest of the console: the registry, and the node's metric filter."""
from __future__ import annotations

import json

import pytest

from engine_console.adapters import load_default_adapters
from engine_console.adapters.base import EngineAdapter
from engine_console.adapters.trtllm import TrtllmAdapter

from .conftest import read, read_json


def test_registry_finds_and_instantiates_the_adapter() -> None:
    reg = load_default_adapters(("engine_console.adapters.trtllm",))
    adapter = reg.get("trtllm")
    assert isinstance(adapter, TrtllmAdapter) and isinstance(adapter, EngineAdapter)
    assert adapter.display_name == "TensorRT-LLM" and "trtllm" in reg
    assert [a.id for a in reg.list()] == ["trtllm"]


def test_adapter_implements_the_whole_port(adapter: TrtllmAdapter) -> None:
    assert not getattr(type(adapter), "__abstractmethods__", None)
    assert adapter.image_for("x86_64") == adapter.default_image == adapter.image_for("aarch64")


def _node_filter() -> object:
    agent = pytest.importorskip("ecnode.agent")
    fn = getattr(agent, "filter_metrics", None)
    if fn is None:
        pytest.skip("the node no longer exposes filter_metrics")
    return fn


def test_the_nodes_metric_filter_keeps_what_the_adapter_parses(adapter: TrtllmAdapter) -> None:
    """The node keeps a line only if its metric name starts with one of `metric_prefixes`. `/metrics` is a single JSON
    line without metric names; the filter reads the text before the first `{` as the name, which is `[`."""
    node_filter = _node_filter()
    body = json.dumps([read_json("iter_stats_1.2.1.json")["plain"]], separators=(",", ":"))   # what the server sends
    assert "\n" not in body
    kept = node_filter(body, adapter.metric_prefixes)  # type: ignore[operator]
    assert kept.strip() == body
    assert adapter.parse_metrics(kept)["requests_running"] == 3
    # a filter without "[" would silently drop the only data this engine's /metrics carries
    assert node_filter(body, ("trtllm_",)) == ""  # type: ignore[operator]
    prom = read("collector_1.2.1.prom")
    kept_prom = node_filter("# HELP python_info x\npython_info 1\n" + prom, adapter.metric_prefixes)  # type: ignore[operator]
    assert "python_info" not in kept_prom and "trtllm_e2e_request_latency_seconds_bucket" in kept_prom
    assert adapter.parse_metrics(kept_prom).keys() == adapter.parse_metrics(prom).keys()


def test_prefixes_are_short_strings(adapter: TrtllmAdapter) -> None:
    assert adapter.metric_prefixes and all(isinstance(p, str) and 0 < len(p) <= 64 for p in adapter.metric_prefixes)
