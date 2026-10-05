"""The real library and the real lifecycle wired together (the other core tests use a fake library).

security review 2026-10-04 L2: the library asks the lifecycle who uses a copy, and the lifecycle asked the library where
an instance's model is, which asked the lifecycle who uses it... With one ready managed instance whose model was in the
library, every library read hit RecursionError, which was swallowed and logged with a ~1000-frame traceback."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from ecnode import proto

from engine_console.adapters import load_default_adapters
from engine_console.config import Settings
from engine_console.container import build_container
from engine_console.domain.fleet import HostConfig
from engine_console.services.common import EventBus, now
from engine_console.services.fleet import FleetService

from .fakefleet import FakeNode, make_hello
from .fakes import FakeHub


class _Node(FakeNode):
    async def call(self, op: str, args: dict[str, Any] | None = None, *, timeout: float = 30.0) -> dict[str, Any]:
        if op == "models.list":
            entries = [{"root": "hf", "rel": "hub/models--org--m", "format": "hf", "name": "org/m", "revision": "a" * 40,
                        "size": 10**10, "files": 4, "mtime": 1.0}]
            return proto.clean({"generation": 1, "entries": entries, "next_cursor": None, "roots": []},
                               proto.RESULTS["models.list"])
        if op in ("models.scan", "metrics.configure"):
            return {}
        return await super().call(op, args, timeout=timeout)

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    def reconnect(self) -> None:
        pass


async def test_l2_library_reads_with_an_active_instance_do_not_recurse(tmp_path: Path,
                                                                       caplog: pytest.LogCaptureFixture) -> None:
    cfg = Settings(data_dir=tmp_path / "data", hf_cache_dir=tmp_path / "hf", otlp_enabled=False, background_tasks=False,
                   frontend_dir=tmp_path / "fe", hf_token=None, hf_token_file=tmp_path / "no-token",
                   hosts_file=tmp_path / "hosts.json", _env_file=None)
    host = HostConfig(id="local", label="L")
    node = _Node(host, hello=make_hello(hostname="L", roots=[{"id": "hf", "path": "/data/hf", "kind": "hf-home",
                                                              "writable": True, "label": "HF"}]))
    fleet = FleetService(cfg, EventBus(), hosts=[host], link_factory=lambda h: node)
    cont = build_container(cfg, adapters=load_default_adapters(), fleet=fleet, hub=FakeHub())
    await cont.library.refresh("local")
    cont.store.execute(
        "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,port,container_name,image,state,created_at,"
        "host_id,model) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("inst_aaaaaaaaaaaa", "m", "vllm", "org/m", "{}", "[0]", "[]", 18001, "ec-m-aaaaaa", "vllm/vllm-openai:v1",
         "ready", now(), "local", json.dumps({"format": "hf", "repo_id": "org/m"})))
    with caplog.at_level(logging.ERROR, logger="engine_console"):
        models = cont.library.models()
        fleet.summaries()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert [c.in_use_by for m in models for c in m.copies] == [["inst_aaaaaaaaaaaa"]]
    assert cont.lifecycle.get("inst_aaaaaaaaaaaa").model_location is not None
