"""Image pins per host: host pin -> engine pin -> None, and the shape rules of PUT /settings `host_image_pins`."""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from engine_console.domain.errors import BadRequest
from engine_console.domain.models import SettingsPatch
from engine_console.services.settings import SettingsService, clean_host_image_pins
from engine_console.services.store import Store

from .conftest import make_settings

DIGEST = "sha256:" + "a" * 64


_stores: list[Store] = []


@pytest.fixture(autouse=True)
def _close_stores() -> Iterator[None]:
    yield
    for store in _stores:
        store.close()
    _stores.clear()


def svc(tmp: Path) -> SettingsService:
    store = Store(":memory:")
    _stores.append(store)
    return SettingsService(store, make_settings(tmp), lambda: {"fake/engine:2", "fake/engine:1"})


def put(s: SettingsService, **kw: object) -> dict[str, object]:
    return s.update(SettingsPatch(**kw))   # type: ignore[arg-type]


def test_no_pins_by_default(tmp_path: Path) -> None:
    s = svc(tmp_path)
    assert s.host_image_pins() == {} and s.image_pin("vllm") is None and s.image_pin("vllm", "spark") is None
    assert s.effective()["host_image_pins"] == {}


def test_a_host_pin_beats_the_engine_pin_which_beats_nothing(tmp_path: Path) -> None:
    s = svc(tmp_path)
    put(s, image_pins={"vllm": "fake/engine:2"})
    assert s.image_pin("vllm") == "fake/engine:2" and s.image_pin("vllm", "spark") == "fake/engine:2"      # the global pin applies
    put(s, host_image_pins={"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}})
    assert s.image_pin("vllm", "spark") == "vllm/vllm-openai:v0.12.0-arm64"
    assert s.image_pin("vllm", "local") == "fake/engine:2" and s.image_pin("vllm") == "fake/engine:2"
    assert s.image_pin("sglang", "spark") is None                                                    # no pin for another engine
    assert s.effective()["host_image_pins"] == {"spark": {"vllm": "vllm/vllm-openai:v0.12.0-arm64"}}
    assert s.effective()["image_pins"] == {"vllm": "fake/engine:2"}                                    # untouched by host pins


def test_local_is_the_default_host(tmp_path: Path) -> None:
    s = svc(tmp_path)
    put(s, host_image_pins={"local": {"vllm": "a/b:1"}})
    assert s.image_pin("vllm") == "a/b:1" and s.image_pin("vllm", host_id="local") == "a/b:1"


def test_a_put_replaces_the_whole_table_and_empty_clears_it(tmp_path: Path) -> None:
    s = svc(tmp_path)
    put(s, host_image_pins={"spark": {"vllm": "a/b:1"}, "gpu2": {"sglang": "c/d:2"}})
    put(s, host_image_pins={"gpu2": {"sglang": "c/d:3"}})
    assert s.host_image_pins() == {"gpu2": {"sglang": "c/d:3"}}
    put(s, idle_ttl_s=10)                                              # a patch without the field leaves it alone
    assert s.host_image_pins() == {"gpu2": {"sglang": "c/d:3"}}
    put(s, host_image_pins={})
    assert s.host_image_pins() == {}


def test_a_host_with_no_pins_is_dropped(tmp_path: Path) -> None:
    s = svc(tmp_path)
    put(s, host_image_pins={"spark": {}, "gpu2": {"vllm": "a/b:1"}})
    assert s.host_image_pins() == {"gpu2": {"vllm": "a/b:1"}}


def test_host_pins_are_not_held_to_the_hubs_allowlist(tmp_path: Path) -> None:
    """Whether a machine may run an image is that machine's node policy (an arm64-only image is not in the hub's
    list). The global pin keeps the v1 rule."""
    s = svc(tmp_path)
    put(s, host_image_pins={"spark": {"vllm": "registry.example/team/vllm-arm64:1.2.3"}})
    assert s.image_pin("vllm", "spark") == "registry.example/team/vllm-arm64:1.2.3"
    with pytest.raises(BadRequest) as ei:
        put(s, image_pins={"vllm": "registry.example/team/vllm-arm64:1.2.3"})
    assert ei.value.code == "image_not_allowed"
    with pytest.raises(BadRequest) as ei2:
        put(s, image_pins={"vllm": "fake/engine:latest"})
    assert ei2.value.code == "invalid_setting"


@pytest.mark.parametrize("image", ["a/b:1", "vllm/vllm-openai:v0.12.0", f"repo/name@{DIGEST}", "ghcr.io/ggml-org/llama.cpp:server-cuda-b6000",
                                   "registry.example:5000/team/img:1.0+build.5", "a" * 200 + ":1"])
def test_pinned_tags_and_digests_are_accepted(image: str) -> None:
    assert clean_host_image_pins({"spark": {"vllm": image}}) == {"spark": {"vllm": image}}


@pytest.mark.parametrize("image", ["a/b", "a/b:latest", "latest", "", "a/b:1 --privileged", "a/b:1\n--rm", "-a/b:1", "a/b:1;rm -rf /",
                                   "a b:1", "$(x):1", "a/b:" + "x" * 300, "é/b:1", "a/b:1\x00"])
def test_images_that_are_not_pinned_references_are_refused(image: str) -> None:
    with pytest.raises(BadRequest) as ei:
        clean_host_image_pins({"spark": {"vllm": image}})
    assert ei.value.code == "invalid_setting"


@pytest.mark.parametrize("host", ["Spark", "-x", "a b", "a" * 33, "", "a/b", "a\n", "ü", "UPPER", "a_b"])
def test_host_ids_follow_the_host_id_rule(host: str) -> None:
    with pytest.raises(BadRequest) as ei:
        clean_host_image_pins({host: {"vllm": "a/b:1"}})
    assert ei.value.code == "invalid_setting"
    assert clean_host_image_pins({"a" * 32: {"vllm": "a/b:1"}, "gpu-2": {"vllm": "a/b:1"}, "0x": {"vllm": "a/b:1"}})


@pytest.mark.parametrize("engine", ["", "has space", "-lead", "a" * 65, "a/b", "a\n", "é"])
def test_engine_ids_are_checked_too(engine: str) -> None:
    with pytest.raises(BadRequest):
        clean_host_image_pins({"spark": {engine: "a/b:1"}})
    assert clean_host_image_pins({"spark": {"llama.cpp": "a/b:1", "trt_llm-2": "a/b:1", "V1": "a/b:1"}})


def test_the_table_is_bounded() -> None:
    with pytest.raises(BadRequest):
        clean_host_image_pins({f"h{i}": {"vllm": "a/b:1"} for i in range(33)})
    with pytest.raises(BadRequest):
        clean_host_image_pins({"spark": {f"e{i}": "a/b:1" for i in range(17)}})
    assert len(clean_host_image_pins({f"h{i}": {"vllm": "a/b:1"} for i in range(32)})) == 32


def test_a_refused_change_changes_nothing(tmp_path: Path) -> None:
    s = svc(tmp_path)
    put(s, host_image_pins={"spark": {"vllm": "a/b:1"}})
    with pytest.raises(BadRequest):
        put(s, idle_ttl_s=99, host_image_pins={"spark": {"vllm": "a/b:latest"}})
    assert s.host_image_pins() == {"spark": {"vllm": "a/b:1"}} and s.idle_ttl_s() == 0


def test_a_damaged_stored_value_reads_as_no_pins(tmp_path: Path) -> None:
    s = svc(tmp_path)
    for raw in ('"just a string"', "[1, 2]", '{"spark": "x"}', '{"spark": {"vllm": 5}}', '{"spark": {"vllm": "a/b:1"}, "x": 1}'):
        s._db.execute("INSERT INTO settings(key,value) VALUES('host_image_pins',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (raw,))
        pins = s.host_image_pins()
        assert all(isinstance(h, str) and all(isinstance(e, str) and isinstance(i, str) for e, i in v.items()) for h, v in pins.items())
    assert json.loads('{"spark": {"vllm": "a/b:1"}}') and s.image_pin("vllm", "spark") == "a/b:1"
    s._db.execute("UPDATE settings SET value='\"x\"' WHERE key='host_image_pins'")
    assert s.host_image_pins() == {} and s.image_pin("vllm", "spark") is None


def test_v1_methods_are_all_still_there(tmp_path: Path) -> None:
    s = svc(tmp_path)
    for name in ("engine_image_allowed", "hf_token", "token_status", "hf_cache_dir", "hub_dir", "idle_ttl_s", "default_gpu_ids",
                 "image_pin", "effective", "update", "create_key", "list_keys", "revoke_key", "authenticate", "ensure_bootstrap_key"):
        assert callable(getattr(s, name)), name
    assert s.hub_dir() == s.hf_cache_dir() / "hub" and s.idle_ttl_s() == 0 and s.default_gpu_ids() == []
    assert s.token_status() == "[unset]" and s.hf_token() is None
    key = s.create_key("k", "admin")
    assert key.secret and s.authenticate(key.secret) is not None and s.list_keys()[0].secret is None
