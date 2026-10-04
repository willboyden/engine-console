"""GET /learn, and the engines API that merges learn-mode explanations into the parameter catalogue."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine_console.adapters.base import Explain
from engine_console.domain.learn import LearnBundle
from engine_console.learn import load_bundle

from .conftest import Env
from .fakes import FakeMultiAdapter

P = "/api/v1"


def doc(section: str, items: dict[str, object]) -> str:
    return json.dumps({"version": 1, section: items})


def ex(summary: str, level: str = "advanced") -> dict[str, object]:
    return {"summary": summary, "details": "d", "when_to_change": "w", "risks": "r", "example": "e", "terms": ["kv_cache"],
            "level": level}


@pytest.fixture
def content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LearnBundle:
    """A small learn bundle on disk (real files, loaded by the real loader) served in place of the packaged content."""
    root = tmp_path / "learn"
    (root / "params").mkdir(parents=True)
    (root / "presets").mkdir()
    (root / "glossary.json").write_text(doc("terms", {"kv_cache": {"title": "KV cache", "short": "Memory of the conversation.",
                                                                    "long": "", "see_also": []}}))
    (root / "metrics.json").write_text(doc("metrics", {"requests_running": ex("How many requests run right now.")}))
    (root / "concepts.json").write_text(doc("concepts", {"state.ready": ex("The engine answers requests.", "basic")}))
    (root / "params" / "fake.json").write_text(doc("params", {"max_model_len": ex("The longest conversation.", "basic"),
                                                                "tp": ex("Cards one model is split across.")}))
    (root / "params" / "fakecpp.json").write_text(doc("params", {"max_model_len": ex("The longest conversation (cpp).")}))
    (root / "params" / "ghost.json").write_text(doc("params", {"x": ex("An engine that is not installed.")}))
    (root / "presets" / "fake.json").write_text(doc("presets", {"balanced": ex("A bit of everything.", "basic")}))
    (root / "presets" / "ghost.json").write_text(doc("presets", {"p": ex("not installed")}))
    bundle = load_bundle(root)
    import engine_console.api.routers.system as system
    monkeypatch.setattr(system, "learn_bundle", lambda: bundle)
    return bundle


def test_learn_serves_the_bundle_filtered_to_registered_engines(client: TestClient, env: Env, content: LearnBundle) -> None:
    body = client.get(f"{P}/learn").json()
    assert body["version"] == 1 and set(body["glossary"]) == {"kv_cache"} and set(body["metrics"]) == {"requests_running"}
    assert set(body["concepts"]) == {"state.ready"}
    assert set(body["params"]) == {"fake"} and set(body["presets"]) == {"fake"}          # `ghost` and `fakecpp` are not registered
    assert body["params"]["fake"]["max_model_len"]["summary"] == "The longest conversation."
    env.container.adapters.register(FakeMultiAdapter())
    body = client.get(f"{P}/learn").json()
    assert set(body["params"]) == {"fake", "fakecpp"} and "ghost" not in json.dumps(body)
    LearnBundle.model_validate(body)                                                       # the documented shape


def test_learn_with_no_packaged_content_is_an_empty_valid_bundle(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import engine_console.api.routers.system as system
    monkeypatch.setattr(system, "learn_bundle", lambda: LearnBundle())
    body = client.get(f"{P}/learn").json()
    assert body == LearnBundle().model_dump()


def test_params_carry_their_explain_and_a_level(client: TestClient, content: LearnBundle) -> None:
    by_key = {p["key"]: p for p in client.get(f"{P}/engines/fake/params").json()}
    assert by_key["max_model_len"]["explain"]["summary"] == "The longest conversation."
    assert by_key["tp"]["explain"]["level"] == "advanced"
    assert by_key["kv_dtype"]["explain"] is None                                          # no explanation yet: null, not invented
    assert all(p["level"] in ("basic", "advanced") for p in by_key.values())
    assert by_key["max_model_len"]["level"] == "basic" and by_key["trust_remote_code"]["level"] == "basic"


def test_a_level_the_adapter_set_is_kept_and_advanced_params_default_to_advanced(client: TestClient, env: Env, content: LearnBundle) -> None:
    fake = env.container.adapters.get("fake")
    catalog = fake.param_catalog()
    catalog[0] = catalog[0].model_copy(update={"level": "expert"})
    catalog[1] = catalog[1].model_copy(update={"advanced": True})
    fake.param_catalog = lambda: catalog  # type: ignore[method-assign]
    by_key = {p["key"]: p for p in client.get(f"{P}/engines/fake/params").json()}
    assert by_key["max_model_len"]["level"] == "expert" and by_key["tp"]["level"] == "advanced"


def test_the_catalogue_itself_is_not_mutated_by_serving_it(client: TestClient, env: Env, content: LearnBundle) -> None:
    client.get(f"{P}/engines/fake/params")
    assert all(p.explain is None and p.level is None for p in env.container.adapters.get("fake").param_catalog())


def test_unknown_engine_is_a_404_problem(client: TestClient) -> None:
    r = client.get(f"{P}/engines/nope/params")
    assert r.status_code == 404 and r.json()["code"] == "unknown_engine"


def test_engines_list_gains_formats_images_by_arch_multi_model_and_preset_info(client: TestClient, env: Env, content: LearnBundle) -> None:
    env.container.adapters.register(FakeMultiAdapter())
    eng = {e["id"]: e for e in client.get(f"{P}/engines").json()}
    fake, cpp = eng["fake"], eng["fakecpp"]
    assert fake["model_formats"] == ["hf"] and fake["images_by_arch"] == {} and fake["multi_model"] is False
    assert cpp["model_formats"] == ["gguf", "hf", "ollama", "plain"] and cpp["images_by_arch"] == {"aarch64": "fake/engine:1.0-arm64"}
    assert [p["name"] for p in fake["presets"]] == ["balanced", "long-context"]
    assert fake["presets_info"]["balanced"]["summary"] == "A bit of everything."          # from the learn content
    assert fake["default_image"] == "fake/engine:1.0" and fake["display_name"] == "Fake Engine"


def test_preset_info_falls_back_to_the_adapters_own_when_learn_has_none(client: TestClient, env: Env, content: LearnBundle) -> None:
    class WithInfo(FakeMultiAdapter):
        def preset_info(self) -> dict[str, Explain]:
            return {"balanced": Explain(summary="from the adapter")}

    env.container.adapters.register(WithInfo())
    eng = {e["id"]: e for e in client.get(f"{P}/engines").json()}
    assert eng["fakecpp"]["presets_info"] == {"balanced": Explain(summary="from the adapter").model_dump()}
    assert eng["fake"]["presets_info"]["balanced"]["summary"] == "A bit of everything."


def test_default_image_shows_the_settings_pin(client: TestClient) -> None:
    client.put(f"{P}/settings", json={"image_pins": {"fake": "fake/engine:2.0"}})
    assert {e["id"]: e for e in client.get(f"{P}/engines").json()}["fake"]["default_image"] == "fake/engine:2.0"
