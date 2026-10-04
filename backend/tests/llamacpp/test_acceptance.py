"""The adapter's launch lines against the LIVE parser check of the pinned image (fixtures/parser_acceptance_b10156.json,
produced by live_check.py: no model, no GPU, no network, a model path that does not exist)."""
from __future__ import annotations

import json

from engine_console.adapters.llamacpp import LlamaCppAdapter
from engine_console.adapters.llamacpp_params import DEFAULT_IMAGE

from .acceptance_cases import NEGATIVE_CASES, cases
from .helpers import FIX

RERUN = "the launch the adapter builds changed since the live parser check: run `uv run python tests/llamacpp/live_check.py`"


def record() -> dict:  # type: ignore[type-arg]
    return json.loads((FIX / "parser_acceptance_b10156.json").read_text(encoding="utf-8"))


def test_the_recorded_run_used_the_pinned_image() -> None:
    assert record()["image"] == DEFAULT_IMAGE


def test_every_launch_line_the_adapter_builds_was_accepted_by_the_pinned_parser(adapter: LlamaCppAdapter) -> None:
    rec = record()
    now = cases(adapter)
    assert [(c["name"], c["argv"], c["env"], c["files"]) for c in now] == [
        (c["name"], c["argv"], c["env"], c["files"]) for c in rec["cases"]], RERUN
    refused = [c["name"] for c in rec["cases"] if not c["accepted"]]
    assert not refused, refused


def test_the_check_covers_the_whole_catalog(adapter: LlamaCppAdapter) -> None:
    covered = {c["name"].split("=", 1)[0] for c in record()["cases"] if "=" in c["name"]}
    keys = {p.key for p in adapter.param_catalog()}
    assert keys == covered, sorted(keys ^ covered)
    names = {c["name"] for c in record()["cases"]}
    for p in adapter.param_catalog():
        for choice in p.choices or []:
            assert f"{p.key}={choice!r}" in names, (p.key, choice)          # every enum choice reached the parser
    assert {f"preset:{n}" for n in adapter.presets()} <= names


def test_the_check_can_tell_a_rejection_from_an_acceptance() -> None:
    controls = record()["negative_controls"]
    assert [c["name"] for c in controls] == [name for name, _ in NEGATIVE_CASES]
    assert all(c["rejected"] for c in controls)
    lines = {c["name"]: c["error_line"] for c in controls}
    assert lines["unknown flag"] == "error: invalid argument: --no-such-flag"
    assert "unknown value for --fit: 'maybe'" in lines["bad enum --fit"]
    assert "Unsupported cache type: fp7" in lines["bad enum --cache-type-k"]
    assert lines["bad enum at verbosity 0"] == lines["bad enum --fit"]                    # errors are printed at verbosity 0 too


def test_the_one_silenced_case_is_marked(adapter: LlamaCppAdapter) -> None:
    silenced = [c for c in record()["cases"] if c.get("note")]
    assert [c["name"] for c in silenced] == ["log_verbosity=0"] and silenced[0]["accepted"]


def test_secrets_in_the_recorded_env_are_the_test_value_only() -> None:
    for c in record()["cases"]:
        assert set(c["env"]) <= {"LLAMA_API_KEY"}
        assert c["env"].get("LLAMA_API_KEY", "probe-key-not-a-secret") == "probe-key-not-a-secret"
