"""Write backend/crates/ec-hub/tests/data/wse_runspec_vectors.json: what the Python hub's pure lifecycle helpers answer for
many inputs, so the Rust port is pinned to them value for value (tests/test_wse_vectors.rs replays every vector):

* services/runspec.py: check_ref, display_of, hf_layout, hf_mounts, model_mounts, split_env, engine_labels,
  build_run_spec and validate_run_spec (the exact error texts);
* services/discovery.py: slug, display_name, clean_text, parse_generic_metrics;
* services/externals.py: script_instance_id, external_instance_id.

Deterministic (fixed seed). Run with the Python reference (gen/pyref.py):

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/wse_runspec_vectors.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from engine_console.adapters.base import LaunchSpec
from engine_console.domain.errors import ProblemError
from engine_console.domain.refs import ModelRef
from engine_console.services import discovery, externals, runspec

OUT = Path(__file__).resolve().parents[2] / "crates" / "ec-hub" / "tests" / "data" / "wse_runspec_vectors.json"

REFS: list[dict[str, Any]] = [
    {"format": "hf", "repo_id": "Qwen/Qwen3-32B"}, {"format": "hf", "repo_id": "a--b/c"}, {"format": "hf"},
    {"format": "hf", "repo_id": "../x/y"}, {"format": "hf", "repo_id": "org/na.me_1-2"}, {"format": "hf", "repo_id": "org/"},
    {"format": "gguf", "root_id": "g", "rel": "a/b/model-Q4.gguf"}, {"format": "gguf", "root_id": "g", "rel": "/abs/m.gguf"},
    {"format": "gguf", "root_id": "g", "rel": "../m.gguf"}, {"format": "gguf", "root_id": "g", "rel": "a//m.gguf"},
    {"format": "gguf", "root_id": "g", "rel": "m.bin"}, {"format": "gguf", "rel": "m.gguf"},
    {"format": "gguf", "root_id": "a b", "rel": "m.gguf"}, {"format": "gguf", "root_id": "g"},
    {"format": "gguf", "root_id": "g", "rel": "a\\b.gguf"}, {"format": "gguf", "root_id": "g", "rel": "M.GGUF"},
    {"format": "gguf", "root_id": "g", "rel": "unsloth/m/Model-Q4_K_M-00001-of-00004.gguf"},
    {"format": "gguf", "root_id": "g", "rel": "models--a--b/snapshots/abc/m-Q8_0.gguf"},
    {"format": "gguf", "root_id": "x" * 33, "rel": "m.gguf"}, {"format": "gguf", "root_id": "g", "rel": "a/./m.gguf"},
    {"format": "plain", "root_id": "p", "rel": "org/m"}, {"format": "plain", "root_id": "p", "rel": ""},
    {"format": "plain", "root_id": "p"}, {"format": "plain", "root_id": "p", "rel": "a/../b"}, {"format": "plain", "rel": "a"},
    {"format": "plain", "root_id": "p", "rel": "org/my-model/"},
    {"format": "ollama", "name": "llama3:8b"}, {"format": "ollama", "name": "hf.co/org/repo:Q4_K_M"},
    {"format": "ollama"}, {"format": "ollama", "name": ""}, {"format": "ollama", "name": "llama3; rm -rf /"},
    {"format": "ollama", "name": "x" * 200}, {"format": "ollama", "name": "-x"},
]

SPEC: dict[str, Any] = {"name": "ec-x", "instance": "inst_1", "image": "i:1", "argv": [], "container_port": 8000,
                        "labels": {}, "env": {}, "secret_env": {}, "mounts": [], "volumes": [], "shm_size": "16g"}
BAD: list[dict[str, Any]] = [
    {}, {"name": "../x"}, {"name": "ec-A"}, {"name": "ec-" + "a" * 60}, {"instance": "a b"}, {"instance": ""},
    {"container_port": 0}, {"container_port": 65536}, {"container_port": True}, {"container_port": "8000"},
    {"container_port": 8000.0}, {"shm_size": "16 g"}, {"shm_size": "16G"}, {"shm_size": "100g"}, {"shm_size": "99999m"},
    {"shm_size": "512m"}, {"shm_size": "1048577k"}, {"env": {"1BAD": "x"}}, {"env": {"A": "x" * 4097}},
    {"env": {f"V{i}": "1" for i in range(129)}}, {"secret_env": {f"S{i}": "1" for i in range(17)}},
    {"labels": {"x": "y"}}, {"labels": {"engine-console.a": "v" * 257}},
    {"labels": {f"engine-console.l{i}": "v" for i in range(33)}},
    {"mounts": [{"root": "hf", "rel": "../x", "dest": "/m"}]}, {"mounts": [{"root": "hf", "rel": "a", "dest": "rel/path"}]},
    {"mounts": [{"root": "hf", "rel": "a", "dest": "/a/../b"}]}, {"mounts": [{"rel": "", "dest": "/m"}]},
    {"mounts": [{"root": "h'f", "rel": "a\\b", "dest": "/m"}]}, {"mounts": [{"root": "hf", "dest": "/m"}] * 17},
    {"volumes": [{"name": "evil", "dest": "/x"}]}, {"volumes": [{"name": "ec-ok", "dest": "x"}]},
    {"volumes": [{"name": "ec-ok", "dest": "/x"}] * 9}, {"argv": ["a\x00b"]}, {"argv": ["x"] * 513},
    {"argv": ["x" * 4097]}, {"files": {"relative": "x"}}, {"files": {f"/f{i}": "x" for i in range(5)}},
    {"files": {"/big": "x" * (128 * 1024 + 1)}}, {"files": {"/euro": "\u20ac" * 43691}}, {"files": {"/euro-ok": "\u20ac" * 43690}}, {"gpu_uuids": ["g"] * 17},
    {"ulimits": {"memlock": "-1"}}, {"ulimits": {"stack": "lots"}}, {"ulimits": {"nofile": "1:2:3"}},
    {"ulimits": {"stack": "-1", "nofile": "65536:65536"}}, {"name": "../x", "instance": "a b", "shm_size": "x"},
]


def check(ref: dict[str, Any]) -> dict[str, Any]:
    r = ModelRef.model_validate(ref)
    try:
        runspec.check_ref(r)
    except ProblemError as e:
        return {"ok": False, "status": e.status, "code": e.code, "detail": e.detail}
    out: dict[str, Any] = {"ok": True, "display": runspec.display_of(r)}
    if r.format != "hf":
        mounts, path = runspec.model_mounts(r, copy_rel=None, ollama_root=None)
        out["mounts"] = mounts
        out["model_path"] = path
    return out


def main() -> None:
    rng = random.Random(20261028)
    v: dict[str, Any] = {"refs": [{"ref": ref, **check(ref)} for ref in REFS]}
    v["hf_layout"] = [{"kind": k, "rel": rel, "out": runspec.hf_layout(k, rel)}
                      for k in (None, "hf-home", "hf-hub", "auto", "plain")
                      for rel in ("models--a--b", "hub/models--a--b", "", "x")]
    v["hf_mounts"] = [{"root": "hf", "kind": k, "rel": rel, "path": p, "out": runspec.hf_mounts("hf", k, rel, p)}
                      for k in (None, "hf-home", "hf-hub") for rel in ("models--a--b", "hub/x")
                      for p in ("/root/.cache/huggingface", "/root/.cache/huggingface/", "/hf//")]
    ollama = []
    for copy_rel in (None, "", "manifests/registry.ollama.ai/library/llama3/8b", "models/manifests/x", "blobs/x"):
        for root_path in ("/data/ollama", "/data/ollama/models", "/data/ollama/models/", "/models"):
            mounts, path = runspec.model_mounts(ModelRef(format="ollama", name="llama3:8b"), copy_rel=copy_rel,
                                                ollama_root=("ollama", root_path))
            ollama.append({"copy_rel": copy_rel, "root_path": root_path, "mounts": mounts, "model_path": path})
    v["ollama_mounts"] = ollama
    envs = [{"HF_HOME": "/hf", "VLLM_API_KEY": "s", "MY_TOKEN": "t", "PASSWORD_FILE": "p", "X": "1"},
            {"HF_HUB_OFFLINE": "0"}, {}, {"Secret_Sauce": "v", "Keyring": "k", "N": 5, "B": True, "F": 1.5}]
    v["split_env"] = [{"env": e, "offline": off, "out": list(runspec.split_env(e, offline=off))}
                      for e in envs for off in (True, False)]
    labels = []
    for ref in REFS[:12] + REFS[16:30]:
        r = ModelRef.model_validate(ref)
        for uuids in ([], ["GPU-a"], ["GPU-" + "u" * 40] * 8):
            labels.append({"ref": ref, "uuids": uuids, "out": runspec.engine_labels(
                instance="inst_1", engine="fake", display="d" * rng.choice((1, 256, 257)), ref=r, port=18000,
                name="n", gpu_ids=[0, 1][:len(uuids)], gpu_uuids=uuids)})
    v["labels"] = labels
    launch_cases = [{}, {"files": {"/etc/a.yaml": "x: 1\n"}}, {"ulimits": {"stack": "67108864"}},
                    {"files": {"/a": "1"}, "ulimits": {"nofile": "1:2"}, "shm_size": "32g", "ipc_host": False}]
    specs = []
    for lc in launch_cases:
        launch = LaunchSpec(image="img:1", argv=["--model", "m"], container_port=8000, health_path="/health",
                            metrics_path="/metrics", **lc)
        spec = runspec.build_run_spec(name="ec-t-1", instance="inst_1", image="img:1", launch=launch, gpu_uuids=["G0"],
                                      labels={"engine-console.role": "engine"}, env={"A": "1"}, secret_env={"K": "x"},
                                      mounts=[{"root": "hf", "rel": "hub", "dest": "/m", "ro": True}],
                                      volumes=[("ec-vllm-cache", "/root/.cache/vllm")], ipc_host=True)
        specs.append({"launch": lc, "spec": spec})
    v["run_specs"] = specs
    validations = []
    for bad in BAD:
        spec = {**SPEC, **bad}
        for mx in ((None, 64.0, 8.0, 0.5) if "shm_size" in bad or not bad else (None,)):
            validations.append({"spec": spec, "max_shm_gib": mx, "errs": runspec.validate_run_spec(spec, max_shm_gib=mx)})
    v["validate"] = validations
    alphabet = "aZ09 -_.:/!@é\x00\x1b\n\t~#Ü" + " "
    strings = ["", "My Engine!", "../../etc", "x" * 99, "a\x00b\x1b[0m c", "!!!", "  .-lead-.  ", "Ü-ber", "İstanbul"]
    strings += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 70))) for _ in range(150)]
    v["text"] = [{"s": s, "slug": discovery.slug(s), "display_name": discovery.display_name(s),
                  "clean_text": discovery.clean_text(s), "clean_text_5": discovery.clean_text(s, 5),
                  "ext_id": externals.external_instance_id("gpu-2", s), "scr_id": externals.script_instance_id("local", s)}
                 for s in strings]
    metrics_texts = ["llamacpp:requests_processing 3\nllamacpp:requests_deferred{a=\"b\"} 1\n",
                     "llamacpp:tokens_predicted_total 9.5e3\nllamacpp:prompt_tokens_total +Inf\n",
                     "llamacpp:requests_processing\t7\n", "llamacpp:requests_processing  8 123\n",
                     "llamacpp:requests_processing 1_000\nllamacpp:requests_processing 2\n", "# x\n\n bad 1\n"]
    v["generic_metrics"] = [{"engine": e, "text": t, "out": [[k, x] for k, x in discovery.parse_generic_metrics(e, t).items()
                                                            if x == x and abs(x) != float("inf")],
                             "nonfinite": [k for k, x in discovery.parse_generic_metrics(e, t).items()
                                           if not (x == x and abs(x) != float("inf"))]}
                            for e in ("llamacpp", "vllm") for t in metrics_texts]
    OUT.write_text(json.dumps(v, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: " + ", ".join(f"{k} {len(x)}" for k, x in v.items()))


if __name__ == "__main__":
    main()
