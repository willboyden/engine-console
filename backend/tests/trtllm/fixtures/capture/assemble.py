"""Host side, standard library only: turn the raw outputs of the capture scripts into the committed fixtures.

    python assemble.py <raw dir> <fixtures dir>

The raw dir holds (names are the ones README.md uses): introspect.txt, iterstats.txt, logs.txt (stdout of
logs_and_prom.py), logs.err (its stderr), serve_help.txt and top_help.txt (stdout of `trtllm-serve serve --help` and
`trtllm-serve --help`). NVIDIA's entrypoint prints a licence banner before the real output, so every reader here
cuts the text before the first line that is exactly "{" (JSON) or before "Usage:" (help)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TRIM_CLASSES = ["TorchLlmArgs", "KvCacheConfig", "CudaGraphConfig", "MoeConfig", "AttentionDpConfig",
                "TorchCompileConfig", "SchedulerConfig", "DynamicBatchConfig", "Nvfp4GemmConfig",
                "MTPDecodingConfig", "EagleDecodingConfig", "NGramDecodingConfig", "DraftTargetDecodingConfig",
                "DecodingBaseConfig"]
SERVE_KEYS = ("name", "opts", "secondary_opts", "kind", "type", "required", "multiple", "is_flag", "default")


def json_after_banner(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index("\n{\n") + 1:])  # type: ignore[no-any-return]


def write_json(path: Path, doc: object) -> None:
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(raw: Path, out: Path) -> None:
    for src, dst in (("serve_help.txt", "serve_help_1.2.1.txt"), ("top_help.txt", "top_help_1.2.1.txt")):
        text = (raw / src).read_text(encoding="utf-8")
        (out / dst).write_text(text[text.index("Usage:"):].rstrip() + "\n", encoding="utf-8")

    full = json_after_banner(raw / "introspect.txt")
    classes = full["classes"]
    assert isinstance(classes, dict)
    serve = []
    for p in full["commands"]["serve"]:  # type: ignore[index]
        item = {k: p[k] for k in SERVE_KEYS if k in p}
        if "choices" in p:
            item["choices"] = p["choices"]
        serve.append(item)
    write_json(out / "llm_args_1.2.1.json", {
        "_provenance": ("Introspected inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1 (amd64, "
                        "sha256:33cd085b772947bd22b7273886539331420404e5d2a4a039945241945ff927b9): pydantic field "
                        "tables of the LLM API argument classes, the click options of `trtllm-serve serve`, the "
                        "parser registries and the PyTorch backend's MODEL_CLASS_MAPPING. Descriptions are left "
                        "out."),
        "tensorrt_llm_version": full["tensorrt_llm_version"],
        "classes": {name: {f: {k: v[k] for k in ("type", "default", "status")} for f, v in sorted(classes[name].items())}
                    for name in TRIM_CLASSES},
        "enums": full["enums"], "build_config_defaults": full["build_config_defaults"],
        "reasoning_parsers": full["reasoning_parsers"], "tool_parsers": full["tool_parsers"],
        "serve_params": serve, "architectures": full["architectures"]})

    stats = json_after_banner(raw / "iterstats.txt")
    write_json(out / "iter_stats_1.2.1.json", {
        "_provenance": ("Built inside the pinned image from IterationStats/KvCacheStats/... with the same "
                        "assignments py_executor._update_iter_stats makes, then serialized by the engine's own "
                        "BaseWorker._stats_serializer. Key names and layout are the engine's; every VALUE is an "
                        "arbitrary test number. Not captured from a running server."), **stats})

    logs = (raw / "logs.txt").read_text(encoding="utf-8")

    def section(name: str) -> str:
        m = re.search(rf"##SECTION {name}\n(.*?)(?=\n##SECTION |\Z)", logs, re.S)
        return m.group(1) if m else ""

    prom = [ln for ln in section("prometheus").splitlines() if "trtllm_" in ln]
    (out / "collector_1.2.1.prom").write_text("\n".join(prom) + "\n", encoding="utf-8")
    lines = [ln for ln in section("logger").splitlines() if ln.strip()]
    uvicorn = [ln for ln in (raw / "logs.err").read_text(encoding="utf-8").splitlines() if ln.startswith("INFO:")]
    (out / "startup_1.2.1.log").write_text("\n".join([*lines, *uvicorn, section("tqdm").strip("\n")]) + "\n",
                                           encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
