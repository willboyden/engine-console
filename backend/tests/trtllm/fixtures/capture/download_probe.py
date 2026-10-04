"""The engine's own offline model resolution (`llmapi.utils.download_hf_model`) against a read-only HF cache laid out
the way the console writes it. No model: the cache holds one config.json."""
import json
import os
import sys

from tensorrt_llm.llmapi.utils import download_hf_model

out = {"HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "HF_HOME": os.environ.get("HF_HOME")}
try:
    path = download_hf_model("example-org/tiny")
    out["resolved"] = str(path)
    out["files"] = sorted(os.listdir(path))
except Exception as e:
    out["error"] = f"{type(e).__name__}: {str(e)[:160]}"
try:
    download_hf_model("example-org/absent")
except Exception as e:
    out["absent_model_error"] = f"{type(e).__name__}: {str(e)[:120]}"
try:
    open("/hf/hub/probe", "w").close()
    out["cache_writable"] = True
except OSError as e:
    out["cache_writable"] = f"no ({type(e).__name__})"
json.dump(out, sys.stdout, indent=1, sort_keys=True)
