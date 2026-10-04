"""Check, with the engine's own functions, three behaviors the adapter relies on. No model, no weights.
 1. The YAML merge: kv_cache_config is deep-merged with the CLI's --free_gpu_memory_fraction; scheduler_config is
    replaced as a whole, so the CLI's dynamic batch settings vanish unless the YAML restates them.
 2. get_attention_backend: any name other than the four known ones silently becomes TRTLLM.
 3. load_chat_template: how an inline template and a bare word are treated."""
import json
import os
import sys

os.environ["TLLM_LOG_LEVEL"] = "error"

from tensorrt_llm._torch.attention_backend.utils import get_attention_backend  # noqa: E402
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE  # noqa: E402
from tensorrt_llm.commands import serve as serve_module  # noqa: E402
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict  # noqa: E402
from tensorrt_llm.serve.chat_utils import load_chat_template  # noqa: E402

out = {}
base, _ = serve_module.get_llm_args(model="example-org/m", free_gpu_memory_fraction=0.85)
merged = update_llm_args_with_extra_dict(dict(base), {"kv_cache_config": {"enable_block_reuse": False}})
kv = merged["kv_cache_config"]
out["kv_cache_merge"] = {"free_gpu_memory_fraction": kv.free_gpu_memory_fraction, "enable_block_reuse": kv.enable_block_reuse}
merged = update_llm_args_with_extra_dict(dict(base), {"scheduler_config": {"context_chunking_policy": "EQUAL_PROGRESS"}})
sched = merged["scheduler_config"]
out["scheduler_replaced"] = {"cli_value_before": type(base["scheduler_config"]).__name__,
                             "cli_default_had_dynamic_batch": base["scheduler_config"].dynamic_batch_config is not None,
                             "after_yaml": type(sched).__name__, "after_yaml_content": sched}
out["attention_backends"] = {n: get_attention_backend(n).__name__ for n in
                             ("TRTLLM", "FLASHINFER", "VANILLA", "FLASHINFER_STAR_ATTENTION", "NOPE", "trtllm")}
out["flashinfer_available"] = bool(IS_FLASHINFER_AVAILABLE)
templates = {}
for label, value in (("jinja braces", "{{ messages[0]['content'] }}"), ("newline only", "line one\nline two"),
                     ("bare word", "template.jinja"), ("absolute missing path", "/nonexistent/chat.jinja")):
    try:
        templates[label] = "literal: " + repr(load_chat_template(value))[:40]
    except Exception as e:
        templates[label] = f"{type(e).__name__}: {str(e)[:70]}"
out["chat_template"] = templates
json.dump(out, sys.stdout, indent=1, sort_keys=True)
