"""Run the engine's own `DecoderModelForCausalLM.infer_max_seq_len` on a table of model configs, so the adapter's
re-implementation can be compared with it (tests/trtllm/test_trtllm_compat.py). The method only reads `self.config`,
so a namespace stands in for the model: no weights, no GPU memory. Prints one JSON list."""
import json
import os
import sys
import types

os.environ["TLLM_LOG_LEVEL"] = "error"

from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM  # noqa: E402


def ns(d):
    return types.SimpleNamespace(**{k: (ns(v) if k == "text_config" else v) for k, v in d.items()})


CASES = [
    {"max_position_embeddings": 131072},
    {"max_position_embeddings": 131072, "rope_scaling": None},
    {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 2.0}},
    {"max_position_embeddings": 4096, "rope_scaling": {"rope_type": "dynamic", "factor": 2.5}},
    {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear"}},
    {"max_position_embeddings": 32768, "rope_scaling": {"rope_type": "yarn", "factor": 4.0}},
    {"max_position_embeddings": 32768, "rope_scaling": {"type": "su", "factor": 2.0}},
    {"max_position_embeddings": 32768, "rope_scaling": {"type": "longrope", "factor": 3.0}},
    {"max_position_embeddings": 8192, "rope_scaling": {"rope_type": "llama3", "factor": 8.0}},
    {"rope_scaling": {"type": "linear", "factor": 2.0}},
    {},
    {"text_config": {"max_position_embeddings": 40960}},
    {"max_position_embeddings": 4096, "text_config": {"max_position_embeddings": 40960}},
    {"text_config": {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 2.0}}},
    {"max_position_embeddings": 4096, "rope_scaling": {"type": "linear", "factor": 1.0}},
]
out = []
for case in CASES:
    model = types.SimpleNamespace(config=ns(case))
    out.append({"config": case, "inferred": DecoderModelForCausalLM.infer_max_seq_len(model)})
json.dump(out, sys.stdout, indent=1, sort_keys=True)
