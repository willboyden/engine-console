"""Introspect the pinned image: the LLM API argument classes, the `trtllm-serve serve` click options, the parser
registries and the PyTorch backend's model registry. Prints one JSON document (see README.md for how the committed
fixture `llm_args_1.2.1.json` is derived from it). Read-only: no model is loaded."""
import enum
import json
import sys

import click
from pydantic_core import PydanticUndefined
from tensorrt_llm import version
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm.commands import serve as serve_module
from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi import llm_args as la
from tensorrt_llm.llmapi.reasoning_parser import ReasoningParserFactory
from tensorrt_llm.serve.tool_parser import ToolParserFactory

CLASSES = ["TorchLlmArgs", "KvCacheConfig", "CudaGraphConfig", "MoeConfig", "AttentionDpConfig",
           "TorchCompileConfig", "SchedulerConfig", "DynamicBatchConfig", "Nvfp4GemmConfig",
           "GuidedDecodingConfig", "MTPDecodingConfig", "EagleDecodingConfig", "NGramDecodingConfig",
           "DraftTargetDecodingConfig", "DecodingBaseConfig", "PeftCacheConfig", "CacheTransceiverConfig",
           "KvCacheConnectorConfig", "MoeLoadBalancerConfig"]


def plain(value):
    if isinstance(value, enum.Enum):
        return value.value if isinstance(value.value, str) else value.name
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(x) for k, x in value.items()}
    return repr(value)


def fields(cls):
    out = {}
    for name, f in cls.model_fields.items():
        extra = f.json_schema_extra if isinstance(f.json_schema_extra, dict) else {}
        default = f.default
        if default is PydanticUndefined:
            if f.default_factory is None:
                default = "<required>"
            else:
                made = f.default_factory()
                default = {"factory": plain(made.model_dump() if hasattr(made, "model_dump") else made)}
        out[name] = {"type": str(f.annotation).replace("typing.", ""), "default": plain(default),
                     "description": f.description, "status": extra.get("status"), "alias": f.alias,
                     "exclude": bool(f.exclude)}
    return out


doc = {"tensorrt_llm_version": version.__version__,
       "classes": {c: fields(getattr(la, c)) for c in CLASSES},
       "enums": {"CapacitySchedulerPolicy": [e.value for e in la.CapacitySchedulerPolicy],
                 "ContextChunkingPolicy": [e.value for e in la.ContextChunkingPolicy],
                 "SamplerType": [e.value for e in la.SamplerType],
                 "LoadFormat": [e.name for e in la.LoadFormat]},
       "build_config_defaults": {k: plain(BuildConfig.model_fields[k].default)
                                 for k in ("max_batch_size", "max_num_tokens", "max_seq_len", "max_beam_width")},
       "reasoning_parsers": sorted(ReasoningParserFactory.parsers),
       "tool_parsers": sorted(ToolParserFactory.parsers),
       "architectures": sorted(MODEL_CLASS_MAPPING)}
commands = {}
for name, cmd in serve_module.main.commands.items():
    params = []
    for p in cmd.params:
        info = {"name": p.name, "opts": list(p.opts), "secondary_opts": list(p.secondary_opts),
                "kind": type(p).__name__, "type": p.type.name, "required": p.required, "multiple": p.multiple,
                "is_flag": getattr(p, "is_flag", False), "default": plain(p.default)}
        if isinstance(p.type, click.Choice):
            info["choices"] = list(p.type.choices)
        params.append(info)
    commands[name] = params
doc["commands"] = commands
json.dump(doc, sys.stdout, indent=1, sort_keys=True)
