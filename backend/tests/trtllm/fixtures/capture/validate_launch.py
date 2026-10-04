"""Replay generated launches through the pinned image's own parsers, with no model and no weights:
 1. click parses the argv with the real `trtllm-serve serve` command object;
 2. the steps of `serve()` run: get_llm_args -> YAML merge (`update_llm_args_with_extra_dict`) -> TorchLlmArgs
    validation (strict pydantic models, unknown keys are errors) -> Mapping validation.
Input: /work/cases.json (written by make_cases.py), a list of {"name", "argv", "yaml"}. Prints one JSON list."""
import json
import sys

import yaml
from tensorrt_llm.commands import serve as serve_module
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict

NAMED = ["tokenizer", "custom_tokenizer", "backend", "max_beam_width", "max_batch_size", "max_num_tokens",
         "max_seq_len", "tensor_parallel_size", "pipeline_parallel_size", "context_parallel_size",
         "moe_expert_parallel_size", "moe_cluster_parallel_size", "gpus_per_node", "free_gpu_memory_fraction",
         "num_postprocess_workers", "trust_remote_code", "revision", "reasoning_parser",
         "fail_fast_on_attention_window_too_large", "otlp_traces_endpoint", "enable_chunked_prefill"]


def plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(x) for k, x in value.items()}
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    return str(value)


def resolved(args, mapping):
    kv = args.kv_cache_config
    return {
        "free_gpu_memory_fraction": kv.free_gpu_memory_fraction, "kv_dtype": kv.dtype,
        "enable_block_reuse": kv.enable_block_reuse, "enable_partial_reuse": kv.enable_partial_reuse,
        "max_tokens": kv.max_tokens, "max_gpu_total_bytes": kv.max_gpu_total_bytes,
        "host_cache_size": kv.host_cache_size, "tokens_per_block": kv.tokens_per_block,
        "max_attention_window": kv.max_attention_window, "mamba_ssm_cache_dtype": kv.mamba_ssm_cache_dtype,
        "cuda_graph_config": plain(args.cuda_graph_config), "attn_backend": args.attn_backend,
        "enable_iter_perf_stats": args.enable_iter_perf_stats, "max_stats_len": args.max_stats_len,
        "scheduler": plain(args.scheduler_config), "moe_backend": args.moe_config.backend,
        "speculative": type(args.speculative_config).__name__ if args.speculative_config else None,
        "torch_compile": plain(args.torch_compile_config), "max_batch_size": args.max_batch_size,
        "max_num_tokens": args.max_num_tokens, "tp": args.tensor_parallel_size, "pp": args.pipeline_parallel_size,
        "moe_ep": mapping.moe_ep_size, "moe_tp": mapping.moe_tp_size, "moe_cluster": mapping.moe_cluster_size,
        "stream_interval": args.stream_interval, "sampler_type": str(args.sampler_type),
        "guided": args.guided_decoding_backend, "allreduce": args.allreduce_strategy,
        "enable_attention_dp": args.enable_attention_dp, "autotuner": args.enable_autotuner,
        "orchestrator": args.orchestrator_type, "chunked_prefill": args.enable_chunked_prefill,
        "reasoning_parser": args.reasoning_parser, "tokenizer_mode": args.tokenizer_mode,
        "nvfp4_backends": args.nvfp4_gemm_config.allowed_backends, "print_iter_log": args.print_iter_log,
    }


def main():
    command = serve_module.main.commands["serve"]
    with open("/work/cases.json", encoding="utf-8") as fh:
        cases = json.load(fh)
    results = []
    for case in cases:
        result = {"name": case["name"]}
        results.append(result)
        try:
            ctx = command.make_context("serve", case["argv"][2:])
            params = dict(ctx.params)
            result["click"] = "ok"
        except Exception as e:
            result["click"] = f"{type(e).__name__}: {e}"
            continue
        try:
            llm_args, dropped = serve_module.get_llm_args(model=params["model"], **{k: params[k] for k in NAMED})
            result["dropped_by_get_llm_args"] = sorted(dropped)
            extra = yaml.safe_load(case["yaml"]) if case.get("yaml") else {}
            llm_args = update_llm_args_with_extra_dict(llm_args, extra)
            llm_args.pop("build_config", None)
            llm_args["skip_tokenizer_init"] = True      # validation only: do not load a tokenizer for a fake model
            args = TorchLlmArgs.from_kwargs(**llm_args)
            result["schema"] = "ok"
            result["resolved"] = resolved(args, args.parallel_config.to_mapping())
        except Exception as e:
            result["schema"] = f"{type(e).__name__}: {str(e)[:600]}"
    json.dump(results, sys.stdout, indent=1, sort_keys=True)


main()
