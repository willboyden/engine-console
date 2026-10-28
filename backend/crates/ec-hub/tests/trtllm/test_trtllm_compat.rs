//! Port of backend/tests/trtllm/test_trtllm_compat.py: compatibility rules: every code has a positive and a negative
//! case, and every message carries a fix.

use std::collections::BTreeMap;

use crate::a2::{hardware, hw_with, model, python_add_codes, read_json, rust_add_codes, trt, with};
use ec_hub::adapters::base::{Compat, EngineAdapter, Hardware, ModelInfo, Obj};
use ec_hub::adapters::trtllm::{TrtllmAdapter, inferred_max_seq_len};
use serde_json::{Value, json};

fn p(v: Value) -> Obj {
    crate::a2::p(v)
}

fn codes(m: &ModelInfo, params: Value, hw: &Hardware) -> BTreeMap<String, Compat> {
    TrtllmAdapter::new()
        .compatibility(m, &p(params), hw)
        .into_iter()
        .map(|c| (c.code.clone(), c))
        .collect()
}

#[test]
fn test_clean_launch_has_no_findings_trtllm() {
    assert!(codes(&trt::dense(), json!({}), &trt::hw1()).is_empty());
}

#[test]
fn test_unsupported_architecture_blocks_trtllm() {
    let hw1 = trt::hw1();
    let new = model(
        json!({"repo_id": "example-org/newfam-30B", "architectures": ["Qwen3_5MoeForConditionalGeneration"]}),
    );
    let c = &codes(&new, json!({}), &hw1)["unsupported_architecture"];
    assert!(c.level == "block" && c.message.contains("vLLM or SGLang"));
    assert!(c.message.contains("Qwen3_5MoeForConditionalGeneration"));
    let ok = model(json!({"repo_id": "a/b", "architectures": ["Qwen3ForCausalLM"]}));
    assert!(!codes(&ok, json!({}), &hw1).contains_key("unsupported_architecture"));
    assert!(
        !codes(&model(json!({"repo_id": "a/b"})), json!({}), &hw1)
            .contains_key("unsupported_architecture")
    );
    // only the FIRST architecture counts (AutoModelForCausalLM.from_config reads architectures[0])
    let two = model(json!({"repo_id": "a/b", "architectures": ["WeirdModel", "LlamaForCausalLM"]}));
    assert!(codes(&two, json!({}), &hw1).contains_key("unsupported_architecture"));
}

#[test]
fn test_world_size_rules_trtllm() {
    let (hw, hw1, dense) = (trt::hw(), trt::hw1(), trt::dense());
    assert_eq!(
        codes(&dense, json!({"tensor_parallel_size": 4}), &hw)["world_exceeds_gpus"].level,
        "block"
    );
    assert!(
        codes(
            &dense,
            json!({"tensor_parallel_size": 2, "pipeline_parallel_size": 2}),
            &hw
        )
        .contains_key("world_exceeds_gpus")
    );
    assert!(
        !codes(&dense, json!({"tensor_parallel_size": 2}), &hw).contains_key("world_exceeds_gpus")
    );
    let unused = &codes(&dense, json!({}), &hw)["gpus_unused"];
    assert!(unused.level == "warn" && unused.message.contains("2 GPUs"));
    assert!(!codes(&dense, json!({"pipeline_parallel_size": 2}), &hw).contains_key("gpus_unused"));
    assert!(!codes(&dense, json!({}), &hw1).contains_key("gpus_unused"));
}

#[test]
fn test_tensor_parallel_over_pcie_prefers_pipeline_trtllm() {
    let (hw, dense) = (trt::hw(), trt::dense());
    let c = &codes(&dense, json!({"tensor_parallel_size": 2}), &hw)["tp_over_pcie"];
    assert!(
        c.level == "warn"
            && c.message.to_lowercase().contains("pipeline")
            && c.message.contains("RTX PRO 6000")
    );
    assert!(!codes(&dense, json!({"pipeline_parallel_size": 2}), &hw).contains_key("tp_over_pcie"));
}

#[test]
fn test_heads_must_divide_trtllm() {
    let hw8 = hardware(
        json!({"gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7], "gpu_uuids": ["g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7"],
        "gpu_names": ["x", "x", "x", "x", "x", "x", "x", "x"], "gpu_total_gib": [95.6, 95.6, 95.6, 95.6, 95.6, 95.6, 95.6, 95.6],
        "gpu_free_gib": [94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0, 94.0], "compute_capability": "12.0", "host_ram_gib": 512.0}),
    );
    let odd = with(&trt::dense(), json!({"num_attention_heads": 36}));
    assert_eq!(
        codes(&odd, json!({"tensor_parallel_size": 8}), &hw8)["tp_heads_indivisible"].level,
        "block"
    );
    assert!(
        !codes(&odd, json!({"tensor_parallel_size": 4}), &hw8).contains_key("tp_heads_indivisible")
    );
    assert!(
        !codes(
            &odd,
            json!({"tensor_parallel_size": 8, "enable_attention_dp": true}),
            &hw8
        )
        .contains_key("tp_heads_indivisible")
    );
}

#[test]
fn test_speculation_conflicts_trtllm() {
    let (hw, dense) = (trt::hw(), trt::dense());
    assert_eq!(
        codes(
            &dense,
            json!({"pipeline_parallel_size": 2, "spec_decoding_type": "MTP"}),
            &hw
        )["pp_with_speculation"]
            .level,
        "block"
    );
    assert!(
        codes(
            &dense,
            json!({"pipeline_parallel_size": 2, "speculative_config": {"decoding_type": "Eagle"}}),
            &hw
        )
        .contains_key("pp_with_speculation")
    );
    assert!(
        !codes(
            &dense,
            json!({"pipeline_parallel_size": 2, "spec_decoding_type": "AUTO"}),
            &hw
        )
        .contains_key("pp_with_speculation")
    );
    let clash = codes(
        &dense,
        json!({"sampler_type": "TRTLLMSampler", "spec_decoding_type": "MTP"}),
        &hw,
    );
    let c = &clash["trtllm_sampler_conflict"];
    assert!(c.level == "block" && c.message.contains("auto"));
    assert!(
        !codes(&dense, json!({"sampler_type": "TRTLLMSampler"}), &hw)
            .contains_key("trtllm_sampler_conflict")
    );
}

#[test]
fn test_moe_backend_rules_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    let c = &codes(&dense, json!({"moe_backend": "TRTLLM"}), &hw1)["moe_backend_trtllm_sm120"];
    assert!(c.level == "block" && c.message.contains("CUTLASS") && c.message.contains("SM120"));
    for backend in ["DEEPGEMM", "CUTEDSL", "WIDEEP", "TRITON"] {
        assert_eq!(
            codes(&dense, json!({"moe_backend": backend}), &hw1)["moe_backend_datacenter_only"]
                .level,
            "warn"
        );
    }
    let triton = codes(
        &dense,
        json!({"moe_backend": "TRITON", "tensor_parallel_size": 2, "moe_expert_parallel_size": 2}),
        &hw1,
    );
    let t = &triton["moe_backend_triton_ep"];
    assert!(t.level == "block" && t.message.contains("Hopper"));
    assert!(
        !codes(&dense, json!({"moe_backend": "TRITON"}), &hw1)
            .contains_key("moe_backend_triton_ep")
    );
    let hopper = hw_with(&hw1, json!({"compute_capability": "9.0"}));
    let found = codes(
        &dense,
        json!({"moe_backend": "TRITON", "moe_expert_parallel_size": 2}),
        &hopper,
    );
    assert!(
        !found.contains_key("moe_backend_triton_ep")
            && !found.contains_key("moe_backend_datacenter_only")
    );
    assert!(codes(&dense, json!({"moe_backend": "CUTLASS"}), &hw1).is_empty());
    assert!(
        !codes(&dense, json!({"moe_backend": "TRTLLM"}), &hopper)
            .contains_key("moe_backend_trtllm_sm120")
    );
    let gb10 = hw_with(&hw1, json!({"compute_capability": "12.1"})); // sm_121 also raises in the engine
    assert!(
        codes(&dense, json!({"moe_backend": "TRTLLM"}), &gb10)
            .contains_key("moe_backend_trtllm_sm120")
    );
}

#[test]
fn test_quantization_rules_trtllm() {
    let hw1 = trt::hw1();
    let block = model(
        json!({"repo_id": "a/b-FP8", "quantization": "fp8", "raw_config": {"quantization_config": {
        "quant_method": "fp8", "weight_block_size": [128, 128]}}}),
    );
    assert_eq!(
        codes(&block, json!({}), &hw1)["fp8_block_scaling_sm120"].level,
        "warn"
    );
    let per_tensor = model(
        json!({"repo_id": "a/b-FP8", "quantization": "fp8", "raw_config": {"quantization_config": {"quant_method": "fp8"}}}),
    );
    assert!(!codes(&per_tensor, json!({}), &hw1).contains_key("fp8_block_scaling_sm120"));
    let q = |name: &str| model(json!({"repo_id": "a/b", "quantization": name}));
    assert!(codes(&q("awq"), json!({}), &hw1).contains_key("awq_gptq_sm120"));
    assert!(codes(&q("gptq"), json!({}), &hw1).contains_key("awq_gptq_sm120"));
    assert!(
        !codes(
            &q("awq"),
            json!({}),
            &hw_with(&hw1, json!({"compute_capability": "9.0"}))
        )
        .contains_key("awq_gptq_sm120")
    );
    let plain = model(json!({"repo_id": "a/b"}));
    assert_eq!(
        codes(&plain, json!({"kv_cache_dtype": "nvfp4"}), &hw1)["nvfp4_kv_sm120"].level,
        "warn"
    );
    assert!(!codes(&plain, json!({"kv_cache_dtype": "fp8"}), &hw1).contains_key("nvfp4_kv_sm120"));
    assert!(codes(&q("bitsandbytes"), json!({}), &hw1).contains_key("quant_format_unlisted"));
    assert!(!codes(&q("nvfp4"), json!({}), &hw1).contains_key("quant_format_unlisted"));
    let mx = model(json!({"repo_id": "acme/moe-mxfp4", "quantization": "mxfp4"}));
    assert!(codes(&mx, json!({}), &hw1).contains_key("mxfp4_family_unlisted"));
    let oss = model(json!({"repo_id": "openai/gpt-oss-120b", "quantization": "mxfp4"}));
    assert!(!codes(&oss, json!({}), &hw1).contains_key("mxfp4_family_unlisted"));
    // NVFP4 itself is listed for sm_120: dense and MoE alike, so no finding
    let nv = model(json!({"repo_id": "a/b-NVFP4", "quantization": "nvfp4", "is_moe": true}));
    assert!(codes(&nv, json!({}), &hw1).is_empty());
}

#[test]
fn test_attention_backend_rules_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    assert_eq!(
        codes(&dense, json!({"attention_backend": "VANILLA"}), &hw1)["attention_backend_vanilla"]
            .level,
        "warn"
    );
    let c = &codes(&dense, json!({"attention_backend": "FLASHINFER"}), &hw1)["attention_backend_flashinfer_reuse"];
    assert!(c.message.contains("enable_block_reuse=false"));
    assert!(
        !codes(
            &dense,
            json!({"attention_backend": "FLASHINFER", "enable_block_reuse": false}),
            &hw1
        )
        .contains_key("attention_backend_flashinfer_reuse")
    );
    assert!(codes(&dense, json!({"attention_backend": "TRTLLM"}), &hw1).is_empty());
}

#[test]
fn test_max_seq_len_is_capped_by_the_engine_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    let c = &codes(&dense, json!({"max_seq_len": 262_144}), &hw1)["max_seq_len_capped"];
    assert!(c.level == "warn" && c.message.contains("131072") && c.message.contains("extend"));
    assert!(
        !codes(&dense, json!({"max_seq_len": 131_072}), &hw1).contains_key("max_seq_len_capped")
    );
    assert!(!codes(&dense, json!({}), &hw1).contains_key("max_seq_len_capped"));
}

#[test]
fn test_inferred_length_matches_the_engines_own_function_trtllm() {
    // fixtures/infer_max_seq_len_1.2.1.json is the output of the engine's own `infer_max_seq_len` per config
    let doc = read_json("trtllm", "infer_max_seq_len_1.2.1.json");
    let rows = doc["cases"].as_array().expect("cases");
    assert!(rows.len() >= 15);
    for row in rows {
        let m = model(json!({"repo_id": "a/b", "raw_config": row["config"]}));
        let got = inferred_max_seq_len(&m).expect("no raise");
        if row["config"]
            .as_object()
            .is_some_and(serde_json::Map::is_empty)
        {
            assert!(got.is_none()); // no config at all: the adapter claims nothing
            continue;
        }
        assert_eq!(
            got.map(|i| i.to_string()),
            Some(row["inferred"].to_string()),
            "{}",
            row["config"]
        );
    }
}

#[test]
fn test_inferred_length_drives_the_capped_warning_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    let linear = with(
        &dense,
        json!({"raw_config": {"max_position_embeddings": 131_072,
                                                    "rope_scaling": {"type": "linear", "factor": 2.0}}}),
    );
    assert!(
        !codes(&linear, json!({"max_seq_len": 262_144}), &hw1).contains_key("max_seq_len_capped")
    );
    assert!(
        codes(&linear, json!({"max_seq_len": 262_145}), &hw1).contains_key("max_seq_len_capped")
    );
    let yarn = with(
        &dense,
        json!({"raw_config": {"max_position_embeddings": 131_072,
                                                  "rope_scaling": {"rope_type": "yarn", "factor": 4.0}}}),
    );
    assert!(codes(&yarn, json!({"max_seq_len": 262_144}), &hw1).contains_key("max_seq_len_capped"));
    let bare = model(json!({"repo_id": "a/b", "raw_config": {"hidden_size": 64}})); // the engine falls back to 2048
    assert!(
        codes(&bare, json!({"max_seq_len": 4096}), &hw1)["max_seq_len_capped"]
            .message
            .contains("2048")
    );
    assert!(
        !codes(
            &model(json!({"repo_id": "a/b"})),
            json!({"max_seq_len": 1_000_000}),
            &hw1
        )
        .contains_key("max_seq_len_capped")
    );
}

#[test]
fn test_memory_rules_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    assert_eq!(
        codes(&dense, json!({"free_gpu_memory_fraction": 0.98}), &hw1)["kv_cache_fraction_high"]
            .level,
        "warn"
    );
    assert!(
        !codes(&dense, json!({"free_gpu_memory_fraction": 0.9}), &hw1)
            .contains_key("kv_cache_fraction_high")
    );
    let huge = with(
        &dense,
        json!({"weight_bytes": 140_i64 * 1024 * 1024 * 1024}),
    );
    let c = &codes(&huge, json!({}), &hw1)["weights_exceed_gpu"];
    assert!(
        c.level == "block"
            && c.message.contains("FP8")
            && c.message.contains("loads a checkpoint as it is")
    );
    let two = hardware(
        json!({"gpu_ids": [0, 1], "gpu_uuids": ["a", "b"], "gpu_names": ["x", "x"], "gpu_total_gib": [95.6, 95.6],
        "gpu_free_gib": [94.0, 94.0], "compute_capability": "12.0", "host_ram_gib": 246.0}),
    );
    assert!(
        !codes(&huge, json!({"tensor_parallel_size": 2}), &two).contains_key("weights_exceed_gpu")
    );
    let busy = hw_with(&hw1, json!({"gpu_free_gib": [30.0]}));
    let free = &codes(&dense, json!({}), &busy)["insufficient_free_vram"];
    assert!(
        free.level == "block" && free.message.contains("30.0 GiB") && free.message.contains("Stop")
    );
    assert!(!codes(&dense, json!({}), &hw1).contains_key("insufficient_free_vram"));
}

#[test]
fn test_parser_rules_trtllm() {
    let hw1 = trt::hw1();
    let qwen = model(
        json!({"repo_id": "Qwen/Qwen3-32B", "model_type": "qwen3", "pipeline_tag": "text-generation"}),
    );
    let missing = codes(&qwen, json!({}), &hw1);
    let tool = &missing["tool_parser_mismatch"].message;
    assert!(tool.contains("qwen3_coder") && tool.contains("qwen3"));
    assert!(
        missing["reasoning_parser_mismatch"]
            .message
            .contains("reasoning_parser=qwen3")
    );
    assert!(
        codes(
            &qwen,
            json!({"tool_parser": "qwen3_coder", "reasoning_parser": "qwen3"}),
            &hw1
        )
        .is_empty()
    );
    let wrong = codes(
        &qwen,
        json!({"tool_parser": "kimi_k2", "reasoning_parser": "deepseek-r1"}),
        &hw1,
    );
    assert!(
        wrong["tool_parser_mismatch"]
            .message
            .contains("does not match")
    );
    assert!(
        wrong["reasoning_parser_mismatch"]
            .message
            .contains("use qwen3")
    );
    let coder = model(
        json!({"repo_id": "Qwen/Qwen3-Coder-30B", "model_type": "qwen3_moe", "pipeline_tag": "text-generation"}),
    );
    let c = codes(&coder, json!({}), &hw1);
    assert!(
        c["tool_parser_mismatch"]
            .message
            .contains("tool_parser=qwen3_coder.")
            && !c.contains_key("reasoning_parser_mismatch")
    );
    let kimi =
        model(json!({"repo_id": "moonshotai/Kimi-K2-Instruct", "pipeline_tag": "text-generation"}));
    assert!(
        codes(&kimi, json!({}), &hw1)["tool_parser_mismatch"]
            .message
            .contains("tool_parser=kimi_k2")
    );
    let r1 = model(
        json!({"repo_id": "deepseek-ai/DeepSeek-R1-0528", "architectures": ["DeepseekV3ForCausalLM"]}),
    );
    let both = codes(&r1, json!({}), &hw1);
    assert!(both["tool_parser_mismatch"].message.contains("deepseek_v3"));
    assert!(
        both["reasoning_parser_mismatch"]
            .message
            .contains("deepseek-r1")
    );
    let llama =
        model(json!({"repo_id": "meta-llama/Llama-3.3-70B", "pipeline_tag": "text-generation"}));
    assert!(codes(&llama, json!({}), &hw1).is_empty()); // no family claim without certainty
    let embed =
        model(json!({"repo_id": "Qwen/Qwen3-Embedding-8B", "pipeline_tag": "feature-extraction"}));
    assert!(!codes(&embed, json!({}), &hw1).contains_key("tool_parser_mismatch"));
}

#[test]
fn test_checkpoint_read_ahead_needs_host_ram_trtllm() {
    // The weight loader reads files into host memory before loading (weight_loader.prefetch_files).
    let (hw1, dense) = (trt::hw1(), trt::dense());
    let big = with(
        &dense,
        json!({"weight_bytes": 120_i64 * 1024 * 1024 * 1024}),
    );
    let c = &codes(&big, json!({}), &hw1)["weights_prefetch_host_ram"]; // 120 GiB of 246 GiB
    assert!(
        c.level == "warn"
            && c.message.contains("120 GiB")
            && c.message.contains("246 GiB")
            && c.message.contains("90 %")
    );
    assert!(!codes(&dense, json!({}), &hw1).contains_key("weights_prefetch_host_ram")); // 36 GiB
    assert!(
        !codes(
            &with(&dense, json!({"weight_bytes": null})),
            json!({}),
            &hw1
        )
        .contains_key("weights_prefetch_host_ram")
    );
    let small_host = hw_with(&hw1, json!({"host_ram_gib": 64.0}));
    assert!(codes(&dense, json!({}), &small_host).contains_key("weights_prefetch_host_ram")); // 36 > 0.4 x 64
}

#[test]
fn test_misc_rules_trtllm() {
    let (hw1, dense) = (trt::hw1(), trt::dense());
    let c = codes(
        &dense,
        json!({"trust_remote_code": true, "enable_iter_perf_stats": false, "enable_cuda_graphs": false}),
        &hw1,
    );
    for k in ["trust_remote_code", "iter_stats_off", "cuda_graphs_off"] {
        assert!(c.contains_key(k), "{k}");
    }
    assert!(!codes(&dense, json!({}), &hw1).contains_key("iter_stats_off")); // the console default is on
    let arm = hw_with(&hw1, json!({"arch": "aarch64"}));
    assert!(codes(&dense, json!({}), &arm).contains_key("image_arch_unverified"));
    let other = json!({"image": "nvcr.io/nvidia/tensorrt-llm/release:1.2.1-arm64"});
    assert!(!codes(&dense, other, &arm).contains_key("image_arch_unverified"));
    let c = &codes(
        &dense,
        json!({"image": "nvcr.io/nvidia/tensorrt-llm/release:1.3.0"}),
        &hw1,
    )["image_version_unverified"];
    assert!(c.level == "warn" && c.message.contains("1.2.1") && c.message.contains("1.3.0"));
    let a = TrtllmAdapter::new();
    assert!(
        !codes(&dense, json!({"image": a.default_image()}), &hw1)
            .contains_key("image_version_unverified")
    );
    let vl = model(
        json!({"repo_id": "Qwen/Qwen2.5-VL-7B", "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                         "pipeline_tag": "image-text-to-text"}),
    );
    assert!(codes(&vl, json!({}), &hw1).contains_key("multimodal_block_reuse"));
    assert!(
        !codes(&vl, json!({"enable_block_reuse": false}), &hw1)
            .contains_key("multimodal_block_reuse")
    );
}

#[test]
fn test_every_message_is_actionable_and_every_code_is_found_by_the_learn_scan_trtllm() {
    // The learn gate finds `add("<level>", "<code>", ...)` literals; the Rust module emits exactly the Python codes.
    let py: BTreeMap<String, String> = python_add_codes("trtllm.py")
        .into_iter()
        .map(|(l, c)| (c, l))
        .collect();
    let rs: std::collections::BTreeSet<String> = rust_add_codes("trtllm.rs").into_iter().collect();
    assert!(py.len() >= 25);
    assert!(
        py.values()
            .all(|l| ["ok", "warn", "block"].contains(&l.as_str()))
    );
    assert!(
        py.keys()
            .all(|c| *c == c.to_lowercase() && !c.contains(' '))
    );
    assert_eq!(
        py.keys()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        rs
    );
}
