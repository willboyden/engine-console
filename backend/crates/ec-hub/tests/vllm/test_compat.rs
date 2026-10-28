//! Port of backend/tests/vllm/test_compat.py.

use std::collections::BTreeMap;

use ec_hub::adapters::base::{Compat, EngineAdapter, Hardware, ModelInfo};
use ec_hub::adapters::vllm::VllmAdapter;
use regex::RegexBuilder;
use serde_json::{Value, json};

use crate::ws_a1_kit::{dense, hw_vllm, model, obj};

fn codes(a: &VllmAdapter, m: &ModelInfo, p: Value, hw: &Hardware) -> BTreeMap<String, Compat> {
    a.compatibility(m, &obj(p), hw)
        .into_iter()
        .map(|c| (c.code.clone(), c))
        .collect()
}

fn on(hw: &Hardware, cc: &str) -> Hardware {
    Hardware {
        compute_capability: cc.into(),
        ..hw.clone()
    }
}

#[test]
fn test_nvfp4_moe_warns_unless_backend_chosen_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let m = ModelInfo {
        quantization: Some("nvfp4".into()),
        is_moe: true,
        ..model("nvidia/x-30B-A3B-NVFP4")
    };
    let c = &codes(&a, &m, json!({}), &hw)["nvfp4_moe_sm120"];
    assert!(
        c.level == "warn"
            && c.message.contains("FP8")
            && c.message.contains("SGLang")
            && c.message.contains("flashinfer_b12x")
    );
    assert!(!codes(&a, &m, json!({"moe_backend": "cutlass"}), &hw).contains_key("nvfp4_moe_sm120"));
    // the premise is sm_120-specific
    assert!(!codes(&a, &m, json!({}), &on(&hw, "10.0")).contains_key("nvfp4_moe_sm120"));
}

#[test]
fn test_nvfp4_dense_is_clean_vllm() {
    let m = ModelInfo {
        quantization: Some("nvfp4".into()),
        ..model("unsloth/Qwen3.8-27B-NVFP4")
    };
    assert!(!codes(&VllmAdapter::new(), &m, json!({}), &hw_vllm()).contains_key("nvfp4_moe_sm120"));
}

#[test]
fn test_mxfp4_moe_blocks_except_gpt_oss_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let bad = ModelInfo {
        quantization: Some("mxfp4".into()),
        is_moe: true,
        ..model("acme/moe-mxfp4")
    };
    assert_eq!(
        codes(&a, &bad, json!({}), &hw)["mxfp4_moe_sm120"].level,
        "block"
    );
    assert!(!codes(&a, &bad, json!({}), &on(&hw, "10.0")).contains_key("mxfp4_moe_sm120"));
    let oss = ModelInfo {
        quantization: Some("mxfp4".into()),
        is_moe: true,
        ..model("openai/gpt-oss-120b")
    };
    let c = codes(&a, &oss, json!({}), &hw);
    assert!(!c.contains_key("mxfp4_moe_sm120") && c["gpt_oss_prefer_ollama"].level == "warn");
    assert!(c["gpt_oss_prefer_ollama"].message.contains("llama.cpp"));
}

#[test]
fn test_gated_vllm() {
    let m = ModelInfo {
        gated: true,
        ..model("google/gemma-4-31B")
    };
    let c = codes(&VllmAdapter::new(), &m, json!({}), &hw_vllm());
    assert!(c["gated_needs_hf_token"].message.contains("HF_TOKEN"));
}

#[test]
fn test_tp_exceeds_gpus_and_divisibility_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    assert_eq!(
        codes(&a, &d, json!({"tensor_parallel_size": 4}), &hw)["tp_exceeds_gpus"].level,
        "block"
    );
    let c = &codes(&a, &d, json!({"tensor_parallel_size": 2}), &hw)["tp_over_pcie"];
    assert!(c.level == "warn" && c.message.contains("NVLink"));
    let odd = ModelInfo {
        num_attention_heads: Some(36),
        ..d.clone()
    };
    let eight = Hardware {
        gpu_ids: (0..8).collect(),
        ..hw.clone()
    };
    assert!(
        codes(
            &a,
            &odd,
            json!({"tensor_parallel_size": 8, "data_parallel_size": 1}),
            &eight
        )
        .contains_key("tp_heads_indivisible")
    );
    assert!(
        codes(
            &a,
            &d,
            json!({"data_parallel_size": 2, "tensor_parallel_size": 2}),
            &hw
        )
        .contains_key("tp_exceeds_gpus")
    );
}

#[test]
fn test_fp8_kv_with_flash_attn_is_refused_on_sm120_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    let c = &codes(
        &a,
        &d,
        json!({"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"}),
        &hw,
    )["kv_fp8_flash_attn"];
    assert!(
        c.level == "block"
            && c.message.contains("FLASHINFER")
            && c.message.contains("FlashAttention 2")
    );
    assert!(
        !codes(
            &a,
            &d,
            json!({"kv_cache_dtype": "fp8", "attention_backend": "FLASHINFER"}),
            &hw
        )
        .contains_key("kv_fp8_flash_attn")
    );
    // FA3 on Hopper handles fp8 KV
    assert!(
        !codes(
            &a,
            &d,
            json!({"kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN"}),
            &on(&hw, "9.0")
        )
        .contains_key("kv_fp8_flash_attn")
    );
}

#[test]
fn test_kv_dtype_must_be_supported_by_the_chosen_backend_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    let c = codes(
        &a,
        &d,
        json!({"kv_cache_dtype": "fp8_per_token_head", "attention_backend": "FLASHINFER"}),
        &hw,
    );
    let m = &c["kv_dtype_backend_mismatch"];
    assert!(m.level == "block" && m.message.contains("fp8_e5m2"));
    assert!(
        !codes(
            &a,
            &d,
            json!({"kv_cache_dtype": "fp8_per_token_head", "attention_backend": "TRITON_ATTN"}),
            &hw
        )
        .contains_key("kv_dtype_backend_mismatch")
    );
    // without an explicit backend vLLM picks one that fits, so nothing is said
    assert!(
        !codes(&a, &d, json!({"kv_cache_dtype": "fp8_per_token_head"}), &hw)
            .contains_key("kv_dtype_backend_mismatch")
    );
}

#[test]
fn test_fp8_kv_accuracy_note_names_the_engine_warning_vllm() {
    let c = codes(
        &VllmAdapter::new(),
        &dense(),
        json!({"kv_cache_dtype": "fp8"}),
        &hw_vllm(),
    );
    assert!(c["kv_fp8_accuracy"].message.contains("uncalibrated"));
}

#[test]
fn test_mla_backends_checked_against_sm120_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    for b in [
        "CUTLASS_MLA",
        "FLASHMLA",
        "FLASHINFER_MLA",
        "FLASH_ATTN_MLA",
    ] {
        let c = &codes(&a, &d, json!({"attention_backend": b}), &hw)["mla_backend_arch"];
        assert!(c.level == "block" && c.message.contains("TRITON_MLA"));
        assert!(
            !codes(&a, &d, json!({"attention_backend": b}), &on(&hw, "10.0"))
                .contains_key("mla_backend_arch")
        );
    }
    assert!(
        !codes(&a, &d, json!({"attention_backend": "TRITON_MLA"}), &hw)
            .contains_key("mla_backend_arch")
    );
    assert!(
        !codes(
            &a,
            &d,
            json!({"attention_backend": "FLASHINFER_MLA_SPARSE_SM120"}),
            &hw
        )
        .contains_key("mla_backend_arch")
    );
}

#[test]
fn test_kernel_backends_for_other_hardware_are_blocked_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    for p in [
        json!({"moe_backend": "aiter"}),
        json!({"linear_backend": "xpu"}),
        json!({"moe_backend": "hpc"}),
    ] {
        assert_eq!(
            codes(&a, &d, p, &hw)["kernel_backend_not_for_gpu"].level,
            "block"
        );
    }
    assert!(
        !codes(&a, &d, json!({"moe_backend": "hpc"}), &on(&hw, "9.0"))
            .contains_key("kernel_backend_not_for_gpu")
    );
    assert!(
        !codes(&a, &d, json!({"moe_backend": "flashinfer_b12x"}), &hw)
            .contains_key("kernel_backend_not_for_gpu")
    );
}

#[test]
fn test_deep_gemm_needs_fp8_block_weights_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    assert_eq!(
        codes(&a, &d, json!({"moe_backend": "deep_gemm"}), &hw)["deep_gemm_fp8_only"].level,
        "warn"
    );
    let fp8 = ModelInfo {
        quantization: Some("fp8".into()),
        ..d.clone()
    };
    assert!(
        !codes(&a, &fp8, json!({"moe_backend": "deep_gemm"}), &hw)
            .contains_key("deep_gemm_fp8_only")
    );
    // DeepGEMM does run on 12.x
    assert!(
        !codes(&a, &d, json!({"moe_backend": "deep_gemm"}), &hw).contains_key("deep_gemm_sm120")
    );
}

#[test]
fn test_xxhash_prefix_hash_needs_a_package_the_image_lacks_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    for algo in ["xxhash", "xxhash_cbor"] {
        let c = &codes(&a, &d, json!({"prefix_caching_hash_algo": algo}), &hw)["prefix_hash_xxhash_missing"];
        assert!(c.level == "warn" && c.message.contains("sha256"));
    }
    assert!(
        !codes(
            &a,
            &d,
            json!({"prefix_caching_hash_algo": "sha256_cbor"}),
            &hw
        )
        .contains_key("prefix_hash_xxhash_missing")
    );
}

#[test]
fn test_tool_and_reasoning_parser_hints_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let m = ModelInfo {
        pipeline_tag: Some("text-generation".into()),
        ..model("Qwen/Qwen3.6-35B-A3B")
    };
    let c = codes(&a, &m, json!({}), &hw);
    assert!(
        c["tool_parser_missing"]
            .message
            .contains("tool_call_parser=qwen3_xml")
    );
    assert!(
        c["reasoning_parser_missing"]
            .message
            .contains("reasoning_parser=qwen3")
    );
    let c2 = codes(
        &a,
        &m,
        json!({"tool_call_parser": "qwen3_xml", "enable_auto_tool_choice": true, "reasoning_parser": "qwen3"}),
        &hw,
    );
    assert!(
        !c2.contains_key("tool_parser_missing") && !c2.contains_key("reasoning_parser_missing")
    );
    assert!(!c2.contains_key("tool_parser_dialect_mismatch"));
    assert!(
        codes(&a, &m, json!({"tool_call_parser": "qwen3_xml"}), &hw)
            .contains_key("tool_parser_inactive")
    );
    let coder = ModelInfo {
        pipeline_tag: Some("text-generation".into()),
        ..model("Qwen/Qwen3-Coder-30B")
    };
    assert!(!codes(&a, &coder, json!({}), &hw).contains_key("reasoning_parser_missing"));
}

#[test]
fn test_tool_parser_dialect_mismatch_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let tg = |r: &str| ModelInfo {
        pipeline_tag: Some("text-generation".into()),
        ..model(r)
    };
    let xml = tg("Qwen/Qwen3.8-27B");
    let bad = codes(
        &a,
        &xml,
        json!({"tool_call_parser": "hermes", "enable_auto_tool_choice": true}),
        &hw,
    );
    let m = &bad["tool_parser_dialect_mismatch"];
    assert!(m.level == "warn" && m.message.contains("qwen3_xml"));
    for ok in ["qwen3_xml", "qwen3_coder"] {
        // one parser class in the pinned image
        assert!(
            !codes(
                &a,
                &xml,
                json!({"tool_call_parser": ok, "enable_auto_tool_choice": true}),
                &hw
            )
            .contains_key("tool_parser_dialect_mismatch")
        );
    }
    // Qwen3-8B is the older JSON dialect: the "3.x" regex must not swallow a size suffix such as "-8B"
    let old = tg("Qwen/Qwen3-8B");
    assert!(
        !codes(
            &a,
            &old,
            json!({"tool_call_parser": "hermes", "enable_auto_tool_choice": true}),
            &hw
        )
        .contains_key("tool_parser_dialect_mismatch")
    );
    assert!(
        codes(
            &a,
            &old,
            json!({"tool_call_parser": "qwen3_xml", "enable_auto_tool_choice": true}),
            &hw
        )
        .contains_key("tool_parser_dialect_mismatch")
    );
    let oss = tg("openai/gpt-oss-20b");
    assert!(
        !codes(
            &a,
            &oss,
            json!({"tool_call_parser": "openai", "enable_auto_tool_choice": true}),
            &hw
        )
        .contains_key("tool_parser_dialect_mismatch")
    );
}

#[test]
fn test_quant_override_on_prequantized_blocks_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let m = ModelInfo {
        quantization: Some("nvfp4".into()),
        ..model("unsloth/Qwen3.8-27B-NVFP4")
    };
    assert_eq!(
        codes(&a, &m, json!({"quantization": "fp8"}), &hw)["quant_override_prequantized"].level,
        "block"
    );
    assert!(!codes(&a, &m, json!({}), &hw).contains_key("quant_override_prequantized"));
}

#[test]
fn test_fp8_recommended_for_big_bf16_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let m = ModelInfo {
        weight_bytes: Some(90 * 1024 * 1024 * 1024),
        ..model("a/b")
    };
    assert!(codes(&a, &m, json!({}), &hw).contains_key("fp8_recommended"));
    assert!(!codes(&a, &m, json!({"quantization": "fp8"}), &hw).contains_key("fp8_recommended"));
}

#[test]
fn test_max_len_rules_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    assert_eq!(
        codes(&a, &d, json!({"max_model_len": 262_144}), &hw)["max_len_exceeds_model"].level,
        "block"
    );
    let ok = codes(
        &a,
        &d,
        json!({"max_model_len": 262_144, "hf_overrides": {"a": 1}}),
        &hw,
    );
    assert!(
        ok["yarn_short_context_quality"].level == "warn"
            && !ok.contains_key("max_len_exceeds_model")
    );
    assert!(
        !codes(
            &a,
            &d,
            json!({"max_model_len": 262_144, "allow_long_max_model_len": true}),
            &hw
        )
        .contains_key("max_len_exceeds_model")
    );
}

#[test]
fn test_free_vram_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    let busy = Hardware {
        gpu_free_gib: vec![9.7, 94.0],
        ..hw.clone()
    };
    let c = &codes(&a, &d, json!({"gpu_memory_utilization": 0.9}), &busy)["insufficient_free_vram"];
    assert!(c.level == "block" && c.message.contains("0.09"));
    assert!(
        !codes(&a, &d, json!({"gpu_memory_utilization": 0.05}), &busy)
            .contains_key("insufficient_free_vram")
    );
    assert!(
        !codes(
            &a,
            &d,
            json!({"kv_cache_memory_bytes": 7_000_000_000_i64}),
            &busy
        )
        .contains_key("insufficient_free_vram")
    );
    assert_eq!(
        codes(&a, &d, json!({"gpu_memory_utilization": 0.97}), &hw)["util_very_high"].level,
        "warn"
    );
}

/// Image source (config/compilation.py): with full CUDA graphs, max_num_seqs above the Mamba cache blocks stops
/// start-up, whatever the memory share.
#[test]
fn test_hybrid_models_need_an_explicit_max_num_seqs_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let hybrid = ModelInfo {
        architectures: vec!["NemotronHForCausalLM".into()],
        is_moe: true,
        quantization: Some("nvfp4".into()),
        ..model("example-org/hybrid-moe-30B-A3B-NVFP4")
    };
    for share in [0.3, 0.9] {
        let c = &codes(&a, &hybrid, json!({"gpu_memory_utilization": share}), &hw)["hybrid_max_num_seqs"];
        assert!(
            c.level == "warn"
                && c.message.contains("max_num_seqs")
                && c.message.contains("Mamba cache blocks")
        );
    }
    assert!(
        !codes(&a, &hybrid, json!({"max_num_seqs": 256}), &hw).contains_key("hybrid_max_num_seqs")
    );
    // no full graphs, no check
    assert!(
        !codes(&a, &hybrid, json!({"enforce_eager": true}), &hw)
            .contains_key("hybrid_max_num_seqs")
    );
    let dense = ModelInfo {
        architectures: vec!["LlamaForCausalLM".into()],
        ..model("a/dense")
    };
    assert!(
        !codes(&a, &dense, json!({"gpu_memory_utilization": 0.3}), &hw)
            .contains_key("hybrid_max_num_seqs")
    );
    let qwen_next = ModelInfo {
        architectures: vec!["Qwen3NextForCausalLM".into()],
        is_moe: true,
        ..model("Qwen/Qwen3-Next-80B-A3B")
    };
    assert!(codes(&a, &qwen_next, json!({}), &hw).contains_key("hybrid_max_num_seqs"));
}

#[test]
fn test_image_family_note_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let omni = model("nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4");
    let c = &codes(&a, &omni, json!({}), &hw)["image_pin_nemotron_omni"];
    assert!(c.level == "warn" && c.message.contains("0.20.0") && c.message.contains("0.27.1"));
    let laguna = model("poolside/Laguna-S-2.1-NVFP4");
    let c = codes(&a, &laguna, json!({}), &hw);
    assert!(!c.contains_key("image_too_old_laguna") && !c.contains_key("image_old_qwen38"));
}

#[test]
fn test_misc_rules_vllm() {
    let (a, hw, d) = (VllmAdapter::new(), hw_vllm(), dense());
    let c = codes(
        &a,
        &d,
        json!({"speculative_config": {"method": "dflash"}, "enforce_eager": true, "trust_remote_code": true,
               "enable_sleep_mode": true, "cpu_offload_gb": 4}),
        &hw,
    );
    for k in [
        "eager_slow",
        "trust_remote_code",
        "sleep_mode_dev_router",
        "cpu_offload_slow",
    ] {
        assert!(c.contains_key(k), "{k}");
    }
    // the rule had no evidence behind it and was dropped
    assert!(!c.contains_key("dflash_max_num_seqs"));
    assert!(
        c["sleep_mode_dev_router"]
            .message
            .contains("VLLM_SERVER_DEV_MODE")
    );
    assert!(
        codes(&a, &d, json!({"enable_expert_parallel": true}), &hw).contains_key("ep_dense_model")
    );
}

/// The repository is public: no site-specific wording in anything a user is shown.
#[test]
fn test_every_message_is_generic_text_vllm() {
    let (a, hw) = (VllmAdapter::new(), hw_vllm());
    let models = [
        dense(),
        ModelInfo {
            quantization: Some("nvfp4".into()),
            is_moe: true,
            gated: true,
            ..model("nvidia/x-NVFP4")
        },
        ModelInfo {
            quantization: Some("mxfp4".into()),
            is_moe: true,
            ..model("openai/gpt-oss-120b")
        },
        ModelInfo {
            architectures: vec!["NemotronHForCausalLM".into()],
            ..model("nvidia/Nemotron-3-Nano-Omni-30B")
        },
    ];
    let params = [
        json!({}),
        json!({"tensor_parallel_size": 2, "kv_cache_dtype": "fp8", "attention_backend": "FLASH_ATTN",
               "gpu_memory_utilization": 0.3, "moe_backend": "deep_gemm", "enable_sleep_mode": true,
               "speculative_config": {"method": "dflash"}, "tool_call_parser": "hermes", "max_model_len": 10_000_000}),
    ];
    let banned = RegexBuilder::new(r"\blab\b|this repo|lightning|/home/|@[a-z0-9-]+\.[a-z]{2,}")
        .case_insensitive(true)
        .build()
        .unwrap();
    let mut seen = 0;
    for m in &models {
        for p in &params {
            for c in a.compatibility(m, &obj(p.clone()), &hw) {
                seen += 1;
                assert!(!banned.is_match(&c.message), "{c:?}");
            }
        }
    }
    assert!(seen > 10);
}
