//! Port of backend/tests/llamacpp/test_compat.py: compatibility() of the llama.cpp adapter, every rule, its level and
//! the fix it names.

use std::collections::{BTreeMap, BTreeSet};

use crate::a2::{hw_with, llama, model, p, rust_add_codes};
use ec_hub::adapters::base::{Compat, EngineAdapter, Hardware, ModelInfo};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use serde_json::{Value, json};

/// every code the adapter can emit (the learn gate needs a `check.<code>` concept for each; the report lists them)
const EXPECTED_CODES: &[&str] = &[
    "blackwell_cuda_build",
    "fit_auto_adjust",
    "kv_quant_needs_flash_attn",
    "tensor_split_mismatch",
    "main_gpu_out_of_range",
    "split_mode_none_multi_gpu",
    "tp_over_pcie",
    "split_mode_tensor_experimental",
    "cpu_offload_slow",
    "context_beyond_native",
    "ctx_per_slot_small",
    "kv_exceeds_vram",
    "api_key_blocks_metrics",
    "jinja_disabled",
    "spec_type_needs_model",
    "moe_experts_on_cpu",
];

fn codes(m: &ModelInfo, params: Value, hw: &Hardware) -> BTreeMap<String, Compat> {
    let found = LlamaCppAdapter::new().compatibility(m, &p(params), hw);
    let unique: BTreeSet<&String> = found.iter().map(|c| &c.code).collect();
    assert_eq!(unique.len(), found.len(), "a code was emitted twice");
    found.into_iter().map(|c| (c.code.clone(), c)).collect()
}

fn bare() -> ModelInfo {
    model(json!({"repo_id": "x"}))
}

#[test]
fn test_the_emitted_codes_are_exactly_the_documented_ones_llamacpp() {
    // Python walks the module's AST for add(<level>, "<code>", ...); the Rust twin scans llamacpp.rs the same way.
    let emitted: BTreeSet<String> = rust_add_codes("llamacpp.rs").into_iter().collect();
    let want: BTreeSet<String> = EXPECTED_CODES.iter().map(|c| (*c).to_string()).collect();
    assert_eq!(emitted, want);
}

#[test]
fn test_blackwell_note_only_on_compute_capability_12_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(&dense, json!({}), &hw)["blackwell_cuda_build"];
    assert!(c.level == "ok" && c.message.contains("sm_120a") && c.message.contains("CUDA 12.8"));
    let older = hw_with(&hw, json!({"compute_capability": "8.9"}));
    assert!(!codes(&dense, json!({}), &older).contains_key("blackwell_cuda_build"));
}

#[test]
fn test_engine_fit_is_explained_when_something_is_left_unset_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(&dense, json!({}), &hw)["fit_auto_adjust"];
    assert!(
        c.level == "ok"
            && c.message.contains("ctx_size and n_gpu_layers")
            && c.message.contains("1024 MiB")
    );
    let only_layers = &codes(&dense, json!({"ctx_size": 8192}), &hw)["fit_auto_adjust"];
    assert!(
        only_layers.message.contains("n_gpu_layers") && !only_layers.message.contains("ctx_size")
    );
    assert!(
        !codes(
            &dense,
            json!({"ctx_size": 8192, "n_gpu_layers": "all"}),
            &hw
        )
        .contains_key("fit_auto_adjust")
    );
    assert!(
        codes(&dense, json!({"fit_target": 2048}), &hw)["fit_auto_adjust"]
            .message
            .contains("2048 MiB")
    );
}

#[test]
fn test_a_missing_fit_with_no_context_is_a_warning_with_the_cost_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let off = &codes(&dense, json!({"fit": "off"}), &hw)["fit_auto_adjust"];
    assert!(
        off.level == "warn"
            && off.message.contains("fit is off")
            && off.message.contains("10.0 GiB")
    );
    assert!(off.message.contains("ctx_size"));
    for (params, needle) in [
        (json!({"tensor_split": "3,1"}), "tensor_split"),
        (json!({"override_tensor": "exps=CPU"}), "override_tensor"),
        (json!({"split_mode": "row"}), "split_mode=row"),
        (json!({"split_mode": "tensor"}), "split_mode=tensor"),
    ] {
        let c = &codes(&dense, params, &hw)["fit_auto_adjust"];
        assert!(
            c.level == "warn"
                && c.message.contains(needle)
                && c.message.contains("turns the automatic fit off")
        );
    }
    assert!(
        !codes(&dense, json!({"fit": "off", "ctx_size": 8192}), &hw)
            .contains_key("fit_auto_adjust")
    );
    assert!(
        !codes(&bare(), json!({"fit": "off"}), &hw)["fit_auto_adjust"]
            .message
            .contains("GiB")
    );
}

#[test]
fn test_quantized_value_cache_needs_flash_attention_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(
        &dense,
        json!({"cache_type_v": "q8_0", "flash_attn": "off"}),
        &hw,
    )["kv_quant_needs_flash_attn"];
    assert!(
        c.level == "block"
            && c.message
                .contains("V cache quantization requires flash_attn")
    );
    assert!(c.message.contains("flash_attn to on"));
    for params in [
        json!({"cache_type_v": "q8_0", "flash_attn": "auto"}),
        json!({"cache_type_v": "q8_0"}),
        json!({"cache_type_k": "q4_0", "flash_attn": "off"}),
        json!({"cache_type_v": "bf16", "flash_attn": "off"}),
    ] {
        assert!(
            !codes(&dense, params.clone(), &hw).contains_key("kv_quant_needs_flash_attn"),
            "{params}"
        );
    }
}

#[test]
fn test_gpu_placement_rules_llamacpp() {
    let (hw, hw2, dense) = (llama::hw(), llama::hw2(), llama::dense());
    assert_eq!(
        codes(&dense, json!({"tensor_split": "3,1"}), &hw)["tensor_split_mismatch"].level,
        "warn"
    );
    assert!(
        !codes(&dense, json!({"tensor_split": "3,1"}), &hw2).contains_key("tensor_split_mismatch")
    );
    assert!(
        !codes(&dense, json!({"tensor_split": "1,0"}), &hw).contains_key("tensor_split_mismatch")
    );
    assert_eq!(
        codes(&dense, json!({"main_gpu": 1}), &hw)["main_gpu_out_of_range"].level,
        "block"
    );
    assert!(!codes(&dense, json!({"main_gpu": 1}), &hw2).contains_key("main_gpu_out_of_range"));
    assert!(!codes(&dense, json!({"main_gpu": 0}), &hw).contains_key("main_gpu_out_of_range"));
    let idle = &codes(&dense, json!({"split_mode": "none"}), &hw2)["split_mode_none_multi_gpu"];
    assert!(idle.level == "warn" && idle.message.contains("2 are selected"));
    assert!(
        !codes(&dense, json!({"split_mode": "none"}), &hw)
            .contains_key("split_mode_none_multi_gpu")
    );
}

#[test]
fn test_row_and_tensor_splits_warn_about_pcie_llamacpp() {
    let (hw, hw2, dense) = (llama::hw(), llama::hw2(), llama::dense());
    for mode in ["row", "tensor"] {
        let c = &codes(&dense, json!({"split_mode": mode}), &hw2)["tp_over_pcie"];
        assert!(c.level == "warn" && c.message.contains("NVLink") && c.message.contains("PCIe"));
        assert!(!codes(&dense, json!({"split_mode": mode}), &hw).contains_key("tp_over_pcie"));
    }
    assert!(!codes(&dense, json!({"split_mode": "layer"}), &hw2).contains_key("tp_over_pcie"));
    assert!(!codes(&dense, json!({}), &hw2).contains_key("tp_over_pcie"));
}

#[test]
fn test_tensor_split_mode_is_experimental_and_needs_flash_attention_llamacpp() {
    let (hw2, dense) = (llama::hw2(), llama::dense());
    let c = &codes(&dense, json!({"split_mode": "tensor"}), &hw2)["split_mode_tensor_experimental"];
    assert!(c.level == "warn" && c.message.contains("EXPERIMENTAL"));
    let blocked = &codes(
        &dense,
        json!({"split_mode": "tensor", "flash_attn": "off"}),
        &hw2,
    )["split_mode_tensor_experimental"];
    assert!(blocked.level == "block" && blocked.message.contains("flash_attn to on or auto"));
    assert!(
        !codes(&dense, json!({"split_mode": "row"}), &hw2)
            .contains_key("split_mode_tensor_experimental")
    );
}

#[test]
fn test_cpu_offload_is_called_slow_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let part = &codes(&dense, json!({"n_gpu_layers": 20}), &hw)["cpu_offload_slow"];
    assert!(part.level == "warn" && part.message.contains("20 of 65 layers"));
    for params in [
        json!({"n_gpu_layers": 0}),
        json!({"device": "none"}),
        json!({"n_gpu_layers": "0"}),
    ] {
        assert!(
            codes(&dense, params, &hw)["cpu_offload_slow"]
                .message
                .contains("whole model runs on the CPU")
        );
    }
    for params in [
        json!({}),
        json!({"n_gpu_layers": "all"}),
        json!({"n_gpu_layers": "auto"}),
        json!({"n_gpu_layers": 65}),
        json!({"n_gpu_layers": 999}),
    ] {
        assert!(
            !codes(&dense, params.clone(), &hw).contains_key("cpu_offload_slow"),
            "{params}"
        );
    }
    // layer count unknown
    assert!(!codes(&bare(), json!({"n_gpu_layers": 20}), &hw).contains_key("cpu_offload_slow"));
}

#[test]
fn test_context_past_the_trained_length_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(
        &dense,
        json!({"ctx_size": 131_072, "n_gpu_layers": "all"}),
        &hw,
    )["context_beyond_native"];
    assert!(c.level == "warn" && c.message.contains("40960"));
    assert!(c.message.contains("rope_scaling=yarn") && c.message.contains("rope_scale=4"));
    assert!(c.message.contains("yarn_orig_ctx=40960"));
    let scaled = &codes(
        &dense,
        json!({"ctx_size": 131_072, "rope_scaling": "yarn", "rope_scale": 4}),
        &hw,
    )["context_beyond_native"];
    assert!(
        scaled.message.contains("stretches") && !scaled.message.contains("rope_scaling=yarn with")
    );
    assert!(!codes(&dense, json!({"ctx_size": 40960}), &hw).contains_key("context_beyond_native"));
    assert!(!codes(&dense, json!({}), &hw).contains_key("context_beyond_native"));
    // slots divide the context unless they share one buffer
    assert!(
        !codes(&dense, json!({"ctx_size": 131_072, "parallel": 4}), &hw)
            .contains_key("context_beyond_native")
    );
    assert!(
        codes(
            &dense,
            json!({"ctx_size": 131_072, "parallel": 4, "kv_unified": true}),
            &hw
        )
        .contains_key("context_beyond_native")
    );
    assert!(
        codes(&dense, json!({"ctx_size": 131_072, "parallel": -1}), &hw)
            .contains_key("context_beyond_native")
    );
}

#[test]
fn test_context_per_slot_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(&dense, json!({"ctx_size": 8192, "parallel": 8}), &hw)["ctx_per_slot_small"];
    assert!(
        c.level == "warn" && c.message.contains("1024 tokens") && c.message.contains("kv_unified")
    );
    for params in [
        json!({"ctx_size": 8192, "parallel": 8, "kv_unified": true}),
        json!({"ctx_size": 65536, "parallel": 8}),
        json!({"ctx_size": 8192}),
        json!({"ctx_size": 8192, "parallel": 1}),
    ] {
        assert!(
            !codes(&dense, params.clone(), &hw).contains_key("ctx_per_slot_small"),
            "{params}"
        );
    }
}

#[test]
fn test_kv_cache_against_free_vram_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c = &codes(
        &dense,
        json!({"ctx_size": 400_000, "n_gpu_layers": "all"}),
        &hw,
    )["kv_exceeds_vram"];
    assert!(c.level == "warn" && c.message.contains("97.7 GiB") && c.message.contains("90.0 GiB"));
    assert!(c.message.contains("q8_0"));
    for params in [
        json!({"ctx_size": 400_000, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}),
        json!({"ctx_size": 400_000, "kv_offload": false}),
        json!({"ctx_size": 32768}),
        json!({}), // context unset: the engine's fit decides
    ] {
        assert!(
            !codes(&dense, params.clone(), &hw).contains_key("kv_exceeds_vram"),
            "{params}"
        );
    }
    let small = hw_with(&hw, json!({"gpu_free_gib": [10.0]}));
    assert!(codes(&dense, json!({"ctx_size": 65536}), &small).contains_key("kv_exceeds_vram")); // 16 GiB into 10 free
}

#[test]
fn test_kv_check_uses_the_main_gpu_when_not_splitting_llamacpp() {
    let (hw2, dense) = (llama::hw2(), llama::dense());
    // 97.7 GiB into 130 GiB free over two GPUs
    assert!(!codes(&dense, json!({"ctx_size": 400_000}), &hw2).contains_key("kv_exceeds_vram"));
    let one = codes(
        &dense,
        json!({"ctx_size": 400_000, "split_mode": "none"}),
        &hw2,
    );
    assert!(
        one.contains_key("kv_exceeds_vram") && one["kv_exceeds_vram"].message.contains("90.0 GiB")
    );
}

#[test]
fn test_api_key_warns_that_metrics_are_lost_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let c =
        &codes(&dense, json!({"api_key": "distinctive-key-value"}), &hw)["api_key_blocks_metrics"];
    assert!(c.level == "warn" && c.message.contains("401") && c.message.contains("/metrics"));
    assert!(!c.message.contains("distinctive-key-value")); // the value itself never appears
    assert!(!codes(&dense, json!({}), &hw).contains_key("api_key_blocks_metrics"));
}

#[test]
fn test_jinja_disabled_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    assert_eq!(
        codes(&dense, json!({"jinja": false}), &hw)["jinja_disabled"].level,
        "warn"
    );
    assert!(!codes(&dense, json!({"jinja": true}), &hw).contains_key("jinja_disabled"));
    assert!(!codes(&dense, json!({}), &hw).contains_key("jinja_disabled"));
}

#[test]
fn test_speculative_types_need_their_model_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    for t in ["draft-simple", "draft-eagle3", "draft-dflash"] {
        let c = &codes(&dense, json!({"spec_type": t}), &hw)["spec_type_needs_model"];
        assert!(c.level == "warn" && c.message.contains("spec_draft_model"));
        let with_model = json!({"spec_type": t, "spec_draft_model": "/models/d.gguf"});
        assert!(!codes(&dense, with_model, &hw).contains_key("spec_type_needs_model"));
    }
    assert!(
        codes(&dense, json!({"spec_type": "draft-mtp"}), &hw)["spec_type_needs_model"]
            .message
            .contains("MTP layers")
    );
    for t in ["none", "ngram-mod", "ngram-simple"] {
        assert!(
            !codes(&dense, json!({"spec_type": t}), &hw).contains_key("spec_type_needs_model"),
            "{t}"
        );
    }
}

#[test]
fn test_moe_experts_on_cpu_llamacpp() {
    let (hw, dense, moe) = (llama::hw(), llama::dense(), llama::moe());
    let c = &codes(&moe, json!({"cpu_moe": true}), &hw)["moe_experts_on_cpu"];
    assert!(
        c.level == "ok" && c.message.contains("host RAM") && c.message.contains("load_mode none")
    );
    let none =
        &codes(&moe, json!({"cpu_moe": true, "load_mode": "none"}), &hw)["moe_experts_on_cpu"];
    assert!(!none.message.contains("load_mode none"));
    for params in [
        json!({"n_cpu_moe": 20}),
        json!({"override_tensor": "exps=CPU"}),
        json!({"override_tensor": r"blk\.0\.=CPU"}),
    ] {
        assert_eq!(
            codes(&moe, params.clone(), &hw)["moe_experts_on_cpu"].level,
            "ok",
            "{params}"
        );
    }
    let wrong = &codes(&dense, json!({"cpu_moe": true}), &hw)["moe_experts_on_cpu"];
    assert!(wrong.level == "warn" && wrong.message.contains("not one"));
    assert!(!codes(&moe, json!({}), &hw).contains_key("moe_experts_on_cpu"));
    assert!(!codes(&moe, json!({"n_cpu_moe": 0}), &hw).contains_key("moe_experts_on_cpu"));
    assert!(
        !codes(&moe, json!({"override_tensor": "blk.0=CUDA0"}), &hw)
            .contains_key("moe_experts_on_cpu")
    );
}

#[test]
fn test_a_clean_launch_has_only_notes_llamacpp() {
    let (hw, dense) = (llama::hw(), llama::dense());
    let found = LlamaCppAdapter::new().compatibility(
        &dense,
        &p(json!({"ctx_size": 16384, "n_gpu_layers": "all", "flash_attn": "on"})),
        &hw,
    );
    assert!(
        found
            .iter()
            .map(|c| c.code.as_str())
            .eq(["blackwell_cuda_build"])
            && found[0].level == "ok"
    );
    assert!(
        found
            .iter()
            .all(|c| ["ok", "warn", "block"].contains(&c.level.as_str()) && !c.message.is_empty())
    );
}
