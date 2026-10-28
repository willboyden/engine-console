//! Model fixtures for the fit estimator (port of backend/tests/core/model_fixtures.py).
//!
//! APPROXIMATIONS, as in Python: close to each model family's public config, not byte-exact copies of Hub files.
//! Each builder returns the fixture; tests change fields on the returned value (Python's `**kw` overrides).

#![allow(dead_code, unreachable_pub, clippy::needless_pass_by_value)]

use ec_hub::adapters::base::{FloatMap, Hardware, ModelInfo};
use serde_json::{Value, json};

pub const GIB: f64 = 1_073_741_824.0;

fn obj(v: Value) -> serde_json::Map<String, Value> {
    v.as_object().cloned().unwrap_or_default()
}

/// Qwen3-32B-style dense GQA: 64 layers, 64 heads / 8 KV heads, head_dim 128 (~65.5 GB in bf16).
pub fn qwen3_32b_dense() -> ModelInfo {
    ModelInfo {
        repo_id: "Qwen/Qwen3-32B".into(),
        architectures: vec!["Qwen3ForCausalLM".into()],
        model_type: Some("qwen3".into()),
        num_params: Some(32_762_000_000),
        weight_bytes: Some(65_530_000_000),
        dtype: Some("bfloat16".into()),
        hidden_size: Some(5120),
        num_layers: Some(64),
        num_attention_heads: Some(64),
        num_kv_heads: Some(8),
        head_dim: Some(128),
        max_position_embeddings: Some(40960),
        raw_config: obj(
            json!({"model_type": "qwen3", "num_hidden_layers": 64, "use_sliding_window": false,
                               "sliding_window": null}),
        ),
        ..ModelInfo::default()
    }
}

/// Qwen3-30B-A3B-style MoE: 48 layers, 32 heads / 4 KV heads, 128 experts (~61 GB bf16).
pub fn qwen3_30b_a3b_moe() -> ModelInfo {
    ModelInfo {
        repo_id: "Qwen/Qwen3-30B-A3B".into(),
        architectures: vec!["Qwen3MoeForCausalLM".into()],
        model_type: Some("qwen3_moe".into()),
        num_params: Some(30_500_000_000),
        num_active_params: Some(3_300_000_000),
        is_moe: true,
        weight_bytes: Some(61_000_000_000),
        dtype: Some("bfloat16".into()),
        hidden_size: Some(2048),
        num_layers: Some(48),
        num_attention_heads: Some(32),
        num_kv_heads: Some(4),
        head_dim: Some(128),
        max_position_embeddings: Some(40960),
        raw_config: obj(json!({"model_type": "qwen3_moe", "num_experts": 128})),
        ..ModelInfo::default()
    }
}

/// Llama-3.3-70B-Instruct-style: 80 layers, 64 heads / 8 KV heads, head_dim 128 (~141 GB bf16).
pub fn llama33_70b() -> ModelInfo {
    ModelInfo {
        repo_id: "meta-llama/Llama-3.3-70B-Instruct".into(),
        architectures: vec!["LlamaForCausalLM".into()],
        model_type: Some("llama".into()),
        num_params: Some(70_553_000_000),
        weight_bytes: Some(141_107_000_000),
        dtype: Some("bfloat16".into()),
        hidden_size: Some(8192),
        num_layers: Some(80),
        num_attention_heads: Some(64),
        num_kv_heads: Some(8),
        head_dim: Some(128),
        max_position_embeddings: Some(131_072),
        gated: true,
        raw_config: obj(json!({"model_type": "llama"})),
        ..ModelInfo::default()
    }
}

/// gpt-oss-120b-style: 36 layers alternating sliding(128)/full, 64 heads / 8 KV heads, head_dim 64, MXFP4.
pub fn gpt_oss_120b() -> ModelInfo {
    let layer_types: Vec<&str> = (0..36)
        .map(|i| {
            if i % 2 == 0 {
                "sliding_attention"
            } else {
                "full_attention"
            }
        })
        .collect();
    ModelInfo {
        repo_id: "openai/gpt-oss-120b".into(),
        architectures: vec!["GptOssForCausalLM".into()],
        model_type: Some("gpt_oss".into()),
        num_params: Some(116_800_000_000),
        num_active_params: Some(5_100_000_000),
        is_moe: true,
        weight_bytes: Some(65_000_000_000),
        quantization: Some("mxfp4".into()),
        hidden_size: Some(2880),
        num_layers: Some(36),
        num_attention_heads: Some(64),
        num_kv_heads: Some(8),
        head_dim: Some(64),
        max_position_embeddings: Some(131_072),
        sliding_window: Some(128),
        raw_config: obj(
            json!({"model_type": "gpt_oss", "layer_types": layer_types, "num_local_experts": 128}),
        ),
        ..ModelInfo::default()
    }
}

/// DeepSeek-V2-Lite-style MLA + MoE: 27 layers, kv_lora_rank 512, qk_rope_head_dim 64 (~31 GB bf16).
pub fn deepseek_v2_lite_mla() -> ModelInfo {
    ModelInfo {
        repo_id: "deepseek-ai/DeepSeek-V2-Lite".into(),
        architectures: vec!["DeepseekV2ForCausalLM".into()],
        model_type: Some("deepseek_v2".into()),
        num_params: Some(15_700_000_000),
        is_moe: true,
        weight_bytes: Some(31_400_000_000),
        dtype: Some("bfloat16".into()),
        hidden_size: Some(2048),
        num_layers: Some(27),
        num_attention_heads: Some(16),
        num_kv_heads: Some(16),
        head_dim: None,
        max_position_embeddings: Some(163_840),
        kv_lora_rank: Some(512),
        raw_config: obj(
            json!({"model_type": "deepseek_v2", "kv_lora_rank": 512, "qk_rope_head_dim": 64,
                               "n_routed_experts": 64}),
        ),
        ..ModelInfo::default()
    }
}

/// `hw(n, total, free)`: `free` None = every card free; one value = the same on every card; a list = per card.
pub fn hw_with(n: usize, total: f64, free: Option<Vec<f64>>) -> Hardware {
    let frees = match free {
        Some(f) if f.len() == 1 && n != 1 => vec![f[0]; n],
        Some(f) => f,
        None => vec![total; n],
    };
    Hardware {
        gpu_ids: (0..i64::try_from(n).unwrap_or(0)).collect(),
        gpu_uuids: (0..n).map(|i| format!("GPU-{i}")).collect(),
        gpu_names: vec!["RTX PRO 6000 Blackwell".into(); n],
        gpu_total_gib: vec![total; n],
        gpu_free_gib: frees,
        compute_capability: "12.0".into(),
        host_ram_gib: 246.0,
        host_id: "local".into(),
        arch: "x86_64".into(),
        unified_memory: false,
    }
}

pub fn hw(n: usize) -> Hardware {
    hw_with(n, 96.0, None)
}

/// A memory model as Python's dict literal (`{"mem_fraction": 0.9, ...}`).
pub fn mem(pairs: &[(&str, f64)]) -> FloatMap {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

/// `{**a, **b}`
pub fn with(a: &FloatMap, b: &[(&str, f64)]) -> FloatMap {
    let mut out = a.clone();
    for (k, v) in b {
        match out.iter_mut().find(|(kk, _)| kk == k) {
            Some(slot) => slot.1 = *v,
            None => out.push(((*k).to_string(), *v)),
        }
    }
    out
}
