//! Port of backend/tests/vllm/test_parse.py, plus native property tests (the parsers never panic on any input, and the
//! output keys stay within the canonical set).

use std::sync::{Arc, Mutex};

use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::vllm::VllmAdapter;
use ec_hub::adapters::vllm_parse::{histogram_quantile, parse_samples};
use proptest::prelude::*;
use serde_json::{Value, json};

use crate::ws_a1_kit::{approx, fixture, get, keys};

fn log(line: &str) -> Value {
    VllmAdapter::new()
        .parse_startup_log(line)
        .map_or(Value::Null, Value::Object)
}

const CANONICAL: &[&str] = &[
    "requests_running",
    "requests_waiting",
    "kv_cache_usage_pct",
    "prefix_cache_hit_pct",
    "prompt_tokens_total",
    "generation_tokens_total",
    "ttft_p50_s",
    "ttft_p95_s",
    "itl_p50_s",
    "e2e_p50_s",
    "e2e_p95_s",
    "prompt_tps",
    "generation_tps",
    "preemptions_total",
    "spec_decode_accept_pct",
];

#[test]
fn test_prometheus_text_parsing_edge_cases_vllm() {
    let s = parse_samples(
        "# HELP x\nvllm:a{l=\"q\\\"x\"} 1.5\nbad line\nvllm:b 2 1712345\nvllm:c{} NaN\n",
    );
    assert_eq!(
        s[0],
        (
            "vllm:a".to_string(),
            vec![("l".to_string(), "q\"x".to_string())],
            1.5
        )
    );
    assert_eq!(s[1], ("vllm:b".to_string(), vec![], 2.0));
    assert!(s[2].2.is_nan());
}

#[test]
fn test_histogram_quantile_exact_vllm() {
    let b = [
        (0.1, 0.0),
        (0.2, 50.0),
        (0.5, 100.0),
        (f64::INFINITY, 100.0),
    ];
    assert!(approx(histogram_quantile(0.5, &b).unwrap(), 0.2));
    assert!(approx(histogram_quantile(0.75, &b).unwrap(), 0.35));
    assert_eq!(
        histogram_quantile(0.5, &[(1.0, 0.0), (f64::INFINITY, 0.0)]),
        None
    );
    // +Inf bucket -> last finite
    assert_eq!(
        histogram_quantile(0.99, &[(1.0, 1.0), (f64::INFINITY, 10.0)]),
        Some(1.0)
    );
}

#[test]
fn test_real_v0271_scrape_vllm() {
    let m = VllmAdapter::new().parse_metrics(&fixture("vllm", "metrics_v0.27.1.prom"));
    assert!(get(&m, "requests_running") == Some(0.0) && get(&m, "requests_waiting") == Some(0.0));
    assert_eq!(get(&m, "kv_cache_usage_pct"), Some(0.0));
    assert_eq!(get(&m, "prefix_cache_hit_pct"), Some(0.0));
    assert!(
        get(&m, "prompt_tokens_total") == Some(1106.0)
            && get(&m, "generation_tokens_total") == Some(7723.0)
    );
    assert_eq!(get(&m, "preemptions_total"), Some(0.0));
    assert!(approx(
        get(&m, "spec_decode_accept_pct").unwrap(),
        100.0 * 4512.0 / 6402.0
    ));
    let (t50, t95) = (
        get(&m, "ttft_p50_s").unwrap(),
        get(&m, "ttft_p95_s").unwrap(),
    );
    assert!(0.02 < t50 && t50 <= t95 && t95 < 1.0);
    let itl = get(&m, "itl_p50_s").unwrap();
    assert!(0.0 < itl && itl < 0.1);
    let (e50, e95) = (get(&m, "e2e_p50_s").unwrap(), get(&m, "e2e_p95_s").unwrap());
    assert!(0.0 < e50 && e50 <= e95);
    assert!(get(&m, "prompt_tps").is_none()); // first sample: no rate yet
    assert!(keys(&m).iter().all(|k| CANONICAL.contains(k)));
}

#[test]
fn test_rates_from_two_scrapes_vllm() {
    let t = Arc::new(Mutex::new(100.0_f64));
    let clock = t.clone();
    let a = VllmAdapter::with_clock(Arc::new(move || *clock.lock().unwrap()));
    let tpl = |p: i64, g: i64| {
        format!(
            "vllm:prompt_tokens_total{{model_name=\"m\"}} {p}\nvllm:generation_tokens_total{{model_name=\"m\"}} {g}\n"
        )
    };
    a.parse_metrics(&tpl(1000, 5000));
    *t.lock().unwrap() = 102.0;
    let m = a.parse_metrics(&tpl(1400, 5600));
    assert!(
        approx(get(&m, "prompt_tps").unwrap(), 200.0)
            && approx(get(&m, "generation_tps").unwrap(), 300.0)
    );
    *t.lock().unwrap() = 104.0;
    let m = a.parse_metrics(&tpl(10, 20)); // counter reset (restart): no negative rate
    assert!(get(&m, "prompt_tps").is_none() && get(&m, "generation_tps").is_none());
}

#[test]
fn test_legacy_names_and_multi_engine_vllm() {
    let text = [
        "vllm:gpu_cache_usage_perc{engine=\"0\"} 0.25",
        "vllm:gpu_cache_usage_perc{engine=\"1\"} 0.75",
        "vllm:num_requests_running{engine=\"0\"} 2",
        "vllm:num_requests_running{engine=\"1\"} 3",
        "vllm:gpu_prefix_cache_hits_total 30",
        "vllm:gpu_prefix_cache_queries_total 120",
        "vllm:time_per_output_token_seconds_bucket{le=\"0.05\"} 10",
        "vllm:time_per_output_token_seconds_bucket{le=\"+Inf\"} 10",
        "vllm:spec_decode_draft_acceptance_rate 0.6",
        "vllm:num_preemptions_total 4",
        "vllm:prompt_tokens_total 5",
    ]
    .join("\n");
    let m = VllmAdapter::new().parse_metrics(&text);
    assert!(
        get(&m, "kv_cache_usage_pct") == Some(50.0) && get(&m, "requests_running") == Some(5.0)
    );
    assert_eq!(get(&m, "prefix_cache_hit_pct"), Some(25.0));
    assert!(approx(get(&m, "itl_p50_s").unwrap(), 0.025));
    assert!(approx(get(&m, "spec_decode_accept_pct").unwrap(), 60.0));
    assert_eq!(get(&m, "preemptions_total"), Some(4.0));
}

#[test]
fn test_missing_metrics_omitted_vllm() {
    let a = VllmAdapter::new();
    assert!(a.parse_metrics("").is_empty());
    assert!(a.parse_metrics("process_cpu_seconds_total 3\n").is_empty());
    let m = a.parse_metrics("vllm:num_requests_waiting{model_name=\"z\"} 7\n");
    assert_eq!(m, vec![("requests_waiting".to_string(), 7.0)]);
}

#[test]
fn test_startup_log_real_lines_vllm() {
    let parsed: Vec<Value> = fixture("vllm", "startup_v0.27.1.log")
        .lines()
        .map(log)
        .collect();
    for want in [
        json!({"phase": "loading_weights", "pct": 0.0}),
        json!({"phase": "loading_weights", "pct": 50.0}),
        json!({"phase": "loading_weights", "pct": 100.0, "weights_gib": 22.13, "load_seconds": 3.962537}),
        json!({"kv_cache_tokens": 534_714}),
        json!({"max_concurrency": 2.04, "max_model_len": 262_144}),
        json!({"phase": "ready", "pct": 100.0}),
    ] {
        assert!(parsed.contains(&want), "{want}");
    }
    assert!(
        parsed
            .iter()
            .any(|x| x.get("weights_gib") == Some(&json!(22.13)))
    );
    assert!(
        parsed
            .iter()
            .any(|x| x.get("phase") == Some(&json!("compiling")))
    );
    assert!(
        parsed
            .iter()
            .any(|x| x.get("graph_gib") == Some(&json!(0.59)))
    );
    assert!(
        parsed
            .iter()
            .any(|x| x.get("phase") == Some(&json!("warming_up"))
                && x["init_seconds"] == json!(15.73))
    );
}

#[test]
fn test_tqdm_takes_last_redraw_vllm() {
    let line = "(EngineCore pid=421) Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):   0%|   | 0/43 \
                [00:00<?, ?it/s]Capturing CUDA graphs (mixed prefill-decode, PIECEWISE):  47%|▏ | 20/43 [00:00<00:04]";
    assert_eq!(log(line), json!({"phase": "capturing_graphs", "pct": 47.0}));
    let real = log(&fixture("vllm", "tqdm_line.log")); // truncated real v0.27.1 line
    let pct = real["pct"].as_f64().unwrap();
    assert!(real["phase"] == "capturing_graphs" && 0.0 < pct && pct < 100.0);
    assert_eq!(
        log("Loading safetensors checkpoint shards: 100% Completed | 2/2 [00:01<00:00]"),
        json!({"phase": "loading_weights", "pct": 100.0})
    );
    assert_eq!(
        log("INFO Graph capturing finished in 3 secs, took 0.59 GiB")["graph_gib"],
        json!(0.59)
    );
    assert_eq!(
        log("[gpu_worker.py] Available KV cache memory: 60.11 GiB"),
        json!({"kv_cache_gib": 60.11})
    );
}

#[test]
fn test_stats_line_and_errors_vllm() {
    let s = log(
        "(APIServer pid=1) INFO 09-24 02:41:33 [loggers.py:310] Engine 000: Avg prompt throughput: 5.4 tokens/s, \
         Avg generation throughput: 110.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 2.8%, \
         Prefix cache hit rate: 0.0%",
    );
    assert_eq!(
        s,
        json!({"stats": {"prompt_tps": 5.4, "generation_tps": 110.6, "running": 1, "waiting": 0,
                         "kv_cache_pct": 2.8, "prefix_hit_pct": 0.0}})
    );
    assert_eq!(
        log("torch.OutOfMemoryError: CUDA out of memory. Tried")["error"],
        "cuda_oom"
    );
    assert_eq!(
        log("RuntimeError: No CUDA runtime is found")["error"],
        "no_gpu"
    );
    assert_eq!(
        log(
            "ValueError: ... (9.6 GiB KV cache is needed, which is larger than the available KV cache memory (2.45 GiB)."
        )["error"],
        "kv_cache_too_small"
    );
    assert_eq!(log("INFO random unrelated line"), Value::Null);
    assert_eq!(log(""), Value::Null);
}

/// Deferred and Preemptions only appear when non-zero (format string in vllm/v1/metrics/loggers.py, v0.27.1).
#[test]
fn test_stats_line_with_the_optional_segments_vllm() {
    let s = log(
        "(APIServer pid=1) INFO 10-04 09:00:00 [loggers.py:310] Engine 000: Avg prompt throughput: 0.0 tokens/s, \
         Avg generation throughput: 41.5 tokens/s, Running: 4 reqs, Waiting: 2 reqs, Deferred: 1 reqs, \
         Preemptions: 3, GPU KV cache usage: 91.2%, Prefix cache hit rate: 55.5%",
    );
    assert_eq!(
        s,
        json!({"stats": {"prompt_tps": 0.0, "generation_tps": 41.5, "running": 4, "waiting": 2, "deferred": 1,
                         "preemptions": 3, "kv_cache_pct": 91.2, "prefix_hit_pct": 55.5}})
    );
}

/// Wording copied from the pinned image's source; none of these was provoked on a live engine.
#[test]
fn test_failure_lines_read_from_the_image_source_vllm() {
    let e = log(
        "ValueError: max_num_seqs (1024) exceeds available Mamba cache blocks (316). Each decode sequence requires \
         one Mamba cache block, so CUDA graph capture cannot proceed. Please lower max_num_seqs to at most 316 or \
         increase gpu_memory_utilization.",
    );
    assert!(e["error"] == "max_num_seqs_too_high" && e["max_num_seqs_limit"] == 316);
    let e = log(
        "ValueError: Selected backend FLASH_ATTN is not valid for this configuration. Reason: \
         ['kv_cache_dtype not supported']",
    );
    assert!(e["error"] == "attention_backend_invalid" && e["backend"] == "FLASH_ATTN");
    let e = log(
        "ValueError: To serve at least one request with the model's max seq len (262144), (9.14 GiB KV cache is \
         needed, which is larger than the available KV cache memory (2.45 GiB). Based on the available memory, the \
         estimated maximum model length is 70000. Try increasing `gpu_memory_utilization` or decreasing \
         `max_model_len`",
    );
    assert!(e["error"] == "kv_cache_too_small" && e["suggested_max_model_len"] == 70000);
    let e = log(
        "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`",
    );
    assert_eq!(e["error"], "kv_cache_too_small");
    let e = log(
        "ValueError: Free memory on device cuda:0 (35.0/95.6 GiB) on startup is less than desired GPU memory \
         utilization (0.9, 86.1 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.",
    );
    assert_eq!(e["error"], "insufficient_free_vram");
    let e = log(
        "RuntimeError: Failed to infer device type, please set the environment variable `VLLM_LOGGING_LEVEL=DEBUG`",
    );
    assert_eq!(e["error"], "no_gpu");
}

/// The v2 model runner drops the word "memory" from the loading line (v1/worker/gpu/model_runner.py).
#[test]
fn test_new_model_runner_loading_line_vllm() {
    assert_eq!(
        log("INFO Model loading took 22.13 GiB and 3.962537 seconds"),
        json!({"phase": "loading_weights", "pct": 100.0, "weights_gib": 22.13, "load_seconds": 3.962537})
    );
}

/// Two scrapes of the same cumulative histogram give the same quantile: the adapter does not diff them.
#[test]
fn test_histogram_buckets_are_lifetime_totals_vllm() {
    let a = VllmAdapter::new();
    let text = "vllm:time_to_first_token_seconds_bucket{le=\"0.1\",model_name=\"m\"} 10\n\
                vllm:time_to_first_token_seconds_bucket{le=\"0.5\",model_name=\"m\"} 90\n\
                vllm:time_to_first_token_seconds_bucket{le=\"+Inf\",model_name=\"m\"} 100\n";
    assert_eq!(
        get(&a.parse_metrics(text), "ttft_p50_s"),
        get(&a.parse_metrics(text), "ttft_p50_s")
    );
}

// ---- Rust-native property tests ----
proptest! {
    #[test]
    fn prop_vllm_parsers_never_panic_and_keep_canonical_keys(
        text in "(\\PC|[ \\n\\r\\t{}=\",#_:.0-9a-z+-]){0,400}",
        names in proptest::collection::vec(prop::sample::select(vec![
            "vllm:num_requests_running", "vllm:prompt_tokens_total", "vllm:time_to_first_token_seconds_bucket",
            "vllm:kv_cache_usage_perc", "vllm:spec_decode_num_draft_tokens_total", "vllm:x"]), 0..6),
        vals in proptest::collection::vec("[-+0-9.eNaInf_]{1,8}", 0..6),
    ) {
        let a = VllmAdapter::new();
        let mut t = text;
        for (n, v) in names.iter().zip(&vals) {
            t.push_str(&format!("\n{n}{{le=\"{v}\"}} {v}"));
        }
        let m = a.parse_metrics(&t);
        prop_assert!(m.iter().all(|(k, _)| CANONICAL.contains(&k.as_str())));
        prop_assert!(m.iter().all(|(_, v)| !v.is_nan()));
        for line in t.lines() {
            let _ = a.parse_startup_log(line);
        }
        let _ = a.parse_startup_log(&t);
    }
}
