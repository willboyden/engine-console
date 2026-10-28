//! Port of backend/tests/trtllm/test_trtllm_parse.py: metrics and start-up log parsing against the fixtures from the
//! pinned image (tests/data/trtllm).

use crate::a2::{approx, fget, keys, read, read_json, same};
use ec_hub::adapters::base::EngineAdapter;
use ec_hub::adapters::trtllm::TrtllmAdapter;
use ec_hub::adapters::trtllm_parse::py::{PyVal, json_loads};
use ec_hub::adapters::trtllm_parse::{
    histogram_quantile, iteration_stats, parse_log_line, parse_metrics_text, parse_samples,
};
use serde_json::{Map, Value, json};

const CANONICAL: [&str; 15] = [
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

fn adapter() -> TrtllmAdapter {
    TrtllmAdapter::new()
}

fn stats() -> Value {
    read_json("trtllm", "iter_stats_1.2.1.json")
}

/// `stats_body(**kw)`: one object of the captured iteration-stats shape with a few fields replaced.
fn stats_body(kw: Value) -> Value {
    let mut base = stats()["plain"].clone();
    let Value::Object(kw) = kw else {
        panic!("kw must be an object")
    };
    for (k, v) in kw {
        base[&k] = v;
    }
    base
}

fn dumps(v: &Value) -> String {
    serde_json::to_string(v).expect("json")
}

fn log(line: &str) -> Option<Value> {
    adapter().parse_startup_log(line).map(Value::Object)
}

fn contains(parsed: &[Option<Value>], want: &Value) -> bool {
    parsed.iter().flatten().any(|p| same(p, want, false))
}

// --------------------------------------------------------------- JSON /metrics
#[test]
fn test_metrics_endpoint_json_from_the_engine_serializer_trtllm() {
    let m = adapter().parse_metrics(&dumps(&json!([stats()["plain"]])));
    assert!(fget(&m, "requests_running") == Some(3.0) && fget(&m, "requests_waiting") == Some(2.0));
    assert!(approx(
        fget(&m, "kv_cache_usage_pct").unwrap_or(f64::NAN),
        25.0
    )); // 250 of 1000 blocks
    assert!(approx(
        fget(&m, "prefix_cache_hit_pct").unwrap_or(f64::NAN),
        25.0
    )); // cacheHitRate 0.25 is a fraction
    assert!(approx(
        fget(&m, "generation_tps").unwrap_or(f64::NAN),
        2.0 / 0.0125
    )); // 2 generation requests, 12.5 ms
    assert_eq!(fget(&m, "itl_p50_s"), None); // this step also has a context request
    assert!(keys(&m).iter().all(|k| CANONICAL.contains(k)));
    assert!(fget(&m, "prompt_tps").is_none() && fget(&m, "prompt_tokens_total").is_none());
}

#[test]
fn test_speculation_and_decode_only_iterations_trtllm() {
    let doc = stats();
    let spec = &doc["with_spec_and_requests"];
    let m = adapter().parse_metrics(&dumps(&json!([spec])));
    assert!(approx(
        fget(&m, "spec_decode_accept_pct").unwrap_or(f64::NAN),
        100.0 * 5.0 / 8.0
    ));
    let mut decode = stats_body(json!({}));
    decode["inflightBatchingStats"]["numContextRequests"] = json!(0);
    decode["iterLatencyMS"] = json!(20.0);
    let with_latency = |l: f64| {
        let mut d = decode.clone();
        d["iterLatencyMS"] = json!(l);
        d
    };
    let m = adapter().parse_metrics(&dumps(&json!([
        decode,
        with_latency(30.0),
        with_latency(40.0)
    ])));
    assert!(approx(fget(&m, "itl_p50_s").unwrap_or(f64::NAN), 0.030)); // median step time of decode-only steps
    let mut spec_decode = decode.clone();
    spec_decode["specDecodingStats"] = spec["specDecodingStats"].clone();
    assert_eq!(
        fget(
            &adapter().parse_metrics(&dumps(&json!([spec_decode]))),
            "itl_p50_s"
        ),
        None
    );
}

#[test]
fn test_window_rate_uses_the_timestamps_trtllm() {
    let first = stats_body(json!({"timestamp": "10-04-2026 09:00:00.000000"}));
    let last =
        stats_body(json!({"timestamp": "10-04-2026 09:00:01.000000", "iterLatencyMS": 500.0}));
    let m = adapter().parse_metrics(&dumps(&json!([first, last])));
    // 4 generation tokens over (1.0 s + 0.5 s) of wall time
    assert!(approx(
        fget(&m, "generation_tps").unwrap_or(f64::NAN),
        4.0 / 1.5
    ));
    let broken = stats_body(json!({"timestamp": "not a time"}));
    assert!(
        fget(
            &adapter().parse_metrics(&dumps(&json!([broken, broken]))),
            "generation_tps"
        )
        .unwrap_or(0.0)
            > 0.0
    );
}

#[test]
fn test_cache_hit_rate_falls_back_to_block_counters_trtllm() {
    let mut body = stats_body(json!({}));
    let kv = body["kvCacheStats"].as_object_mut().expect("kv");
    kv.remove("cacheHitRate");
    kv.insert("reusedBlocks".into(), json!(4));
    kv.insert("missedBlocks".into(), json!(3121));
    let m = adapter().parse_metrics(&dumps(&json!([body])));
    assert!(approx(
        fget(&m, "prefix_cache_hit_pct").unwrap_or(f64::NAN),
        100.0 * 4.0 / 3125.0
    ));
}

#[test]
fn test_empty_and_odd_bodies_never_raise_trtllm() {
    for text in [
        "",
        "   ",
        "[]",
        "{}",
        "null",
        "42",
        "\"x\"",
        "not json",
        "[1, 2]",
        "# HELP x\n",
        "process_cpu_seconds_total 3\n",
    ] {
        assert!(adapter().parse_metrics(text).is_empty(), "{text}");
    }
    assert!(
        adapter()
            .parse_metrics(
                r#"[{"numActiveRequests": "many", "kvCacheStats": {"maxNumBlocks": 0}}]"#
            )
            .is_empty()
    );
    let nan = r#"[{"numActiveRequests": NaN, "numQueuedRequests": 1}]"#;
    assert_eq!(
        adapter().parse_metrics(nan),
        vec![("requests_waiting".to_string(), 1.0)]
    );
}

#[test]
fn test_clipped_body_keeps_the_complete_objects_trtllm() {
    let one = dumps(&stats_body(json!({})));
    let half: String = one.chars().take(one.chars().count() / 2).collect();
    let clipped = format!("[{one}, {one}, {half}"); // the node caps a scrape at 512 KiB
    assert_eq!(iteration_stats(&clipped).expect("parses").len(), 2);
    assert_eq!(
        fget(&adapter().parse_metrics(&clipped), "requests_running"),
        Some(3.0)
    );
    let parsed = json_loads(&one).expect("one");
    assert_eq!(
        iteration_stats(&format!("{one}{one}")).expect("parses"),
        vec![parsed.clone(), parsed.clone()]
    ); // concatenated
    assert_eq!(iteration_stats(&one).expect("parses"), vec![parsed]); // a bare object
}

#[test]
fn test_parser_keeps_no_state_between_scrapes_trtllm() {
    let body = dumps(&json!([stats_body(json!({}))]));
    let a = adapter();
    assert_eq!(a.parse_metrics(&body), a.parse_metrics(&body));
    assert_eq!(
        a.parse_metrics(&body),
        TrtllmAdapter::new().parse_metrics(&body)
    );
}

// --------------------------------------------------------------- Prometheus text (return_perf_metrics)
#[test]
fn test_prometheus_text_from_the_engine_collector_trtllm() {
    let m = adapter().parse_metrics(&read("trtllm", "collector_1.2.1.prom"));
    let g = |k: &str| fget(&m, k).unwrap_or(f64::NAN);
    // four synthetic requests were observed: e2e 0.6/1.2/3.4/9.8 s, ttft 0.04/0.08/0.21/0.9 s
    assert!(
        0.5 < g("e2e_p50_s")
            && g("e2e_p50_s") <= 1.5
            && 5.0 <= g("e2e_p95_s")
            && g("e2e_p95_s") <= 10.0
    );
    assert!(
        0.04 < g("ttft_p50_s")
            && g("ttft_p50_s") < 0.1
            && 0.5 < g("ttft_p95_s")
            && g("ttft_p95_s") <= 1.0
    );
    assert!(0.01 < g("itl_p50_s") && g("itl_p50_s") < 0.05);
    assert!(g("e2e_p50_s") <= g("e2e_p95_s") && g("ttft_p50_s") <= g("ttft_p95_s"));
    assert!(
        keys(&m).iter().all(|k| CANONICAL.contains(k)) && fget(&m, "requests_running").is_none()
    );
}

#[test]
fn test_histogram_helpers_trtllm() {
    let s = parse_samples(
        "# HELP x\ntrtllm_a{l=\"q\\\"x\"} 1.5\nbad line\ntrtllm_b 2 1712345\ntrtllm_c{} NaN\n",
    );
    assert_eq!(
        (s[0].0.as_str(), s[0].1.clone(), s[0].2),
        ("trtllm_a", vec![("l".to_string(), "q\"x".to_string())], 1.5)
    );
    assert_eq!(
        (s[1].0.as_str(), s[1].1.is_empty(), s[1].2),
        ("trtllm_b", true, 2.0)
    );
    assert!(s[2].2.is_nan());
    let b = [
        (0.1, 0.0),
        (0.2, 50.0),
        (0.5, 100.0),
        (f64::INFINITY, 100.0),
    ];
    assert!(approx(histogram_quantile(0.5, &b).unwrap_or(f64::NAN), 0.2));
    assert!(approx(
        histogram_quantile(0.75, &b).unwrap_or(f64::NAN),
        0.35
    ));
    assert_eq!(
        histogram_quantile(0.5, &[(1.0, 0.0), (f64::INFINITY, 0.0)]),
        None
    );
    assert_eq!(
        histogram_quantile(0.99, &[(1.0, 1.0), (f64::INFINITY, 10.0)]),
        Some(1.0)
    );
}

// --------------------------------------------------------------- start-up log
fn parsed_fixture() -> Vec<Option<Value>> {
    let text = read("trtllm", "startup_1.2.1.log").replace('\r', "\n");
    ec_hub::services::common::py_splitlines(&text)
        .into_iter()
        .map(log)
        .collect()
}

#[test]
fn test_startup_progress_from_fixture_lines_trtllm() {
    let parsed = parsed_fixture();
    for want in [
        json!({"phase": "loading_weights", "pct": 0.0}), // download / prefetch
        json!({"phase": "loading_weights", "weights_gib": 14.96}),
        json!({"phase": "loading_weights", "pct": 100.0, "load_seconds": 21.34}),
        json!({"phase": "loading_weights", "pct": 25.0}),
        json!({"phase": "loading_weights", "pct": 100.0}), // tqdm
        json!({"max_model_len": 131_072}),
        json!({"max_model_len": 131_072, "warning": "max_seq_len_capped", "requested": 262_144}),
        json!({"max_model_len": 98304, "warning": "max_seq_len_reduced_by_kv_cache"}),
        json!({"kv_cache_gib": 64.8}),
        json!({"phase": "warming_up", "step": "autotuner"}),
        json!({"phase": "capturing_graphs", "graphs_total": 34}),
        json!({"phase": "capturing_graphs", "batch_size": 128}),
        json!({"phase": "capturing_graphs", "batch_size": 1}),
        json!({"phase": "compiling"}),
        json!({"phase": "warming_up", "step": "warmup", "tokens": 8192}),
        json!({"phase": "ready", "pct": 100.0}), // uvicorn, real lines
    ] {
        assert!(contains(&parsed, &want), "{want}");
    }
}

#[test]
fn test_ready_is_only_the_server_startup_line_trtllm() {
    let text = read("trtllm", "startup_1.2.1.log");
    let ready: Vec<&str> = ec_hub::services::common::py_splitlines(&text)
        .into_iter()
        .filter(|l| log(l).is_some_and(|p| p["phase"] == "ready"))
        .collect();
    assert_eq!(ready, vec!["INFO:     Application startup complete."]);
    assert_eq!(log("INFO:     Waiting for application startup."), None);
    assert_eq!(
        log("INFO:     Uvicorn running on http://127.0.0.1:18999 (Press CTRL+C to quit)"),
        None
    );
}

#[test]
fn test_per_step_log_line_trtllm() {
    let text = read("trtllm", "startup_1.2.1.log");
    let line = ec_hub::services::common::py_splitlines(&text)
        .into_iter()
        .find(|x| x.contains("iter = 154"))
        .expect("line");
    let want = json!({"stats": {"iter": 154, "rank": 0, "host_step_ms": 9.8, "scheduled": 3, "num_ctx_requests": 0,
                                "num_ctx_tokens": 0, "num_generation_tokens": 3}});
    assert!(same(&log(line).expect("parsed"), &want, false));
}

#[test]
fn test_rank_prefix_does_not_matter_trtllm() {
    let got = log("[10/04/2026-18:20:34] [TRT-LLM] [RANK 1] [I] Use 7.48 GB for model weights.")
        .expect("parsed");
    assert!(same(
        &got,
        &json!({"phase": "loading_weights", "weights_gib": 7.48}),
        false
    ));
    let got = log("[TensorRT-LLM][INFO] [MemUsageChange] Allocated 64.80 GiB for max tokens in paged KV cache (528384).")
        .expect("parsed");
    assert!(same(
        &got,
        &json!({"kv_cache_gib": 64.8, "kv_cache_tokens": 528_384}),
        false
    ));
}

#[test]
fn test_tqdm_redraws_take_the_last_percentage_trtllm() {
    let line = "\rLoading safetensors weights in parallel:   0%|          | 0/4 [00:00<?, ?it/s]\rLoading safetensors weights in parallel:  75%|███████▌  | 3/4 [00:00<00:00,  6.4it/s]";
    assert!(same(
        &log(line).expect("parsed"),
        &json!({"phase": "loading_weights", "pct": 75.0}),
        false
    ));
    let got = log("Loading bin weights in parallel: 50%|").expect("parsed");
    assert!(same(
        &got,
        &json!({"phase": "loading_weights", "pct": 50.0}),
        false
    ));
}

#[test]
fn test_failures_are_named_trtllm() {
    // Failure texts copied from the image's sources (synthetic lines: no failing start was run).
    let err = |line: &str| log(line).map(|p| p["error"].clone());
    assert_eq!(
        err("RuntimeError: Executor creation failed due to insufficient GPU memory."),
        Some(json!("cuda_oom"))
    );
    let total = log("Total GPU memory (GiB): 94.99").expect("parsed");
    assert!(same(
        &total,
        &json!({"error": "cuda_oom", "total_gib": 94.99, "message": "Total GPU memory (GiB): 94.99"}),
        true
    ));
    let free =
        log("Free GPU memory before component creation attempt (GiB): 12.31").expect("parsed");
    assert_eq!(free["free_gib"], json!(12.31));
    assert_eq!(
        err("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"),
        Some(json!("cuda_oom"))
    );
    let arch = log("ValueError: Unknown architecture for AutoModelForCausalLM: FooForCausalLM")
        .expect("parsed");
    assert!(
        arch["error"] == "unsupported_architecture" && arch["architecture"] == "FooForCausalLM"
    );
    assert_eq!(
        err("NotImplementedError: TRTLLMGenFusedMoE does not support SM120 and above."),
        Some(json!("kernel_unsupported_on_gpu"))
    );
    assert_eq!(
        err("ValueError: No GPU devices found on the node"),
        Some(json!("no_gpu"))
    );
    assert_eq!(
        err("ImportError: libcuda.so.1: cannot open shared object file: No such file or directory"),
        Some(json!("no_gpu"))
    );
    assert_eq!(
        err(
            "RuntimeError: Failed to bind socket to 0.0.0.0:8000: [Errno 98] Address already in use"
        ),
        Some(json!("port_in_use"))
    );
    assert_eq!(
        err(
            "RuntimeError: Failed, NCCL error /app/x.cpp:12 'unhandled cuda error (run with NCCL_DEBUG=INFO for details)'"
        ),
        Some(json!("nccl_error"))
    );
    assert_eq!(
        err("AssertionError: Impossible to fit in any sequence in kvCache"),
        Some(json!("kv_cache_too_small"))
    );
    assert_eq!(
        err(
            "Cannot find an appropriate cached snapshot folder for the specified revision on the local disk and outgoing traffic has been disabled."
        ),
        Some(json!("model_not_in_cache"))
    );
    assert_eq!(
        err("RuntimeError: No weight files found in /x."),
        Some(json!("no_weights"))
    );
}

#[test]
fn test_ordinary_lines_carry_nothing_trtllm() {
    for line in [
        "",
        "[10/04/2026-18:20:34] [TRT-LLM] [I] Validating KV Cache config against kv_cache_dtype=\"auto\"",
        "[TensorRT-LLM] TensorRT LLM version: 1.2.1",
        "INFO:     Started server process [1]",
        "INFO:     Shutting down",
    ] {
        assert_eq!(log(line), None, "{line}");
    }
}

#[test]
fn test_metrics_text_entry_point_matches_the_adapter_trtllm() {
    let body = dumps(&json!([stats_body(json!({}))]));
    assert_eq!(
        parse_metrics_text(&body).expect("parses"),
        adapter().parse_metrics(&body)
    );
    let line = "INFO:     Application startup complete.";
    let entry: Option<Map<String, Value>> = parse_log_line(line).expect("parses");
    assert_eq!(entry.map(Value::Object), log(line));
    let _ = PyVal::None;
}
