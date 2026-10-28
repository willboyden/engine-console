//! The library and data rows of backend/tests/core/test_api_smoke.py (HF + fit, profiles, bench, chat and the arena
//! over HTTP); bench, chat and the arena run against a ready instance of the lifecycle.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

#[path = "common/kit.rs"]
mod kit;

use kit::fakefleet::{HttpHandler, HttpResp, http_handler, qwen_cfg};
use kit::{Resp, TestEnv};
use serde_json::{Value, json};

const P: &str = "/api/v1";

/// test_api_smoke.problem_code: the problem+json content type, `status` equal to the HTTP status, and the code.
fn problem_code(r: &Resp) -> String {
    assert!(
        r.header("content-type")
            .unwrap_or_default()
            .starts_with("application/problem+json"),
        "{}",
        r.text()
    );
    assert_eq!(r.json()["status"], json!(r.status));
    r.code()
}

/// test_api_smoke.sse_engine
fn sse_engine() -> HttpHandler {
    http_handler(|req| {
        let sent: Value = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
        let mut chunks: Vec<Value> = ["Hel", "lo", " there"]
            .iter()
            .map(|w| json!({"choices": [{"delta": {"content": w}}]}))
            .collect();
        chunks.push(json!({"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 3}}));
        if sent["stream"] == json!(true) {
            let mut text: String = chunks
                .iter()
                .map(|c| format!("data: {}\n\n", ec_hub::pyjson::dumps(c)))
                .collect();
            text.push_str("data: [DONE]\n\n");
            HttpResp::bytes(200, text.into_bytes()).header("content-type", "text/event-stream")
        } else {
            HttpResp::json(
                200,
                &json!({"choices": [{"message": {"content": "Hello there", "reasoning_content": "hmm"}}],
                        "usage": {"prompt_tokens": 20, "completion_tokens": 3}}),
            )
        }
    })
}

/// test_api_smoke.make_ready_instance
async fn make_ready_instance(env: &mut TestEnv, name: &str, repo: &str) -> Value {
    env.seed_local_model(repo, None, 65_000_000_000, "local");
    env.engine_up();
    let gpus = if name == "qwen" {
        json!([0])
    } else {
        json!([1])
    };
    let r = env
        .post(&format!("{P}/instances"))
        .json(&json!({"engine": "fake", "repo_id": repo, "name": name, "params": {"max_model_len": 8192},
                      "gpu_ids": gpus}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let inst = r.json();
    let container = inst["container_name"].as_str().unwrap().to_string();
    env.node
        .st()
        .logs
        .insert(container, "loading weights 40%\n".into());
    env.container().lifecycle.tick().await.unwrap();
    let got = env
        .get(&format!("{P}/instances/{}", inst["id"].as_str().unwrap()))
        .send()
        .await
        .json();
    assert_eq!(got["state"], "ready", "{got}");
    got
}

// ---- HF + fit -------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_hf_search_detail_and_fit() {
    let mut env = TestEnv::new();
    env.hub.add(
        "Qwen/Qwen3-32B",
        &[("config.json", b"{}"), ("model.safetensors", &[b'x'; 100])],
        Some(qwen_cfg()),
        &[("BF16", 32_762_000_000)],
    );
    let s = env
        .get(&format!("{P}/hf/search?q=qwen&limit=5"))
        .send()
        .await
        .json();
    assert_eq!(s["items"][0]["repo_id"], "Qwen/Qwen3-32B");
    // round(32_762_000_000 * 2 / 1024**3, 2)
    assert_eq!(s["items"][0]["approx_size_gib"], json!(61.02));
    let none = env
        .get(&format!("{P}/hf/search?q=qwen&max_gib=1"))
        .send()
        .await
        .json();
    assert_eq!(none["items"], json!([]));
    let d = env
        .get(&format!("{P}/hf/models/Qwen/Qwen3-32B"))
        .send()
        .await
        .json();
    assert!(
        d["info"]["num_layers"] == 64 && d["info"]["num_kv_heads"] == 8,
        "{d}"
    );
    assert!(d["card"].as_str().unwrap().contains("model card"));
    let missing = env
        .get(&format!("{P}/hf/models/does/not-exist"))
        .send()
        .await;
    assert_eq!(problem_code(&missing), "not_found");
    let r = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"max_model_len": 8192}}))
        .send()
        .await;
    assert_eq!(r.status, 200, "{}", r.text());
    let fit = r.json();
    assert!(
        fit["verdict"] == "fits" || fit["verdict"] == "tight",
        "{fit}"
    );
    assert_eq!(fit["per_gpu"][0]["uuid"], "GPU-uuid-0");
    let bad = env
        .post(&format!("{P}/fit"))
        .json(&json!({"engine": "fake", "repo_id": "Qwen/Qwen3-32B", "params": {"bogus": 1}}))
        .send()
        .await;
    assert!(bad.status == 422 && problem_code(&bad) == "invalid_params");
    assert_eq!(bad.json()["errors"], json!(["unknown parameter 'bogus'"]));
    env.hub.gated.lock().unwrap().push("meta/gated".into());
    let gated = env.get(&format!("{P}/hf/models/meta/gated")).send().await;
    assert_eq!(problem_code(&gated), "hf_gated");
}

// ---- profiles -------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_profiles_crud_diff_yaml() {
    let mut env = TestEnv::new();
    let r = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "long", "engine": "fake", "preset": "long-context", "params": {"tp": 2}}))
        .send()
        .await;
    assert_eq!(r.status, 201, "{}", r.text());
    let a = r.json();
    assert_eq!(
        a["params"],
        json!({"mem_fraction": 0.9, "max_model_len": 8192, "tp": 2})
    );
    let b = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "fast", "engine": "fake", "params": {"tp": 1, "kv_dtype": "fp8"}}))
        .send()
        .await
        .json();
    let dup = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "long", "engine": "fake"}))
        .send()
        .await;
    assert_eq!(problem_code(&dup), "profile_exists");
    let preset = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "x", "engine": "fake", "preset": "zzz"}))
        .send()
        .await;
    assert_eq!(problem_code(&preset), "unknown_preset");
    let invalid = env
        .post(&format!("{P}/profiles"))
        .json(&json!({"name": "x", "engine": "fake", "params": {"bad": 1}}))
        .send()
        .await;
    assert_eq!(invalid.status, 422);
    let (aid, bid) = (a["id"].as_str().unwrap(), b["id"].as_str().unwrap());
    let d = env
        .get(&format!("{P}/profiles/{aid}/diff/{bid}"))
        .send()
        .await
        .json();
    assert_eq!(d["changed"], json!({"tp": {"a": 2, "b": 1}}));
    assert_eq!(
        d["only_a"],
        json!({"mem_fraction": 0.9, "max_model_len": 8192})
    );
    assert_eq!(d["only_b"], json!({"kv_dtype": "fp8"}));
    let y = env.get(&format!("{P}/profiles/{aid}/export")).send().await;
    assert!(y.text().contains("max_model_len: 8192"), "{}", y.text());
    assert!(
        y.header("content-type")
            .unwrap_or_default()
            .starts_with("application/yaml")
    );
    let imp = env
        .post(&format!("{P}/profiles/import"))
        .header("content-type", "text/yaml")
        .raw(y.text().replace("name: long", "name: long2"))
        .send()
        .await;
    assert!(
        imp.status == 201 && imp.json()["name"] == "long2",
        "{}",
        imp.text()
    );
    let list_yaml = env
        .post(&format!("{P}/profiles/import"))
        .header("content-type", "text/yaml")
        .raw("- just\n- a list")
        .send()
        .await;
    assert_eq!(problem_code(&list_yaml), "invalid_yaml");
    let put = env
        .put(&format!("{P}/profiles/{bid}"))
        .json(&json!({"name": "fast2", "engine": "fake", "params": {"tp": 1}}))
        .send()
        .await
        .json();
    assert_eq!(put["name"], "fast2");
    let two = env
        .get(&format!("{P}/profiles?limit=2"))
        .send()
        .await
        .json();
    assert_eq!(two["items"].as_array().unwrap().len(), 2);
    let one = env
        .get(&format!("{P}/profiles?limit=1"))
        .send()
        .await
        .json();
    assert_eq!(one["next_cursor"], "1");
    assert_eq!(
        env.delete(&format!("{P}/profiles/{bid}"))
            .send()
            .await
            .status,
        204
    );
    let gone = env.get(&format!("{P}/profiles/{bid}/export")).send().await;
    assert_eq!(problem_code(&gone), "not_found");
}

// ---- bench ----------------------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_bench() {
    let mut env = TestEnv::new();
    let inst = make_ready_instance(&mut env, "qwen", "Qwen/Qwen3-32B").await;
    let iid = inst["id"].as_str().unwrap().to_string();
    env.engine_route("/v1/chat/completions", sse_engine());
    let r = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": iid, "suite": {"concurrency": [1, 2], "prompt_tokens": 28, "max_tokens": 4,
                                                   "rounds": 1, "prefix_cache": true}}))
        .send()
        .await;
    assert_eq!(r.status, 202, "{}", r.text());
    let bid = r.json()["id"].as_str().unwrap().to_string();
    env.container().bench.wait(&bid).await;
    let run = env.get(&format!("{P}/bench/{bid}")).send().await.json();
    assert_eq!(run["state"], "completed", "{run}");
    let levels: Vec<Value> = run["results"]["concurrency"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["concurrency"].clone())
        .collect();
    assert_eq!(levels, [json!(1), json!(2)]);
    assert!(
        run["results"]["single"]["errors"] == 0
            && !run["results"]["single"]["ttft_p50_s"].is_null()
    );
    assert!(
        run["results"]["concurrency"][1]["throughput_tps"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(run["results"]["prefix_cache"].get("speedup").is_some());
    assert_eq!(
        env.get(&format!("{P}/bench")).send().await.json()["items"][0]["id"],
        json!(bid)
    );
    let snap = env
        .get(&format!("{P}/bench/{bid}/stream?once=true"))
        .send()
        .await;
    assert!(snap.text().contains("event: snapshot"), "{}", snap.text());
    let unknown = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": iid, "suite": "nope"}))
        .send()
        .await;
    assert_eq!(problem_code(&unknown), "unknown_suite");
    env.post(&format!("{P}/instances/{iid}/stop")).send().await;
    let stopped = env
        .post(&format!("{P}/bench"))
        .json(&json!({"instance_id": iid}))
        .send()
        .await;
    assert_eq!(problem_code(&stopped), "instance_not_ready");
}

// ---- chat / arena / usage -------------------------------------------------------------------------------------------
#[tokio::test]
async fn test_chat_conversations_prompts() {
    let mut env = TestEnv::new();
    let inst = make_ready_instance(&mut env, "qwen", "Qwen/Qwen3-32B").await;
    let iid = inst["id"].as_str().unwrap().to_string();
    env.engine_route("/v1/chat/completions", sse_engine());
    let conv = env
        .post(&format!("{P}/conversations"))
        .json(&json!({"title": "t", "instance_id": iid}))
        .send()
        .await
        .json();
    let cid = conv["id"].as_str().unwrap().to_string();
    let msgs = json!([{"role": "user", "content": "hi"}]);
    let r = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": iid, "conversation_id": cid, "messages": msgs}))
        .send()
        .await
        .json();
    assert_eq!(r["choices"][0]["message"]["content"], "Hello there");
    let s = env
        .post(&format!("{P}/chat/completions"))
        .json(
            &json!({"instance_id": iid, "conversation_id": cid, "messages": msgs, "stream": true}),
        )
        .send()
        .await;
    assert!(
        s.header("content-type")
            .unwrap_or_default()
            .starts_with("text/event-stream")
    );
    let text = s.text();
    assert!(text.contains("Hel") && text.contains("[DONE]"), "{text}");
    let forwarded: Value =
        serde_json::from_slice(&env.engine_requests().last().unwrap().body).unwrap();
    assert!(forwarded["model"] == "Qwen/Qwen3-32B" && forwarded.get("instance_id").is_none());
    assert_eq!(forwarded["stream_options"], json!({"include_usage": true}));
    let full = env
        .get(&format!("{P}/conversations/{cid}"))
        .send()
        .await
        .json();
    let roles: Vec<Value> = full["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["role"].clone())
        .collect();
    assert_eq!(
        roles,
        [
            json!("user"),
            json!("assistant"),
            json!("user"),
            json!("assistant")
        ]
    );
    assert!(
        full["messages"][3]["content"] == "Hello there"
            && full["messages"][1]["reasoning"] == "hmm"
    );
    assert_eq!(
        env.get(&format!("{P}/conversations")).send().await.json()["items"][0]["id"],
        json!(cid)
    );
    assert_eq!(
        env.delete(&format!("{P}/conversations/{cid}"))
            .send()
            .await
            .status,
        204
    );
    let missing = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"messages": msgs}))
        .send()
        .await;
    assert_eq!(problem_code(&missing), "missing_instance_id");
    let unknown = env
        .post(&format!("{P}/chat/completions"))
        .json(&json!({"instance_id": "nope", "messages": msgs}))
        .send()
        .await;
    assert_eq!(problem_code(&unknown), "not_found");
    let u = env
        .get(&format!("{P}/usage?group_by=model"))
        .send()
        .await
        .json();
    assert!(
        u[0]["key"] == "Qwen/Qwen3-32B" && u[0]["requests"] == 2 && u[0]["prompt_tokens"] == 40,
        "{u}"
    );
    assert_eq!(
        env.get(&format!("{P}/usage?group_by=day"))
            .send()
            .await
            .status,
        200
    );
    let hour = env
        .get(&format!("{P}/usage?group_by=hour"))
        .send()
        .await
        .json();
    assert!(hour[0]["key"].as_str().unwrap().ends_with(":00Z"));
    let bad = env.get(&format!("{P}/usage?group_by=x")).send().await;
    assert_eq!(problem_code(&bad), "invalid_group_by");
    let csv = env.get(&format!("{P}/usage/export.csv")).send().await;
    assert!(
        csv.header("content-type")
            .unwrap_or_default()
            .starts_with("text/csv")
    );
    assert!(
        csv.text()
            .lines()
            .next()
            .unwrap()
            .starts_with("key,requests")
    );
    let pr = env
        .post(&format!("{P}/prompts"))
        .json(&json!({"title": "Sys", "content": "You are terse.", "tags": ["a"]}))
        .send()
        .await
        .json();
    let pid = pr["id"].as_str().unwrap();
    let put = env
        .put(&format!("{P}/prompts/{pid}"))
        .json(&json!({"title": "Sys2", "content": "c"}))
        .send()
        .await
        .json();
    assert_eq!(put["title"], "Sys2");
    assert_eq!(
        env.get(&format!("{P}/prompts")).send().await.json()["items"][0]["title"],
        "Sys2"
    );
    assert_eq!(
        env.delete(&format!("{P}/prompts/{pid}"))
            .send()
            .await
            .status,
        204
    );
}

#[tokio::test]
async fn test_arena_blind_vote_and_elo() {
    let mut env = TestEnv::new();
    let a = make_ready_instance(&mut env, "qwen", "Qwen/Qwen3-32B").await;
    let b = make_ready_instance(&mut env, "qwen-b", "org/other").await;
    env.engine_route("/v1/chat/completions", sse_engine());
    let m = env
        .post(&format!("{P}/arena/matches"))
        .json(&json!({"prompt": "hello", "instance_a": a["id"], "instance_b": b["id"]}))
        .send()
        .await
        .json();
    assert!(
        m["blind"] == true && m["model_a"].is_null() && m["response_a"] == "Hello there",
        "{m}"
    );
    let mid = m["id"].as_str().unwrap().to_string();
    let v = env
        .post(&format!("{P}/arena/matches/{mid}/vote"))
        .json(&json!({"winner": "a"}))
        .send()
        .await
        .json();
    assert!(
        v["model_a"] == "Qwen/Qwen3-32B" && v["winner"] == "a",
        "{v}"
    );
    let again = env
        .post(&format!("{P}/arena/matches/{mid}/vote"))
        .json(&json!({"winner": "b"}))
        .send()
        .await;
    assert_eq!(problem_code(&again), "already_voted");
    let lb = env
        .get(&format!("{P}/arena/leaderboard"))
        .send()
        .await
        .json();
    assert_eq!(
        (&lb[0]["model"], &lb[0]["rating"], &lb[0]["wins"]),
        (&json!("Qwen/Qwen3-32B"), &json!(1016.0), &json!(1))
    );
    assert_eq!(
        (&lb[1]["model"], &lb[1]["rating"], &lb[1]["losses"]),
        (&json!("org/other"), &json!(984.0), &json!(1))
    );
    let same = make_ready_instance(&mut env, "qwen-c", "Qwen/Qwen3-32B").await;
    let m2 = env
        .post(&format!("{P}/arena/matches"))
        .json(&json!({"prompt": "x", "instance_a": a["id"], "instance_b": same["id"]}))
        .send()
        .await
        .json();
    env.post(&format!(
        "{P}/arena/matches/{}/vote",
        m2["id"].as_str().unwrap()
    ))
    .json(&json!({"winner": "a"}))
    .send()
    .await;
    // self-play leaves ratings alone
    assert_eq!(
        env.get(&format!("{P}/arena/leaderboard"))
            .send()
            .await
            .json()[0]["rating"],
        json!(1016.0)
    );
    let selfmatch = env
        .post(&format!("{P}/arena/matches"))
        .json(&json!({"prompt": "x", "instance_a": a["id"], "instance_b": a["id"]}))
        .send()
        .await;
    assert_eq!(problem_code(&selfmatch), "same_instance");
}
