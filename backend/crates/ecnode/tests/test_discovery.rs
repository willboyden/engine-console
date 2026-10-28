//! node/tests/test_discovery.py: inventory, endpoints, samples and metrics: redaction on the node, no env values,
//! loopback probes only. This part: the tests that need no Agent (classification, container views, probes,
//! endpoint plans, script status). Probes reach only a FakeEngine or a closed port.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

#[path = "common/wsd_agent.rs"]
mod wsd_agent;
#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use ecnode::discovery::{
    self, GetOpts, PortProbe, Probe, classify, container_view, parse_params, script_status,
};
use ecnode::policy::{Root, parse};
use serde_json::{Value, json};
use wsd_kit::{
    ARGV_SECRET, C, ENV_VALUE, FakeEngine, GPU_A, GPU_B, cid_engine, closed_port, container,
};

fn no_roots() -> Vec<(Root, PathBuf)> {
    Vec::new()
}

#[test]
fn test_classify_signatures_and_non_engines() {
    assert_eq!(
        classify("vllm/vllm-openai:v0.11.0", "").as_deref(),
        Some("vllm")
    );
    assert_eq!(
        classify("lmsysorg/sglang:latest", "").as_deref(),
        Some("sglang")
    );
    assert_eq!(
        classify("ghcr.io/ggml-org/llama.cpp:server-cuda", "").as_deref(),
        Some("llamacpp")
    );
    assert_eq!(
        classify("nvcr.io/nvidia/tensorrt-llm/release:1.0", "").as_deref(),
        Some("trtllm")
    );
    assert_eq!(
        classify("ollama/ollama:0.12", "").as_deref(),
        Some("ollama")
    );
    assert_eq!(
        classify(
            "custom/image:1",
            "python3 -m sglang.launch_server --model-path x"
        )
        .as_deref(),
        Some("sglang")
    );
    for img in [
        "vllm/vllm-router:1",
        "litellm/litellm:1",
        "nginx:1",
        "grafana/grafana:11",
    ] {
        assert_eq!(classify(img, "vllm serve x"), None, "{img}");
    }
}

#[test]
fn test_container_view_redacts_argv_and_never_reads_env() {
    let tmp = tempfile::tempdir().unwrap();
    let hf = tmp.path().join("hf");
    std::fs::create_dir(&hf).unwrap();
    let src = hf
        .join("hub")
        .join("models--a--b")
        .to_string_lossy()
        .into_owned();
    let doc = container(
        "e1",
        &cid_engine(),
        "vllm/vllm-openai:v1",
        &[
            "vllm",
            "serve",
            "org/m",
            "--api-key",
            ARGV_SECRET,
            "--max-model-len",
            "8192",
            "--enforce-eager",
        ],
        C {
            ports: Some(
                json!({"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18001"}],
                               "9000/tcp": [{"HostIp": "10.0.0.5", "HostPort": "9000"}]}),
            ),
            mounts: Some(
                json!([{"Type": "bind", "Source": src, "Destination": "/m", "RW": false},
                                {"Type": "volume", "Source": "/var/lib/x", "Destination": "/c", "RW": true}]),
            ),
            device_requests: Some(json!([{"Driver": "", "Count": 0, "DeviceIDs": ["1", GPU_A]}])),
            ..C::default()
        },
    );
    let roots = vec![(Root::new("hf", &hf.to_string_lossy()), hf.clone())];
    let by_index = HashMap::from([
        ("0".to_string(), GPU_A.to_string()),
        ("1".to_string(), GPU_B.to_string()),
    ]);
    let v = container_view(&doc, &roots, &by_index, None).unwrap();
    let blob = serde_json::to_string(&v.info).unwrap();
    assert!(!blob.contains(ARGV_SECRET) && !blob.contains(ENV_VALUE) && !blob.contains("Env"));
    assert_eq!(
        Value::Object(v.info["params"].as_object().unwrap().clone()),
        json!({"model": "org/m", "api_key": "[redacted]", "max_model_len": "8192", "enforce_eager": "true"})
    );
    assert_eq!(v.info["gpus"], json!([GPU_B, GPU_A]));
    assert_eq!(v.info["all_gpus"], false);
    assert_eq!(
        v.info["mounts"][0],
        json!({"source": src, "dest": "/m", "ro": true, "root": "hf", "rel": "hub/models--a--b"})
    );
    assert_eq!(v.info["mounts"][1]["root"], Value::Null);
    assert_eq!(v.loopback_ports, vec![18001]);
    assert_eq!(v.published, vec![(8000, 18001), (9000, 9000)]);
    let created = v.info["created"].as_f64().unwrap();
    assert!((created - 1_788_256_800.12).abs() < 0.01);
    // the result survives the hub's own whitelist unchanged in shape
    let cleaned = ec_proto::clean(
        &Value::Object(v.info.clone()),
        &ec_proto::specs::container(),
    );
    assert_eq!(cleaned["argv"], v.info["argv"]);
}

#[test]
fn test_all_gpus_and_cdi_devices() {
    let none = HashMap::new();
    let doc = container(
        "e",
        &"1".repeat(64),
        "x:1",
        &["x"],
        C {
            device_requests: Some(json!([{"Driver": "cdi", "DeviceIDs": ["nvidia.com/gpu=all"]}])),
            ..C::default()
        },
    );
    assert_eq!(
        container_view(&doc, &no_roots(), &none, None).unwrap().info["all_gpus"],
        true
    );
    let doc = container(
        "e",
        &"1".repeat(64),
        "x:1",
        &["x"],
        C {
            device_requests: Some(json!([{"Count": -1}])),
            ..C::default()
        },
    );
    assert_eq!(
        container_view(&doc, &no_roots(), &none, None).unwrap().info["all_gpus"],
        true
    );
}

#[test]
fn test_parse_params_display_only() {
    let argv: Vec<String> = ["vllm", "serve", "m", "--tp=2", "--x", "--y", "1"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(
        Value::Object(parse_params(&argv)),
        json!({"model": "m", "tp": "2", "x": "true", "y": "1"})
    );
}

#[test]
fn test_script_status_unknown_without_patterns() {
    let pol = parse(
        &json!({"scripts": [{"name": "s", "actions": [{"name": "up", "argv": ["/bin/true"]}]}]}),
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        script_status(&pol, &["x".to_string()], 1.0)[0]["status"],
        "unknown"
    );
}

#[test]
fn test_http_get_never_follows_redirects() {
    let engine = FakeEngine::start();
    let p = discovery::http_get(engine.port, "/v1/redirect", &GetOpts::default());
    assert_eq!(p.status, Some(302));
    assert!(engine.seen().iter().all(|s| s.path == "/v1/redirect"));
    // a loopback port nobody listens on
    assert_eq!(
        discovery::http_get(closed_port(), "/v1/models", &GetOpts::default()).status,
        None
    );
}

fn ok_probe(status: i64) -> Probe {
    Probe {
        status: Some(status),
        body: br#"{"data": [{"id": "org/m"}]}"#.to_vec(),
    }
}

#[test]
fn test_slow_metrics_engines_are_never_probed_on_metrics_and_are_scraped_by_health() {
    let doc = container(
        "trt",
        &"cd".repeat(32),
        "nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
        &["trtllm-serve", "serve", "org/m", "--port", "8000"],
        C {
            ports: Some(json!({"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18077"}]})),
            ..C::default()
        },
    );
    let view = container_view(&doc, &no_roots(), &HashMap::new(), None).unwrap();
    assert_eq!(view.info["engine"], "trtllm");
    let loud: Arc<Mutex<Vec<u16>>> = Arc::default();
    let quiet: Arc<Mutex<Vec<u16>>> = Arc::default();
    let prober: discovery::Prober = {
        let loud = Arc::clone(&loud);
        Arc::new(move |p| {
            loud.lock().unwrap().push(p);
            PortProbe {
                port: p,
                models: ok_probe(200),
                health: ok_probe(200),
                metrics: ok_probe(200),
            }
        })
    };
    let quiet_prober: discovery::Prober = {
        let quiet = Arc::clone(&quiet);
        Arc::new(move |p| {
            quiet.lock().unwrap().push(p);
            PortProbe {
                port: p,
                models: ok_probe(200),
                health: ok_probe(200),
                metrics: Probe::default(),
            }
        })
    };
    let eps = discovery::build_endpoints(&[view], &[], &prober, 8, &quiet_prober);
    assert_eq!(*quiet.lock().unwrap(), vec![18077]);
    assert!(loud.lock().unwrap().is_empty()); // /metrics is never touched by discovery
    assert_eq!(eps[0]["engine"], "trtllm");
    assert_eq!(eps[0]["health_status"], 200);
    assert_eq!(eps[0]["metrics_status"], Value::Null);
}

fn pp(port: u16, models: Probe, health: Probe, metrics: Probe) -> PortProbe {
    PortProbe {
        port,
        models,
        health,
        metrics,
    }
}

fn probe(status: i64, body: &[u8]) -> Probe {
    Probe {
        status: Some(status),
        body: body.to_vec(),
    }
}

#[test]
fn test_m4_a_web_apps_catch_all_page_is_not_an_engine() {
    // security review 2026-10-04 M4: an SPA answering 200 + HTML on every path must not qualify as an engine
    let html = probe(200, b"<!doctype html><html>app</html>");
    let listing = probe(200, br#"{"object": "list", "data": [{"id": "m"}]}"#);
    let locked = probe(401, b"");
    assert!(!pp(8080, html.clone(), html.clone(), html.clone()).qualifies());
    assert!(
        !pp(
            8080,
            probe(200, br#"{"data": "x"}"#),
            html.clone(),
            html.clone()
        )
        .qualifies()
    );
    assert!(pp(8000, listing, html, Probe::default()).qualifies());
    let vllm_metrics = probe(
        200,
        b"# HELP vllm:num_requests_running\nvllm:num_requests_running 0\n",
    );
    assert!(pp(8000, locked, Probe::default(), vllm_metrics).qualifies()); // an engine with an API key
}

#[test]
fn test_m4_a_login_protected_app_with_a_health_page_is_not_an_engine() {
    // security re-audit 2026-10-04: 401 on /v1/models plus 200 on /health is any login-protected web app
    let locked = probe(401, b"");
    let health = probe(200, br#"{"status": "ok"}"#);
    let plain_metrics = probe(200, b"# HELP http_requests_total\nhttp_requests_total 3\n");
    assert!(!pp(8080, locked.clone(), health.clone(), Probe::default()).qualifies());
    assert!(!pp(8080, locked, health, plain_metrics).qualifies());
}

#[test]
fn test_container_view_python_dynamic_typing_edges() {
    // Rust-only: values docker never sends but a hostile daemon could, read the way Python reads them
    let none = HashMap::new();
    let doc = json!({
        "Id": 5, "Name": null, "Config": "not a dict",
    });
    // Name null is str(None): a container called "None"
    assert_eq!(
        container_view(&doc, &no_roots(), &none, None).unwrap().info["name"],
        "None"
    );
    assert!(container_view(&json!({"Name": "/"}), &no_roots(), &none, None).is_none());
    let doc = json!({
        "Name": "/x", "Id": "abc", "Path": "/bin/x",
        "State": {"Status": "exited", "Pid": true, "ExitCode": 3, "Running": 1},
        "HostConfig": {"NetworkMode": 7, "PortBindings": {"80/tcp": [{"HostPort": 8080, "HostIp": ""},
                                                                    {"HostPort": "08081"},
                                                                    {"HostPort": "x"}, "junk"],
                                                          "bad/tcp": [{"HostPort": "1"}]}},
        "NetworkSettings": {"Ports": {}},
        "Mounts": [{"Source": "/a", "Destination": "/b"}, {"Source": "/c", "RW": 0}, "junk"],
    });
    let v = container_view(&doc, &no_roots(), &none, None).unwrap();
    assert_eq!(v.info["status"], "exited (exit 3)");
    assert_eq!(v.info["pid"], Value::Null); // a bool is not a pid
    assert!(!v.running); // Running must be exactly true
    assert_eq!(v.info["network_mode"], "7");
    assert_eq!(v.info["id"], "abc");
    assert_eq!(
        v.info["ports"],
        json!([{"host_ip": "", "host_port": 8080, "container_port": 80},
               {"host_ip": "", "host_port": 8081, "container_port": 80}])
    );
    assert_eq!(v.published, vec![(80, 8080)]); // setdefault: the first binding wins
    assert_eq!(v.loopback_ports, vec![8080, 8081]);
    assert_eq!(v.info["mounts"][0]["ro"], false);
    assert_eq!(v.info["mounts"][1]["ro"], true);
    assert_eq!(v.info["status"], "exited (exit 3)");
    let with_status = container_view(&doc, &no_roots(), &none, Some("Up 3 minutes")).unwrap();
    assert_eq!(with_status.info["status"], "Up 3 minutes");
}

#[test]
fn test_bare_ports_hints_and_gateway_plans() {
    // Rust-only: build_endpoints' plan order, the gateway claiming its engine's name, and bare-port hints
    let views = vec![
        container_view(
            &container(
                "eng",
                &"1".repeat(64),
                "vllm/vllm-openai:v1",
                &["vllm", "serve", "m"],
                C {
                    labels: Some(json!({"engine-console": "1", "engine-console.role": "engine",
                                        "engine-console.instance": "i1", "engine-console.engine": "vllm"})),
                    ..C::default()
                },
            ),
            &no_roots(),
            &HashMap::new(),
            None,
        )
        .unwrap(),
        container_view(
            &container(
                "eng-gw",
                &"2".repeat(64),
                "nginx@sha256:00",
                &["/bin/sh"],
                C {
                    labels: Some(json!({"engine-console": "1", "engine-console.role": "gateway",
                                        "engine-console.instance": "i1"})),
                    ports: Some(json!({"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18005"}]})),
                    ..C::default()
                },
            ),
            &no_roots(),
            &HashMap::new(),
            None,
        )
        .unwrap(),
    ];
    let prober: discovery::Prober = Arc::new(|p| PortProbe {
        port: p,
        models: if p == 11434 {
            probe(200, br#"{"data": []}"#)
        } else {
            Probe::default()
        },
        health: Probe::default(),
        metrics: Probe::default(),
    });
    let eps = discovery::build_endpoints(&views, &[18005, 11434, 11434, 9], &prober, 8, &prober);
    let got: Vec<(Value, Value, Value)> = eps
        .iter()
        .map(|e| {
            (
                e["port"].clone(),
                e["container"].clone(),
                e["engine"].clone(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (json!(18005), json!("eng"), json!("vllm")),
            (json!(11434), Value::Null, json!("ollama")),
        ]
    );
}

fn golden() -> Value {
    serde_json::from_str(include_str!("data/wsd_discovery_vectors.json")).unwrap()
}

fn golden_roots(g: &Value) -> Vec<(Root, PathBuf)> {
    g["roots"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            (
                Root::new(r[0].as_str().unwrap(), r[1].as_str().unwrap()),
                PathBuf::from(r[2].as_str().unwrap()),
            )
        })
        .collect()
}

fn golden_index(g: &Value) -> HashMap<String, String> {
    g["by_index"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect()
}

#[test]
fn test_classify_and_params_match_python_goldens() {
    // Rust-only: tests/gen/wsd_discovery_vectors.py ran discovery.py on these (tests/data/wsd_discovery_vectors.json)
    let g = golden();
    for c in g["classify"].as_array().unwrap() {
        let got = classify(c[0].as_str().unwrap(), c[1].as_str().unwrap());
        assert_eq!(json!(got), c[2], "{}", c[0]);
    }
    for c in g["params"].as_array().unwrap() {
        assert_eq!(
            Value::Object(parse_params(&strings(&c[0]))),
            c[1],
            "{}",
            c[0]
        );
    }
}

fn view_json(v: Option<&discovery::View>) -> Value {
    match v {
        None => Value::Null,
        Some(v) => json!({
            "info": v.info, "pid": v.pid, "running": v.running, "host_network": v.host_network,
            "loopback_ports": v.loopback_ports,
            "published": v.published.iter().map(|(c, h)| json!([c, h])).collect::<Vec<_>>(),
        }),
    }
}

#[test]
fn test_container_views_match_python_goldens() {
    // Rust-only: 20 ordinary and hostile inspect documents x 3 statuses. Where Python raises (a dict Mounts, a
    // string State...), the Rust node must still answer without panicking (it reads around the bad field).
    let g = golden();
    let (roots, by_index) = (golden_roots(&g), golden_index(&g));
    let docs = g["docs"].as_array().unwrap();
    let mut compared = 0;
    for case in g["views"].as_array().unwrap() {
        let doc = &docs[usize::try_from(case[0].as_u64().unwrap()).unwrap()];
        let status = case[1].as_str();
        let got = container_view(doc, &roots, &by_index, status);
        if case[2].get("raises").is_some() {
            continue;
        }
        assert_eq!(
            view_json(got.as_ref()),
            case[2],
            "doc {} status {status:?}",
            case[0]
        );
        compared += 1;
    }
    assert_eq!(compared, 57);
}

#[test]
fn test_probe_heuristics_match_python_goldens() {
    let g = golden();
    let bodies: Vec<Vec<u8>> = g["bodies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| hex::decode(b.as_str().unwrap()).unwrap())
        .collect();
    let st = |v: &Value| v.as_i64();
    for c in g["probes"].as_array().unwrap() {
        let met = c[4].as_str().unwrap().as_bytes().to_vec();
        let p = PortProbe {
            port: u16::try_from(c[0].as_u64().unwrap()).unwrap(),
            models: Probe {
                status: st(&c[1]),
                body: bodies[usize::try_from(c[2].as_u64().unwrap()).unwrap()].clone(),
            },
            health: Probe {
                status: st(&c[3]),
                body: Vec::new(),
            },
            metrics: Probe {
                status: if met.is_empty() { None } else { Some(200) },
                body: met,
            },
        };
        let got = json!({"any": p.any_answer(), "names": p.model_names(), "hint": p.hint(),
                         "listing": p.models_listing(), "qualifies": p.qualifies()});
        assert_eq!(got, c[5], "{c}");
    }
}

fn scripted(port: u16) -> PortProbe {
    let p = |s: Option<i64>, b: &[u8]| Probe {
        status: s,
        body: b.to_vec(),
    };
    let (models, health, metrics) = match port % 5 {
        0 => (
            p(Some(200), br#"{"data": [{"id": "m"}]}"#),
            p(Some(200), b""),
            p(Some(200), b"vllm:x"),
        ),
        1 => (p(Some(401), b""), p(None, b""), p(None, b"")),
        2 => (p(Some(200), b"<html>"), p(Some(200), b""), p(None, b"")),
        3 => (p(None, b""), p(None, b""), p(Some(200), b"sglang:x")),
        _ => (p(None, b""), p(None, b""), p(None, b"")),
    };
    PortProbe {
        port,
        models,
        health,
        metrics,
    }
}

#[test]
fn test_build_endpoints_matches_python_golden() {
    // Rust-only: the golden documents' views through build_endpoints with a scripted prober (no network at all)
    let g = golden();
    let (roots, by_index) = (golden_roots(&g), golden_index(&g));
    let docs = g["docs"].as_array().unwrap();
    let views: Vec<discovery::View> = g["ep_docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| {
            let doc = &docs[usize::try_from(i.as_u64().unwrap()).unwrap()];
            container_view(doc, &roots, &by_index, None).unwrap()
        })
        .collect();
    let prober: discovery::Prober = Arc::new(scripted);
    let eps = discovery::build_endpoints(
        &views,
        &[8000, 8000, 11434, 18000, 30000, 30001, 30002, 30003, 30004],
        &prober,
        3,
        &prober,
    );
    let got: Vec<Value> = eps.into_iter().map(Value::Object).collect();
    // Python's golden minus the bare port 30003, whose /metrics merely says "sglang:" while /v1/models never answers:
    // Python made it an endpoint on that hint alone; the node requires a real listing (security audit F3)
    let want: Vec<Value> = g["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| !(e["container"].is_null() && e["port"] == 30003))
        .cloned()
        .collect();
    assert_eq!(want.len(), 5);
    assert_eq!(Value::Array(got), Value::Array(want));
}

// ---- agent-dependent tests ----
mod agent_dependent {
    //! The tests of test_discovery.py that need an Agent (inventory, samples, scraping).
    use std::collections::BTreeSet;

    use ecnode::agent::{filter_metrics, shrink_inventory};
    use ecnode::util::json_size;
    use serde_json::{Value, json};

    use super::wsd_kit::{
        ARGV_SECRET, C, CID_EXT, ENV_VALUE, FakeEngine, GATEWAY_IMAGE, GPU_A, GPU_B, METRICS_TEXT,
        cid_engine, container, manage_policy, obj, w,
    };
    use crate::wsd_agent::World;

    fn names_of(msg: &ec_proto::Obj) -> BTreeSet<String> {
        msg["containers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap().to_string())
            .collect()
    }

    #[test]
    fn test_inventory_endpoints_gateways_and_bare_ports() {
        let w = World::new();
        let engine = FakeEngine::start();
        w.prober.allow(engine.port);
        let mut state = w.fb.state();
        // the console engine publishes nothing (internal network); its gateway publishes the fake engine's port
        state["containers"].as_array_mut().unwrap().push(container(
            "ec-owned-1-gw",
            &"9".repeat(64),
            GATEWAY_IMAGE,
            &["/bin/sh", "-c", "nginx"],
            C {
                labels: Some(json!({"engine-console": "1", "engine-console.role": "gateway",
                                    "engine-console.instance": "inst_1"})),
                ports: Some(json!({"8080/tcp": [{"HostIp": "127.0.0.1", "HostPort": engine.port.to_string()}]})),
                pid: Some(4545),
                ..C::default()
            },
        ));
        w.fb.set_state(&state);
        let scripts = json!([
            {"name": "sg", "containers": ["other-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]},
            {"name": "none", "containers": ["nothing-*"], "actions": [{"name": "up", "argv": ["/bin/true"]}]}]);
        let a = w.agent(Some(&manage_policy(
            &w.machine,
            json!({"discovery_ports": [engine.port, 1], "scripts": scripts}),
        )));
        a.hello(false);
        let inv = a.refresh();
        assert_eq!(
            names_of(&inv.msg),
            BTreeSet::from(
                ["ec-owned-1", "other-engine", "grafana", "ec-owned-1-gw"].map(String::from)
            )
        );
        let eps = inv.msg["endpoints"].as_array().unwrap();
        // the gateway claims the port: no second bare endpoint
        assert_eq!(eps.len(), 1);
        let ep = &eps[0];
        assert_eq!(ep["port"], engine.port);
        assert_eq!(
            (ep["container"].clone(), ep["engine"].clone()),
            (json!("ec-owned-1"), json!("vllm"))
        );
        assert_eq!(ep["models"], json!(["org/model-a"]));
        assert_eq!(
            (
                ep["health_status"].clone(),
                ep["models_status"].clone(),
                ep["metrics_status"].clone()
            ),
            (json!(200), json!(200), json!(200))
        );
        let tried: BTreeSet<u16> = w.prober.tried().into_iter().collect();
        assert!(tried.is_subset(&BTreeSet::from([engine.port, 1]))); // nothing else was ever probed
        assert_eq!(
            inv.gateway_ports,
            std::collections::HashSet::from([engine.port])
        );
        assert!(a.allowed_ports(false).contains(&engine.port));
        let st: Vec<&Value> = inv.msg["scripts"].as_array().unwrap().iter().collect();
        let sg = st.iter().find(|s| s["name"] == "sg").unwrap();
        assert_eq!(sg["status"], "up");
        assert!(sg["detail"].as_str().unwrap().contains("other-engine"));
        assert_eq!(
            st.iter().find(|s| s["name"] == "none").unwrap()["status"],
            "down"
        );
        let blob = Value::Object(inv.msg.clone()).to_string();
        assert!(
            !blob.contains(ENV_VALUE)
                && !blob.contains(ARGV_SECRET)
                && !blob.contains("topsecretvalue")
        );
        // docker info's proxy credentials never leave
        assert!(
            !Value::Object(a.hello(false))
                .to_string()
                .contains("proxy.invalid")
        );
    }

    #[test]
    fn test_bare_discovery_port_with_an_engine() {
        let w = World::new();
        let engine = FakeEngine::start();
        w.prober.allow(engine.port);
        w.fb.update(json!({"containers": []}));
        let a = w.agent(Some(&manage_policy(
            &w.machine,
            json!({"discovery_ports": [engine.port]}),
        )));
        let eps = a.refresh().msg["endpoints"].clone();
        assert_eq!(
            eps,
            json!([{"port": engine.port, "container": null, "engine": "vllm", "models": ["org/model-a"],
                    "health_status": 200, "models_status": 200, "metrics_status": 200, "auth_required": false}])
        );
    }

    #[test]
    fn test_inventory_without_docker_still_probes() {
        let w = World::new();
        let engine = FakeEngine::start();
        w.prober.allow(engine.port);
        w.fb.update(json!({"daemon_down": true}));
        let a = w.agent(Some(&manage_policy(
            &w.machine,
            json!({"discovery_ports": [engine.port]}),
        )));
        let inv = a.refresh();
        assert_eq!(inv.msg["containers"], json!([]));
        assert!(
            inv.error.as_deref().is_some_and(|e| e.contains("daemon")),
            "{:?}",
            inv.error
        );
        let ports: Vec<i64> = inv.msg["endpoints"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| e["port"].as_i64().unwrap())
            .collect();
        assert_eq!(ports, vec![i64::from(engine.port)]);
        let facts = a.hello(false)["docker"].clone();
        assert_eq!(facts["available"], false);
        assert!(facts["error"].as_str().unwrap().contains("Cannot connect"));
    }

    #[test]
    fn test_sample_maps_gpu_processes_to_containers() {
        let w = World::new();
        let a = w.agent(Some(&manage_policy(&w.machine, json!({}))));
        a.hello(false);
        a.refresh();
        let s1 = a.sample(1);
        assert_eq!(
            s1["gpus"][GPU_A]["procs"],
            json!([{"pid": 4242, "used_bytes": 2048_i64 * 1024 * 1024, "name": "vllm", "container": "ec-owned-1"}])
        );
        assert_eq!(s1["gpus"][GPU_B]["procs"][0]["container"], Value::Null); // a pid with no container
        assert_eq!(s1["containers"]["ec-owned-1"]["mem"]["source"], "cgroup");
        assert_eq!(s1["containers"]["other-engine"]["mem"]["source"], "rss");
        assert!(s1["containers"].get("grafana").is_none()); // only engine containers
        let cg = w.machine.paths.cgroup().join(format!(
            "user.slice/user-1000.slice/user@1000.service/user.slice/docker-{}.scope",
            cid_engine()
        ));
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(cg.join("cpu.stat"), "usage_usec 6000000\n").unwrap();
        let s2 = a.sample(2);
        let pct = s2["containers"]["ec-owned-1"]["cpu_pct"].as_f64();
        assert!(pct.is_some_and(|p| p > 0.0), "{pct:?}");
        let pushed = ec_proto::sanitize_push(&s2).unwrap().unwrap();
        assert_eq!(pushed["gpus"][GPU_A]["procs"][0]["container"], "ec-owned-1");
    }

    #[test]
    fn test_sample_attribution_and_memory_for_every_cgroup_form() {
        let world = World::new();
        let (gw_id, sys_id) = ("c".repeat(64), "d".repeat(64));
        let mut state = world.fb.state();
        let cs = state["containers"].as_array_mut().unwrap();
        cs.push(container(
            "ec-owned-1-gw",
            &gw_id,
            GATEWAY_IMAGE,
            &["/bin/sh"],
            C {
                pid: Some(4545),
                labels: Some(
                    json!({"engine-console": "1", "engine-console.role": "gateway",
                                    "engine-console.instance": "inst_1"}),
                ),
                ..C::default()
            },
        ));
        cs.push(container(
            "rootful-vllm",
            &sys_id,
            "vllm/vllm-openai:v1",
            &["vllm", "serve", "m"],
            C {
                pid: Some(4646),
                ..C::default()
            },
        ));
        world.fb.set_state(&state);
        let proc = world.machine.paths.proc.clone();
        for (pid, cg) in [
            (4545, format!("/docker/{gw_id}")),
            (4646, format!("/system.slice/docker-{sys_id}.scope")),
        ] {
            let d = proc.join(pid.to_string());
            w(&d.join("cgroup"), &format!("0::{cg}\n"));
            w(&d.join("comm"), "proc\n");
            w(
                &d.join("stat"),
                &format!("{pid} (proc) S 1 {pid} {pid} 0 -1 0\n"),
            );
            w(&d.join("status"), "VmRSS:\t  100 kB\n");
            let cgd = w_cgroup(&world, &cg);
            w(&cgd.join("memory.current"), "1000\n");
            w(
                &cgd.join("memory.stat"),
                "anon 600\nfile 400\nshmem 0\nkernel 0\n",
            );
        }
        world.fb.smi(
            json!({"apps": [{"pid": 4646, "used_memory": "100", "gpu_uuid": GPU_B},
                                 {"pid": 4242, "used_memory": "200", "gpu_uuid": GPU_A}]}),
        );
        let a = world.agent(Some(&manage_policy(&world.machine, json!({}))));
        a.hello(false);
        a.refresh();
        let s = a.sample(1);
        assert_eq!(s["gpus"][GPU_B]["procs"][0]["container"], "rootful-vllm"); // system.slice/docker-<id>.scope
        assert_eq!(s["gpus"][GPU_A]["procs"][0]["container"], "ec-owned-1"); // user.slice/.../docker-<id>.scope
        assert_eq!(
            s["containers"]["ec-owned-1-gw"]["mem"],
            json!({"current": 1000, "anon": 600, "file": 400, "shmem": 0, "kernel": 0, "source": "cgroup"})
        );
        assert_eq!(s["containers"]["rootful-vllm"]["mem"]["source"], "cgroup");
        let keys: BTreeSet<&str> = s["containers"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from([
                "ec-owned-1",
                "other-engine",
                "ec-owned-1-gw",
                "rootful-vllm"
            ])
        );
        let _ = CID_EXT;
    }

    fn w_cgroup(world: &World, cg: &str) -> std::path::PathBuf {
        world
            .machine
            .paths
            .cgroup()
            .join(cg.trim_start_matches('/'))
    }

    #[test]
    fn test_metrics_filter_keeps_families_and_their_help_lines() {
        let out = filter_metrics(
            METRICS_TEXT,
            &["vllm:".to_string()],
            ec_proto::MAX_METRICS_TEXT,
        );
        assert_eq!(
            out.lines().collect::<Vec<_>>(),
            vec![
                "# HELP vllm:num_requests_running Requests running.",
                "# TYPE vllm:num_requests_running gauge",
                "vllm:num_requests_running{model_name=\"org/model-a\"} 1.0",
            ]
        );
        let big: String = (0..100_000).map(|i| format!("vllm:x{i} 1\n")).collect();
        assert!(filter_metrics(&big, &["vllm:".to_string()], 1000).len() <= 1000);
    }

    #[test]
    fn test_scrape_needs_configuration_and_uses_prefixes() {
        let w = World::new();
        let engine = FakeEngine::start();
        w.prober.allow(engine.port);
        w.fb.update(json!({"containers": []}));
        let a = w.agent(Some(&manage_policy(
            &w.machine,
            json!({"discovery_ports": [engine.port]}),
        )));
        a.refresh();
        assert!(a.scrape().is_empty()); // not configured yet
        a.configure_metrics(Some(0.2), vec![("vllm".into(), vec!["vllm:".into()])]);
        assert_eq!(a.metrics_conf().interval_s, Some(1.0)); // clamped to 1-60 s
        let msgs = a.scrape();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["status"], 200);
        let text = msgs[0]["text"].as_str().unwrap();
        assert!(text.contains("vllm:num_requests_running") && !text.contains("process_cpu"));
        assert_eq!(
            ec_proto::sanitize_push(&msgs[0]).unwrap().unwrap()["port"],
            engine.port
        );
    }

    #[test]
    fn test_huge_inventories_are_shrunk_to_one_frame() {
        let big = |name: String, engine: Option<&str>| {
            let params: ec_proto::Obj = (0..60)
                .map(|i| (format!("p{i}"), json!("v".repeat(200))))
                .collect();
            let labels: ec_proto::Obj = (0..48)
                .map(|i| (format!("l{i}"), json!("w".repeat(256))))
                .collect();
            json!({"name": name, "engine": engine, "managed": false, "argv": vec!["x".repeat(1000); 400],
                   "params": params, "labels": labels})
        };
        let containers: Vec<Value> = (0..256)
            .map(|i| big(format!("c{i}"), if i < 2 { Some("vllm") } else { None }))
            .collect();
        let msg = obj(json!({"t": 1.0, "containers": containers, "endpoints": [], "scripts": []}));
        let out = shrink_inventory(msg, 900 * 1024);
        assert!(json_size(&Value::Object(out.clone())) <= 900 * 1024);
        let names: BTreeSet<&str> = out["containers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c["name"].as_str().unwrap())
            .collect();
        assert!(names.contains("c0") && names.contains("c1")); // engines are kept
        let mut framed = obj(json!({"type": "inventory"}));
        framed.extend(out);
        assert!(ec_proto::encode_json(&framed).unwrap().len() <= ec_proto::MAX_JSON);
    }
}
