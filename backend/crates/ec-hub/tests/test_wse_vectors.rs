//! Replays tests/data/wse_runspec_vectors.json (written by backend/parity/gen/wse_runspec_vectors.py from the Python
//! hub): the pure lifecycle helpers answer value for value what Python answers (ref checks and their problem texts,
//! display names, mounts, env split, labels, the run spec, the node's validation messages, text hygiene, derived ids,
//! the generic metrics map).

#![allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines
)]

use ec_hub::adapters::base::LaunchSpec;
use ec_hub::domain::errors::Obj;
use ec_hub::domain::refs::ModelRef;
use ec_hub::services::discovery::{
    clean_text, clean_text200, display_name, parse_generic_metrics, slug,
};
use ec_hub::services::externals::{external_instance_id, script_instance_id};
use ec_hub::services::runspec::{
    LabelArgs, RunArgs, build_run_spec, check_ref, display_of, engine_labels, hf_layout, hf_mounts,
    model_mounts, split_env, validate_run_spec,
};
use serde_json::{Value, json};

fn vectors() -> Value {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/wse_runspec_vectors.json"
    ))
    .expect("vectors");
    serde_json::from_str(&text).expect("json")
}

fn obj(v: &Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

fn mounts_json(m: &[Obj]) -> Value {
    Value::Array(m.iter().cloned().map(Value::Object).collect())
}

#[test]
fn test_refs_display_names_and_mounts_match_python() {
    let v = vectors();
    for c in v["refs"].as_array().unwrap() {
        let r: ModelRef = serde_json::from_value(c["ref"].clone()).unwrap();
        match check_ref(&r) {
            Err(p) => {
                assert_eq!(c["ok"], json!(false), "{}", c["ref"]);
                assert_eq!(
                    json!([p.status, p.code, p.detail]),
                    json!([c["status"], c["code"], c["detail"]]),
                    "{}",
                    c["ref"]
                );
            }
            Ok(()) => {
                assert_eq!(c["ok"], json!(true), "{}", c["ref"]);
                assert_eq!(json!(display_of(&r)), c["display"], "{}", c["ref"]);
                if r.format != "hf" {
                    let (m, p) = model_mounts(&r, None, None);
                    assert_eq!(mounts_json(&m), c["mounts"], "{}", c["ref"]);
                    assert_eq!(json!(p), c["model_path"], "{}", c["ref"]);
                }
            }
        }
    }
}

#[test]
fn test_hf_layouts_and_mounts_match_python() {
    let v = vectors();
    for c in v["hf_layout"].as_array().unwrap() {
        assert_eq!(
            json!(hf_layout(c["kind"].as_str(), c["rel"].as_str().unwrap())),
            c["out"],
            "{c}"
        );
    }
    for c in v["hf_mounts"].as_array().unwrap() {
        let m = hf_mounts(
            "hf",
            c["kind"].as_str(),
            c["rel"].as_str().unwrap(),
            c["path"].as_str().unwrap(),
        );
        assert_eq!(mounts_json(&m), c["out"], "{c}");
    }
    let llama = ModelRef {
        format: "ollama".into(),
        repo_id: None,
        root_id: None,
        rel: None,
        name: Some("llama3:8b".into()),
    };
    for c in v["ollama_mounts"].as_array().unwrap() {
        let (m, p) = model_mounts(
            &llama,
            c["copy_rel"].as_str(),
            Some(("ollama", c["root_path"].as_str().unwrap())),
        );
        assert_eq!(
            (mounts_json(&m), json!(p)),
            (c["mounts"].clone(), c["model_path"].clone()),
            "{c}"
        );
    }
}

#[test]
fn test_env_split_and_labels_match_python() {
    let v = vectors();
    for c in v["split_env"].as_array().unwrap() {
        let (env, secret) = split_env(&obj(&c["env"]), c["offline"].as_bool().unwrap());
        let secret: Obj = secret.into_iter().map(|(k, x)| (k, json!(x))).collect();
        assert_eq!(json!([env, secret]), c["out"], "{c}");
    }
    for c in v["labels"].as_array().unwrap() {
        let r: ModelRef = serde_json::from_value(c["ref"].clone()).unwrap();
        let uuids: Vec<String> = c["uuids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|u| u.as_str().unwrap().to_string())
            .collect();
        let ids: Vec<i64> = [0, 1].into_iter().take(uuids.len()).collect();
        // the generator's display was one of three lengths; it is the only label value that varies
        let display_len = c["out"]
            .get("engine-console.model")
            .and_then(Value::as_str)
            .map_or(257, |d| d.chars().count());
        let display = "d".repeat(display_len);
        let got = engine_labels(&LabelArgs {
            instance: "inst_1",
            engine: "fake",
            display: &display,
            r: &r,
            port: 18000,
            name: "n",
            gpu_ids: &ids,
            gpu_uuids: &uuids,
        });
        assert_eq!(Value::Object(got), c["out"], "{c}");
    }
}

#[test]
fn test_run_specs_match_python() {
    let v = vectors();
    for c in v["run_specs"].as_array().unwrap() {
        let mut l = json!({"image": "img:1", "argv": ["--model", "m"], "container_port": 8000, "health_path": "/health",
                           "metrics_path": "/metrics"});
        for (k, x) in obj(&c["launch"]) {
            l[k] = x;
        }
        let launch: LaunchSpec = serde_json::from_value(l).unwrap();
        let labels = obj(&json!({"engine-console.role": "engine"}));
        let env = obj(&json!({"A": "1"}));
        let secret = obj(&json!({"K": "x"}));
        let mounts = vec![obj(
            &json!({"root": "hf", "rel": "hub", "dest": "/m", "ro": true}),
        )];
        let spec = build_run_spec(&RunArgs {
            name: "ec-t-1",
            instance: "inst_1",
            image: "img:1",
            launch: &launch,
            gpu_uuids: &["G0".to_string()],
            labels: &labels,
            env: &env,
            secret_env: &secret,
            mounts: &mounts,
            volumes: &[("ec-vllm-cache".to_string(), "/root/.cache/vllm".to_string())],
            ipc_host: true,
        });
        assert_eq!(Value::Object(spec.clone()), c["spec"], "{}", c["launch"]);
        // the same keys in the same order, as the node receives them
        let want: Vec<&String> = c["spec"].as_object().unwrap().keys().collect();
        assert_eq!(spec.keys().collect::<Vec<_>>(), want);
    }
}

/// Security control: the hub refuses exactly what the node would refuse, with Python's messages.
#[test]
fn test_validation_messages_match_python() {
    let v = vectors();
    for c in v["validate"].as_array().unwrap() {
        let errs = validate_run_spec(&obj(&c["spec"]), c["max_shm_gib"].as_f64());
        assert_eq!(
            json!(errs),
            c["errs"],
            "max_shm_gib={} spec keys={:?}",
            c["max_shm_gib"],
            c["spec"].as_object().map(|o| o.len())
        );
    }
}

#[test]
fn test_text_hygiene_and_derived_ids_match_python() {
    let v = vectors();
    for c in v["text"].as_array().unwrap() {
        let s = c["s"].as_str().unwrap();
        assert_eq!(json!(slug(s)), c["slug"], "{s:?}");
        assert_eq!(json!(display_name(s)), c["display_name"], "{s:?}");
        assert_eq!(json!(clean_text200(s)), c["clean_text"], "{s:?}");
        assert_eq!(json!(clean_text(s, 5)), c["clean_text_5"], "{s:?}");
        assert_eq!(
            json!(external_instance_id("gpu-2", s)),
            c["ext_id"],
            "{s:?}"
        );
        assert_eq!(json!(script_instance_id("local", s)), c["scr_id"], "{s:?}");
    }
}

#[test]
fn test_generic_metrics_match_python() {
    let v = vectors();
    for c in v["generic_metrics"].as_array().unwrap() {
        let got = parse_generic_metrics(c["engine"].as_str().unwrap(), c["text"].as_str().unwrap());
        let finite: Vec<Value> = got
            .iter()
            .filter(|(_, x)| x.is_finite())
            .map(|(k, x)| json!([k, x]))
            .collect();
        let nonfinite: Vec<Value> = got
            .iter()
            .filter(|(_, x)| !x.is_finite())
            .map(|(k, _)| json!(k))
            .collect();
        assert_eq!(json!(finite), c["out"], "{c}");
        assert_eq!(json!(nonfinite), c["nonfinite"], "{c}");
    }
}

// ---- externals: Python's derived instances for the same node state ---------------------------------------------
mod externals_vectors {
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use ec_hub::domain::errors::{NodeError, Obj};
    use ec_hub::domain::fleet::{HostConfig, LinkStatus, NodeClient, NodeStream};
    use ec_hub::services::externals::{HostCtx, build_host_instances};
    use serde_json::{Value, json};

    /// A node that only answers what it last pushed (the builders never call it).
    struct SnapNode {
        host: HostConfig,
        link: LinkStatus,
        latest: HashMap<String, Arc<Obj>>,
    }

    #[async_trait]
    impl NodeClient for SnapNode {
        fn host(&self) -> &HostConfig {
            &self.host
        }
        fn link(&self) -> LinkStatus {
            self.link.clone()
        }
        fn latest(&self, kind: &str) -> Option<Arc<Obj>> {
            self.latest.get(kind).cloned()
        }
        async fn call(&self, op: &str, _args: Option<Obj>, _t: Duration) -> Result<Obj, NodeError> {
            panic!("the builders never call the node ({op})")
        }
        fn open(&self, op: &str, _args: Option<Obj>) -> Box<dyn NodeStream> {
            panic!("the builders never open a stream ({op})")
        }
    }

    #[test]
    fn test_derived_instances_match_python() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/wse_externals_vectors.json"
        ))
        .expect("vectors");
        let cases: Vec<Value> = serde_json::from_str(&text).expect("json");
        assert_eq!(cases.len(), 7);
        for c in cases {
            let host_id = c["host_id"].as_str().unwrap().to_string();
            let mut latest = HashMap::new();
            for kind in ["hello", "sample", "inventory", "disks"] {
                if let Value::Object(o) = &c[kind] {
                    latest.insert(kind.to_string(), Arc::new(o.clone()));
                }
            }
            let node = SnapNode {
                host: HostConfig {
                    id: host_id.clone(),
                    label: c["label"].as_str().unwrap().into(),
                    ssh: None,
                    port: None,
                    ssh_config: false,
                    enabled: true,
                },
                link: serde_json::from_value(c["link"].clone()).unwrap(),
                latest,
            };
            let others = c["others"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(h, v)| {
                    let addrs = v[1]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|a| a.as_str().unwrap().to_string())
                        .collect();
                    (h.clone(), (v[0].as_str().unwrap().to_string(), addrs))
                })
                .collect();
            let ctx = HostCtx {
                host_id: host_id.clone(),
                label: c["label"].as_str().unwrap().into(),
                node: Arc::new(node),
                others,
            };
            let busy: Vec<(String, String)> = c["busy"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect();
            let mut seen = HashMap::new();
            let got = build_host_instances(&ctx, &mut seen, 1000.0, &busy);
            let want = c["instances"].as_array().unwrap();
            assert_eq!(got.len(), want.len(), "{}", c["name"]);
            for (g, w) in got.iter().zip(want) {
                assert_eq!(
                    serde_json::to_value(g).unwrap(),
                    *w,
                    "{} {}",
                    c["name"],
                    w["id"]
                );
            }
            let seen_json: serde_json::Map<String, Value> =
                seen.into_iter().map(|(k, v)| (k, json!(v))).collect();
            assert_eq!(Value::Object(seen_json), c["first_seen"], "{}", c["name"]);
        }
    }
}
