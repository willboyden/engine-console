//! Port of backend/tests/llamacpp/test_acceptance.py (with acceptance_cases.py): the adapter's launch lines against the
//! LIVE parser check of the pinned image (tests/data/llamacpp/parser_acceptance_b10156.json, made by live_check.py: no
//! model, no GPU, no network, a model path that does not exist). The Rust adapter must build exactly the recorded argv.

use std::collections::BTreeSet;

use crate::a2::{llama, read_json};
use ec_hub::adapters::base::{EngineAdapter, LaunchContext, Obj, ParamSpec};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::llamacpp_params::{DEFAULT_IMAGE, NEGATABLE_KEYS};
use ec_hub::adapters::trtllm_parse::py::PyVal;
use serde_json::{Value, json};

const MODEL_PATH: &str = "/models/does-not-exist.gguf";
const TEMPLATE_TEXT: &str = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}\n{% endfor %}";
const RERUN: &str = "the launch the adapter builds changed since the live parser check: run tests/llamacpp/live_check.py \
                     of the Python reference (backend/parity/lib/python-release.mjs) with its backend venv";

/// Negative controls: each must be REJECTED by the parser (only the names are compared here).
const NEGATIVE_CASES: &[&str] = &[
    "unknown flag",
    "bad enum --fit",
    "bad enum --cache-type-k",
    "bad enum --split-mode",
    "bad enum --flash-attn",
    "bad enum --spec-type",
    "missing value",
    "bad enum at verbosity 0",
];

fn record() -> Value {
    read_json("llamacpp", "parser_acceptance_b10156.json")
}

fn string_values(key: &str) -> Vec<Value> {
    let v: Vec<&str> = match key {
        "n_gpu_layers" | "spec_draft_ngl" => vec!["all", "auto", "12"],
        "tensor_split" => vec!["3,1"],
        "device" => vec!["none"],
        "override_tensor" => vec!["exps=CPU"],
        "override_kv" => vec!["tokenizer.ggml.add_bos_token=bool:false"],
        "mmproj" => vec!["/models/mmproj.gguf"],
        "lora" => vec!["/models/a.gguf,/models/b.gguf"],
        "lora_scaled" => vec!["/models/a.gguf:0.5"],
        "spec_draft_model" => vec!["/models/draft.gguf"],
        "chat_template" => vec!["chatml", "llama3"],
        "chat_template_jinja" => vec![TEMPLATE_TEXT],
        "cors_origins" => vec!["localhost", "*", "https://chat.example.org"],
        "api_key" => vec!["probe-key-not-a-secret"],
        other => panic!("KeyError: {other}"),
    };
    v.into_iter().map(Value::from).collect()
}

/// `values_for`: all choices, both bool states, the default and the minimum. `ParamSpec.min` is a float in Python
/// (`float | None`), which is why a minimum shows as `0.0` in a case name.
fn values_for(spec: &ParamSpec) -> Vec<Value> {
    match spec.type_.as_str() {
        "bool" if NEGATABLE_KEYS.contains(&spec.key.as_str()) => vec![json!(true), json!(false)],
        "bool" => vec![json!(true)],
        "enum" => spec
            .choices
            .clone()
            .unwrap_or_default()
            .into_iter()
            .map(Value::from)
            .collect(),
        "string" => string_values(&spec.key),
        "string_list" => match spec.key.as_str() {
            "served_model_name" => vec![json!(["extra-a", "extra-b"])],
            other => panic!("KeyError: {other}"),
        },
        "json" => match spec.key.as_str() {
            "chat_template_kwargs" => vec![json!({"enable_thinking": false})],
            other => panic!("KeyError: {other}"),
        },
        _ => {
            let base = if !spec.default.is_null() {
                spec.default.clone()
            } else if let Some(m) = spec.min {
                json!(m)
            } else {
                json!(1)
            };
            let mut out = vec![base.clone()];
            if let Some(m) = spec.min
                && base.as_f64() != Some(m)
            {
                #[allow(clippy::cast_possible_truncation)]
                out.push(if spec.type_ == "int" {
                    json!(m as i64)
                } else {
                    json!(m)
                });
            }
            out
        }
    }
}

/// `cases(adapter)`: name, argv, env and file names of every case, exactly as the adapter builds them today.
fn cases(adapter: &LlamaCppAdapter) -> Vec<Value> {
    let hw = llama::hw();
    let mut out = Vec::new();
    let mut add = |name: String, params: Obj| {
        let ctx = LaunchContext {
            model: "m".into(),
            model_format: "gguf".into(),
            model_path: Some(MODEL_PATH.into()),
            served_name: "demo".into(),
            hf_cache_container_path: "/hf".into(),
            params,
            hw: hw.clone(),
        };
        let spec = adapter
            .py_build_launch_ctx(&ctx)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut files: Vec<String> = spec.files.keys().cloned().collect();
        files.sort();
        out.push(json!({"name": name, "argv": spec.argv, "env": spec.env, "files": files}));
    };
    add("baseline".into(), Obj::new());
    for spec in adapter.param_catalog() {
        for v in values_for(&spec) {
            let mut params = Obj::new();
            params.insert(spec.key.clone(), v.clone());
            add(
                format!("{}={}", spec.key, PyVal::from_json(&v).repr()),
                params,
            );
        }
    }
    for (name, params) in adapter.presets() {
        add(format!("preset:{name}"), params);
    }
    out
}

#[test]
fn test_the_recorded_run_used_the_pinned_image_llamacpp() {
    assert_eq!(record()["image"], DEFAULT_IMAGE);
}

#[test]
fn test_every_launch_line_the_adapter_builds_was_accepted_by_the_pinned_parser_llamacpp() {
    let rec = record();
    let now = cases(&LlamaCppAdapter::new());
    let pick = |c: &Value| json!([c["name"], c["argv"], c["env"], c["files"]]);
    let recorded: Vec<Value> = rec["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(pick)
        .collect();
    let built: Vec<Value> = now.iter().map(pick).collect();
    assert_eq!(built.len(), recorded.len(), "{RERUN}");
    for (b, r) in built.iter().zip(&recorded) {
        assert_eq!(b, r, "{RERUN}");
    }
    let refused: Vec<&Value> = rec["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|c| c["accepted"] != true)
        .map(|c| &c["name"])
        .collect();
    assert!(refused.is_empty(), "{refused:?}");
}

#[test]
fn test_the_check_covers_the_whole_catalog_llamacpp() {
    let a = LlamaCppAdapter::new();
    let rec = record();
    let names: BTreeSet<String> = rec["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["name"].as_str().expect("name").to_string())
        .collect();
    let covered: BTreeSet<String> = names
        .iter()
        .filter_map(|n| n.split_once('=').map(|(k, _)| k.to_string()))
        .collect();
    let keys: BTreeSet<String> = a.param_catalog().into_iter().map(|p| p.key).collect();
    assert_eq!(
        keys,
        covered,
        "{:?}",
        keys.symmetric_difference(&covered).collect::<Vec<_>>()
    );
    for p in a.param_catalog() {
        for choice in p.choices.clone().unwrap_or_default() {
            // every enum choice reached the parser
            let name = format!("{}={}", p.key, PyVal::Str(choice.clone()).repr());
            assert!(names.contains(&name), "{} {choice}", p.key);
        }
    }
    for (n, _) in a.presets() {
        assert!(names.contains(&format!("preset:{n}")), "{n}");
    }
}

#[test]
fn test_the_check_can_tell_a_rejection_from_an_acceptance_llamacpp() {
    let rec = record();
    let controls = rec["negative_controls"].as_array().expect("controls");
    assert!(
        controls
            .iter()
            .map(|c| c["name"].as_str().expect("name"))
            .eq(NEGATIVE_CASES.iter().copied())
    );
    assert!(controls.iter().all(|c| c["rejected"] == true));
    let line = |name: &str| {
        controls
            .iter()
            .find(|c| c["name"] == name)
            .expect("control")["error_line"]
            .as_str()
            .expect("line")
            .to_string()
    };
    assert_eq!(
        line("unknown flag"),
        "error: invalid argument: --no-such-flag"
    );
    assert!(line("bad enum --fit").contains("unknown value for --fit: 'maybe'"));
    assert!(line("bad enum --cache-type-k").contains("Unsupported cache type: fp7"));
    // errors are printed at verbosity 0 too
    assert_eq!(line("bad enum at verbosity 0"), line("bad enum --fit"));
}

#[test]
fn test_the_one_silenced_case_is_marked_llamacpp() {
    let rec = record();
    let silenced: Vec<&Value> = rec["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .filter(|c| {
            c.get("note")
                .is_some_and(|n| !n.is_null() && n != "" && n != false)
        })
        .collect();
    assert!(
        silenced
            .iter()
            .map(|c| c["name"].as_str().expect("name"))
            .eq(["log_verbosity=0"])
    );
    assert_eq!(silenced[0]["accepted"], true);
}

#[test]
fn test_secrets_in_the_recorded_env_are_the_test_value_only_llamacpp() {
    for c in record()["cases"].as_array().expect("cases") {
        let env = c["env"].as_object().expect("env");
        assert!(env.keys().all(|k| k == "LLAMA_API_KEY"));
        assert_eq!(
            env.get("LLAMA_API_KEY")
                .and_then(Value::as_str)
                .unwrap_or("probe-key-not-a-secret"),
            "probe-key-not-a-secret"
        );
    }
}
