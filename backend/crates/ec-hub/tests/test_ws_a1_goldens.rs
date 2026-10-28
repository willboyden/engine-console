//! Differential replay of the Python vLLM and SGLang adapters: every vector in tests/data/ws_a1/*.json is an
//! input and what the Python reference returned or raised (generator: backend/parity/gen/ws_a1_adapter_goldens.py, seeded
//! property-style draws over the catalogs, models, hardware, Prometheus text and log lines, plus every fixture).
//!
//! Where Python raises an uncaught exception (validate / compatibility / memory_model / host_memory_gib /
//! build_launch_ctx), the port must panic (the HTTP layer answers both with the same 500). Where Python raises inside a
//! parser (its callers catch it), the port returns nothing: an empty metrics map, no log progress (see the adapters'
//! module docs).

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::print_stderr
)]

#[path = "common/ws_a1_kit.rs"]
mod ws_a1_kit;

use std::sync::{Arc, Mutex};

use ec_hub::adapters::base::{EngineAdapter, Hardware, LaunchContext, ModelInfo, Obj};
use ec_hub::adapters::sglang::SglangAdapter;
use ec_hub::adapters::sglang_parsers;
use ec_hub::adapters::vllm::VllmAdapter;
use ec_hub::adapters::vllm_parse::{self, py};
use serde_json::{Map, Value};
use ws_a1_kit::{data_dir, floatmap_matches, golden_float, plain, raises};

fn load(name: &str) -> Value {
    let p = data_dir().join(format!("{name}.json"));
    serde_json::from_str(&std::fs::read_to_string(&p).expect("golden file")).expect("golden json")
}

fn as_obj(v: &Value) -> Obj {
    v.as_object().cloned().unwrap_or_default()
}

fn model_of(v: &Value) -> ModelInfo {
    serde_json::from_value(plain(v)).expect("ModelInfo")
}

fn hw_of(v: &Value) -> Hardware {
    serde_json::from_value(plain(v)).expect("Hardware")
}

/// Collect failures instead of stopping at the first one, so one run shows every divergence.
#[derive(Default)]
struct Report {
    fails: Vec<String>,
    checked: usize,
}

impl Report {
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        self.checked += 1;
        if !ok && self.fails.len() < 40 {
            self.fails.push(what());
        } else if !ok {
            self.fails.push(String::new());
        }
    }

    fn finish(self, name: &str) {
        assert!(self.checked > 0, "{name}: nothing checked");
        let shown: Vec<&String> = self.fails.iter().filter(|f| !f.is_empty()).collect();
        assert!(
            self.fails.is_empty(),
            "{name}: {} of {} vectors differ from Python:\n{}",
            self.fails.len(),
            self.checked,
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n---\n")
        );
    }
}

fn identity_and_catalog(a: &dyn EngineAdapter, g: &Value, r: &mut Report) {
    let id = &g["identity"];
    r.check(id["id"] == a.id(), || "id".into());
    r.check(id["display_name"] == a.display_name(), || {
        "display_name".into()
    });
    r.check(id["default_image"] == a.default_image(), || {
        "default_image".into()
    });
    r.check(id["default_port"] == a.default_port(), || {
        "default_port".into()
    });
    let mut f = a.model_formats();
    f.sort();
    r.check(id["model_formats"] == serde_json::json!(f), || {
        "model_formats".into()
    });
    r.check(
        id["images_by_arch"] == Value::Object(Map::new()) && a.images_by_arch().is_empty(),
        || "images_by_arch".into(),
    );
    r.check(id["multi_model"] == a.multi_model(), || {
        "multi_model".into()
    });
    r.check(
        id["metric_prefixes"] == serde_json::json!(a.metric_prefixes().unwrap_or_default()),
        || "metric_prefixes".into(),
    );
    r.check(
        id["preset_info"] == Value::Object(Map::new()) && a.preset_info().is_empty(),
        || "preset_info".into(),
    );
    for (arch, img) in id["image_for"].as_object().unwrap() {
        r.check(*img == a.image_for(arch), || format!("image_for {arch}"));
    }
    let cat = serde_json::to_value(a.param_catalog()).unwrap();
    let want = &g["catalog"];
    r.check(
        cat.as_array().map(Vec::len) == want.as_array().map(Vec::len),
        || "catalog length".into(),
    );
    for (i, (x, y)) in cat
        .as_array()
        .unwrap()
        .iter()
        .zip(want.as_array().unwrap())
        .enumerate()
    {
        r.check(x == y, || format!("catalog[{i}]\n rust {x}\n python {y}"));
    }
    let presets: Vec<Value> = a
        .presets()
        .into_iter()
        .map(|(n, p)| serde_json::json!([n, p]))
        .collect();
    r.check(Value::Array(presets.clone()) == g["presets"], || {
        format!("presets {presets:?}")
    });
}

fn validate_vectors(a: &dyn EngineAdapter, g: &Value, r: &mut Report) {
    for row in g["validate"].as_array().unwrap() {
        let p = as_obj(&row["params"]);
        if row.get("raises").is_some() {
            r.check(raises(|| a.validate(&p)), || {
                format!("validate should raise: {p:?}")
            });
        } else {
            let got = a.validate(&p);
            r.check(serde_json::json!(got) == row["errors"], || {
                format!("validate {p:?}\n rust {got:?}\n python {}", row["errors"])
            });
        }
    }
}

fn launch_vectors(a: &dyn EngineAdapter, g: &Value, r: &mut Report) {
    for row in g["launch"].as_array().unwrap() {
        let ctx: LaunchContext = serde_json::from_value(plain(&row["ctx"])).expect("LaunchContext");
        if row.get("raises").is_some() {
            r.check(raises(|| a.build_launch_ctx(&ctx)), || {
                format!("launch should raise: {:?}", ctx.params)
            });
            continue;
        }
        match a.build_launch_ctx(&ctx) {
            Ok(spec) => {
                let got = serde_json::to_value(&spec).unwrap();
                r.check(row.get("spec") == Some(&got), || {
                    format!(
                        "launch {:?}\n rust {got}\n python {}",
                        ctx.params,
                        row.get("spec").unwrap_or(&row["error"])
                    )
                });
            }
            Err(p) => {
                r.check(
                    row.get("error").and_then(Value::as_str) == Some(p.detail.as_str())
                        && p.status == 422
                        && p.code == "invalid_params",
                    || {
                        format!(
                            "launch {:?}\n rust error {p}\n python {}",
                            ctx.params,
                            row.get("spec").unwrap_or(&row["error"])
                        )
                    },
                );
            }
        }
    }
}

fn compat_vectors(a: &dyn EngineAdapter, g: &Value, r: &mut Report) {
    let rows = g["compat"].as_array().unwrap();
    for row in rows {
        let (m, p, hw) = (
            model_of(&row["model"]),
            as_obj(&row["params"]),
            hw_of(&row["hw"]),
        );
        if row.get("raises").is_some() {
            r.check(raises(|| a.compatibility(&m, &p, &hw)), || {
                format!("compat should raise: {p:?}")
            });
            continue;
        }
        let got = serde_json::to_value(a.compatibility(&m, &p, &hw)).unwrap();
        r.check(got == row["out"], || {
            format!(
                "compat {} {p:?} cc={:?}\n rust {got}\n python {}",
                m.repo_id, hw.compute_capability, row["out"]
            )
        });
    }
    for (section, host) in [("memory", false), ("host_memory", true)] {
        for row in g[section].as_array().unwrap() {
            let src = &rows[usize::try_from(row["i"].as_u64().unwrap()).unwrap()];
            let (m, p, hw) = (
                model_of(&src["model"]),
                as_obj(&src["params"]),
                hw_of(&src["hw"]),
            );
            let run = || {
                if host {
                    a.host_memory_gib(&m, &p, &hw)
                } else {
                    a.memory_model(&m, &p)
                }
            };
            if row.get("raises").is_some() {
                r.check(raises(run), || format!("{section} should raise: {p:?}"));
                continue;
            }
            let got = run();
            let res = floatmap_matches(&got, &row["out"]);
            r.check(res.is_ok(), || {
                format!("{section} {p:?}\n{}", res.clone().err().unwrap_or_default())
            });
        }
    }
}

fn metrics_vectors(make: &dyn Fn() -> Box<dyn EngineAdapter>, g: &Value, r: &mut Report) {
    for row in g["metrics"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let got = make().parse_metrics(text);
        if row.get("raises").is_some() {
            // Python raises (its caller drops the scrape); the port reports nothing
            r.check(got.is_empty(), || {
                format!("metrics should be empty (Python raised): {text:?}")
            });
            continue;
        }
        let res = floatmap_matches(&got, &row["out"]);
        r.check(res.is_ok(), || {
            format!(
                "metrics {text:?}\n{}",
                res.clone().err().unwrap_or_default()
            )
        });
    }
}

fn log_vectors(a: &dyn EngineAdapter, g: &Value, r: &mut Report) {
    for row in g["log"].as_array().unwrap() {
        let line = row["line"].as_str().unwrap();
        let got = a.parse_startup_log(line);
        if row.get("raises").is_some() || row.get("nonfinite").is_some() {
            r.check(got.is_none(), || {
                format!("log should give None (Python raised / non-finite): {line:?}")
            });
            continue;
        }
        let got_v = got.map_or(Value::Null, Value::Object);
        r.check(got_v == row["out"], || {
            format!("log {line:?}\n rust {got_v}\n python {}", row["out"])
        });
    }
}

#[test]
fn test_vllm_matches_python_goldens() {
    let g = load("vllm");
    let a = VllmAdapter::new();
    let mut r = Report::default();
    identity_and_catalog(&a, &g, &mut r);
    validate_vectors(&a, &g, &mut r);
    launch_vectors(&a, &g, &mut r);
    compat_vectors(&a, &g, &mut r);
    metrics_vectors(&|| Box::new(VllmAdapter::new()), &g, &mut r);
    log_vectors(&a, &g, &mut r);
    // rate state across scrapes, with the test's clock
    for case in g["rates"].as_array().unwrap() {
        let now = Arc::new(Mutex::new(0.0_f64));
        let clock = now.clone();
        let a = VllmAdapter::with_clock(Arc::new(move || *clock.lock().unwrap()));
        for step in case["steps"].as_array().unwrap() {
            *now.lock().unwrap() = step["t"].as_f64().unwrap();
            let text = step["text"].as_str().unwrap();
            let got = a.parse_metrics(text);
            if step.get("raises").is_some() {
                r.check(got.is_empty(), || {
                    format!("rates should be empty: {text:?}")
                });
            } else {
                let res = floatmap_matches(&got, &step["out"]);
                r.check(res.is_ok(), || {
                    format!(
                        "rates t={} {text:?}\n{}",
                        step["t"],
                        res.clone().err().unwrap_or_default()
                    )
                });
            }
        }
    }
    r.finish("vllm");
}

#[test]
fn test_sglang_matches_python_goldens() {
    let g = load("sglang");
    let a = SglangAdapter::new();
    let mut r = Report::default();
    identity_and_catalog(&a, &g, &mut r);
    validate_vectors(&a, &g, &mut r);
    launch_vectors(&a, &g, &mut r);
    compat_vectors(&a, &g, &mut r);
    metrics_vectors(&|| Box::new(SglangAdapter::new()), &g, &mut r);
    log_vectors(&a, &g, &mut r);
    r.finish("sglang");
}

#[test]
fn test_python_semantics_match_goldens() {
    let g = load("py");
    let mut r = Report::default();
    for row in g["sum"].as_array().unwrap() {
        let vals: Vec<f64> = row[0]
            .as_array()
            .unwrap()
            .iter()
            .map(golden_float)
            .collect();
        let want = golden_float(&row[1]);
        let got = py::sum(&vals);
        r.check(
            got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
            || format!("sum {vals:?}: rust {got:?} python {want:?}"),
        );
    }
    for row in g["float"].as_array().unwrap() {
        let s = row[0].as_str().unwrap();
        let got = py::float_str(s);
        let want = row[1]
            .as_str()
            .map(|x| golden_float(&Value::String(x.into())));
        r.check(
            match (got, want) {
                (Some(a), Some(b)) => a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()),
                (None, None) => true,
                _ => false,
            },
            || format!("float({s:?}): rust {got:?} python {want:?}"),
        );
    }
    for row in g["int"].as_array().unwrap() {
        let s = row[0].as_str().unwrap();
        let got = py::int_str(s).map(|b| b.0);
        let want = row[1].as_str().map(str::to_string);
        r.check(got == want, || {
            format!("int({s:?}): rust {got:?} python {want:?}")
        });
    }
    for row in g["loads"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        match py::JsonDoc::loads(text) {
            Ok(doc) => r.check(
                row["dumps"].as_str() == Some(doc.dumps_compact().as_str())
                    && row["container"].as_bool() == Some(doc.is_container()),
                || format!("loads {text:?}: rust {} python {row}", doc.dumps_compact()),
            ),
            Err(e) => r.check(row["msg"].as_str() == Some(e.0), || {
                format!("loads {text:?}: rust {e:?} python {row}")
            }),
        }
    }
    for row in g["round"].as_array().unwrap() {
        let x = golden_float(&row[0]);
        let got = [
            py::repr(py::round(x, 4)),
            py::fixed(x, 1),
            py::fixed(x, 2),
            py::fmt_g(x),
        ];
        let want: Vec<&str> = row.as_array().unwrap()[1..]
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        r.check(
            got.iter().map(String::as_str).collect::<Vec<_>>() == want,
            || format!("round/format {x:?}: rust {got:?} python {want:?}"),
        );
    }
    for row in g["sorted"].as_array().unwrap() {
        let items: Vec<(f64, usize)> = row["keys"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, k)| (golden_float(k), i))
            .collect();
        let got: Vec<usize> = py::sorted_by_float_key(&items)
            .into_iter()
            .map(|(_, i)| i)
            .collect();
        let want: Vec<usize> = row["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        r.check(got == want, || {
            format!("sorted {items:?}: rust {got:?} python {want:?}")
        });
    }
    for row in g["parse_samples"].as_array().unwrap() {
        let text = row["text"].as_str().unwrap();
        let got: Vec<Value> = vllm_parse::parse_samples(text)
            .into_iter()
            .map(|(n, l, v)| {
                let labels: Map<String, Value> =
                    l.into_iter().map(|(k, x)| (k, Value::String(x))).collect();
                serde_json::json!([n, labels, py::repr(v)])
            })
            .collect();
        r.check(Value::Array(got.clone()) == row["out"], || {
            format!(
                "parse_samples {text:?}\n rust {got:?}\n python {}",
                row["out"]
            )
        });
    }
    for row in g["quantile"].as_array().unwrap() {
        let q = row["q"].as_f64().unwrap();
        let b: Vec<(f64, f64)> = row["buckets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (golden_float(&p[0]), golden_float(&p[1])))
            .collect();
        let v = vllm_parse::histogram_quantile(q, &b).map(py::repr);
        let s = sglang_parsers::histogram_quantile(q, &b).map(py::repr);
        r.check(
            row["vllm"].as_str() == v.as_deref() && row["sglang"].as_str() == s.as_deref(),
            || {
                format!(
                    "quantile {q} {b:?}: rust {v:?}/{s:?} python {}/{}",
                    row["vllm"], row["sglang"]
                )
            },
        );
    }
    r.finish("py");
}
