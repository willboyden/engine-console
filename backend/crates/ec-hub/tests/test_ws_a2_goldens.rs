//! Differential goldens of the TensorRT-LLM, llama.cpp and Ollama adapters: the Python adapters' own answers
//! (backend/parity/gen/goldens_ws_a2.py, tests/data/ws_a2/) replayed against the Rust TensorRT-LLM, llama.cpp and
//! Ollama adapters. Thousands of seeded cases per engine: catalogs and presets, validate(), build_launch_ctx() (argv,
//! env order, files, every LaunchSpec field, the error text), compatibility(), memory_model() and host_memory_gib(),
//! the log and metrics parsers, and the Python behaviours under them (json.loads messages, yaml.safe_dump, float(),
//! isdigit/int, strptime).
//!
//! Where Python raises, the Rust `try_*` form must return the same exception kind (addendum A5); nothing is skipped.

#![allow(
    clippy::panic,
    clippy::expect_used,
    clippy::print_stderr,
    clippy::too_many_lines
)]

#[path = "common/a2_kit.rs"]
mod a2;

use a2::{assert_same, floatmap_json, golden, same};
use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchContext, LaunchSpec, ModelInfo, Obj,
};
use ec_hub::adapters::llamacpp::LlamaCppAdapter;
use ec_hub::adapters::ollama::OllamaAdapter;
use ec_hub::adapters::trtllm::{TrtllmAdapter, yaml};
use ec_hub::adapters::trtllm_parse::py::{self, PyResult, PyVal};
use ec_hub::adapters::{llamacpp, ollama, trtllm};
use serde_json::{Value, json};

/// The raising forms of the three adapters (addendum A5), as one interface for the replay.
trait Try {
    fn validate(&self, p: &Obj) -> PyResult<Vec<String>>;
    fn launch(&self, c: &LaunchContext) -> PyResult<LaunchSpec>;
    fn compat(&self, m: &ModelInfo, p: &Obj, h: &Hardware) -> PyResult<Vec<Compat>>;
    fn memory(&self, m: &ModelInfo, p: &Obj) -> PyResult<FloatMap>;
    fn host(&self, m: &ModelInfo, p: &Obj, h: &Hardware) -> PyResult<FloatMap>;
    fn log(&self, line: &str) -> PyResult<Option<Obj>>;
    fn metrics(&self, text: &str) -> PyResult<FloatMap>;
}

macro_rules! impl_try {
    ($t:ty) => {
        impl Try for $t {
            fn validate(&self, p: &Obj) -> PyResult<Vec<String>> {
                self.try_validate(p)
            }
            fn launch(&self, c: &LaunchContext) -> PyResult<LaunchSpec> {
                self.py_build_launch_ctx(c)
            }
            fn compat(&self, m: &ModelInfo, p: &Obj, h: &Hardware) -> PyResult<Vec<Compat>> {
                self.try_compatibility(m, p, h)
            }
            fn memory(&self, m: &ModelInfo, p: &Obj) -> PyResult<FloatMap> {
                self.try_memory_model(m, p)
            }
            fn host(&self, m: &ModelInfo, p: &Obj, h: &Hardware) -> PyResult<FloatMap> {
                self.try_host_memory_gib(m, p, h)
            }
            fn log(&self, line: &str) -> PyResult<Option<Obj>> {
                self.try_parse_startup_log(line)
            }
            fn metrics(&self, text: &str) -> PyResult<FloatMap> {
                self.try_parse_metrics(text)
            }
        }
    };
}
impl_try!(TrtllmAdapter);
impl_try!(LlamaCppAdapter);
impl_try!(OllamaAdapter);

fn tried(id: &str) -> Box<dyn Try> {
    match id {
        "trtllm" => Box::new(TrtllmAdapter::new()),
        "llamacpp" => Box::new(LlamaCppAdapter::new()),
        _ => Box::new(OllamaAdapter::new()),
    }
}

fn engine(id: &str) -> std::sync::Arc<dyn EngineAdapter> {
    match id {
        "trtllm" => trtllm::adapter(),
        "llamacpp" => llamacpp::adapter(),
        _ => ollama::adapter(),
    }
}

fn arr<'a>(v: &'a Value, key: &str) -> &'a Vec<Value> {
    v.get(key)
        .and_then(Value::as_array)
        .unwrap_or_else(|| panic!("no list {key}"))
}

/// Every mismatch of a run, reported together.
#[derive(Default)]
struct Fails(Vec<String>);

impl Fails {
    fn same(&mut self, got: &Value, want: &Value, ordered: bool, what: &str) {
        if !same(got, want, ordered) {
            self.0.push(format!(
                "{what}\n  got:  {}\n  want: {}",
                serde_json::to_string(got).unwrap_or_default(),
                serde_json::to_string(want).unwrap_or_default()
            ));
        }
    }

    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            self.0.push(what());
        }
    }

    fn done(self, what: &str) {
        let n = self.0.len();
        let shown: Vec<&String> = self.0.iter().take(25).collect();
        assert!(
            n == 0,
            "{what}: {n} mismatch(es)\n{}",
            shown
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join("\n")
        );
    }
}

fn raised(v: &Value) -> Option<&str> {
    v.get("$raise").and_then(Value::as_str)
}

// ---------------------------------------------------------------- catalogs
#[test]
fn golden_catalogs_presets_and_identity_match_python() {
    let doc = golden("catalogs.json");
    for want in arr(&doc, "engines") {
        let id = want["id"].as_str().expect("id");
        let a = engine(id);
        assert_eq!(a.display_name(), want["display_name"], "{id}");
        assert_eq!(a.default_image(), want["default_image"], "{id}");
        assert_eq!(Value::from(a.default_port()), want["default_port"], "{id}");
        let mut formats = a.model_formats();
        formats.sort();
        assert_eq!(json!(formats), want["model_formats"], "{id}");
        let by_arch: serde_json::Map<String, Value> = a
            .images_by_arch()
            .into_iter()
            .map(|(k, v)| (k, Value::from(v)))
            .collect();
        assert_eq!(Value::Object(by_arch), want["images_by_arch"], "{id}");
        assert_eq!(Value::from(a.multi_model()), want["multi_model"], "{id}");
        assert_eq!(
            json!(a.metric_prefixes().unwrap_or_default()),
            want["metric_prefixes"],
            "{id}"
        );
        assert!(a.ported(), "{id}");
        let cat = serde_json::to_value(a.param_catalog()).expect("catalog");
        let want_cat = want["catalog"].as_array().expect("catalog");
        assert_eq!(
            cat.as_array().map(Vec::len),
            Some(want_cat.len()),
            "{id}: catalog size"
        );
        for (got, w) in cat.as_array().into_iter().flatten().zip(want_cat) {
            assert_same(got, w, false, &format!("{id}: param {}", w["key"]));
        }
        let presets: Vec<Value> = a
            .presets()
            .into_iter()
            .map(|(n, p)| json!([n, Value::Object(p)]))
            .collect();
        assert_same(
            &json!(presets),
            &want["presets"],
            true,
            &format!("{id}: presets"),
        );
        assert!(
            a.preset_info().is_empty() && want["preset_info"] == json!({}),
            "{id}: preset_info"
        );
        for (arch, img) in want["image_for"].as_object().expect("image_for") {
            assert_eq!(Value::from(a.image_for(arch)), *img, "{id} {arch}");
        }
    }
}

// ---------------------------------------------------------------- per-engine cases
struct Tables {
    models: Vec<ModelInfo>,
    hws: Vec<Hardware>,
}

fn tables(doc: &Value) -> Tables {
    Tables {
        models: arr(doc, "models")
            .iter()
            .map(|m| serde_json::from_value(m.clone()).expect("model"))
            .collect(),
        hws: arr(doc, "hws")
            .iter()
            .map(|h| serde_json::from_value(h.clone()).expect("hw"))
            .collect(),
    }
}

fn idx(v: &Value) -> usize {
    usize::try_from(v.as_u64().expect("index")).expect("index")
}

fn obj_of(v: &Value) -> Obj {
    v.as_object().cloned().expect("params object")
}

/// Compare a `try_*` answer with Python's: the value, or the same exception kind (and, for ValueError, its text when
/// `msg` is asked for).
fn check<T>(
    f: &mut Fails,
    got: PyResult<T>,
    want: &Value,
    ordered: bool,
    msg: bool,
    what: &str,
    json: impl Fn(&T) -> Value,
) {
    match (got, raised(want)) {
        (Ok(v), None) => f.same(&json(&v), want, ordered, what),
        (Err(e), Some(kind)) => f.check(
            e.kind == kind
                && (!msg
                    || kind != "ValueError"
                    || e.msg == want["msg"].as_str().unwrap_or_default()),
            || format!("{what}\n  got:  {e}\n  want: {want}"),
        ),
        (Ok(v), Some(_)) => {
            f.0.push(format!("{what}\n  got:  {}\n  want: {want}", json(&v)));
        }
        (Err(e), None) => f.0.push(format!("{what}\n  got:  {e}\n  want: {want}")),
    }
}

fn run_engine_cases(id: &str) {
    let doc = golden(&format!("{id}_cases.json"));
    let t = tables(&doc);
    let a = engine(id);
    let tr = tried(id);
    let mut f = Fails::default();

    for (i, case) in arr(&doc, "validate").iter().enumerate() {
        let params = obj_of(&case["params"]);
        let what = format!("{id} validate #{i} {}", case["params"]);
        check(
            &mut f,
            tr.validate(&params),
            &case["out"],
            true,
            true,
            &what,
            |v| json!(v),
        );
        if raised(&case["out"]).is_none() {
            f.same(&json!(a.validate(&params)), &case["out"], true, &what);
        }
    }

    for (i, case) in arr(&doc, "launch").iter().enumerate() {
        let mut ctx = case["ctx"].clone();
        ctx["hw"] = serde_json::to_value(&t.hws[idx(&case["hw"])]).expect("hw");
        let ctx: LaunchContext = serde_json::from_value(ctx).expect("ctx");
        let what = format!("{id} launch #{i} {}", case["ctx"]);
        check(
            &mut f,
            tr.launch(&ctx),
            &case["out"],
            true,
            true,
            &what,
            |s| serde_json::to_value(s).expect("spec"),
        );
        // the trait form: a ValueError is the lifecycle's 422 invalid_params with the same text
        if raised(&case["out"]).is_none_or(|k| k == "ValueError") {
            match (a.build_launch_ctx(&ctx), raised(&case["out"])) {
                (Ok(spec), None) => {
                    f.same(
                        &serde_json::to_value(&spec).expect("spec"),
                        &case["out"],
                        true,
                        &what,
                    );
                }
                (Err(problem), Some(_)) => f.check(
                    (problem.status, problem.code.as_str()) == (422, "invalid_params")
                        && problem.detail == case["out"]["msg"].as_str().unwrap_or_default(),
                    || format!("{what}\n  got:  {problem:?}\n  want: {}", case["out"]),
                ),
                (got, _) => {
                    f.0.push(format!("{what}\n  got:  {got:?}\n  want: {}", case["out"]));
                }
            }
        }
    }

    for (i, case) in arr(&doc, "bridge").iter().enumerate() {
        let args = &case["args"];
        let got = a.build_launch(
            args["model"].as_str().unwrap_or_default(),
            &Obj::new(),
            &t.hws[0],
            args["served_name"].as_str().unwrap_or_default(),
            args["hf"].as_str().unwrap_or_default(),
        );
        match (got, raised(&case["out"])) {
            (Ok(spec), None) => f.same(
                &serde_json::to_value(&spec).expect("spec"),
                &case["out"],
                true,
                "bridge",
            ),
            (Err(p), Some(_)) => f.check(
                p.detail == case["out"]["msg"].as_str().unwrap_or_default(),
                || format!("{id} bridge #{i}: {p:?} != {}", case["out"]),
            ),
            (got, _) => f.0.push(format!(
                "{id} bridge #{i}: got {got:?}, want {}",
                case["out"]
            )),
        }
    }

    for (i, case) in arr(&doc, "compat").iter().enumerate() {
        let (m, h) = (&t.models[idx(&case["model"])], &t.hws[idx(&case["hw"])]);
        let params = obj_of(&case["params"]);
        let what = format!("{id} compat #{i} {} {}", case["model"], case["params"]);
        check(
            &mut f,
            tr.compat(m, &params, h),
            &case["out"],
            true,
            false,
            &what,
            |c| serde_json::to_value(c).expect("compat"),
        );
    }

    for (i, case) in arr(&doc, "memory").iter().enumerate() {
        let (m, h) = (&t.models[idx(&case["model"])], &t.hws[idx(&case["hw"])]);
        let params = obj_of(&case["params"]);
        let what = format!("{id} memory #{i} {} {}", case["model"], case["params"]);
        check(
            &mut f,
            tr.memory(m, &params),
            &case["memory_model"],
            true,
            false,
            &what,
            floatmap_json,
        );
        check(
            &mut f,
            tr.host(m, &params, h),
            &case["host_memory_gib"],
            true,
            false,
            &what,
            floatmap_json,
        );
    }
    f.done(id);
}

#[test]
fn golden_trtllm_cases_match_python() {
    run_engine_cases("trtllm");
}

#[test]
fn golden_llamacpp_cases_match_python() {
    run_engine_cases("llamacpp");
}

#[test]
fn golden_ollama_cases_match_python() {
    run_engine_cases("ollama");
}

// ---------------------------------------------------------------- parsers
#[test]
fn golden_log_and_metrics_parsers_match_python() {
    let doc = golden("parse_cases.json");
    let mut f = Fails::default();
    for id in ["trtllm", "llamacpp", "ollama"] {
        let (a, tr) = (engine(id), tried(id));
        for (i, case) in arr(&doc, &format!("{id}_log")).iter().enumerate() {
            let line = case["line"].as_str().expect("line");
            let what = format!("{id} log #{i} {line:?}");
            check(
                &mut f,
                tr.log(line),
                &case["out"],
                true,
                false,
                &what,
                |o| o.clone().map_or(Value::Null, Value::Object),
            );
            if raised(&case["out"]).is_none() {
                let plain = a.parse_startup_log(line).map_or(Value::Null, Value::Object);
                f.same(&plain, &case["out"], true, &what);
            }
        }
    }
    for id in ["trtllm", "llamacpp"] {
        let tr = tried(id);
        for (i, case) in arr(&doc, &format!("{id}_metrics")).iter().enumerate() {
            let text = case["text"].as_str().expect("text");
            let what = format!("{id} metrics #{i} {text:?}");
            check(
                &mut f,
                tr.metrics(text),
                &case["out"],
                true,
                false,
                &what,
                floatmap_json,
            );
        }
    }
    f.done("parsers");
}

// ---------------------------------------------------------------- Python behaviours
/// The golden as Python objects: py::json_loads keeps big ints exact; {"$f": ...} becomes a float.
fn pyvals(name: &str) -> PyVal {
    let text = a2::read("ws_a2", name);
    unmark(py::json_loads(&text).expect("golden parses"))
}

fn unmark(v: PyVal) -> PyVal {
    match v {
        PyVal::Dict(d) => {
            if d.len() == 1 && d[0].0 == "$f" {
                return PyVal::Float(match &d[0].1 {
                    PyVal::Str(s) if s == "nan" => f64::NAN,
                    PyVal::Str(s) if s == "inf" => f64::INFINITY,
                    _ => f64::NEG_INFINITY,
                });
            }
            PyVal::Dict(d.into_iter().map(|(k, x)| (k, unmark(x))).collect())
        }
        PyVal::List(a) => PyVal::List(a.into_iter().map(unmark).collect()),
        other => other,
    }
}

fn pyval_same(a: &PyVal, b: &PyVal) -> bool {
    match (a, b) {
        (PyVal::Float(x), PyVal::Float(y)) => {
            x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan())
        }
        (PyVal::List(x), PyVal::List(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| pyval_same(p, q))
        }
        (PyVal::Dict(x), PyVal::Dict(y)) => {
            x.len() == y.len()
                && x.iter()
                    .zip(y)
                    .all(|((k1, v1), (k2, v2))| k1 == k2 && pyval_same(v1, v2))
        }
        _ => a == b,
    }
}

#[test]
fn golden_json_loads_matches_python_messages_and_values() {
    let doc = pyvals("py_cases.json");
    let PyVal::List(cases) = doc.get("json_loads").cloned().expect("json_loads") else {
        panic!("json_loads")
    };
    let (mut checked, mut surrogates) = (0, 0);
    for case in &cases {
        let Some(PyVal::Str(s)) = case.get("s") else {
            panic!("s")
        };
        let out = case.get("out").expect("out");
        let got = py::json_loads(s).map_err(|e| (e.kind, e.msg));
        if let Some(PyVal::Str(kind)) = out.get("$raise") {
            assert_eq!(got.as_ref().err().map(|e| &e.0), Some(kind), "{s:?}");
        } else if out.get("surrogate").is_some() {
            // D-A2-2: a lone surrogate is a Python str but not a Rust one
            assert!(got.is_err(), "{s:?}");
            surrogates += 1;
        } else if let Some(PyVal::Str(msg)) = out.get("err") {
            assert_eq!(got.as_ref().err().map(|e| &e.1), Some(msg), "{s:?}");
        } else if let Some(want) = out.get("ok") {
            let v = got.unwrap_or_else(|e| panic!("{s:?}: {e:?}"));
            assert!(pyval_same(&v, want), "{s:?}: {v:?} != {want:?}");
            let dumps = out.get("dumps").expect("dumps");
            assert_eq!(
                Some(&PyVal::Str(v.dumps_compact())),
                dumps.get("compact"),
                "{s:?}"
            );
            assert_eq!(
                Some(&PyVal::Bool(v.all_finite())),
                dumps.get("strict_ok"),
                "{s:?}"
            );
        } else {
            panic!("{s:?}: unexpected {out:?}");
        }
        checked += 1;
    }
    assert!(checked > 1000 && surrogates < 10, "{checked} {surrogates}");
}

#[test]
fn golden_yaml_safe_dump_matches_pyyaml() {
    let doc = pyvals("py_cases.json");
    let PyVal::List(cases) = doc.get("yaml").cloned().expect("yaml") else {
        panic!("yaml")
    };
    assert!(cases.len() > 1200);
    for case in &cases {
        let tree = case.get("tree").expect("tree");
        let Some(PyVal::Str(want)) = case.get("text") else {
            panic!("text")
        };
        assert_eq!(&yaml::safe_dump(tree), want, "{}", tree.repr());
    }
}

#[test]
fn golden_float_isdigit_int_and_strptime_match_python() {
    // read with the Python-compatible parser: some recorded ints are thousands of digits long
    let doc = pyvals("py_cases.json");
    let list = |k: &str| match doc.get(k) {
        Some(PyVal::List(a)) => a.clone(),
        other => panic!("{k}: {other:?}"),
    };
    let text = |v: &PyVal, k: &str| match v.get(k) {
        Some(PyVal::Str(s)) => s.clone(),
        other => panic!("{k}: {other:?}"),
    };
    for case in list("float") {
        let s = text(&case, "s");
        let got = py::float(&s).map_or(PyVal::None, PyVal::Float);
        let want = case.get("out").cloned().unwrap_or(PyVal::None);
        let want = match want {
            PyVal::Int(d) => PyVal::Float(d.parse().unwrap_or(f64::NAN)),
            other => other,
        };
        assert!(
            pyval_same(&got, &want),
            "float({s:?}) = {got:?}, want {want:?}"
        );
    }
    for case in list("isdigit") {
        let s = text(&case, "s");
        assert_eq!(
            Some(&PyVal::Bool(py::isdigit(&s))),
            case.get("isdigit"),
            "isdigit({s:?})"
        );
        let got = py::int_str(&s).map_or(PyVal::None, |i| PyVal::Int(i.to_string()));
        assert_eq!(Some(&got), case.get("int"), "int({s:?})");
    }
    let zeros: Vec<PyVal> = list("nd_zeros");
    let ours: Vec<PyVal> = py::ND_ZEROS
        .iter()
        .map(|z| PyVal::int(i128::from(*z)))
        .collect();
    assert_eq!(zeros, ours);
    for case in list("strptime") {
        let s = text(&case, "s");
        let got = py::strptime_us(&s).map_or(PyVal::None, |us| PyVal::int(i128::from(us)));
        assert_eq!(Some(&got), case.get("us"), "strptime({s:?})");
    }
}

#[test]
fn golden_rendered_yaml_is_the_launch_file() {
    // the options file in a launch is the header plus safe_dump of yaml_tree()
    let a = TrtllmAdapter::new();
    let tree = a
        .yaml_tree(&a2::p(json!({"kv_cache_dtype": "fp8"})))
        .expect("tree");
    assert_eq!(
        TrtllmAdapter::render_yaml(&tree),
        "# Generated by Engine Console for `trtllm-serve --extra_llm_api_options`.\n# Edit the launch parameters, not \
         this file.\nenable_iter_perf_stats: true\nkv_cache_config:\n  dtype: fp8\nmax_stats_len: 100\n"
    );
}
