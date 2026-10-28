//! Python exceptions raised by adapters (PORTING.md §A5). Rust-native: the Python behaviour pinned here
//! is "an uncaught exception is the catch-all 500 on an HTTP path, and a value the caller can catch everywhere else".
//! `pycompat::catch` turns a raised exception into `Err` and lets any other panic through; the `try_*` defaults of
//! `EngineAdapter` use it; an exception on an HTTP path is the generic 500 and the server keeps serving.

#![allow(clippy::panic, clippy::expect_used, clippy::unwrap_used)]

#[path = "common/kit.rs"]
mod kit;

use std::sync::Arc;

use ec_hub::adapters::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use ec_hub::adapters::pycompat::{self, PyExc, catch, exc, or_raise, raise};
use ec_hub::domain::errors::Problem;
use kit::{FakeAdapter, TestEnv};

const P: &str = "/api/v1";

/// The fake engine, raising a TypeError from the methods named in `raises` (as a Python adapter would on bad input).
struct Raising {
    raises: &'static [&'static str],
}

impl Raising {
    fn check(&self, method: &str) {
        if self.raises.contains(&method) {
            raise(
                "TypeError",
                &format!("{method}: unsupported operand type(s)"),
            );
        }
    }
}

impl EngineAdapter for Raising {
    fn id(&self) -> &str {
        "raising"
    }
    fn display_name(&self) -> &str {
        "Raising"
    }
    fn default_image(&self) -> &str {
        FakeAdapter.default_image()
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        self.check("param_catalog");
        FakeAdapter.param_catalog()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        FakeAdapter.presets()
    }
    fn validate(&self, params: &Obj) -> Vec<String> {
        self.check("validate");
        FakeAdapter.validate(params)
    }
    fn build_launch(
        &self,
        model: &str,
        params: &Obj,
        hw: &Hardware,
        served_name: &str,
        hf_cache_container_path: &str,
    ) -> Result<LaunchSpec, Problem> {
        self.check("build_launch");
        FakeAdapter.build_launch(model, params, hw, served_name, hf_cache_container_path)
    }
    fn compatibility(&self, model: &ModelInfo, params: &Obj, hw: &Hardware) -> Vec<Compat> {
        self.check("compatibility");
        FakeAdapter.compatibility(model, params, hw)
    }
    fn memory_model(&self, model: &ModelInfo, params: &Obj) -> FloatMap {
        self.check("memory_model");
        FakeAdapter.memory_model(model, params)
    }
    fn parse_metrics(&self, prometheus_text: &str) -> FloatMap {
        self.check("parse_metrics");
        FakeAdapter.parse_metrics(prometheus_text)
    }
    fn parse_startup_log(&self, line: &str) -> Option<Obj> {
        self.check("parse_startup_log");
        FakeAdapter.parse_startup_log(line)
    }
}

fn hardware() -> Hardware {
    serde_json::from_value(serde_json::json!({
        "gpu_ids": [0], "gpu_uuids": ["GPU-0"], "gpu_names": ["Test GPU"], "gpu_total_gib": [96.0],
        "gpu_free_gib": [90.0], "compute_capability": "12.0", "host_ram_gib": 256.0
    }))
    .expect("hardware")
}

fn model() -> ModelInfo {
    serde_json::from_value(serde_json::json!({"repo_id": "org/m"})).unwrap_or_default()
}

#[test]
fn test_catch_turns_a_raised_exception_into_err() {
    pycompat::install_quiet_panic_hook();
    assert_eq!(catch(|| 41 + 1), Ok(42));
    let e: Result<i32, PyExc> = catch(|| raise("ZeroDivisionError", "integer modulo by zero"));
    assert_eq!(e, Err(exc("ZeroDivisionError", "integer modulo by zero")));
    assert_eq!(
        e.unwrap_err().to_string(),
        "ZeroDivisionError: integer modulo by zero"
    );
}

#[test]
fn test_catch_lets_any_other_panic_through() {
    pycompat::install_quiet_panic_hook();
    // a genuine bug is not a Python exception: catch must not swallow it
    let outer = std::panic::catch_unwind(|| catch(|| -> i32 { panic!("a real bug") }));
    let payload = outer.expect_err("the bug propagates");
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"a real bug"));
}

#[test]
fn test_or_raise_round_trips_through_catch() {
    pycompat::install_quiet_panic_hook();
    assert_eq!(or_raise(Ok::<_, PyExc>(7)), 7);
    let back = catch(|| or_raise::<i32>(Err(exc("ValueError", "bad"))));
    assert_eq!(back, Err(exc("ValueError", "bad")));
}

#[test]
fn test_the_try_forms_return_a_raised_exception_as_err() {
    pycompat::install_quiet_panic_hook();
    let a = Raising {
        raises: &[
            "validate",
            "compatibility",
            "memory_model",
            "parse_metrics",
            "parse_startup_log",
            "build_launch",
        ],
    };
    let (m, params, hw) = (model(), Obj::new(), hardware());
    let kind = |r: Result<(), PyExc>| r.unwrap_err().kind;
    assert_eq!(kind(a.try_validate(&params).map(|_| ())), "TypeError");
    assert_eq!(
        kind(a.try_compatibility(&m, &params, &hw).map(|_| ())),
        "TypeError"
    );
    assert_eq!(
        kind(a.try_memory_model(&m, &params).map(|_| ())),
        "TypeError"
    );
    assert_eq!(kind(a.try_parse_metrics("x 1").map(|_| ())), "TypeError");
    assert_eq!(
        kind(a.try_parse_startup_log("line").map(|_| ())),
        "TypeError"
    );
    // host_memory_gib does not raise here (the default): Ok, even though other methods do
    assert!(a.try_host_memory_gib(&m, &params, &hw).is_ok());
}

#[test]
fn test_the_try_forms_equal_the_plain_methods_when_nothing_raises() {
    pycompat::install_quiet_panic_hook();
    let a = Raising { raises: &[] };
    let (m, params, hw) = (model(), Obj::new(), hardware());
    assert_eq!(a.try_validate(&params).unwrap(), a.validate(&params));
    assert_eq!(
        a.try_compatibility(&m, &params, &hw).unwrap(),
        a.compatibility(&m, &params, &hw)
    );
    assert_eq!(
        a.try_memory_model(&m, &params).unwrap(),
        a.memory_model(&m, &params)
    );
    assert_eq!(a.try_parse_metrics("x 1").unwrap(), a.parse_metrics("x 1"));
}

#[tokio::test]
async fn test_an_exception_on_an_http_path_is_the_generic_500_and_the_server_carries_on() {
    pycompat::install_quiet_panic_hook();
    let mut env = TestEnv::new();
    env.adapters = vec![
        Arc::new(FakeAdapter),
        Arc::new(Raising {
            raises: &["param_catalog"],
        }),
    ];
    let r = env.get(&format!("{P}/engines/raising/params")).send().await;
    assert_eq!(r.status, 500, "{}", r.text());
    assert_eq!(r.code(), "internal_error");
    assert!(!r.text().contains("TypeError"), "{}", r.text()); // the exception stays in the log
    // the next request is served normally
    let ok = env.get(&format!("{P}/engines/fake/params")).send().await;
    assert_eq!(ok.status, 200, "{}", ok.text());
}

#[test]
fn test_every_adapter_raises_the_one_shared_exception_type() {
    // The first port of the TensorRT-LLM, llama.cpp and Ollama adapters shipped its own PyExc: `catch` could not
    // recognise it and re-raised it as a bug, which would have killed a poller on the first exception of one of those
    // engines. One type for every adapter (§A5).
    pycompat::install_quiet_panic_hook();
    let e: PyExc = ec_hub::adapters::trtllm_parse::py::exc("TypeError", "boom");
    assert_eq!(e, exc("TypeError", "boom"));
    let caught: Result<(), PyExc> =
        catch(|| ec_hub::adapters::trtllm_parse::py::raise("TypeError", "boom"));
    assert_eq!(caught, Err(exc("TypeError", "boom")));
}
