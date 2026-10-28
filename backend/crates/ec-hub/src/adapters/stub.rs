//! Placeholder adapter: the real identity of an engine (id, name, pinned image, port, formats), everything else empty
//! or `not_ported`. No engine uses it since every adapter is ported; the routes answer 501 for an engine that would.

use super::base::{
    Compat, EngineAdapter, FloatMap, Hardware, LaunchSpec, ModelInfo, Obj, ParamSpec,
};
use crate::domain::errors::Problem;

pub struct StubAdapter {
    pub id: &'static str,
    pub display_name: &'static str,
    pub default_image: &'static str,
    pub default_port: i64,
    pub model_formats: &'static [&'static str],
    pub multi_model: bool,
    pub metric_prefixes: Option<&'static [&'static str]>,
    /// python module named in the 501 detail
    pub module: &'static str,
}

impl EngineAdapter for StubAdapter {
    fn id(&self) -> &str {
        self.id
    }
    fn display_name(&self) -> &str {
        self.display_name
    }
    fn default_image(&self) -> &str {
        self.default_image
    }
    fn default_port(&self) -> i64 {
        self.default_port
    }
    fn model_formats(&self) -> Vec<String> {
        self.model_formats
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
    fn multi_model(&self) -> bool {
        self.multi_model
    }
    fn ported(&self) -> bool {
        false
    }
    fn metric_prefixes(&self) -> Option<Vec<String>> {
        self.metric_prefixes
            .map(|p| p.iter().map(|s| (*s).to_string()).collect())
    }
    fn param_catalog(&self) -> Vec<ParamSpec> {
        Vec::new()
    }
    fn presets(&self) -> Vec<(String, Obj)> {
        Vec::new()
    }
    fn validate(&self, _params: &Obj) -> Vec<String> {
        Vec::new()
    }
    fn build_launch(
        &self,
        _m: &str,
        _p: &Obj,
        _hw: &Hardware,
        _s: &str,
        _h: &str,
    ) -> Result<LaunchSpec, Problem> {
        Err(Problem::new(
            501,
            "not_ported",
            format!("not ported: {}", self.module),
        ))
    }
    fn compatibility(&self, _m: &ModelInfo, _p: &Obj, _hw: &Hardware) -> Vec<Compat> {
        Vec::new()
    }
    fn memory_model(&self, _m: &ModelInfo, _p: &Obj) -> FloatMap {
        Vec::new()
    }
    fn parse_metrics(&self, _text: &str) -> FloatMap {
        Vec::new()
    }
    fn parse_startup_log(&self, _line: &str) -> Option<Obj> {
        None
    }
}
