//! Python: backend/src/engine_console/services/fitting.py
//! (tests/core/test_fit*.py with services/fit.rs).
//!
//! Glue between the pure estimator (services/fit.rs) and the world: adapter memory model + compat, per-host
//! hardware, resident engines; unified-memory hosts budget one pool. The routes, the lifecycle and the
//! /models route use it.
//!
//! Adapter exceptions (PORTING.md §A5): the adapter calls use the `try_*` forms, and an exception becomes the
//! generic 500 problem Python's catch-all answers. That is what an HTTP caller of `/fit` sees either way, and it lets
//! the non-HTTP-style callers react as Python's `except` clauses do: the lifecycle's `_resolve_gpus` falls back to the
//! requested GPUs, and the `/models` listing skips the engine ("one odd model must not blank the list").
//!
//! Host-aware (contract v2): the estimate is made against ONE host's GPUs and RAM. A unified-memory host (the GPU's
//! memory is the host's RAM) has a single pool, so the RAM the engine process itself needs is taken off the GPU
//! budget instead of being counted as a second, separate pool.

use crate::sync::RecoverRw;
use std::sync::{Arc, RwLock};

use serde_json::Value;

use crate::adapters::AdapterRegistry;
use crate::adapters::base::{Hardware, ModelInfo};
use crate::adapters::pycompat::PyExc;
use crate::domain::errors::{AppError, AppResult, Obj, Problem, internal, unprocessable};
use crate::domain::models::{FitReport, FitRequest, ResidentUse};
use crate::domain::refs::ModelRef;
use crate::services::fit::{Mem, estimate_fit, estimate_host_ram};
use crate::services::hardware::HardwareService;
use crate::services::hf::HfService;
use crate::services::libport::{LibraryPort, NullLibrary};
use crate::services::settings::SettingsService;

/// An exception an adapter raised, as Python's catch-all answers it (and logs it).
fn raised(e: &PyExc) -> AppError {
    tracing::error!(target: "engine_console.services.fitting", "adapter raised {}", e.kind);
    AppError::Problem(internal("unexpected server error"))
}

/// How an estimate failed: a problem the caller sees as such, or a Python exception nobody in the estimate catches
/// (an adapter's, or the estimator's own arithmetic).
enum Fail {
    App(AppError),
    Py(PyExc),
}

impl From<AppError> for Fail {
    fn from(e: AppError) -> Self {
        Fail::App(e)
    }
}

impl From<Problem> for Fail {
    fn from(e: Problem) -> Self {
        Fail::App(AppError::Problem(e))
    }
}

impl Fail {
    /// For the callers that catch (A5): the exception becomes the generic 500 problem as a value.
    fn caught(self) -> AppError {
        match self {
            Fail::App(e) => e,
            Fail::Py(e) => raised(&e),
        }
    }
}

pub type ResidentFn = Arc<dyn Fn(&str) -> Vec<ResidentUse> + Send + Sync>;

pub struct FitService {
    adapters: AdapterRegistry,
    hf: Arc<HfService>,
    hw: Arc<HardwareService>,
    settings: Arc<SettingsService>,
    resident: RwLock<ResidentFn>,
    library: RwLock<Arc<dyn LibraryPort>>,
}

impl std::fmt::Debug for FitService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("FitService")
    }
}

impl FitService {
    pub fn new(
        adapters: AdapterRegistry,
        hf: Arc<HfService>,
        hw: Arc<HardwareService>,
        settings: Arc<SettingsService>,
    ) -> Self {
        Self {
            adapters,
            hf,
            hw,
            settings,
            resident: RwLock::new(Arc::new(|_: &str| Vec::new())),
            library: RwLock::new(Arc::new(NullLibrary)),
        }
    }

    // a poisoned lock (a panic elsewhere while it was held) still hands out its value: a setter is never dropped
    // and a reader never falls back to an empty library (crate::sync)
    pub fn set_resident_provider(&self, f: ResidentFn) {
        *self.resident.write_guard() = f;
    }

    pub fn set_library(&self, library: Arc<dyn LibraryPort>) {
        *self.library.write_guard() = library;
    }

    fn library(&self) -> Arc<dyn LibraryPort> {
        Arc::clone(&self.library.read_guard())
    }

    fn residents(&self, host_id: &str) -> Vec<ResidentUse> {
        let f = Arc::clone(&self.resident.read_guard());
        f(host_id)
    }

    /// The GPUs a launch gets: the explicit choice wins, then the configured default (only indices this host has),
    /// else the `tp` GPUs with the most free VRAM. An exception the adapter's memory model raises comes back as the
    /// generic 500 problem (the lifecycle's `_resolve_gpus` catches it and falls back to the requested GPUs).
    pub fn resolve_gpus(
        &self,
        engine: &str,
        info: &ModelInfo,
        params: &Obj,
        gpu_ids: &[i64],
        host_id: &str,
    ) -> AppResult<Vec<i64>> {
        self.resolve_gpus_raw(engine, info, params, gpu_ids, host_id)
            .map_err(Fail::caught)
    }

    fn resolve_gpus_raw(
        &self,
        engine: &str,
        info: &ModelInfo,
        params: &Obj,
        gpu_ids: &[i64],
        host_id: &str,
    ) -> Result<Vec<i64>, Fail> {
        if !gpu_ids.is_empty() {
            return Ok(gpu_ids.to_vec());
        }
        let hw = self.hw.hardware(host_id, None)?;
        let default: Vec<i64> = self
            .settings
            .default_gpu_ids()?
            .into_iter()
            .filter(|i| hw.gpu_ids.contains(i))
            .collect();
        if !default.is_empty() {
            return Ok(default);
        }
        let mm = self
            .adapters
            .get(engine)?
            .try_memory_model(info, params)
            .map_err(Fail::Py)?;
        // `int(mm.get("tp") or 1)`, at least 1
        let tp = match Mem::from_map(&mm).get("tp").filter(|t| *t != 0.0) {
            Some(t) if t.is_finite() => {
                #[allow(clippy::cast_possible_truncation)]
                let t = t.trunc() as i64;
                t
            }
            Some(_) => {
                return Err(crate::domain::errors::internal("unexpected server error").into());
            }
            None => 1,
        }
        .max(1);
        let mut order: Vec<usize> = (0..hw.gpu_ids.len()).collect();
        // sorted(..., key=lambda i: -free): stable, most free first
        order.sort_by(|a, b| {
            let fa = -hw.gpu_free_gib.get(*a).copied().unwrap_or(0.0);
            let fb = -hw.gpu_free_gib.get(*b).copied().unwrap_or(0.0);
            fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut ids: Vec<i64> = order
            .into_iter()
            .take(usize::try_from(tp).unwrap_or(usize::MAX))
            .map(|i| hw.gpu_ids[i])
            .collect();
        ids.sort_unstable();
        Ok(ids)
    }

    /// The host's own copy when it has one, else (allow_hub) the Hugging Face description of a repo.
    pub async fn info_for(
        &self,
        host_id: &str,
        r: &ModelRef,
        revision: Option<&str>,
        prefer_library: bool,
        allow_hub: bool,
    ) -> AppResult<ModelInfo> {
        if prefer_library && let Some(local) = self.library().model_info(host_id, r).await? {
            return Ok(local);
        }
        if allow_hub
            && r.format == "hf"
            && let Some(repo) = r.repo_id.as_deref().filter(|x| !x.is_empty())
        {
            return self.hf.model_info(repo, revision).await;
        }
        Err(unprocessable(
            "no readable description of this model on that machine (no config.json, or the scan has not seen it \
             yet): rescan the library and retry",
        )
        .code("model_info_unavailable")
        .into())
    }

    /// POST /fit (a viewer passes allow_hub=false).
    pub async fn assess(&self, req: &FitRequest, allow_hub: bool) -> AppResult<FitReport> {
        let r = req.model.clone().unwrap_or_else(|| ModelRef {
            format: "hf".into(),
            repo_id: req.repo_id.clone(),
            root_id: None,
            rel: None,
            name: None,
        });
        let adapter = self.adapters.get(&req.engine)?;
        let formats = adapter.model_formats();
        if !formats.contains(&r.format) {
            let mut sorted = formats.clone();
            sorted.sort();
            return Err(unprocessable(format!(
                "{} cannot load {} models (it loads: {})",
                adapter.display_name(),
                r.format,
                sorted.join(", ")
            ))
            .code("format_unsupported")
            .into());
        }
        // a bare repo id (the Models page's fit badge) is judged from Hugging Face, as in v1; a chosen library copy
        // from the host. Without allow_hub (a viewer) only a copy a machine holds can be judged.
        let info = self
            .info_for(
                &req.host_id,
                &r,
                req.revision.as_deref(),
                req.model.is_some() || !allow_hub,
                allow_hub,
            )
            .await?;
        // nothing on the /fit path catches an exception: Python's catch-all answers it (app.rs, PORTING.md §A5)
        match self.report(
            &req.engine,
            &info,
            &req.params,
            &req.gpu_ids,
            req.concurrency,
            &req.host_id,
        ) {
            Ok(r) => Ok(r),
            Err(Fail::App(e)) => Err(e),
            Err(Fail::Py(e)) => crate::adapters::pycompat::raise_exc(e),
        }
    }

    /// The estimate for a model already described. An exception (an adapter's, or the estimator's arithmetic) comes
    /// back as the generic 500 problem, so the callers that catch it in Python can: the `/models` listing skips the
    /// engine ("one odd model must not blank the list"); the lifecycle's preflight passes it on.
    pub fn assess_info(
        &self,
        engine: &str,
        info: &ModelInfo,
        params: &Obj,
        gpu_ids: &[i64],
        concurrency: i64,
        host_id: &str,
    ) -> AppResult<FitReport> {
        self.report(engine, info, params, gpu_ids, concurrency, host_id)
            .map_err(Fail::caught)
    }

    fn report(
        &self,
        engine: &str,
        info: &ModelInfo,
        params: &Obj,
        gpu_ids: &[i64],
        concurrency: i64,
        host_id: &str,
    ) -> Result<FitReport, Fail> {
        let adapter = self.adapters.get(engine)?;
        let errors = adapter.try_validate(params).map_err(Fail::Py)?;
        if !errors.is_empty() {
            return Err(unprocessable("invalid engine parameters")
                .with("errors", Value::from(errors))
                .into());
        }
        let ids = self.resolve_gpus_raw(engine, info, params, gpu_ids, host_id)?;
        let hw = self
            .hw
            .hardware(host_id, if ids.is_empty() { None } else { Some(&ids) })?;
        let hw_all = self.hw.hardware(host_id, None)?;
        let mem = Mem::from_map(&adapter.try_memory_model(info, params).map_err(Fail::Py)?);
        let compat = adapter
            .try_compatibility(info, params, &hw)
            .map_err(Fail::Py)?;
        let host_ram = estimate_host_ram(
            &adapter
                .try_host_memory_gib(info, params, &hw)
                .map_err(Fail::Py)?,
            self.hw.host_memory(host_id)?.as_ref(),
        );
        let budget_hw = unified_budget(&hw, host_ram.needed_gib);
        let mut report = estimate_fit(
            info,
            &mem,
            &budget_hw,
            concurrency,
            &self.residents(host_id),
            &compat,
            Some(&hw_all),
        )
        // the estimator's Python raises (ZeroDivisionError, int() of a NaN): an exception, not a problem
        .map_err(|p| Fail::Py(crate::adapters::pycompat::exc("ArithmeticError", p.detail)))?;
        let needed = host_ram.needed_gib;
        report.host_ram = Some(host_ram);
        if hw.unified_memory {
            report.notes.push(format!(
                "unified memory: the GPU and the host share one RAM pool, so the {needed:.1} GiB the engine process \
                 needs was taken off the GPU budget"
            ));
        }
        Ok(report)
    }
}

fn unified_budget(hw: &Hardware, host_needed_gib: f64) -> Hardware {
    if !hw.unified_memory || hw.gpu_ids.is_empty() {
        return hw.clone();
    }
    #[allow(clippy::cast_precision_loss)]
    let share = host_needed_gib / hw.gpu_ids.len() as f64;
    let mut out = hw.clone();
    // max(f - share, 0.0): the first argument stays unless 0.0 is greater
    out.gpu_free_gib = hw
        .gpu_free_gib
        .iter()
        .map(|f| {
            let x = f - share;
            if 0.0 > x { 0.0 } else { x }
        })
        .collect();
    out
}
