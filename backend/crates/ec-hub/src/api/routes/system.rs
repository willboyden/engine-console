//! /health, /hardware, /learn, /engines, /hf/*, /fit, the OpenAPI document and the SPA (port of
//! api/routers/system.py and main.py).

use serde_json::{Map, Value, json};

use crate::VERSION;
use crate::api::http::{Ctx, Fail, HResult, ok, with_ct};
use crate::api::schemas::FIT_REQUEST;
use crate::api::statics;
use crate::api::validate::{Params, T};
use crate::domain::errors::{internal, not_ported};
use crate::domain::models::{FitRequest, HOST_ID_PATTERN};
use crate::domain::repo_id::{check_repo_id, check_revision};
use crate::services::hf::SearchParams;

pub(crate) const OPENAPI_JSON: &str = include_str!("../../../assets/openapi.json");

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub(crate) fn host_q() -> T {
    T::Str {
        min: None,
        max: None,
        pattern: Some(HOST_ID_PATTERN),
    }
}

/// Learn bundle errors are a broken build, never a client error.
fn learn_bundle(ctx: &Ctx) -> Result<std::sync::Arc<crate::domain::learn::LearnBundle>, Fail> {
    ctx.c
        .learn_bundle()
        .map_err(|_| Fail::from(internal("unexpected server error")))
}

pub async fn openapi(ctx: Ctx) -> HResult {
    // a plain starlette route: no role dependency, but the middleware already required a key (it is guarded)
    let _ = ctx;
    let v: Value = serde_json::from_str(OPENAPI_JSON)
        .map_err(|_| Fail::from(internal("unexpected server error")))?;
    Ok(with_ct(
        200,
        "application/json",
        serde_json::to_vec(&v).unwrap_or_default(),
    ))
}

pub async fn health(ctx: Ctx) -> HResult {
    let cfg = &ctx.c.cfg;
    let proxied = cfg.egress_proxy.as_deref().is_some_and(|p| !p.is_empty());
    let mut warnings: Vec<&str> = Vec::new();
    if !proxied {
        warnings.push(
            "console egress is DIRECT: Hugging Face traffic bypasses any egress proxy (only the in-app allowlist \
             applies). Set EGRESS_PROXY to route it through your proxy.",
        );
    }
    if proxied && cfg.egress_ca_bundle.is_none() {
        warnings.push(
            "EGRESS_PROXY is set but EGRESS_CA_BUNDLE is not: an intercepting proxy's CA is not in the system trust \
             store, so Hugging Face requests will fail TLS verification. Set EGRESS_CA_BUNDLE.",
        );
    }
    if cfg.trust_loopback {
        warnings.push(
            "trust_loopback is ON: any local process or browser-reachable loopback client is treated as admin, and \
             this console can now start containers on every machine in the fleet. Keep the CSRF header and Host \
             checks intact, and set ENGINE_CONSOLE_TRUST_LOOPBACK=false (the default) on shared machines.",
        );
    }
    let engines: Vec<String> = ctx
        .c
        .adapters
        .list()
        .iter()
        .map(|a| a.id().to_string())
        .collect();
    ok(&json!({
        "status": "ok",
        "version": VERSION,
        "engines": engines,
        "docker_context": cfg.docker_context,
        "egress_mode": if proxied { "proxied" } else { "direct" },
        "warnings": warnings,
    }))
}

pub async fn hardware(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let host = p
        .opt_str_q(&ctx.query, "host", &host_q())
        .unwrap_or_else(|| "local".into());
    finish(p)?;
    ok(&ctx.c.hardware.report(&host)?)
}

pub async fn learn(ctx: Ctx) -> HResult {
    learn_impl(&ctx)
}

fn learn_impl(ctx: &Ctx) -> HResult {
    let b = learn_bundle(ctx)?;
    let ids: Vec<String> = ctx
        .c
        .adapters
        .list()
        .iter()
        .map(|a| a.id().to_string())
        .collect();
    let keep = |m: &Map<String, Value>| -> Map<String, Value> {
        m.iter()
            .filter(|(k, _)| ids.contains(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    };
    let mut out = (*b).clone();
    out.params = keep(&b.params);
    out.presets = keep(&b.presets);
    ok(&out)
}

/// 501 while an engine's adapter is the placeholder (its catalog would be served empty).
fn adapters_ported<'a>(
    list: impl IntoIterator<Item = &'a std::sync::Arc<dyn crate::adapters::EngineAdapter>>,
) -> Result<(), Fail> {
    for a in list {
        if !a.ported() {
            return Err(not_ported(&format!("engine_console.adapters.{}", a.id())).into());
        }
    }
    Ok(())
}

pub async fn engines(ctx: Ctx) -> HResult {
    adapters_ported(&ctx.c.adapters.list())?;
    let b = learn_bundle(&ctx)?;
    let mut out = Vec::new();
    for a in ctx.c.adapters.list() {
        let pin = ctx.c.settings.image_pin(a.id(), "local")?;
        let presets: Vec<Value> = a
            .presets()
            .into_iter()
            .map(|(n, p)| json!({"name": n, "params": p}))
            .collect();
        let mut formats = a.model_formats();
        formats.sort();
        let images: Map<String, Value> = a
            .images_by_arch()
            .into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        // `b.presets.get(a.id) or a.preset_info()`: the learn files win when they have any entry for the engine
        let info = match b.presets.get(a.id()) {
            Some(Value::Object(m)) if !m.is_empty() => Value::Object(m.clone()),
            _ => Value::Object(
                a.preset_info()
                    .into_iter()
                    .map(|(k, e)| (k, serde_json::to_value(e).unwrap_or(Value::Null)))
                    .collect(),
            ),
        };
        out.push(json!({
            "id": a.id(),
            "display_name": a.display_name(),
            "default_image": pin.unwrap_or_else(|| a.default_image().to_string()),
            "presets": presets,
            "model_formats": formats,
            "images_by_arch": images,
            "multi_model": a.multi_model(),
            "presets_info": info,
        }));
    }
    ok(&out)
}

pub async fn engine_params(ctx: Ctx) -> HResult {
    let engine = ctx.param("engine");
    let adapter = ctx.c.adapters.get(&engine)?;
    adapters_ported([&adapter])?;
    let b = learn_bundle(&ctx)?;
    let explained = b
        .params
        .get(&engine)
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let mut out = Vec::new();
    for mut p in adapter.param_catalog() {
        p.explain = explained
            .get(&p.key)
            .and_then(|v| serde_json::from_value(v.clone()).ok());
        if p.level.as_deref().is_none_or(str::is_empty) {
            p.level = Some(if p.advanced { "advanced" } else { "basic" }.into());
        }
        out.push(p);
    }
    ok(&out)
}

pub async fn hf_search(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let q = &ctx.query;
    let params = SearchParams {
        q: p.str_q(q, "q", Some("")).unwrap_or_default(),
        task: p.opt_str_q(q, "task", &T::str()),
        library: p.opt_str_q(q, "library", &T::str()),
        quant: p.opt_str_q(q, "quant", &T::str()),
        max_gib: p.opt_float_q(q, "max_gib"),
        gated: p.opt_bool_q(q, "gated"),
        sort: p.str_q(q, "sort", Some("downloads")).unwrap_or_default(),
        limit: p.int_q(q, "limit", 20),
        cursor: p.opt_str_q(q, "cursor", &T::str()),
    };
    finish(p)?;
    ok(&ctx.c.hf.search(&params).await?)
}

pub async fn hf_model(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let repo = p.path(
        "repo_id",
        &ctx.param("repo_id"),
        &T::After(Box::new(T::str()), check_repo_id),
    );
    let rev = p.opt_str_q(
        &ctx.query,
        "revision",
        &T::After(Box::new(T::str()), check_revision),
    );
    finish(p)?;
    ok(&ctx.c.hf.detail(&repo, rev.as_deref()).await?)
}

pub async fn fit(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&FIT_REQUEST), true);
    finish(p)?;
    let req: FitRequest =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    // a viewer may ask for a fit, but only from copies the machines already hold (review finding L4)
    let allow_hub = ctx.principal.role == "admin";
    ok(&ctx.c.fit.assess(&req, allow_hub).await?)
}

/// GET /{path:path}: the app shell and its asset dirs.
pub async fn spa(ctx: Ctx) -> HResult {
    let Some(root) = ctx.c.frontend.clone() else {
        return Ok(crate::api::http::not_found_resp("no such file"));
    };
    let path = ctx.param("path");
    Ok(statics::spa(&root, &path, ctx.header("if-none-match")))
}
