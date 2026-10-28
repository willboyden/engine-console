//! /profiles (port of api/routers/profiles.py).

use serde_json::Value;

use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok, with_ct};
use crate::api::schemas::PROFILE_IN;
use crate::api::validate::{Params, T};
use crate::domain::errors::{bad_request, internal};
use crate::domain::models::ProfileIn;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

async fn profile_in(ctx: &mut Ctx) -> Result<ProfileIn, Fail> {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&PROFILE_IN), true);
    finish(p)?;
    serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))
}

pub async fn list_profiles(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 100);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    let engine = p.opt_str_q(&ctx.query, "engine", &T::str());
    finish(p)?;
    ok(&ctx
        .c
        .profiles
        .list_page(limit, cursor.as_deref(), engine.as_deref())?)
}

pub async fn create_profile(mut ctx: Ctx) -> HResult {
    let body = profile_in(&mut ctx).await?;
    json(201, &ctx.c.profiles.create(&body)?)
}

/// Body is YAML text (yaml content-type) or JSON {"yaml": "..."}; size-capped by the security middleware.
pub async fn import_profile(mut ctx: Ctx) -> HResult {
    let bytes = ctx.read_body().await?;
    let mut raw = String::from_utf8_lossy(&bytes).into_owned();
    if ctx
        .header("content-type")
        .is_some_and(|ct| ct.contains("json"))
    {
        let doc: Value = serde_json::from_str(&raw)
            .map_err(|_| Fail::from(bad_request("invalid JSON body").code("invalid_body")))?;
        raw = match doc {
            Value::Object(o) => match o.get("yaml") {
                Some(Value::String(s)) => s.clone(),
                // `yaml.safe_load(<not a str>)` takes it for a file and fails on `.read`: an exception nobody
                // catches, so Python's catch-all 500 (PORTING.md §A5)
                Some(other) => crate::adapters::pycompat::raise(
                    "AttributeError",
                    &format!("'{}' object has no attribute 'read'", py_type_name(other)),
                ),
                None => String::new(),
            },
            _ => String::new(),
        };
    }
    json(201, &ctx.c.profiles.import_yaml(&raw)?)
}

pub async fn diff(ctx: Ctx) -> HResult {
    ok(&ctx.c.profiles.diff(&ctx.param("a"), &ctx.param("b"))?)
}

pub async fn export(ctx: Ctx) -> HResult {
    let text = ctx.c.profiles.export_yaml(&ctx.param("pid"))?;
    Ok(with_ct(200, "application/yaml", text))
}

pub async fn update_profile(mut ctx: Ctx) -> HResult {
    let body = profile_in(&mut ctx).await?;
    ok(&ctx.c.profiles.update(&ctx.param("pid"), &body)?)
}

pub async fn delete_profile(ctx: Ctx) -> HResult {
    ctx.c.profiles.delete(&ctx.param("pid"))?;
    no_content()
}

/// Python's type name of a JSON value as `json.loads` builds it.
fn py_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(n) if n.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}
