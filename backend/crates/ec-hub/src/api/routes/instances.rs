//! /instances (port of api/routers/instances.py).

use futures::StreamExt;
use serde_json::{Map, Value};

use crate::api::deps::sse_response;
use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok, with_ct};
use crate::api::routes::system::host_q;
use crate::api::schemas::{INSTANCE_CREATE, INSTANCE_PATCH};
use crate::api::validate::{Params, T};
use crate::domain::errors::internal;
use crate::domain::models::{InstanceCreate, InstancePatch};
use crate::services::common::{py_splitlines, sse_text};

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub async fn list_instances(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 100);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    let host = p.opt_str_q(&ctx.query, "host", &host_q());
    finish(p)?;
    ok(&ctx
        .c
        .lifecycle
        .list_page(limit, cursor.as_deref(), host.as_deref())?)
}

async fn create_body(ctx: &mut Ctx) -> Result<InstanceCreate, Fail> {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&INSTANCE_CREATE), true);
    finish(p)?;
    serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))
}

/// The profile's params (when `profile_id` is given; 404 when it does not exist).
fn profile_params(ctx: &Ctx, req: &InstanceCreate) -> Result<Option<Map<String, Value>>, Fail> {
    match &req.profile_id {
        Some(pid) if !pid.is_empty() => Ok(Some(ctx.c.profiles.get(pid)?.params)),
        _ => Ok(None),
    }
}

pub async fn create_instance(mut ctx: Ctx) -> HResult {
    let req = create_body(&mut ctx).await?;
    let pp = profile_params(&ctx, &req)?;
    json(201, &ctx.c.lifecycle.create(&req, pp.as_ref()).await?)
}

/// Dry run of the checks POST /instances performs.
pub async fn preflight(mut ctx: Ctx) -> HResult {
    let req = create_body(&mut ctx).await?;
    let mut params = profile_params(&ctx, &req)?.unwrap_or_default();
    for (k, v) in &req.params {
        params.insert(k.clone(), v.clone());
    }
    ok(&ctx.c.lifecycle.preflight(&req, &params).await?)
}

/// Re-read every node's inventory (what they last pushed).
pub async fn discover(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.list_page(500, None, None)?)
}

pub async fn get_instance(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.get(&ctx.param("iid"))?)
}

pub async fn patch_instance(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&INSTANCE_PATCH), true);
    finish(p)?;
    let patch: InstancePatch =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    ok(&ctx.c.lifecycle.patch(&ctx.param("iid"), &patch)?)
}

pub async fn stop(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.stop(&ctx.param("iid")).await?)
}

pub async fn start(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.start(&ctx.param("iid")).await?)
}

pub async fn restart(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.restart(&ctx.param("iid")).await?)
}

pub async fn remove(ctx: Ctx) -> HResult {
    ctx.c.lifecycle.remove(&ctx.param("iid")).await?;
    no_content()
}

pub async fn logs(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let tail = p.int_q(&ctx.query, "tail", 200);
    finish(p)?;
    let text = ctx.c.lifecycle.logs(&ctx.param("iid"), tail).await?;
    Ok(with_ct(200, "text/plain; charset=utf-8", text))
}

pub async fn logs_stream(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let tail = p.int_q(&ctx.query, "tail", 100);
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    let iid = ctx.param("iid");
    ctx.c.lifecycle.require_managed(&iid)?; // 404/409 before the stream starts
    if once {
        let text = ctx.c.lifecycle.logs(&iid, tail).await?;
        let events: Vec<String> = py_splitlines(&text)
            .iter()
            .map(|l| sse_text(l, Some("log")))
            .collect();
        return sse_response(&ctx, futures::stream::iter(events).boxed());
    }
    let lines = ctx.c.lifecycle.follow_logs(&iid, tail).await?;
    let events = lines
        .take_while(|l| futures::future::ready(l.is_ok()))
        .filter_map(|l| async move { l.ok().map(|line| sse_text(&line, Some("log"))) });
    sse_response(&ctx, events.boxed())
}

pub async fn command(ctx: Ctx) -> HResult {
    ok(&ctx.c.lifecycle.command(&ctx.param("iid")).await?)
}
