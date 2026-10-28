//! /scripts (port of api/routers/scripts.py; contract §10.8). Running one is admin-only.

use futures::StreamExt;

use crate::api::deps::sse_response;
use crate::api::http::{Ctx, Fail, HResult, json, ok};
use crate::api::routes::system::host_q;
use crate::api::schemas::SCRIPT_RUN_REQUEST;
use crate::api::validate::{Params, T};
use crate::domain::models::{HOST_ID_PATTERN, Page};
use crate::services::common::sse_text;

const SCRIPT_NAME: &str = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,31}$";
const JOB_ID: &str = r"^[A-Za-z0-9_\-]{1,64}$";

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

fn pat(p: &'static str) -> T {
    T::Str {
        min: None,
        max: None,
        pattern: Some(p),
    }
}

pub async fn list_scripts(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let host = p.opt_str_q(&ctx.query, "host", &host_q());
    finish(p)?;
    ok(&Page::new(ctx.c.scripts.scripts(host.as_deref())?, None))
}

/// Start one action of a script (checked against the policy the node reported; the node checks again).
pub async fn run_script(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let host = p.path("host_id", &ctx.param("host_id"), &pat(HOST_ID_PATTERN));
    let name = p.path("name", &ctx.param("name"), &pat(SCRIPT_NAME));
    let action = p.path("action", &ctx.param("action"), &pat(SCRIPT_NAME));
    let v = p.body(&body, &T::opt(T::Model(&SCRIPT_RUN_REQUEST)), false);
    finish(p)?;
    let arg = v.get("arg").and_then(|a| a.as_str()).map(str::to_string);
    json(
        202,
        &ctx.c
            .scripts
            .run(&host, &name, &action, arg.as_deref())
            .await?,
    )
}

pub async fn get_job(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let id = p.path("job_id", &ctx.param("job_id"), &pat(JOB_ID));
    finish(p)?;
    ok(&ctx.c.scripts.job(&id)?)
}

/// SSE: every output line as event `line` (history first, then live), then one `end` with the job.
pub async fn job_stream(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let id = p.path("job_id", &ctx.param("job_id"), &pat(JOB_ID));
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    ctx.c.scripts.job(&id)?; // 404 before the stream starts
    let events = ctx
        .c
        .scripts
        .stream(&id, once)?
        .map(|(event, data)| sse_text(&data, Some(&event)));
    sse_response(&ctx, events.boxed())
}
