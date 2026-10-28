//! /library (port of api/routers/library.py).

use serde_json::json;

use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok};
use crate::api::routes::system::host_q;
use crate::api::schemas::{COPY_REQUEST, RESCAN_REQUEST};
use crate::api::validate::{Params, T};
use crate::domain::errors::internal;
use crate::domain::fleet::{CopyRequest, LIBRARY_FORMATS};
use crate::domain::models::Page;
use crate::services::store::paginate;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub async fn list_library(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let q = &ctx.query;
    let host = p.opt_str_q(q, "host", &host_q());
    let format = p.opt_str_q(q, "format", &T::Literal(LIBRARY_FORMATS));
    let text = p.opt_str_q(
        q,
        "q",
        &T::Str {
            min: None,
            max: Some(200),
            pattern: None,
        },
    );
    let limit = p.int_q(q, "limit", 100);
    let cursor = p.opt_str_q(q, "cursor", &T::str());
    finish(p)?;
    let models = ctx
        .c
        .library
        .models(host.as_deref(), format.as_deref(), text.as_deref())?;
    let (limit, off) = paginate(limit, cursor.as_deref());
    let (lim, offu) = (
        usize::try_from(limit).unwrap_or(0),
        usize::try_from(off).unwrap_or(0),
    );
    let items: Vec<_> = models.iter().skip(offu).take(lim).cloned().collect();
    let next = (models.len() > offu + lim).then(|| (off + limit).to_string());
    ok(&Page::new(items, next))
}

pub async fn rescan_library(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::opt(T::Model(&RESCAN_REQUEST)), false);
    finish(p)?;
    let host = v
        .get("host_id")
        .and_then(|h| h.as_str())
        .map(str::to_string);
    let hosts = ctx.c.library.rescan(host.as_deref()).await?;
    json(202, &json!({"hosts": hosts}))
}

pub async fn delete_copy(ctx: Ctx) -> HResult {
    ctx.c.library.delete_copy(&ctx.param("copy_id")).await?;
    no_content()
}

pub async fn copy_copy(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&COPY_REQUEST), true);
    finish(p)?;
    let req: CopyRequest =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    json(
        201,
        &ctx.c.downloads.copy(&ctx.param("copy_id"), &req).await?,
    )
}
