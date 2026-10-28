//! /downloads, /models (port of api/routers/downloads.py).

use futures::StreamExt;
use serde_json::Value;

use crate::api::deps::sse_response;
use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok};
use crate::api::schemas::DOWNLOAD_REQUEST;
use crate::api::validate::{Params, T};
use crate::domain::errors::{internal, not_found};
use crate::domain::fleet::LOCAL;
use crate::domain::models::{DownloadRequest, Page};
use crate::domain::refs::ModelRef;
use crate::domain::repo_id::check_repo_id;
use crate::services::common::sse;
use crate::services::libport::LibraryPort;
use crate::services::store::paginate;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub async fn list_downloads(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 50);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    finish(p)?;
    ok(&ctx.c.downloads.list_page(limit, cursor.as_deref())?)
}

pub async fn create_download(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&DOWNLOAD_REQUEST), true);
    finish(p)?;
    let req: DownloadRequest =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    json(201, &ctx.c.downloads.create(&req).await?)
}

/// SSE: a snapshot of every download, then live progress. `once=true` ends after the snapshot.
pub async fn stream_downloads(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    let snapshot = ctx.c.downloads.list_page(200, None)?.items;
    let first = sse(
        &serde_json::to_value(&snapshot).unwrap_or(Value::Null),
        Some("snapshot"),
    );
    let head = futures::stream::once(async move { first });
    if once {
        return sse_response(&ctx, head.boxed());
    }
    let sub = ctx.c.bus.subscribe("downloads");
    let live = futures::stream::unfold(sub, |mut s| async move {
        let ev = s.recv().await?;
        Some((sse(&ev, Some("progress")), s))
    });
    sse_response(&ctx, head.chain(live).boxed())
}

pub async fn get_download(ctx: Ctx) -> HResult {
    ok(&ctx.c.downloads.get(&ctx.param("did"))?)
}

pub async fn pause(ctx: Ctx) -> HResult {
    ok(&ctx.c.downloads.pause(&ctx.param("did")).await?)
}

pub async fn resume(ctx: Ctx) -> HResult {
    ok(&ctx.c.downloads.resume(&ctx.param("did")).await?)
}

pub async fn cancel(ctx: Ctx) -> HResult {
    ok(&ctx.c.downloads.cancel(&ctx.param("did")).await?)
}

pub async fn delete_download(ctx: Ctx) -> HResult {
    ctx.c.downloads.delete(&ctx.param("did")).await?;
    no_content()
}

/// The v1 shape: the Hugging Face models of every machine, each with the engines that fit it on its own machine.
pub async fn local_models(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 100);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    finish(p)?;
    let (limit, off) = paginate(limit, cursor.as_deref());
    let models = ctx.c.library.local_models()?;
    let (lim, off_u) = (
        usize::try_from(limit).unwrap_or(0),
        usize::try_from(off).unwrap_or(0),
    );
    let mut page: Vec<_> = models.iter().skip(off_u).take(lim).cloned().collect();
    for m in &mut page {
        let Some(info) = ctx
            .c
            .library
            .model_info(&m.host_id, &ModelRef::hf(&m.repo_id))
            .await?
        else {
            continue;
        };
        for a in ctx.c.adapters.list() {
            // one odd model (or a machine that is down) must not blank the list
            if let Ok(r) =
                ctx.c
                    .fit
                    .assess_info(a.id(), &info, &serde_json::Map::new(), &[], 1, &m.host_id)
                && (r.verdict == "fits" || r.verdict == "tight")
            {
                m.engines_that_fit.push(a.id().to_string());
            }
        }
    }
    let next = (models.len() > off_u + lim).then(|| (off + limit).to_string());
    ok(&Page::new(page, next))
}

pub async fn delete_model(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let repo = p.path(
        "repo_id",
        &ctx.param("repo_id"),
        &T::After(Box::new(T::str()), check_repo_id),
    );
    finish(p)?;
    if !crate::services::library::PORTED {
        return Err(crate::domain::errors::not_ported("engine_console.services.library").into());
    }
    let Some(copy) = ctx.c.library.find_copy(LOCAL, &ModelRef::hf(&repo), true) else {
        return Err(not_found(format!("model not in local cache: {repo}")).into());
    };
    ctx.c.library.delete_copy(&copy.id).await?;
    no_content()
}
