//! Route dependencies (port of api/deps.py): roles, stream slots, event-stream responses.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body as AxBody;
use futures::StreamExt;
use futures::stream::BoxStream;
use http::{HeaderValue, Response, StatusCode, header};

use crate::api::http::{Ctx, Fail, HResult};
use crate::api::router::Access;
use crate::api::security::viewer_may;
use crate::domain::errors::{Problem, forbidden, too_many_streams};
use crate::services::common::SseLimiter;
use crate::services::settings::Principal;

/// requests without a key (trusted loopback) share one bucket; key ids never look like this
pub const NO_KEY: &str = "(no key)";
pub const SAFE_METHODS: &[&str] = &["GET", "HEAD", "OPTIONS"];

/// `enforce_roles`, attached to every API router: a mutating route is admin-only unless it is on the exact viewer
/// list, even if the middleware were bypassed or changed (defence in depth). `path` is the percent-decoded path
/// (Starlette's `request.url.path`).
pub fn enforce_roles_for(method: &str, path: &str, p: &Principal) -> Result<(), Fail> {
    if SAFE_METHODS.contains(&method) {
        return Ok(());
    }
    if p.role == "admin" || (p.role == "viewer" && viewer_may(method, path)) {
        return Ok(());
    }
    Err(forbidden("admin role required for this action").into())
}

/// `Admin`: 403 "admin role required" for anything but an admin.
pub fn require_admin_for(p: &Principal) -> Result<(), Fail> {
    if p.role == "admin" {
        Ok(())
    } else {
        Err(forbidden("admin role required").into())
    }
}

/// The role dependencies of a route, in FastAPI's order (the router's `enforce_roles` first, then the endpoint's
/// `_: Admin`): what the request pipeline runs after authentication and before the handler, so a role error comes
/// before any validation error, as a FastAPI dependency's does.
pub fn check_access(access: Access, method: &str, path: &str, p: &Principal) -> Result<(), Fail> {
    match access {
        Access::Open => Ok(()),
        Access::Caller => enforce_roles_for(method, path, p),
        Access::Admin => {
            enforce_roles_for(method, path, p)?;
            require_admin_for(p)
        }
    }
}

/// `enforce_roles` of a request context.
pub fn enforce_roles(ctx: &Ctx) -> Result<(), Fail> {
    enforce_roles_for(&ctx.method, &ctx.path, &ctx.principal)
}

/// `require_admin` of a request context.
pub fn require_admin(ctx: &Ctx) -> Result<(), Fail> {
    require_admin_for(&ctx.principal)
}

/// The bucket the current caller's streams count against: its API key id.
pub fn stream_key(ctx: &Ctx) -> String {
    ctx.principal
        .key_id
        .clone()
        .unwrap_or_else(|| NO_KEY.to_string())
}

/// One held slot; given back when dropped (however the response ends).
pub struct Slot {
    lim: Arc<SseLimiter>,
    key: String,
}

impl Drop for Slot {
    fn drop(&mut self) {
        self.lim.release(&self.key);
    }
}

/// One slot of `lim` for the current caller, or 429 `code` naming the cap that is full (this key's share first).
pub fn take_stream_slot(
    ctx: &Ctx,
    lim: &Arc<SseLimiter>,
    what: &str,
    code: &str,
) -> Result<Slot, Fail> {
    let key = stream_key(ctx);
    if !lim.acquire(&key) {
        let mine = lim.key_full(&key);
        let n = if mine { lim.per_key } else { lim.limit };
        let detail = if mine {
            format!(
                "this API key already has {n} {what} open (the most one key may hold); end one and retry"
            )
        } else {
            format!("{n} {what} are already open on this console (its limit); end one and retry")
        };
        let p: Problem = too_many_streams(detail)
            .code(code)
            .with("scope", if mine { "key" } else { "total" })
            .with("limit", n);
        return Err(p.into());
    }
    Ok(Slot {
        lim: Arc::clone(lim),
        key,
    })
}

/// A text/event-stream response that holds `slot` for as long as it is open.
pub fn slot_stream(body: BoxStream<'static, String>, slot: Slot) -> HResult {
    let stream = body.map(move |s| {
        let _hold = &slot;
        Ok::<_, std::io::Error>(bytes::Bytes::from(s))
    });
    let mut r = Response::new(AxBody::from_stream(stream));
    *r.status_mut() = StatusCode::OK;
    let h = r.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream; charset=utf-8"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    h.insert("x-accel-buffering", HeaderValue::from_static("no"));
    Ok(r)
}

/// End a stream after `idle` without an item.
pub fn until_idle(body: BoxStream<'static, String>, idle: Duration) -> BoxStream<'static, String> {
    futures::stream::unfold(body, move |mut s| async move {
        match tokio::time::timeout(idle, s.next()).await {
            Ok(Some(item)) => Some((item, s)),
            _ => None,
        }
    })
    .boxed()
}

/// `sse_response`: the console's stream caps (429 too_many_streams) and the idle timeout.
pub fn sse_response(ctx: &Ctx, body: BoxStream<'static, String>) -> HResult {
    let slot = take_stream_slot(ctx, &ctx.c.sse, "live streams", "too_many_streams")?;
    let idle = Duration::from_secs_f64(ctx.c.cfg.sse_idle_timeout_s.max(0.0));
    slot_stream(until_idle(body, idle), slot)
}
