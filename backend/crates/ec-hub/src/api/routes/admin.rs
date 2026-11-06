//! /settings, /keys (port of api/routers/admin.py), and the Config files page's read-only /config routes (beyond
//! Python).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::Value;

use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok};
use crate::api::schemas::{CONFIG_CHECK, KEY_IN, SETTINGS_PATCH};
use crate::api::validate::{Params, T};
use crate::domain::errors::{Problem, internal, not_found, too_many_streams};
use crate::domain::models::{KeyIn, SettingsPatch};
use crate::services::config_check;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub async fn get_settings(ctx: Ctx) -> HResult {
    ok(&ctx.c.settings.effective()?)
}

/// `host_image_pins` replaces the whole table ({} clears it); shapes and the pinned-tag rule only.
pub async fn put_settings(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&SETTINGS_PATCH), true);
    finish(p)?;
    let patch: SettingsPatch =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    ok(&ctx.c.settings.update(&patch)?)
}

pub async fn list_keys(ctx: Ctx) -> HResult {
    ok(&ctx.c.settings.list_keys()?)
}

/// The plaintext `secret` is returned exactly once, here.
pub async fn create_key(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&KEY_IN), true);
    finish(p)?;
    let k: KeyIn =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    json(201, &ctx.c.settings.create_key(&k.name, &k.role)?)
}

pub async fn revoke_key(ctx: Ctx) -> HResult {
    ctx.c.settings.revoke_key(&ctx.param("kid"))?;
    no_content()
}

/// GET /config/{file}: what the Config files page may load for one of its files (admin; read-only). hosts.json comes
/// through the loader's own file checks and only when the loader accepts it and nothing in it looks secret; the env
/// file is never served (it may hold a token), only its path; a node policy lives on its node. Plus the templates.
/// The file work (open, fstat, a read of up to 64 KiB, lstat) runs on the blocking pool, not on an async worker.
pub async fn get_config_file(ctx: Ctx) -> HResult {
    let file = ctx.param("file");
    let cfg = Arc::clone(&ctx.c.cfg);
    let name = file.clone();
    let current = tokio::task::spawn_blocking(move || config_check::current(&name, &cfg))
        .await
        .map_err(|_| Fail::from(internal("unexpected server error")))?;
    match current {
        Some(v) => ok(&v),
        None => Err(not_found(format!(
            "no config file '{}' (the editor knows: {})",
            file.chars().take(32).collect::<String>(),
            config_check::FILES.join(", ")
        ))
        .into()),
    }
}

/// The checks that may run at once per console; more answer 429 (each is CPU only, but a draft may be a megabyte).
const MAX_CONFIG_CHECKS: usize = 4;

/// One running check (the container's `config_checks` count); released when dropped.
struct CheckSlot(Arc<AtomicUsize>);

impl Drop for CheckSlot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

fn take_check_slot(running: &Arc<AtomicUsize>) -> Option<CheckSlot> {
    running
        .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
            (n < MAX_CONFIG_CHECKS).then_some(n + 1)
        })
        .ok()
        .map(|_| CheckSlot(Arc::clone(running)))
}

/// How long the page waits for one check (they finish in milliseconds; a check still running after this keeps its
/// slot until it ends, so at most `MAX_CONFIG_CHECKS` ever run).
const CONFIG_CHECK_TIMEOUT: Duration = Duration::from_secs(10);

/// POST /config/validate {file, text}: the hub's and the node's real loaders on a draft (admin). Writes nothing,
/// runs nothing, looks up no path the draft names; the audit keeps metadata only (security::CONTENT_PATHS). No
/// diagnostic carries a value from the hub's own files or environment; a message may quote a key or value of the
/// posted draft (cut short, scrubbed).
/// The check runs on the blocking pool, holding its slot until it ends, under `CONFIG_CHECK_TIMEOUT`.
pub async fn validate_config(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let mut v = p.body(&body, &T::Model(&CONFIG_CHECK), true);
    finish(p)?;
    let file = v
        .get("file")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let text = match v.get_mut("text").map(Value::take) {
        Some(Value::String(t)) => t,
        _ => String::new(),
    };
    let Some(slot) = take_check_slot(&ctx.c.config_checks) else {
        let p: Problem = too_many_streams(format!(
            "{MAX_CONFIG_CHECKS} config checks are already running; retry in a moment"
        ))
        .code("too_many_config_checks");
        return Err(p.into());
    };
    let job = tokio::task::spawn_blocking(move || {
        let _slot = slot;
        config_check::validate(&file, &text)
    });
    match tokio::time::timeout(CONFIG_CHECK_TIMEOUT, job).await {
        Ok(Ok(Some(r))) => ok(&r),
        Ok(Ok(None) | Err(_)) => Err(Fail::from(internal("unexpected server error"))),
        Err(_) => Err(Problem::new(
            503,
            "config_check_timeout",
            format!(
                "the check did not finish within {} s; try a smaller draft",
                CONFIG_CHECK_TIMEOUT.as_secs()
            ),
        )
        .into()),
    }
}
