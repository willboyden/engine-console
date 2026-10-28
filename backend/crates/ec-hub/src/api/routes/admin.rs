//! /settings, /keys (port of api/routers/admin.py).

use crate::api::http::{Ctx, Fail, HResult, json, no_content, ok};
use crate::api::schemas::{KEY_IN, SETTINGS_PATCH};
use crate::api::validate::{Params, T};
use crate::domain::errors::internal;
use crate::domain::models::{KeyIn, SettingsPatch};

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
