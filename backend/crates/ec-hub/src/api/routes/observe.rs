//! Metrics timeseries, benchmarks, usage, audit and the console's own /metrics (port of api/routers/observe.py).

use std::collections::HashMap;

use futures::StreamExt;
use serde_json::{Value, json};

use crate::api::deps::sse_response;
use crate::api::http::{Ctx, Fail, HResult, json, ok, with_ct};
use crate::api::routes::system::host_q;
use crate::api::schemas::BENCH_REQUEST;
use crate::api::validate::{Params, T};
use crate::domain::errors::{AppError, internal};
use crate::domain::fleet::LOCAL;
use crate::domain::models::BenchRequest;
use crate::services::common::sse;
use crate::services::telemetry::METRICS_CONTENT_TYPE;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

pub async fn system_metrics(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let window = p
        .str_q(&ctx.query, "window", Some("15m"))
        .unwrap_or_default();
    let host = p
        .opt_str_q(&ctx.query, "host", &host_q())
        .unwrap_or_else(|| LOCAL.into());
    finish(p)?;
    ok(&ctx.c.metrics.system_series(&window, &host)?)
}

pub async fn instance_metrics(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let window = p
        .str_q(&ctx.query, "window", Some("15m"))
        .unwrap_or_default();
    finish(p)?;
    ok(&ctx.c.metrics.series(&ctx.param("iid"), &window)?)
}

pub async fn metrics_stream(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    let mut first = Vec::new();
    for i in ctx.c.lifecycle.all_instances() {
        if let Some(pt) = ctx.c.metrics.latest(&i.id) {
            let mut m = serde_json::Map::new();
            m.insert("instance_id".into(), Value::String(i.id.clone()));
            if let Ok(Value::Object(o)) = serde_json::to_value(&pt) {
                m.extend(o);
            }
            first.push(sse(&Value::Object(m), Some("metrics")));
        }
    }
    let head = futures::stream::iter(first);
    if once {
        return sse_response(&ctx, head.boxed());
    }
    let sub = ctx.c.bus.subscribe("metrics");
    let live = futures::stream::unfold(sub, |mut s| async move {
        let ev = s.recv().await?;
        Some((sse(&ev, Some("metrics")), s))
    });
    sse_response(&ctx, head.chain(live).boxed())
}

pub async fn start_bench(mut ctx: Ctx) -> HResult {
    let body = ctx.body().await?;
    let mut p = Params::default();
    let v = p.body(&body, &T::Model(&BENCH_REQUEST), true);
    finish(p)?;
    let req: BenchRequest =
        serde_json::from_value(v).map_err(|_| Fail::from(internal("unexpected server error")))?;
    ctx.c.telemetry.bench_run();
    json(
        202,
        &ctx.c
            .bench
            .start(&req.instance_id, &req.suite, req.confirm_external)
            .await?,
    )
}

pub async fn list_bench(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 50);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    let iid = p.opt_str_q(&ctx.query, "instance_id", &T::str());
    finish(p)?;
    ok(&ctx
        .c
        .bench
        .list_page(limit, cursor.as_deref(), iid.as_deref())?)
}

pub async fn get_bench(ctx: Ctx) -> HResult {
    ok(&ctx.c.bench.get(&ctx.param("bid"))?)
}

pub async fn bench_stream(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    let bid = ctx.param("bid");
    let run = ctx.c.bench.get(&bid)?;
    let first = sse(
        &serde_json::to_value(&run).unwrap_or(Value::Null),
        Some("snapshot"),
    );
    let head = futures::stream::once(async move { first });
    if once || run.state != "running" {
        return sse_response(&ctx, head.boxed());
    }
    let sub = ctx.c.bus.subscribe(&format!("bench:{bid}"));
    let live = futures::stream::unfold((sub, false), |(mut s, done)| async move {
        if done {
            return None;
        }
        let ev = s.recv().await?;
        let last = ev.get("event").and_then(Value::as_str) == Some("done");
        Some((sse(&ev, Some("progress")), (s, last)))
    });
    sse_response(&ctx, head.chain(live).boxed())
}

pub async fn usage(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let group_by = p
        .str_q(&ctx.query, "group_by", Some("model"))
        .unwrap_or_default();
    let since = p.opt_float_q(&ctx.query, "since");
    let until = p.opt_float_q(&ctx.query, "until");
    finish(p)?;
    ok(&ctx.c.usage.query(&group_by, since, until)?)
}

pub async fn usage_csv(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let group_by = p
        .str_q(&ctx.query, "group_by", Some("model"))
        .unwrap_or_default();
    finish(p)?;
    let csv = ctx.c.usage.export_csv(&group_by)?;
    let mut r = with_ct(200, "text/csv; charset=utf-8", csv);
    r.headers_mut().insert(
        http::header::CONTENT_DISPOSITION,
        http::HeaderValue::from_static("attachment; filename=\"usage.csv\""),
    );
    Ok(r)
}

pub async fn audit(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let limit = p.int_q(&ctx.query, "limit", 100);
    let cursor = p.opt_str_q(&ctx.query, "cursor", &T::str());
    finish(p)?;
    ok(&ctx.c.audit.list_page(limit, cursor.as_deref())?)
}

/// GET /metrics: the console's own Prometheus text (no role dependency; the middleware requires a key).
pub async fn prometheus(ctx: Ctx) -> HResult {
    let c = &ctx.c;
    let items = c.lifecycle.all_instances();
    let mut states: HashMap<String, i64> = HashMap::new();
    for i in &items {
        *states.entry(i.state.clone()).or_insert(0) += 1;
    }
    c.telemetry.set_instance_states(&states);
    // the console's own machine keeps its v1 unlabelled gauges; "local" may not be in this fleet
    let local_mem = match c.hardware.host_memory(LOCAL) {
        Ok(m) => m,
        Err(AppError::Problem(_)) => None,
        Err(e @ AppError::Node(_)) => return Err(e.into()),
    };
    let per_instance: Vec<(String, f64)> = items
        .iter()
        .filter_map(|i| i.host_memory.as_ref().map(|m| (i.id.clone(), m.total_gib)))
        .collect();
    c.telemetry
        .set_host_memory(local_mem.as_ref(), &per_instance);
    let mut up = Vec::new();
    let mut gpu_used = Vec::new();
    for h in c.fleet.hosts() {
        let state = c
            .fleet
            .node(&h.id)
            .map(|n| n.link().state)
            .unwrap_or_default();
        up.push((h.id.clone(), state == "up"));
        if state == "up" || state == "stale" {
            // no sample yet: leave the host's GPUs out rather than report zero
            if let Ok(gpus) = c.hardware.gpus(&h.id) {
                for g in gpus {
                    gpu_used.push(((h.id.clone(), g.index), g.used_gib));
                }
            }
        }
    }
    let vram: Vec<((String, String, i64), f64)> = items
        .iter()
        .flat_map(|i| {
            i.vram.iter().filter_map(|v| {
                v.used_gib
                    .map(|g| ((i.id.clone(), i.host_id.clone(), v.gpu_index), g))
            })
        })
        .collect();
    c.telemetry.set_fleet(&up, &gpu_used, &vram);
    let _ = json!(null);
    Ok(with_ct(200, METRICS_CONTENT_TYPE, c.telemetry.render()))
}
