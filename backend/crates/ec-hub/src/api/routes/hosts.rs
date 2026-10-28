//! /hosts and the live /fleet/stream (port of api/routers/hosts.py; contract §10.8).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::StreamExt;

use crate::api::deps::sse_response;
use crate::api::http::{Ctx, Fail, HResult, ok};
use crate::api::validate::Params;
use crate::domain::fleet::Fleet;
use crate::domain::models::Page;
use crate::pyjson;
use crate::services::common::{EventBus, sse_text};

/// how often the stream looks for a change nobody announced (live GPU numbers, a retry clock)
pub const STREAM_POLL_S: f64 = 1.0;
/// every host is re-sent at least this often, changed or not
pub const STREAM_REFRESH_S: f64 = 5.0;

fn finish(p: Params) -> Result<(), Fail> {
    if p.ok() {
        Ok(())
    } else {
        Err(Fail::Validation(p.errs))
    }
}

/// Event `host` (a HostSummary, compact JSON) for every host first, then for each host that changed (compared
/// every `poll`, or at once when the bus topic "fleet" announces a change), and for every host at least every
/// `refresh`.
pub fn fleet_events(
    fleet: Arc<dyn Fleet>,
    bus: &Arc<EventBus>,
    once: bool,
    poll: Duration,
    refresh: Duration,
) -> futures::stream::BoxStream<'static, String> {
    let mut last: HashMap<String, (String, Instant)> = HashMap::new();
    let mut due = move |fleet: &Arc<dyn Fleet>| -> Vec<String> {
        let now = Instant::now();
        let mut out = Vec::new();
        for s in fleet.summaries().unwrap_or_default() {
            let data = pyjson::dumps_compact(&serde_json::to_value(&s).unwrap_or_default());
            let fresh = match last.get(&s.id) {
                None => true,
                Some((prev, at)) => *prev != data || now.duration_since(*at) >= refresh,
            };
            if fresh {
                last.insert(s.id.clone(), (data.clone(), now));
                out.push(sse_text(&data, Some("host")));
            }
        }
        out
    };
    if once {
        return futures::stream::iter(due(&fleet)).boxed();
    }
    let sub = bus.subscribe("fleet");
    futures::stream::unfold(
        (fleet, sub, due, true),
        move |(fleet, mut sub, mut due, first)| async move {
            if !first {
                // wait for a poll period, or a change announced on the bus, whichever comes first
                let _ = tokio::time::timeout(poll, sub.recv()).await;
            }
            let chunk = due(&fleet);
            Some((futures::stream::iter(chunk), (fleet, sub, due, false)))
        },
    )
    .flatten()
    .boxed()
}

pub async fn list_hosts(ctx: Ctx) -> HResult {
    ok(&Page::new(ctx.c.fleet.summaries()?, None))
}

pub async fn get_host(ctx: Ctx) -> HResult {
    ok(&ctx.c.fleet.detail(&ctx.param("host_id")).await?)
}

pub async fn reconnect_host(ctx: Ctx) -> HResult {
    ok(&ctx.c.fleet.reconnect(&ctx.param("host_id"))?)
}

pub async fn fleet_stream(ctx: Ctx) -> HResult {
    let mut p = Params::default();
    let once = p.bool_q(&ctx.query, "once", false);
    finish(p)?;
    // a fleet that cannot list its hosts (not ported yet) answers its error instead of an empty stream
    ctx.c.fleet.summaries()?;
    let events = fleet_events(
        Arc::clone(&ctx.c.fleet),
        &ctx.c.bus,
        once,
        Duration::from_secs_f64(STREAM_POLL_S),
        Duration::from_secs_f64(STREAM_REFRESH_S),
    );
    sse_response(&ctx, events)
}
