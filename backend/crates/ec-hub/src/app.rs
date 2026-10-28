//! The HTTP application (port of main.py `create_app` + the pure-ASGI SecurityMiddleware of api/security.py).
//!
//! Every request goes through `handle_request`, in the middleware's order: Host / CSRF / size prechecks,
//! authentication and the viewer rule for guarded paths, routing (Starlette semantics, api/router.rs), the route's
//! role (its declared `Access`, checked here after authentication and before the handler, as FastAPI runs a route's
//! dependencies), the handler, then the security headers (and `Cache-Control: no-store` on /api/), the request
//! metrics, an OTLP span and, for a mutating call on a guarded path, one audit row. For a streamed response the
//! metrics/span/audit run when the stream ends. The listener (limits, peer address, shutdown) is server.rs.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use axum::body::Body as AxBody;
use axum::extract::Request;
use futures::FutureExt;
use http::{HeaderName, HeaderValue};
use serde_json::Map;

use crate::api::deps::check_access;
use crate::api::http::{BodyCapture, Ctx, Resp, fail_response, problem};
use crate::api::router::{Match, Router};
use crate::api::security::{self, KNOWN_METHODS, MUTATING, SECURITY_HEADERS, guarded};
use crate::container::Container;
use crate::server::Limits;
use crate::services::settings::Principal;
use crate::services::telemetry::otel;
use crate::sync::Recover as _;

pub struct AppState {
    pub c: Arc<Container>,
    pub router: Router,
    pub hosts: HashSet<String>,
    /// the listener's limits (the body-read deadline is applied per request here)
    pub limits: Limits,
}

pub fn app_state(c: Arc<Container>) -> Arc<AppState> {
    app_state_with(c, Limits::default())
}

/// `app_state` with the listener's limits given (tests shorten them).
pub fn app_state_with(c: Arc<Container>, limits: Limits) -> Arc<AppState> {
    let router = crate::api::routes::router(c.frontend.as_deref());
    let hosts = security::allowed_hosts(&c.cfg);
    Arc::new(AppState {
        c,
        router,
        hosts,
        limits,
    })
}

/// Starlette's header decoding: the bytes as latin-1 (`bytes.decode("latin-1")`), so every byte is one character and
/// no header that is present can come out empty or "absent" (a non-ASCII Origin is still an Origin).
pub fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

/// Starlette's scope["path"]: the raw path percent-decoded (invalid UTF-8 replaced).
pub fn decode_path(raw: &str) -> String {
    percent_encoding::percent_decode_str(raw)
        .decode_utf8_lossy()
        .into_owned()
}

/// One request from the peer the listener accepted: the trust rules see only the socket's address (never a header).
pub async fn handle_peer(st: &Arc<AppState>, peer: SocketAddr, req: Request) -> Resp {
    handle_request(st, &crate::server::client_host(peer), req).await
}

/// The middleware + router for one request (tests call it without a socket).
pub async fn handle_request(st: &Arc<AppState>, client_host: &str, req: Request) -> Resp {
    let t0 = Instant::now();
    let start = SystemTime::now();
    let c = Arc::clone(&st.c);
    let (parts, body) = req.into_parts();
    let method = parts.method.as_str().to_string();
    let path = decode_path(parts.uri.path());
    let raw_query = parts.uri.query().unwrap_or("").to_string();
    let mut headers: HashMap<String, String> = HashMap::new();
    for (k, v) in &parts.headers {
        headers.insert(k.as_str().to_lowercase(), latin1(v.as_bytes())); // last wins
    }
    let mut principal = Principal::public();
    let mut deny = security::precheck(&c.cfg, &st.hosts, &method, &path, &headers);
    if deny.is_none() && guarded(&path) {
        match security::principal(&c.settings, c.cfg.trust_loopback, client_host, &headers) {
            None => deny = Some((401, "unauthorized", "a valid bearer API key is required")),
            Some(p) => {
                if p.role != "admin"
                    && MUTATING.contains(&method.as_str())
                    && !security::viewer_may(&method, &path)
                {
                    deny = Some((403, "forbidden", "this key has the viewer role (read-only)"));
                }
                principal = p;
            }
        }
    }
    let capture = Arc::new(Mutex::new(BodyCapture::default()));
    let mut route_label: Option<&'static str> = None;
    let mut resp = if let Some((status, code, detail)) = deny {
        problem(status, code, &code.replace('_', " "), detail, &Map::new())
    } else {
        let cap = security::cap(&c.cfg, &path);
        let mut ctx = Ctx::new(
            Arc::clone(&c),
            method.clone(),
            path.clone(),
            raw_query.clone(),
            headers.clone(),
            principal.clone(),
            cap,
            Arc::clone(&capture),
            body,
        );
        ctx.set_body_deadline(t0 + st.limits.body_read);
        match st.router.find(&method, &path) {
            Match::Full(route, params) => {
                route_label = Some(route.path);
                // the route's role dependencies: after authentication, before the handler (and so before any
                // validation error), exactly the problem the dependency raises
                match check_access(route.access, &method, &path, &principal) {
                    Err(f) => fail_response(&c, f),
                    Ok(()) => {
                        ctx.params = params;
                        run_handler(&c, route.handler, ctx).await
                    }
                }
            }
            Match::Partial(route) => {
                route_label = Some(route.path);
                problem(
                    405,
                    "method_not_allowed",
                    "method not allowed",
                    "Method Not Allowed",
                    &Map::new(),
                )
            }
            Match::None => match st.router.slash_redirect(&method, &path) {
                Some(to) => {
                    let mut r = problem(307, "redirect", "redirect", "", &Map::new());
                    *r.body_mut() = AxBody::empty();
                    r.headers_mut().remove(http::header::CONTENT_TYPE);
                    let loc = if raw_query.is_empty() {
                        to
                    } else {
                        format!("{to}?{raw_query}")
                    };
                    if let Ok(v) = HeaderValue::from_str(&loc) {
                        r.headers_mut().insert(http::header::LOCATION, v);
                    }
                    r
                }
                None => problem(404, "not_found", "not found", "Not Found", &Map::new()),
            },
        }
    };
    add_security_headers(&mut resp, &path);
    let status = resp.status().as_u16();
    let label = route_label
        .and_then(metric_route)
        .unwrap_or(if guarded(&path) {
            "unmatched"
        } else {
            "static"
        })
        .to_string();
    let mlabel = if KNOWN_METHODS.contains(&method.as_str()) {
        method.clone()
    } else {
        "OTHER".to_string()
    };
    let audit = guarded(&path) && MUTATING.contains(&method.as_str());
    let finalize = move || {
        let dur = t0.elapsed().as_secs_f64();
        c.telemetry.request(&mlabel, &label, status);
        c.telemetry.observe_latency(&mlabel, &label, dur);
        if c.cfg.otlp_enabled {
            otel::request_span(&mlabel, &label, status, start);
        }
        if audit {
            let (head, total) = {
                let b = capture.guard();
                (b.head.clone(), b.total)
            };
            let params = security::audit_params(&path, &raw_query, &headers, &head, total);
            let actor = principal
                .key_id
                .clone()
                .unwrap_or_else(|| principal.name.clone());
            if c.audit
                .record(&actor, &principal.role, &method, &path, status, &params)
                .is_err()
            {
                // never fail a request because audit storage hiccuped
                tracing::error!(target: "engine_console.api.security", "audit write failed");
            }
        }
    };
    if is_stream(&resp) {
        let (parts, body) = resp.into_parts();
        let wrapped = OnEnd::new(body.into_data_stream(), Box::new(finalize));
        return Resp::from_parts(parts, AxBody::from_stream(wrapped));
    }
    finalize();
    resp
}

/// The handler's response; its failure rendered as a problem; a panic (an uncaught adapter exception, a bug) as
/// Python's catch-all 500. The caller still records the metrics, span and audit row of the request.
async fn run_handler(c: &Arc<Container>, handler: crate::api::router::Handler, ctx: Ctx) -> Resp {
    match std::panic::AssertUnwindSafe(handler(ctx))
        .catch_unwind()
        .await
    {
        Ok(Ok(r)) => r,
        Ok(Err(f)) => fail_response(c, f),
        Err(payload) => {
            match payload.downcast_ref::<crate::adapters::pycompat::PyExc>() {
                // an adapter raised a Python exception nobody caught (PORTING.md §A5): Python's catch-all
                Some(e) => tracing::error!(
                    target: "engine_console.api.errors",
                    "unhandled exception: {}",
                    e.kind
                ),
                None => {
                    tracing::error!(target: "engine_console.api.errors", "unhandled panic");
                }
            }
            problem(
                500,
                "internal_error",
                "internal error",
                "unexpected server error",
                &Map::new(),
            )
        }
    }
}

/// The `route` label of the request metrics: FastAPI keeps the router-relative path of an API route in
/// scope["route"] (the routers are included with the /api/v1 prefix), so "/api/v1/health" counts as "/health".
/// The OpenAPI document is a plain starlette route that sets no scope["route"]: it counts as "unmatched".
pub fn metric_route(template: &'static str) -> Option<&'static str> {
    if template == crate::api::routes::OPENAPI {
        return None;
    }
    Some(template.strip_prefix("/api/v1").unwrap_or(template))
}

fn is_stream(r: &Resp) -> bool {
    r.headers()
        .get(http::header::CONTENT_TYPE)
        .is_some_and(|v| v.as_bytes().starts_with(b"text/event-stream"))
}

/// Replace any security header a handler set with ours, and `Cache-Control: no-store` on /api/ paths.
pub fn add_security_headers(resp: &mut Resp, path: &str) {
    let h = resp.headers_mut();
    let api = path.starts_with("/api/");
    for (name, value) in SECURITY_HEADERS {
        h.insert(
            HeaderName::from_static(name),
            HeaderValue::from_static(value),
        );
    }
    if api {
        h.insert(
            http::header::CACHE_CONTROL,
            HeaderValue::from_static("no-store"),
        );
    }
}

/// A body stream that runs `f` once when it ends or is dropped (the middleware's `finally` for streams).
pub struct OnEnd<S> {
    inner: S,
    f: Option<Box<dyn FnOnce() + Send>>,
}

impl<S> OnEnd<S> {
    pub fn new(inner: S, f: Box<dyn FnOnce() + Send>) -> Self {
        Self { inner, f: Some(f) }
    }
    fn fire(&mut self) {
        if let Some(f) = self.f.take() {
            f();
        }
    }
}

impl<S> Drop for OnEnd<S> {
    fn drop(&mut self) {
        self.fire();
    }
}

impl<S: futures::Stream + Unpin> futures::Stream for OnEnd<S> {
    type Item = S::Item;
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let r = std::pin::Pin::new(&mut self.inner).poll_next(cx);
        if let std::task::Poll::Ready(None) = r {
            self.fire();
        }
        r
    }
}

// ── serving ──────────────────────────────────────────────────────────────────────────────────────────────
/// One background loop (main.py `_loop`): run `f` every `interval`; a round that fails, or panics, is logged
/// (redacted) and the loop goes on.
pub async fn run_loop<F, Fut>(mut f: F, interval: std::time::Duration)
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = crate::domain::errors::AppResult<()>>,
{
    loop {
        match crate::panics::guarded("background loop error", f()).await {
            Some(Ok(())) | None => {}
            Some(Err(_)) => tracing::error!(target: "engine_console", "background loop error"),
        }
        tokio::time::sleep(interval).await;
    }
}

/// The lifespan's start (main.py `lifespan`, before `yield`): the bootstrap key and its warnings, recovery of what
/// the last run left open, the fleet's links, the library, and the background loops (when enabled). Returns the
/// loops' tasks for `lifespan_stop`.
pub async fn lifespan_start(c: &Arc<Container>) -> Vec<tokio::task::JoinHandle<()>> {
    let cfg = Arc::clone(&c.cfg);
    match c.settings.ensure_bootstrap_key(&cfg.data_dir) {
        Ok(Some(p)) => {
            tracing::info!(target: "engine_console", "first start: admin API key written to {} (0600)", p.display());
        }
        Ok(None) => {}
        Err(e) => tracing::error!(target: "engine_console", "bootstrap key: {}", e),
    }
    let existing = cfg
        .data_dir
        .join(crate::services::settings::BOOTSTRAP_KEY_FILE);
    if existing.exists() {
        tracing::warn!(target: "engine_console",
            "bootstrap admin key file {} exists: copy the key somewhere safe (or mint a new one via POST /keys) and DELETE this file after first use",
            existing.display());
    }
    if !cfg.trust_loopback {
        tracing::info!(target: "engine_console", "loopback is not trusted (ENGINE_CONSOLE_TRUST_LOOPBACK=false): the UI asks for an API key");
    }
    let _ = c.downloads.recover();
    let _ = c.bench.recover();
    c.fleet.start().await;
    c.library.start().await;
    let mut tasks = Vec::new();
    if cfg.background_tasks {
        let life = Arc::clone(&c.lifecycle);
        let every = std::time::Duration::from_secs_f64(cfg.supervisor_interval_s.max(0.0));
        tasks.push(tokio::spawn(async move {
            run_loop(
                || {
                    let life = Arc::clone(&life);
                    async move { life.tick().await }
                },
                every,
            )
            .await;
        }));
        let metrics = Arc::clone(&c.metrics);
        tasks.push(tokio::spawn(async move { metrics.run().await })); // each round and message is guarded inside
    }
    tasks
}

/// The lifespan's end (after `yield`), in Python's order.
pub async fn lifespan_stop(c: &Arc<Container>, tasks: Vec<tokio::task::JoinHandle<()>>) {
    for t in &tasks {
        t.abort();
    }
    for t in tasks {
        let _ = t.await;
    }
    // the connections are closed: wait for the chat replies their streams handed off (Python wrote them in the
    // stream's `finally`, before the server finished)
    c.chat.flush().await;
    c.scripts.shutdown().await;
    c.bench.shutdown().await;
    c.downloads.shutdown().await;
    c.library.stop().await;
    c.fleet.stop().await;
}

/// `engine-console` (no subcommand): start-up (bootstrap key, recover, fleet, library, background loops), serve
/// until SIGINT/SIGTERM, then shut everything down in Python's order. Returns the process exit code.
pub async fn serve(c: Arc<Container>) -> i32 {
    let cfg = Arc::clone(&c.cfg);
    let provider = otel::setup_tracing(&cfg);
    let tasks = lifespan_start(&c).await;
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let listener = match tokio::net::TcpListener::bind((cfg.host.as_str(), cfg.port)).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(target: "uvicorn.error", "[Errno {}] error while attempting to bind on address {}: {}", e.raw_os_error().unwrap_or(0), addr, e);
            lifespan_stop(&c, tasks).await;
            return 1;
        }
    };
    tracing::info!(target: "uvicorn.error", "Uvicorn running on http://{} (Press CTRL+C to quit)", addr);
    let state = app_state(Arc::clone(&c));
    // SIGINT/SIGTERM: stop accepting, let requests finish, but never wait more than SHUTDOWN_GRACE for open streams
    let (tx, rx) = crate::server::stop_switch();
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = tx.send(true);
    });
    crate::server::serve_on(listener, state, Limits::default(), rx).await;
    lifespan_stop(&c, tasks).await;
    if let Some(p) = provider {
        let _ = p.shutdown();
    }
    0
}

/// How long a shutdown waits for open responses (SSE streams never end by themselves).
pub const SHUTDOWN_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

async fn shutdown_signal() {
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        () = term => {}
    }
    tracing::info!(target: "uvicorn.error", "Shutting down");
}
