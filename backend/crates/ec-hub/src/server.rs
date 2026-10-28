//! The listener: an accept loop over hyper's HTTP/1.1 connections, with the limits uvicorn has and a few it lacks.
//!
//! Every connection gets the peer address the kernel reported (loopback trust keys on it, never on a header), a
//! header-read timeout, a keep-alive idle timeout, and a slot from a fixed connection cap: past the cap a connection
//! is answered `503` and closed at once instead of queueing. The request body has a total read deadline
//! (`Ctx::read_body`), and a body that ends before its declared length is a 400, never a complete body. Shutdown stops
//! accepting, lets idle connections close and in-flight requests finish, and waits at most `shutdown_grace` for the
//! rest (SSE streams never end by themselves; PORTING.md DV10).

use std::convert::Infallible;
use std::net::SocketAddr;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use axum::body::{Body as AxBody, HttpBody};
use bytes::Bytes;
use futures::FutureExt as _;
use http::{HeaderValue, Response, StatusCode, header};
use hyper::body::{Frame, SizeHint};
use hyper_util::rt::{TokioIo, TokioTimer};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Semaphore, watch};
use tokio::task::JoinSet;

use crate::api::http::Resp;
use crate::app::{AppState, handle_peer};

/// The listener's limits. The defaults are documented on each constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// A request head (request line + headers) must arrive within this time once the server waits for it.
    pub header_read: Duration,
    /// A connection with no request in flight and no new byte for this long is closed.
    pub keep_alive_idle: Duration,
    /// A request body must have arrived completely this long after its request started (`Ctx::read_body`).
    pub body_read: Duration,
    /// Connections served at once; one more is answered 503 and closed.
    pub max_connections: usize,
    /// How long a shutdown waits for open responses.
    pub shutdown_grace: Duration,
}

/// 10 s for a request head. uvicorn has no head timeout at all (a client may hold a half-sent head open for ever);
/// hyper's default is 30 s. A browser sends its head in one write, so 10 s only ever cuts a stalled or hostile client.
pub const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);
/// uvicorn's `timeout_keep_alive` default (5 s).
pub const KEEP_ALIVE_IDLE: Duration = Duration::from_secs(5);
/// 60 s for a whole request body: the largest accepted body (a 32 MiB chat request) needs ~4.5 Mbit/s; every other
/// cap is 1 MiB or less. uvicorn has no body deadline.
pub const BODY_READ_TIMEOUT: Duration = Duration::from_secs(60);
/// 256 connections at once. uvicorn's `limit_concurrency` is off by default; when set it answers 503 like this cap.
/// The console's own streams are capped far lower (8 SSE + 16 chat streams), and a browser opens about 6 per origin.
pub const MAX_CONNECTIONS: usize = 256;

impl Default for Limits {
    fn default() -> Self {
        Self {
            header_read: HEADER_READ_TIMEOUT,
            keep_alive_idle: KEEP_ALIVE_IDLE,
            body_read: BODY_READ_TIMEOUT,
            max_connections: MAX_CONNECTIONS,
            shutdown_grace: crate::app::SHUTDOWN_GRACE,
        }
    }
}

/// What an over-cap connection reads before it is closed (uvicorn's `limit_concurrency` answer).
const OVERLOAD: &[u8] =
    b"HTTP/1.1 503 Service Unavailable\r\ncontent-type: text/plain; charset=utf-8\r\n\
content-length: 19\r\nconnection: close\r\n\r\nService Unavailable";

/// uvicorn's (h11's) answer to a request it cannot parse, e.g. HTTP/1.1 without a Host header or with two.
pub const INVALID_REQUEST: &str = "Invalid HTTP request received.";

// ── per-connection activity ─────────────────────────────────────────────────────────────────────────────────
/// What the idle watchdog of one connection needs: requests in flight, when the last one ended, the last byte read.
struct Activity {
    epoch: Instant,
    inflight: AtomicUsize,
    /// milliseconds since `epoch` when the connection last became idle (accept, or the end of a response)
    idle_since_ms: AtomicU64,
    /// milliseconds since `epoch` of the last byte read
    last_read_ms: AtomicU64,
}

impl Activity {
    fn new() -> Self {
        Self {
            epoch: Instant::now(),
            inflight: AtomicUsize::new(0),
            idle_since_ms: AtomicU64::new(0),
            last_read_ms: AtomicU64::new(0),
        }
    }
    fn now_ms(&self) -> u64 {
        u64::try_from(self.epoch.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

/// One request in flight; dropping it (the response body is done or abandoned) may make the connection idle.
struct InFlight(Arc<Activity>);

impl InFlight {
    fn start(a: &Arc<Activity>) -> Self {
        a.inflight.fetch_add(1, Ordering::SeqCst);
        Self(Arc::clone(a))
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        let now = self.0.now_ms();
        self.0.idle_since_ms.store(now, Ordering::SeqCst);
        self.0.inflight.fetch_sub(1, Ordering::SeqCst);
    }
}

/// The socket, recording when bytes arrive (the idle watchdog must not close a connection that is sending a head).
struct Watched {
    inner: TcpStream,
    act: Arc<Activity>,
}

impl AsyncRead for Watched {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        let before = buf.filled().len();
        let r = Pin::new(&mut self.inner).poll_read(cx, buf);
        if buf.filled().len() > before {
            let now = self.act.now_ms();
            self.act.last_read_ms.store(now, Ordering::SeqCst);
        }
        r
    }
}

impl AsyncWrite for Watched {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write(cx, buf)
    }
    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_flush(cx)
    }
    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<std::io::Result<()>> {
        Pin::new(&mut self.inner).poll_shutdown(cx)
    }
    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[std::io::IoSlice<'_>],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.inner).poll_write_vectored(cx, bufs)
    }
    fn is_write_vectored(&self) -> bool {
        self.inner.is_write_vectored()
    }
}

/// Resolves when the connection has been idle (nothing in flight, no byte read since its last response) for
/// `limit`. A connection that is sending a head is left to the header-read timeout instead.
async fn idle_expired(act: Arc<Activity>, limit: Duration) {
    let limit_ms = u64::try_from(limit.as_millis()).unwrap_or(u64::MAX);
    loop {
        let busy = act.inflight.load(Ordering::SeqCst) > 0;
        let idle_since = act.idle_since_ms.load(Ordering::SeqCst);
        let last_read = act.last_read_ms.load(Ordering::SeqCst);
        let now = act.now_ms();
        if busy || last_read > idle_since {
            tokio::time::sleep(limit).await;
            continue;
        }
        let due = idle_since.saturating_add(limit_ms);
        if now >= due {
            return;
        }
        tokio::time::sleep(Duration::from_millis(due - now)).await;
    }
}

/// A response body that keeps its request counted as in flight until hyper has written or dropped it. Size hint and
/// end-of-stream are the inner body's, so a fixed-size answer keeps its Content-Length (no chunked encoding).
struct Held {
    inner: AxBody,
    _held: InFlight,
}

impl HttpBody for Held {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, axum::Error>>> {
        Pin::new(&mut self.get_mut().inner).poll_frame(cx)
    }
    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }
    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

// ── one request ─────────────────────────────────────────────────────────────────────────────────────────────
/// The plain-text 400 uvicorn sends for a request h11 refuses (no or several Host headers on HTTP/1.1).
fn invalid_request() -> Resp {
    let mut r = Response::new(AxBody::from(INVALID_REQUEST));
    *r.status_mut() = StatusCode::BAD_REQUEST;
    let h = r.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; charset=utf-8"),
    );
    h.insert(header::CONNECTION, HeaderValue::from_static("close"));
    crate::app::add_security_headers(&mut r, "");
    r
}

/// h11's Host rule: HTTP/1.1 needs exactly one Host header; HTTP/1.0 may have none (never two).
pub fn host_header_ok(req: &http::Request<impl Sized>) -> bool {
    let n = req.headers().get_all(header::HOST).iter().count();
    n == 1 || (n == 0 && req.version() < http::Version::HTTP_11)
}

/// Serve one request of a connection: the app's answer, with the request counted as in flight until its response
/// body is done. A panic outside the handlers' own guard is still the generic 500.
async fn serve_request(
    st: Arc<AppState>,
    peer: SocketAddr,
    act: Arc<Activity>,
    req: http::Request<hyper::body::Incoming>,
) -> Result<Resp, Infallible> {
    let held = InFlight::start(&act);
    let req = req.map(AxBody::new);
    let resp = if host_header_ok(&req) {
        match AssertUnwindSafe(handle_peer(&st, peer, req))
            .catch_unwind()
            .await
        {
            Ok(r) => r,
            Err(_) => {
                tracing::error!(target: "engine_console.api.errors", "unhandled panic");
                crate::api::http::problem(
                    500,
                    "internal_error",
                    "internal error",
                    "unexpected server error",
                    &serde_json::Map::new(),
                )
            }
        }
    } else {
        invalid_request()
    };
    // the request stays in flight until hyper has written (or dropped) the whole body
    let (parts, body) = resp.into_parts();
    Ok(Response::from_parts(
        parts,
        AxBody::new(Held {
            inner: body,
            _held: held,
        }),
    ))
}

/// The peer address as the trust rules see it: `scope["client"][0]` (the IP only, as uvicorn reports it).
pub fn client_host(peer: SocketAddr) -> String {
    peer.ip().to_string()
}

// ── the loop ────────────────────────────────────────────────────────────────────────────────────────────────
fn builder(limits: &Limits) -> hyper::server::conn::http1::Builder {
    let mut b = hyper::server::conn::http1::Builder::new();
    b.timer(TokioTimer::new())
        .header_read_timeout(limits.header_read)
        // a client that shuts down its sending side after the body still reads the answer (e.g. the 400 for a
        // body that ended early)
        .half_close(true)
        .keep_alive(true);
    b
}

async fn connection(
    st: Arc<AppState>,
    stream: TcpStream,
    peer: SocketAddr,
    limits: Limits,
    mut stop: watch::Receiver<bool>,
) {
    let _ = stream.set_nodelay(true);
    let act = Arc::new(Activity::new());
    let io = TokioIo::new(Watched {
        inner: stream,
        act: Arc::clone(&act),
    });
    let svc_act = Arc::clone(&act);
    let svc = hyper::service::service_fn(move |req| {
        serve_request(Arc::clone(&st), peer, Arc::clone(&svc_act), req)
    });
    let conn = builder(&limits).serve_connection(io, svc);
    tokio::pin!(conn);
    let idle = idle_expired(Arc::clone(&act), limits.keep_alive_idle);
    tokio::pin!(idle);
    let mut draining = false;
    loop {
        tokio::select! {
            r = conn.as_mut() => {
                if let Err(e) = r {
                    tracing::debug!(target: "uvicorn.error", "connection from {peer}: {e}");
                }
                return;
            }
            stopped = async { stop.wait_for(|v| *v).await.is_ok() }, if !draining => {
                draining = true;
                if stopped {
                    conn.as_mut().graceful_shutdown(); // idle: closes now; busy: after the response
                }
            }
            () = &mut idle => return, // dropping the connection closes the socket
        }
    }
}

/// Accept and serve until `stop` turns true, then drain (at most `limits.shutdown_grace`). Returns when every
/// connection is closed or abandoned.
pub async fn serve_on(
    listener: TcpListener,
    st: Arc<AppState>,
    limits: Limits,
    mut stop: watch::Receiver<bool>,
) {
    let slots = Arc::new(Semaphore::new(limits.max_connections.max(1)));
    let mut conns: JoinSet<()> = JoinSet::new();
    let conn_stop = stop.clone();
    loop {
        tokio::select! {
            () = async { let _ = stop.wait_for(|v| *v).await; } => break,
            Some(_) = conns.join_next(), if !conns.is_empty() => {}
            acc = listener.accept() => match acc {
                Ok((stream, peer)) => match Arc::clone(&slots).try_acquire_owned() {
                    Ok(slot) => {
                        let st = Arc::clone(&st);
                        let stop = conn_stop.clone();
                        conns.spawn(async move {
                            let _slot = slot;
                            let what = format!("connection from {peer}");
                            crate::panics::guarded(&what, connection(st, stream, peer, limits, stop)).await;
                        });
                    }
                    Err(_) => {
                        // over the cap: answer at once and close, never queue
                        refuse(stream);
                        tracing::warn!(target: "uvicorn.error",
                            "connection limit ({}) reached: refused {peer}", limits.max_connections);
                    }
                },
                Err(e) => {
                    // EMFILE / ENFILE / ECONNABORTED: log, back off briefly, keep serving
                    tracing::error!(target: "uvicorn.error", "accept failed: {e}");
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            },
        }
    }
    drop(listener); // stop accepting first, as uvicorn does
    let drained = tokio::time::timeout(limits.shutdown_grace, async {
        while conns.join_next().await.is_some() {}
    })
    .await;
    if drained.is_err() {
        conns.abort_all();
        while conns.join_next().await.is_some() {}
    }
}

/// Answer an over-cap connection with the 503 and close it, without waiting: one non-blocking write into the fresh
/// socket's empty send buffer (a tokio socket has no readiness yet, so this goes through the std socket).
fn refuse(stream: TcpStream) {
    if let Ok(mut s) = stream.into_std() {
        let _ = std::io::Write::write(&mut s, OVERLOAD);
        let _ = s.shutdown(std::net::Shutdown::Write);
    }
}

/// A stop switch for `serve_on` (true = stop).
pub fn stop_switch() -> (watch::Sender<bool>, watch::Receiver<bool>) {
    watch::channel(false)
}
