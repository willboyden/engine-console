//! Python: backend/src/engine_console/services/hf_http.py
//! (tests/core/test_security.py egress cases, tests/data/*).
//!
//! The real `HubClient` over reqwest (rustls) with the ADR 8 egress allowlist: https only on every hop (redirects
//! included), host allowlist (fnmatch, `services::common::fnmatchcase`), the token only in an Authorization header
//! and dropped on cross-origin redirects (reqwest strips it on a change of scheme, host or port, as httpx does),
//! ambient proxy and CA variables never used, and the optional egress proxy (EGRESS_PROXY) with its CA bundle
//! (EGRESS_CA_BUNDLE) failing closed (503 egress_proxy_unavailable, never a direct fallback). Resumable downloads
//! via Range (`ByteStream::offset`).
//!
//! Trust roots, fixed so that no environment variable can steer them (Python: `trust_env=False`): with a CA bundle
//! configured only that bundle (Python: `ssl.create_default_context(cafile=...)`); otherwise the system's CA bundle
//! file read from its fixed path (Python: certifi's Mozilla bundle when direct, the system store when proxied).

use crate::sync::Recover;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use serde_json::{Map, Value};

use crate::domain::errors::{
    AppResult, Problem, egress_denied, egress_proxy_unavailable, gated_model, internal, not_found,
    upstream_error,
};
use crate::domain::ports::{ByteStream, HubClient, HubModel, RepoFile};
use crate::services::gguf_info::py;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EgressConfig {
    pub proxy: Option<String>,
    pub ca_bundle: Option<std::path::PathBuf>,
    pub require_proxy: bool,
}

impl EgressConfig {
    pub fn mode(&self) -> &'static str {
        if self.proxy.as_deref().is_some_and(|p| !p.is_empty()) {
            "proxied"
        } else {
            "direct"
        }
    }

    fn proxy_url(&self) -> Option<&str> {
        self.proxy.as_deref().filter(|p| !p.is_empty())
    }
}

/// `host_allowed`: fnmatch (case-insensitive) against the allowlist patterns.
pub fn host_allowed(host: &str, patterns: &[String]) -> bool {
    let h = host.to_lowercase();
    patterns
        .iter()
        .any(|p| crate::services::common::fnmatchcase(&h, &p.to_lowercase()))
}

pub type TokenFn = Arc<dyn Fn() -> Option<String> + Send + Sync>;

/// httpx's `Timeout(30.0, read=120.0)`: connect 30 s, every read 120 s.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(120);
/// httpx's DEFAULT_MAX_REDIRECTS
const MAX_REDIRECTS: usize = 20;
/// `aiter_bytes(1024 * 1024)`: body chunks of exactly 1 MiB (the last one shorter)
const CHUNK: usize = 1024 * 1024;
/// a buffered (non-streamed) answer larger than this is refused instead of held in memory
const MAX_BODY: usize = 64 * 1024 * 1024;
/// System CA bundles, in OpenSSL's probing order; the first readable one is used (never an environment variable).
pub const SYSTEM_CA_BUNDLES: &[&str] = &[
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/ca-bundle.pem",
    "/etc/pki/tls/cacert.pem",
    "/etc/ssl/cert.pem",
];

/// The egress rule refused a hop (the request hook's EgressDenied).
#[derive(Debug)]
struct Blocked(String);

impl std::fmt::Display for Blocked {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Blocked {}

#[derive(Debug)]
struct TooManyRedirects;

impl std::fmt::Display for TooManyRedirects {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("too many redirects")
    }
}
impl std::error::Error for TooManyRedirects {}

/// `quote(s, safe=safe)` (urllib): unreserved characters and `safe` kept, every other UTF-8 byte as %XX.
pub fn quote(s: &str, safe: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let c = char::from(b);
        if b.is_ascii_alphanumeric() || "_.-~".contains(c) || (b.is_ascii() && safe.contains(c)) {
            out.push(c);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// httpx's query encoding: unreserved characters kept, a space as "+", everything else as %XX.
pub fn query_encode(params: &[(String, String)]) -> String {
    let enc = |s: &str| {
        let mut out = String::new();
        for b in s.bytes() {
            let c = char::from(b);
            if b.is_ascii_alphanumeric() || "_.-~".contains(c) {
                out.push(c);
            } else if b == b' ' {
                out.push('+');
            } else {
                out.push_str(&format!("%{b:02X}"));
            }
        }
        out
    };
    params
        .iter()
        .map(|(k, v)| format!("{}={}", enc(k), enc(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// The egress rule for one request or redirect hop: https only, host on the allowlist.
pub fn check_egress(url: &url::Url, allowed: &[String]) -> Result<(), String> {
    if url.scheme() != "https" {
        return Err("refusing a non-https request/redirect target for Hugging Face traffic".into());
    }
    let host = url_host(url);
    if !host_allowed(&host, allowed) {
        return Err(format!("egress to {host} is not on the HF allowlist"));
    }
    Ok(())
}

/// The host as httpx's `URL.host` gives it (an IPv6 address without brackets).
fn url_host(url: &url::Url) -> String {
    match url.host() {
        Some(url::Host::Ipv6(a)) => a.to_string(),
        Some(h) => h.to_string(),
        None => String::new(),
    }
}

/// `url_host(url)`: the host name of a URL ("" when there is none).
pub fn host_of(url: &str) -> String {
    url::Url::parse(url)
        .map(|u| url_host(&u))
        .unwrap_or_default()
}

/// `public_proxy(url)`: the proxy URL without any userinfo, safe to show.
pub fn public_proxy(url: Option<&str>) -> Option<String> {
    crate::services::settings::public_proxy(url)
}

/// The text of every error in a chain (for classifying transport failures).
fn chain_text(e: &(dyn std::error::Error + 'static)) -> String {
    let mut out = e.to_string();
    let mut cur = e.source();
    while let Some(s) = cur {
        out.push_str(" | ");
        out.push_str(&s.to_string());
        cur = s.source();
    }
    out
}

fn find<'a, T: std::error::Error + 'static>(
    e: &'a (dyn std::error::Error + 'static),
) -> Option<&'a T> {
    let mut cur: Option<&(dyn std::error::Error + 'static)> = Some(e);
    while let Some(x) = cur {
        if let Some(t) = x.downcast_ref::<T>() {
            return Some(t);
        }
        cur = x.source();
    }
    None
}

/// The httpx exception name a reqwest failure corresponds to (the problem text names it, as Python's does).
fn httpx_name(e: &reqwest::Error) -> &'static str {
    let text = chain_text(e);
    if find::<TooManyRedirects>(e).is_some() {
        "TooManyRedirects"
    } else if e.is_connect() && e.is_timeout() {
        "ConnectTimeout"
    } else if e.is_connect() {
        "ConnectError"
    } else if e.is_timeout() {
        "ReadTimeout"
    } else if text.contains("connection closed before message completed")
        || text.contains("incomplete")
        || text.contains("unexpected EOF")
    {
        "RemoteProtocolError"
    } else if e.is_body() || e.is_decode() {
        "ReadError"
    } else {
        "ConnectError"
    }
}

pub struct HfHttpClient {
    endpoint: String,
    allowed: Vec<String>,
    token: TokenFn,
    egress: EgressConfig,
    built: Mutex<Option<reqwest::Client>>,
}

impl std::fmt::Debug for HfHttpClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfHttpClient")
            .field("endpoint", &self.endpoint)
            .finish_non_exhaustive()
    }
}

/// How the client was built (tests check the trust settings, as Python's tests check `_trust_env`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustRoots {
    /// only the configured egress CA bundle
    EgressBundle(std::path::PathBuf),
    /// the system CA bundle file at its fixed path
    SystemBundle(String),
}

/// The most bytes a CA bundle may have (a system bundle is a few hundred KiB).
const MAX_CA_BUNDLE: u64 = 16 * 1024 * 1024;

/// Read a CA bundle file: opened non-blocking (a FIFO at that path cannot hang the hub), required to be a regular file
/// by fstat on the open descriptor, and read under [`MAX_CA_BUNDLE`]. Symlinks are followed on purpose: CA files are
/// often mounted through links, and Python's `ssl.create_default_context(cafile=...)` follows them.
fn read_ca_file(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read as _;
    use std::os::unix::fs::OpenOptionsExt as _;
    let f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(nix::fcntl::OFlag::O_NONBLOCK.bits())
        .open(path)?;
    if !f.metadata()?.is_file() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "not a regular file",
        ));
    }
    let mut out = Vec::new();
    f.take(MAX_CA_BUNDLE + 1).read_to_end(&mut out)?;
    if u64::try_from(out.len()).unwrap_or(u64::MAX) > MAX_CA_BUNDLE {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "too large",
        ));
    }
    Ok(out)
}

fn system_roots() -> Result<(String, Vec<reqwest::Certificate>), Problem> {
    for p in SYSTEM_CA_BUNDLES {
        if let Ok(pem) = read_ca_file(Path::new(p))
            && let Ok(certs) = reqwest::Certificate::from_pem_bundle(&pem)
            && !certs.is_empty()
        {
            return Ok(((*p).to_string(), certs));
        }
    }
    Err(upstream_error(
        "cannot reach Hugging Face: no CA certificate bundle found on this machine",
    ))
}

impl HfHttpClient {
    pub fn new(
        endpoint: &str,
        allowed_hosts: Vec<String>,
        token: TokenFn,
        egress: EgressConfig,
    ) -> Self {
        Self {
            endpoint: endpoint.trim_end_matches('/').to_string(),
            allowed: allowed_hosts,
            token,
            egress,
            built: Mutex::new(None),
        }
    }

    /// The trust roots a client of this configuration uses (the same decision `client()` makes).
    pub fn trust_roots(&self) -> Result<TrustRoots, Problem> {
        match (self.egress.proxy_url(), &self.egress.ca_bundle) {
            (Some(_), Some(b)) => Ok(TrustRoots::EgressBundle(b.clone())),
            _ => system_roots().map(|(p, _)| TrustRoots::SystemBundle(p)),
        }
    }

    /// Fail-closed client factory. Every network operation goes through here, so a missing CA, a missing proxy
    /// (when required) or an unbuildable TLS context stops the request instead of degrading to direct.
    fn client(&self) -> Result<reqwest::Client, Problem> {
        let eg = &self.egress;
        if eg.require_proxy && eg.proxy_url().is_none() {
            return Err(egress_proxy_unavailable(
                "REQUIRE_EGRESS_PROXY is set but EGRESS_PROXY is not: refusing Hugging Face traffic",
            ));
        }
        // a poisoned lock still holds the last client (crate::sync): one panic never breaks every later request
        let mut built = self.built.guard();
        if let Some(c) = built.as_ref() {
            if eg.proxy_url().is_some()
                && let Some(b) = &eg.ca_bundle
                && !b.is_file()
            {
                return Err(egress_proxy_unavailable(
                    "egress CA certificate disappeared: refusing to connect",
                ));
            }
            return Ok(c.clone());
        }
        let allowed = self.allowed.clone();
        let policy = reqwest::redirect::Policy::custom(move |attempt| {
            if attempt.previous().len() > MAX_REDIRECTS {
                return attempt.error(TooManyRedirects);
            }
            match check_egress(attempt.url(), &allowed) {
                Ok(()) => attempt.follow(),
                Err(msg) => attempt.error(Blocked(msg)),
            }
        });
        // no_proxy(): the ambient HTTP(S)_PROXY / ALL_PROXY never steer HF traffic, in either mode
        let mut b = reqwest::Client::builder()
            .no_proxy()
            .redirect(policy)
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(READ_TIMEOUT)
            .http1_only();
        if let Some(proxy) = eg.proxy_url() {
            let roots = match &eg.ca_bundle {
                Some(bundle) => {
                    if !bundle.is_file() {
                        return Err(egress_proxy_unavailable(
                            "egress proxy configured but its CA certificate file is missing: refusing to connect \
                             (no direct fallback)",
                        ));
                    }
                    load_bundle(bundle)?
                }
                // no bundle configured: the system trust store (the proxy's CA must be installed there)
                None => {
                    system_roots()
                        .map_err(|_| {
                            egress_proxy_unavailable("cannot load the egress CA bundle: SSLError")
                        })?
                        .1
                }
            };
            let p = reqwest::Proxy::all(proxy).map_err(|_| internal("unexpected server error"))?;
            b = b.proxy(p).tls_certs_only(roots);
        } else {
            b = b.tls_certs_only(system_roots()?.1);
        }
        let c = b.build().map_err(|e| {
            if eg.proxy_url().is_some() {
                egress_proxy_unavailable(format!(
                    "cannot load the egress CA bundle: {}",
                    if chain_text(&e).contains("certificate") {
                        "SSLError"
                    } else {
                        "ValueError"
                    }
                ))
            } else {
                upstream_error("cannot reach Hugging Face: ConnectError")
            }
        })?;
        *built = Some(c.clone());
        Ok(c)
    }

    fn transport_error(&self, e: &reqwest::Error) -> Problem {
        if let Some(b) = find::<Blocked>(e) {
            return egress_denied(b.0.clone());
        }
        let text = chain_text(e);
        if self.egress.proxy_url().is_some() && e.is_connect() {
            // A TLS verification failure through an intercepting proxy is a CA problem, not a reachability one;
            // saying "unreachable" sends the operator to debug the wrong thing.
            if text.contains("invalid peer certificate")
                || text.contains("CERTIFICATE_VERIFY_FAILED")
            {
                return egress_proxy_unavailable(
                    "TLS verification failed through the egress proxy: set EGRESS_CA_BUNDLE to the proxy's CA \
                     certificate (PEM), or install it in the system trust store. Refusing to connect directly.",
                );
            }
            return egress_proxy_unavailable(
                "egress proxy unreachable: refusing to connect directly",
            );
        }
        upstream_error(format!("cannot reach Hugging Face: {}", httpx_name(e)))
    }

    fn headers(&self, range: Option<String>) -> reqwest::header::HeaderMap {
        let mut h = reqwest::header::HeaderMap::new();
        h.insert(
            reqwest::header::USER_AGENT,
            reqwest::header::HeaderValue::from_static("engine-console"),
        );
        if let Some(tok) = (self.token)().filter(|t| !t.is_empty())
            && let Ok(mut v) = reqwest::header::HeaderValue::from_str(&format!("Bearer {tok}"))
        {
            v.set_sensitive(true);
            h.insert(reqwest::header::AUTHORIZATION, v);
        }
        if let Some(r) = range
            && let Ok(v) = reqwest::header::HeaderValue::from_str(&r)
        {
            h.insert(reqwest::header::RANGE, v);
        }
        h
    }

    /// `_raise`: map an error status to the problem Python raises.
    fn raise_for(
        &self,
        status: u16,
        body: &[u8],
        repo: Option<&str>,
        path: &str,
    ) -> Result<(), Problem> {
        if status < 400 {
            return Ok(());
        }
        let text = String::from_utf8_lossy(body);
        if status == 403 && py::take(&text, 200).contains("egress blocked") {
            return Err(egress_denied(
                "the egress proxy blocked this host; add it to the proxy's allowlist",
            ));
        }
        if status == 401 || status == 403 {
            let tail = repo.map(|r| format!(" {r}")).unwrap_or_default();
            return Err(gated_model(format!(
                "Hugging Face denied access to{tail}: the model is gated or private. Set HF_TOKEN and accept the \
                 license at {}/{}",
                self.endpoint,
                repo.unwrap_or("")
            )));
        }
        if status == 404 {
            return Err(not_found(format!(
                "not found on Hugging Face: {}",
                repo.map_or_else(|| path.to_string(), str::to_string)
            )));
        }
        Err(upstream_error(format!(
            "Hugging Face returned HTTP {status}"
        )))
    }

    /// Send one request (the egress rule checked on the first hop here and on every redirect by the policy).
    async fn send(&self, url: &str, range: Option<String>) -> AppResult<reqwest::Response> {
        let client = self.client()?;
        let parsed = url::Url::parse(url).map_err(|_| internal("unexpected server error"))?;
        check_egress(&parsed, &self.allowed).map_err(egress_denied)?;
        client
            .get(parsed)
            .headers(self.headers(range))
            .send()
            .await
            .map_err(|e| self.transport_error(&e).into())
    }

    /// `_get`: the whole answer, its status checked.
    async fn get(
        &self,
        url: &str,
        params: &[(String, String)],
        repo: Option<&str>,
    ) -> AppResult<Bytes> {
        let full = if params.is_empty() {
            url.to_string()
        } else {
            format!("{url}?{}", query_encode(params))
        };
        let mut resp = self.send(&full, None).await?;
        let status = resp.status().as_u16();
        let path = decoded_path(resp.url());
        let mut buf = BytesMut::new();
        loop {
            match resp.chunk().await {
                Ok(Some(c)) => {
                    if buf.len() + c.len() > MAX_BODY {
                        return Err(upstream_error(format!(
                            "Hugging Face sent more than {} MiB",
                            MAX_BODY / (1024 * 1024)
                        ))
                        .into());
                    }
                    buf.extend_from_slice(&c);
                }
                Ok(None) => break,
                Err(e) => return Err(self.transport_error(&e).into()),
            }
        }
        let body = buf.freeze();
        self.raise_for(status, &body, repo, &path)?;
        Ok(body)
    }
}

fn decoded_path(u: &url::Url) -> String {
    percent_encoding::percent_decode_str(u.path())
        .decode_utf8_lossy()
        .into_owned()
}

fn parse_json(body: &[u8]) -> Result<Value, Problem> {
    serde_json::from_slice(body).map_err(|_| internal("unexpected server error"))
}

#[async_trait]
impl HubClient for HfHttpClient {
    async fn search(
        &self,
        q: &str,
        task: Option<&str>,
        library: Option<&str>,
        quant: Option<&str>,
        sort: &str,
        limit: i64,
        _token_ok: bool,
    ) -> AppResult<Vec<HubModel>> {
        let mut params: Vec<(String, String)> = vec![
            ("limit".into(), limit.to_string()),
            ("sort".into(), sort.into()),
            ("direction".into(), "-1".into()),
        ];
        let mut push = |k: &str, v: Option<&str>| {
            if let Some(v) = v.filter(|v| !v.is_empty()) {
                params.push((k.into(), v.into()));
            }
        };
        push("search", Some(q));
        push("pipeline_tag", task);
        push("library", library);
        push("filter", quant);
        for f in [
            "downloads",
            "likes",
            "pipeline_tag",
            "tags",
            "gated",
            "library_name",
            "safetensors",
        ] {
            params.push(("expand[]".into(), f.into()));
        }
        let body = self
            .get(&format!("{}/api/models", self.endpoint), &params, None)
            .await?;
        // a JSON object iterates over its keys in Python, so every item fails as a model: an error either way
        match parse_json(&body)? {
            Value::Array(items) => Ok(items
                .iter()
                .map(parse_model)
                .collect::<Result<Vec<_>, _>>()?),
            Value::Object(o) if o.is_empty() => Ok(Vec::new()),
            Value::String(s) if s.is_empty() => Ok(Vec::new()),
            _ => Err(internal("unexpected server error").into()),
        }
    }

    async fn model(&self, repo_id: &str, revision: Option<&str>) -> AppResult<HubModel> {
        let rev = quote(revision.unwrap_or("main"), "");
        let body = self
            .get(
                &format!(
                    "{}/api/models/{}/revision/{rev}",
                    self.endpoint,
                    quote(repo_id, "/")
                ),
                &[("blobs".into(), "true".into())],
                Some(repo_id),
            )
            .await?;
        let mut hub = parse_model(&parse_json(&body)?)?;
        let at = hub
            .sha
            .clone()
            .filter(|s| !s.is_empty())
            .or_else(|| revision.filter(|r| !r.is_empty()).map(str::to_string))
            .unwrap_or_else(|| "main".into());
        let cfg_url = format!(
            "{}/{}/resolve/{}/config.json",
            self.endpoint,
            quote(repo_id, "/"),
            quote(&at, "")
        );
        hub.config = match self.get(&cfg_url, &[], Some(repo_id)).await {
            // gated w/o token, or a non-transformers repo: degrade, don't fail
            Ok(b) => match serde_json::from_slice::<Value>(&b) {
                Ok(Value::Object(o)) => Some(o),
                _ => None,
            },
            Err(e) if matches!(e.code(), "not_found" | "hf_gated") => None,
            Err(e) => return Err(e),
        };
        Ok(hub)
    }

    async fn card(&self, repo_id: &str, revision: Option<&str>) -> AppResult<String> {
        let url = format!(
            "{}/{}/raw/{}/README.md",
            self.endpoint,
            quote(repo_id, "/"),
            quote(revision.unwrap_or("main"), "")
        );
        match self.get(&url, &[], Some(repo_id)).await {
            Ok(b) => Ok(String::from_utf8_lossy(&b).into_owned()),
            Err(e) if e.code() == "not_found" => Ok(String::new()),
            Err(e) => Err(e),
        }
    }

    async fn open_file(
        &self,
        repo_id: &str,
        revision: &str,
        path: &str,
        start: i64,
    ) -> AppResult<Box<dyn ByteStream>> {
        let url = format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint,
            quote(repo_id, "/"),
            quote(revision, ""),
            quote(path, "/")
        );
        let range = (start > 0).then(|| format!("bytes={start}-"));
        let mut resp = self.send(&url, range).await?;
        let status = resp.status().as_u16();
        if status >= 400 {
            let req_path = decoded_path(resp.url());
            let mut buf = BytesMut::new();
            while let Ok(Some(c)) = resp.chunk().await {
                if buf.len() < MAX_BODY {
                    buf.extend_from_slice(&c);
                }
            }
            self.raise_for(status, &buf, Some(repo_id), &req_path)?;
        }
        let offset = if status == 206 { start } else { 0 };
        let total = resp
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| s.parse::<i64>().ok())
            .map(|n| n + offset);
        Ok(Box::new(HttpStream {
            resp: Some(resp),
            offset,
            total,
            buf: BytesMut::new(),
            done: false,
        }))
    }
}

/// One file body: re-chunked to exactly 1 MiB like httpx's `aiter_bytes(1 MiB)`. A connection that dies mid-file is
/// an upstream failure the operator can resume (what was buffered below 1 MiB is lost with it).
struct HttpStream {
    resp: Option<reqwest::Response>,
    offset: i64,
    total: Option<i64>,
    buf: BytesMut,
    done: bool,
}

#[async_trait]
impl ByteStream for HttpStream {
    fn total(&self) -> Option<i64> {
        self.total
    }
    fn offset(&self) -> i64 {
        self.offset
    }
    async fn chunk(&mut self) -> AppResult<Option<Bytes>> {
        loop {
            if self.buf.len() >= CHUNK {
                return Ok(Some(self.buf.split_to(CHUNK).freeze()));
            }
            if self.done {
                if self.buf.is_empty() {
                    return Ok(None);
                }
                return Ok(Some(self.buf.split().freeze()));
            }
            let Some(resp) = self.resp.as_mut() else {
                self.done = true;
                continue;
            };
            match resp.chunk().await {
                Ok(Some(c)) => self.buf.extend_from_slice(&c),
                Ok(None) => {
                    self.done = true;
                    self.resp = None;
                }
                Err(e) => {
                    self.resp = None;
                    self.done = true;
                    self.buf.clear();
                    return Err(upstream_error(format!(
                        "the Hugging Face connection dropped mid-download ({}); resume the download to continue",
                        httpx_name(&e)
                    ))
                    .into());
                }
            }
        }
    }
    async fn close(&mut self) {
        self.resp = None;
        self.done = true;
        self.buf.clear();
    }
}

// ---- the HF API's JSON -> HubModel ------------------------------------------------------------------------------
fn bad() -> Problem {
    internal("unexpected server error")
}

/// pydantic's lax int (what Python's API models do with the value later); None for null. A value pydantic refuses
/// is a 500 in Python (the response model fails): an error here too.
fn lax_int(v: Option<&Value>) -> Result<Option<i64>, Problem> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let mut errs = Vec::new();
            crate::api::validate::coerce_int(v, &[], &mut errs)
                .map(Some)
                .ok_or_else(bad)
        }
    }
}

/// `int(x)`: ints, floats truncated toward zero, integer strings.
fn py_int(v: &Value) -> Result<i64, Problem> {
    match v {
        Value::Bool(b) => Ok(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Ok(i);
            }
            let f = n.as_f64().ok_or_else(bad)?;
            if f.is_finite() && f.abs() < 9.2e18 {
                #[allow(clippy::cast_possible_truncation)]
                Ok(f.trunc() as i64)
            } else {
                Err(bad())
            }
        }
        Value::String(s) => crate::services::store::py_int(s).ok_or_else(bad),
        _ => Err(bad()),
    }
}

fn opt_str(v: Option<&Value>) -> Result<Option<String>, Problem> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(_) => Err(bad()),
    }
}

/// `x or {}` for a value used as a mapping: falsy -> empty; a truthy non-object fails as `.get` would in Python.
fn as_map(v: Option<&Value>) -> Result<Map<String, Value>, Problem> {
    match v {
        Some(Value::Object(o)) => Ok(o.clone()),
        Some(x) if py::truthy(x) => Err(bad()),
        _ => Ok(Map::new()),
    }
}

/// `_parse_model`: one model object of the HF API.
pub fn parse_model(m: &Value) -> Result<HubModel, Problem> {
    let Value::Object(m) = m else {
        return Err(bad());
    };
    let st = as_map(m.get("safetensors"))?;
    let card = m.get("cardData");
    let mut license = match card {
        Some(Value::Object(c)) => match c.get("license") {
            Some(Value::String(s)) => Some(s.clone()),
            // a truthy non-string licence makes Python fail later (ModelInfo); here it is shown as unknown
            Some(x) if py::truthy(x) => Some(String::new()),
            _ => None,
        },
        _ => None,
    };
    // `m.get("tags", [])`: a list keeps its strings; a dict iterates its keys and a string its characters
    let tags: Vec<String> = match m.get("tags") {
        None => Vec::new(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|t| t.as_str().map(str::to_string))
            .collect(),
        Some(Value::Object(o)) => o.keys().cloned().collect(),
        Some(Value::String(s)) => s.chars().map(String::from).collect(),
        Some(_) => return Err(bad()),
    };
    if license.as_deref().is_none_or(str::is_empty) {
        let truthy_other = license.as_deref() == Some("");
        if !truthy_other {
            license = tags
                .iter()
                .find_map(|t| t.strip_prefix("license:").map(str::to_string));
        } else {
            license = None;
        }
    }
    let mut files = Vec::new();
    let siblings = match m.get("siblings") {
        Some(Value::Array(a)) => a.clone(),
        Some(x) if py::truthy(x) => return Err(bad()),
        _ => Vec::new(),
    };
    for s in &siblings {
        let Value::Object(s) = s else {
            return Err(bad());
        };
        let lfs = as_map(s.get("lfs"))?;
        let path = match s.get("rfilename") {
            Some(Value::String(p)) => p.clone(),
            _ => return Err(bad()),
        };
        let size_v = py::or_chain(&[s.get("size"), lfs.get("size")]);
        let size = match size_v {
            Some(v) if py::truthy(v) => py_int(v)?,
            _ => 0,
        };
        let sha = py::or_chain(&[lfs.get("sha256"), lfs.get("oid")]);
        let sha256 = match sha {
            Some(Value::String(x)) if !x.is_empty() => Some(x.clone()),
            Some(x) if py::truthy(x) => return Err(bad()),
            // "" (falsy) is kept as Python keeps it only when it is the last value; it reads as no hash either way
            _ => None,
        };
        files.push(RepoFile { path, size, sha256 });
    }
    let mut safetensors_params = Vec::new();
    for (k, v) in as_map(st.get("parameters"))? {
        let n = match v {
            Value::Number(n) if !n.is_f64() => n.as_i64().ok_or_else(bad)?,
            Value::Number(n) => {
                let f = n.as_f64().ok_or_else(bad)?;
                if f.fract() != 0.0 || !f.is_finite() {
                    return Err(bad());
                }
                #[allow(clippy::cast_possible_truncation)]
                let i = f as i64;
                i
            }
            _ => return Err(bad()),
        };
        safetensors_params.push((k, n));
    }
    let repo_id = match py::or_chain(&[m.get("id"), m.get("modelId")]) {
        Some(Value::String(s)) => s.clone(),
        Some(x) if py::truthy(x) => return Err(bad()),
        _ => String::new(),
    };
    Ok(HubModel {
        repo_id,
        sha: opt_str(m.get("sha"))?,
        gated: m.get("gated").is_some_and(py::truthy),
        license,
        pipeline_tag: opt_str(m.get("pipeline_tag"))?,
        library_name: opt_str(m.get("library_name"))?,
        tags,
        files,
        safetensors_params,
        safetensors_total: lax_int(st.get("total"))?,
        config: None,
        downloads: lax_int(m.get("downloads"))?,
        likes: lax_int(m.get("likes"))?,
    })
}

fn load_bundle(bundle: &Path) -> Result<Vec<reqwest::Certificate>, Problem> {
    let pem = read_ca_file(bundle).map_err(|e| {
        egress_proxy_unavailable(format!(
            "cannot load the egress CA bundle: {}",
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                "PermissionError"
            } else {
                "OSError"
            }
        ))
    })?;
    match reqwest::Certificate::from_pem_bundle(&pem) {
        Ok(c) if !c.is_empty() => Ok(c),
        _ => Err(egress_proxy_unavailable(
            "cannot load the egress CA bundle: SSLError",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_quote_and_query_encoding_match_python() {
        assert_eq!(quote("org/na me?x#y", "/"), "org/na%20me%3Fx%23y");
        assert_eq!(
            quote("a b/é?#[]@!$&'()*+,;=~", ""),
            "a%20b%2F%C3%A9%3F%23%5B%5D%40%21%24%26%27%28%29%2A%2B%2C%3B%3D~"
        );
        let q = query_encode(&[
            ("limit".into(), "3".into()),
            ("search".into(), "a b/c?d&e=f+g~é%".into()),
            ("expand[]".into(), "downloads".into()),
            ("filter".into(), "x:y,z".into()),
        ]);
        assert_eq!(
            q,
            "limit=3&search=a+b%2Fc%3Fd%26e%3Df%2Bg~%C3%A9%25&expand%5B%5D=downloads&filter=x%3Ay%2Cz"
        );
    }

    #[test]
    fn test_egress_rule_per_hop() {
        let allowed = vec!["huggingface.co".to_string(), "*.hf.co".to_string()];
        let u = |s: &str| url::Url::parse(s).expect("url");
        assert!(check_egress(&u("https://huggingface.co/x"), &allowed).is_ok());
        assert!(check_egress(&u("https://cas-bridge.xethub.hf.co/x"), &allowed).is_ok());
        assert_eq!(
            check_egress(&u("http://huggingface.co/x"), &allowed),
            Err("refusing a non-https request/redirect target for Hugging Face traffic".into())
        );
        assert_eq!(
            check_egress(&u("https://evil.example/f"), &allowed),
            Err("egress to evil.example is not on the HF allowlist".into())
        );
        assert_eq!(
            check_egress(&u("https://[::1]:9/f"), &allowed),
            Err("egress to ::1 is not on the HF allowlist".into())
        );
    }

    #[test]
    fn test_direct_client_ignores_proxy_env_and_ca_env() {
        // Python checks `_trust_env is False`; here the roots are a fixed file and no system proxy is consulted
        // (`no_proxy()`), whatever HTTPS_PROXY / SSL_CERT_FILE say.
        let c = HfHttpClient::new(
            "https://huggingface.co",
            vec!["huggingface.co".into()],
            Arc::new(|| None),
            EgressConfig::default(),
        );
        match c.trust_roots() {
            Ok(TrustRoots::SystemBundle(p)) => assert!(SYSTEM_CA_BUNDLES.contains(&p.as_str())),
            other => panic!("unexpected roots {other:?}"),
        }
        assert!(c.client().is_ok());
    }

    #[test]
    fn test_ca_file_reader_refuses_fifos_dirs_and_oversized_files() {
        let d = tempfile::tempdir().expect("tmp");
        let fifo = d.path().join("ca.fifo");
        nix::unistd::mkfifo(
            &fifo,
            nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
        )
        .expect("mkfifo");
        // opened non-blocking, so a FIFO with no writer cannot hang the hub; fstat then refuses it
        assert!(read_ca_file(&fifo).is_err());
        assert!(read_ca_file(d.path()).is_err());
        let big = d.path().join("big.pem");
        std::fs::write(&big, vec![b'x'; 16 * 1024 * 1024 + 1]).expect("write");
        assert!(read_ca_file(&big).is_err());
        let ok = d.path().join("ok.pem");
        std::fs::write(&ok, b"pem").expect("write");
        let link = d.path().join("link.pem");
        std::os::unix::fs::symlink(&ok, &link).expect("symlink");
        assert_eq!(read_ca_file(&link).expect("followed"), b"pem");
    }

    #[test]
    fn test_require_proxy_and_missing_ca_fail_closed() {
        let c = HfHttpClient::new(
            "https://huggingface.co",
            vec!["huggingface.co".into()],
            Arc::new(|| None),
            EgressConfig {
                proxy: None,
                ca_bundle: None,
                require_proxy: true,
            },
        );
        let e = c.client().expect_err("refused");
        assert_eq!(
            (e.status, e.code.as_str()),
            (503, "egress_proxy_unavailable")
        );
        let c = HfHttpClient::new(
            "https://huggingface.co",
            vec!["huggingface.co".into()],
            Arc::new(|| None),
            EgressConfig {
                proxy: Some("http://127.0.0.1:9".into()),
                ca_bundle: Some("/nonexistent/ca.pem".into()),
                require_proxy: true,
            },
        );
        let e = c.client().expect_err("refused");
        assert!(e.detail.contains("CA certificate file is missing"));
    }

    #[test]
    fn test_parse_model_shapes() {
        let m = parse_model(&json!({"id": "a/b", "sha": "abc", "gated": "manual", "tags": ["license:mit", 5],
            "safetensors": {"total": 5, "parameters": {"BF16": 5}},
            "siblings": [{"rfilename": "m.safetensors", "size": 10, "lfs": {"sha256": "ff", "size": 10}},
                         {"rfilename": "c.json"}]}))
        .expect("model");
        assert!(m.gated);
        assert_eq!(m.license.as_deref(), Some("mit"));
        assert_eq!(m.files[0].sha256.as_deref(), Some("ff"));
        assert_eq!(m.files[1].size, 0);
        assert_eq!(m.safetensors_params, vec![("BF16".to_string(), 5)]);
        assert!(parse_model(&json!("x")).is_err());
        assert!(parse_model(&json!({"siblings": [{"size": 1}]})).is_err());
    }
}
