//! A recording HTTPS upstream for the HF client tests (the role respx plays in Python).
//!
//! `openssl s_server` in raw mode terminates TLS with the fleet tests' throwaway certificates (wsb_egress.rs
//! `make_certs`): what a client sends appears on its stdout and what is written to its stdin goes back. A reader thread
//! parses each request (request line and headers; the HF client only sends GETs), records it, and answers with the
//! response routed for its path (query string ignored, as respx routes match) or a 404. Responses carry `Connection:
//! close`, so every request is its own connection. [`TlsRecorder::route_then_drop`] answers once and then kills the
//! server, so a response that promised more bytes than it sent ends with a dropped connection (Python's mid-download
//! failures). Binds 127.0.0.1 on an ephemeral port; nothing reaches the network.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::collections::HashMap;
use std::io::{BufRead as _, BufReader, Write as _};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::wsb_egress::{Certs, free_port};

/// One request as the upstream received it.
#[derive(Debug, Clone, Default)]
pub struct Recorded {
    pub method: String,
    /// the raw request target (path and query, exactly as sent)
    pub target: String,
    pub headers: Vec<(String, String)>,
}

impl Recorded {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("")
    }
}

/// A complete HTTP response with `Connection: close` and a Content-Length.
pub fn response(status: u16, headers: &[(&str, &str)], body: &str) -> String {
    let mut out = format!(
        "HTTP/1.1 {status} X\r\nConnection: close\r\nContent-Length: {}\r\n",
        body.len()
    );
    for (k, v) in headers {
        out.push_str(&format!("{k}: {v}\r\n"));
    }
    out.push_str("\r\n");
    out.push_str(body);
    out
}

pub struct TlsRecorder {
    child: Arc<Mutex<Child>>,
    pub port: u16,
    /// path -> (response, drop the connection after it)
    routes: Arc<Mutex<HashMap<String, (String, bool)>>>,
    requests: Arc<Mutex<Vec<Recorded>>>,
}

impl TlsRecorder {
    pub fn start(certs: &Certs) -> Self {
        let port = free_port();
        let mut child = Command::new(sys_tool("openssl"))
            .args(["s_server", "-quiet", "-naccept", "1000", "-accept"])
            .arg(format!("127.0.0.1:{port}"))
            .arg("-cert")
            .arg(&certs.cert)
            .arg("-key")
            .arg(&certs.key)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("s_server");
        let mut stdin = child.stdin.take().expect("stdin");
        let stdout = child.stdout.take().expect("stdout");
        let routes: Arc<Mutex<HashMap<String, (String, bool)>>> = Arc::default();
        let requests: Arc<Mutex<Vec<Recorded>>> = Arc::default();
        let child = Arc::new(Mutex::new(child));
        let (r2, q2, c2) = (
            Arc::clone(&routes),
            Arc::clone(&requests),
            Arc::clone(&child),
        );
        std::thread::spawn(move || {
            let mut rd = BufReader::new(stdout);
            loop {
                let mut first = String::new();
                match rd.read_line(&mut first) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
                let first = first.trim_end();
                if first.is_empty() {
                    continue;
                }
                let mut parts = first.split_whitespace();
                let mut rec = Recorded {
                    method: parts.next().unwrap_or("").to_string(),
                    target: parts.next().unwrap_or("").to_string(),
                    headers: Vec::new(),
                };
                loop {
                    let mut line = String::new();
                    match rd.read_line(&mut line) {
                        Ok(0) | Err(_) => return,
                        Ok(_) => {}
                    }
                    let line = line.trim_end_matches(['\r', '\n']);
                    if line.is_empty() {
                        break;
                    }
                    if let Some((k, v)) = line.split_once(':') {
                        rec.headers
                            .push((k.trim().to_string(), v.trim().to_string()));
                    }
                }
                let (answer, drop_after) = r2
                    .lock()
                    .expect("lock")
                    .get(rec.path())
                    .cloned()
                    .unwrap_or_else(|| (response(404, &[], ""), false));
                q2.lock().expect("lock").push(rec);
                if stdin
                    .write_all(answer.as_bytes())
                    .and_then(|()| stdin.flush())
                    .is_err()
                {
                    return;
                }
                if drop_after {
                    // let the bytes leave, then the connection dies with the server
                    std::thread::sleep(Duration::from_millis(300));
                    let _ = c2.lock().expect("lock").kill();
                    return;
                }
            }
        });
        let end = Instant::now() + Duration::from_secs(10);
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(Instant::now() < end, "s_server does not listen on {port}");
            std::thread::sleep(Duration::from_millis(20));
        }
        // that probe was a connection of its own: s_server saw it open and close with no request
        Self {
            child,
            port,
            routes,
            requests,
        }
    }

    pub fn endpoint(&self) -> String {
        format!("https://127.0.0.1:{}", self.port)
    }

    /// Answer `path` (no query) with `resp` (see [`response`]).
    pub fn route(&self, path: &str, resp: String) {
        self.routes
            .lock()
            .expect("lock")
            .insert(path.to_string(), (resp, false));
    }

    /// Answer `path` with `resp`, then kill the server: the connection drops mid-response.
    pub fn route_then_drop(&self, path: &str, resp: String) {
        self.routes
            .lock()
            .expect("lock")
            .insert(path.to_string(), (resp, true));
    }

    pub fn requests(&self) -> Vec<Recorded> {
        self.requests.lock().expect("lock").clone()
    }
}

impl Drop for TlsRecorder {
    fn drop(&mut self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

/// A system tool the test needs, from the fixed safe PATH (never the caller's PATH, PORTING.md §A12): an absolute path,
/// or one that does not exist (the spawn then fails as it would without the tool).
fn sys_tool(name: &str) -> std::path::PathBuf {
    ec_hub::tools::find_in(name, ec_hub::tools::SAFE_PATH)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("/nonexistent/{name}")))
}
