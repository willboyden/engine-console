//! Egress test fixtures (for the test_network.py egress rows).
//!
//! A throwaway CA and a server certificate for 127.0.0.1 (openssl CLI, in a temp dir), `openssl s_server -HTTP` as
//! the TLS upstream (each file under its docroot is a complete HTTP response), a minimal CONNECT proxy that records
//! the targets it tunnels to, and a plain listener that only counts connections ("nothing may connect here").
//! Everything binds 127.0.0.1 on an ephemeral port; nothing reaches the network.

#![allow(
    dead_code,
    unreachable_pub,
    clippy::expect_used,
    clippy::panic,
    clippy::missing_panics_doc
)]

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};

pub struct Certs {
    pub ca: PathBuf,
    pub cert: PathBuf,
    pub key: PathBuf,
}

fn openssl(dir: &Path, args: &[&str]) {
    let out = Command::new(sys_tool("openssl"))
        .args(args)
        .current_dir(dir)
        .stdin(Stdio::null())
        .output()
        .expect("openssl");
    assert!(
        out.status.success(),
        "openssl {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A CA and a server certificate for IP 127.0.0.1 signed by it, under `dir` (made for this test only).
pub fn make_certs(dir: &Path) -> Certs {
    std::fs::create_dir_all(dir).expect("dir");
    openssl(
        dir,
        &[
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            "ca.key",
            "-out",
            "ca.pem",
            "-days",
            "2",
            "-subj",
            "/CN=wsb-test-ca",
            "-addext",
            "basicConstraints=critical,CA:TRUE",
            "-addext",
            "keyUsage=critical,keyCertSign,cRLSign",
        ],
    );
    openssl(
        dir,
        &[
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            "srv.key",
            "-out",
            "srv.csr",
            "-subj",
            "/CN=127.0.0.1",
        ],
    );
    std::fs::write(
        dir.join("ext.cnf"),
        "subjectAltName=IP:127.0.0.1\nextendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\nkeyUsage=digitalSignature,keyEncipherment\n",
    )
    .expect("ext");
    openssl(
        dir,
        &[
            "x509",
            "-req",
            "-in",
            "srv.csr",
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-out",
            "srv.pem",
            "-days",
            "2",
            "-extfile",
            "ext.cnf",
        ],
    );
    Certs {
        ca: dir.join("ca.pem"),
        cert: dir.join("srv.pem"),
        key: dir.join("srv.key"),
    }
}

/// An ephemeral loopback port that was free a moment ago.
pub fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .expect("port")
}

fn wait_listening(port: u16) {
    let end = Instant::now() + Duration::from_secs(10);
    while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
        assert!(Instant::now() < end, "nothing listens on {port}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// `openssl s_server -HTTP`: GET /a/b answers with the bytes of `<docroot>/a/b` (a complete HTTP response).
pub struct TlsServer {
    child: Child,
    pub port: u16,
    pub docroot: PathBuf,
}

impl TlsServer {
    pub fn start(certs: &Certs, docroot: &Path) -> Self {
        std::fs::create_dir_all(docroot).expect("docroot");
        let port = free_port();
        let child = Command::new(sys_tool("openssl"))
            .args(["s_server", "-quiet", "-HTTP", "-accept"])
            .arg(format!("127.0.0.1:{port}"))
            .arg("-cert")
            .arg(&certs.cert)
            .arg("-key")
            .arg(&certs.key)
            .current_dir(docroot)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("s_server");
        wait_listening(port);
        Self {
            child,
            port,
            docroot: docroot.to_path_buf(),
        }
    }

    /// Serve `response` (status line, headers, body) for `path`.
    pub fn serve(&self, path: &str, response: &str) {
        let p = self.docroot.join(path.trim_start_matches('/'));
        std::fs::create_dir_all(p.parent().expect("parent")).expect("dirs");
        std::fs::write(p, response).expect("response");
    }
}

impl Drop for TlsServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// A listener that accepts and drops connections, counting them.
pub struct Counter {
    pub port: u16,
    pub hits: Arc<AtomicUsize>,
    task: tokio::task::JoinHandle<()>,
}

impl Counter {
    pub async fn start() -> Self {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = l.local_addr().expect("addr").port();
        let hits = Arc::new(AtomicUsize::new(0));
        let h = Arc::clone(&hits);
        let task = tokio::spawn(async move {
            while let Ok((s, _)) = l.accept().await {
                h.fetch_add(1, Ordering::SeqCst);
                drop(s);
            }
        });
        Self { port, hits, task }
    }

    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::SeqCst)
    }
}

impl Drop for Counter {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// A minimal HTTP CONNECT proxy: records each target, tunnels to it (loopback only), answers 502 otherwise.
pub struct ConnectProxy {
    pub port: u16,
    pub targets: Arc<Mutex<Vec<String>>>,
    task: tokio::task::JoinHandle<()>,
}

impl ConnectProxy {
    pub async fn start() -> Self {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let port = l.local_addr().expect("addr").port();
        let targets: Arc<Mutex<Vec<String>>> = Arc::default();
        let t2 = Arc::clone(&targets);
        let task = tokio::spawn(async move {
            while let Ok((s, _)) = l.accept().await {
                let t3 = Arc::clone(&t2);
                tokio::spawn(async move {
                    let mut r = BufReader::new(s);
                    let mut first = String::new();
                    if r.read_line(&mut first).await.is_err() {
                        return;
                    }
                    loop {
                        let mut line = String::new();
                        match r.read_line(&mut line).await {
                            Ok(0) | Err(_) => return,
                            Ok(_) if line == "\r\n" || line == "\n" => break,
                            Ok(_) => {}
                        }
                    }
                    let mut parts = first.split_whitespace();
                    let (method, target) = (
                        parts.next().unwrap_or(""),
                        parts.next().unwrap_or("").to_string(),
                    );
                    t3.lock().expect("lock").push(target.clone());
                    let mut s = r.into_inner();
                    if method != "CONNECT" || !target.starts_with("127.0.0.1:") {
                        let _ = s
                            .write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                            .await;
                        return;
                    }
                    let Ok(mut up) = tokio::net::TcpStream::connect(&target).await else {
                        let _ = s
                            .write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                            .await;
                        return;
                    };
                    if s.write_all(b"HTTP/1.1 200 Connection established\r\n\r\n")
                        .await
                        .is_err()
                    {
                        return;
                    }
                    let _ = tokio::io::copy_bidirectional(&mut s, &mut up).await;
                });
            }
        });
        Self {
            port,
            targets,
            task,
        }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn targets(&self) -> Vec<String> {
        self.targets.lock().expect("lock").clone()
    }
}

impl Drop for ConnectProxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// The block page of the lab's egress proxy (mitmproxy addon_guard) for a host that is not on its allowlist.
pub const BLOCK_PAGE: &str = "HTTP/1.0 403 Forbidden\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\negress blocked by addon_guard: host not in allowlist.txt\n";

pub fn has_openssl() -> bool {
    Command::new(sys_tool("openssl"))
        .arg("version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// A system tool the test needs, from the fixed safe PATH (never the caller's PATH, PORTING.md §A12): an absolute path,
/// or one that does not exist (the spawn then fails as it would without the tool).
fn sys_tool(name: &str) -> std::path::PathBuf {
    ec_hub::tools::find_in(name, ec_hub::tools::SAFE_PATH)
        .unwrap_or_else(|| std::path::PathBuf::from(format!("/nonexistent/{name}")))
}
