//! Node hardening (Rust-only; no Python counterpart): special files and bounded reads, the fixed tool PATH, panics
//! contained, poisoned locks, regex failures that fail closed, no build-time paths in runtime code. The subprocess
//! caps and PATH resolution are in test_util.rs, the end-to-end PATH and panic-hook checks in test_cli.rs.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::os::unix::fs::symlink;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ec_proto::{BAD_ARGS, CONFLICT, Obj};
use ecnode::errors::{HandlerError, OpError};
use ecnode::fsops::{WriteLocks, fs_put_small, fs_read, fs_symlink, fs_write};
use ecnode::hostinfo;
use ecnode::memory::Mem;
use ecnode::policy::{Policy, parse};
use ecnode::util::fsread;
use serde_json::{Value, json};
use wsd_kit::{FakeStream, Machine, obj};

fn fifo(p: &Path) {
    nix::unistd::mkfifo(p, nix::sys::stat::Mode::from_bits_truncate(0o644)).unwrap();
}

/// A unix socket file at `p` (the listener is returned so it stays bound).
fn socket(p: &Path) -> std::os::unix::net::UnixListener {
    std::os::unix::net::UnixListener::bind(p).unwrap()
}

/// Run `f` on another thread; fail the test if it does not return within 5 s (a FIFO open would hang forever).
fn within_5s<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(5))
        .unwrap_or_else(|_| panic!("{what} hung (a FIFO was opened for reading?)"))
}

fn op_err(r: Result<Obj, HandlerError>) -> OpError {
    match r {
        Err(HandlerError::Op(e)) => e,
        other => panic!("expected an OpError, got {other:?}"),
    }
}

// ── bounded reads of files the node does not own ──────────────────────────────────────────────────────────

#[test]
fn test_fsread_refuses_special_files_and_caps_reads() {
    let tmp = tempfile::tempdir().unwrap();
    let d = tmp.path();
    std::fs::write(d.join("f"), "0123456789").unwrap();
    symlink(d.join("f"), d.join("link")).unwrap();
    fifo(&d.join("fifo"));
    symlink(d.join("fifo"), d.join("link-to-fifo")).unwrap();
    let _sock = socket(&d.join("sock"));
    std::fs::create_dir(d.join("dir")).unwrap();
    let dd = d.to_path_buf();
    let res = within_5s("fsread", move || {
        let r = |n: &str, follow| fsread::read_file(&dd.join(n), 4, follow).map_err(|e| e.kind());
        (
            r("f", true),
            r("link", true),
            r("link", false).is_err(),
            r("fifo", true),
            r("link-to-fifo", true),
            r("sock", true),
            r("dir", true),
        )
    });
    use std::io::ErrorKind::InvalidInput;
    assert_eq!(res.0, Ok(b"0123".to_vec()));
    assert_eq!(res.1, Ok(b"0123".to_vec()));
    assert!(res.2, "O_NOFOLLOW: a final symlink is not followed");
    assert_eq!(res.3, Err(InvalidInput));
    assert_eq!(res.4, Err(InvalidInput));
    assert!(res.5.is_err(), "a socket cannot be opened for reading");
    assert_eq!(res.6, Err(InvalidInput));
    // a device is not a regular file either (/dev/null: never reads, never blocks)
    let devnull = fsread::read_file(Path::new("/dev/null"), 4, true).map_err(|e| e.kind());
    assert_eq!(devnull, Err(InvalidInput));
    // a pseudo-file reports size 0: it is read to EOF (or the cap) anyway
    let stat = fsread::read_file(Path::new("/proc/self/stat"), 1 << 16, true).unwrap();
    assert!(!stat.is_empty());
}

#[test]
fn test_proc_sys_etc_reads_never_hang_on_a_fifo() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    let proc = m.paths.proc.clone();
    std::fs::remove_file(proc.join("meminfo")).unwrap();
    fifo(&proc.join("meminfo"));
    std::fs::remove_file(proc.join("loadavg")).unwrap();
    fifo(&proc.join("loadavg"));
    // /etc/os-release is usually a symlink: it is followed (as Python's open does), but to a FIFO it is refused
    let etc = m.paths.etc.clone();
    std::fs::remove_file(etc.join("os-release")).unwrap();
    fifo(&tmp.path().join("os-release-fifo"));
    symlink(tmp.path().join("os-release-fifo"), etc.join("os-release")).unwrap();
    let paths = m.paths.clone();
    let (mem, load, os) = within_5s("memory/hostinfo reads", move || {
        let mem = Mem::new(paths.clone());
        (mem.meminfo(), mem.loadavg(), hostinfo::os_release(&paths))
    });
    assert!(mem.values().all(Value::is_null), "{mem:?}");
    assert_eq!(load, (None, None, None));
    assert_eq!(os, None);
    // a symlink to a regular file is still followed
    std::fs::remove_file(etc.join("os-release")).unwrap();
    std::fs::write(
        tmp.path().join("real-os-release"),
        "PRETTY_NAME=\"Linked 2\"\n",
    )
    .unwrap();
    symlink(tmp.path().join("real-os-release"), etc.join("os-release")).unwrap();
    assert_eq!(hostinfo::os_release(&m.paths).as_deref(), Some("Linked 2"));
}

#[test]
fn test_an_oversized_proc_file_is_read_only_up_to_its_cap() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    // one 8 MiB line, then MemTotal: meminfo reads 64 KiB (as Python's read(65536)), so MemTotal is never seen
    let huge = format!("{}\nMemTotal: 1024 kB\n", "A".repeat(8 << 20));
    std::fs::write(m.paths.proc.join("meminfo"), &huge).unwrap();
    let started = Instant::now();
    let mem = Mem::new(m.paths.clone()).meminfo();
    assert_eq!(mem["total"], Value::Null);
    assert!(started.elapsed() < Duration::from_secs(5));
    // within the cap it is read as usual
    std::fs::write(m.paths.proc.join("meminfo"), "MemTotal: 1024 kB\n").unwrap();
    assert_eq!(Mem::new(m.paths).meminfo()["total"], json!(1024 * 1024));
}

#[test]
fn test_model_metadata_and_gguf_headers_refuse_special_files() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    fifo(&root.join("config.json"));
    fifo(&root.join("model.gguf"));
    let _sock = socket(&root.join("sock.gguf"));
    std::fs::write(root.join("real.json"), "{}").unwrap();
    symlink(root.join("config.json"), root.join("link.json")).unwrap();
    let r = root;
    let (cfg, link, real, gguf_fifo, gguf_sock, gguf_any) =
        within_5s("model metadata", move || {
            (
                ecnode::confine::read_inside(&r, &r.join("config.json"), 4096),
                ecnode::confine::read_inside(&r, &r.join("link.json"), 4096),
                ecnode::confine::read_inside(&r, &r.join("real.json"), 4096),
                ecnode::models::gguf_header(&r, &r.join("model.gguf")),
                ecnode::models::gguf_header(&r, &r.join("sock.gguf")),
                ecnode::gguf::read_file(&r.join("model.gguf"), None, 1 << 20, true),
            )
        });
    assert_eq!((cfg, link), (None, None));
    assert_eq!(real.as_deref(), Some(&b"{}"[..]));
    assert_eq!((gguf_fifo, gguf_sock, gguf_any), (None, None, None));
}

#[test]
fn test_policy_symlink_and_fifo_are_refused_without_hanging() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("p");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::set_permissions(&dir, std::os::unix::fs::PermissionsExt::from_mode(0o755)).unwrap();
    fifo(&dir.join("fifo.json"));
    std::fs::write(dir.join("real.json"), r#"{"mode": "monitor"}"#).unwrap();
    symlink(dir.join("real.json"), dir.join("link.json")).unwrap();
    let d = dir;
    let (f, l) = within_5s("policy load", move || {
        let load = |n: &str| {
            let path = d.join(n).to_string_lossy().into_owned();
            ecnode::policy::load(
                Some(path.as_str()),
                Some("/nonexistent"),
                "/nonexistent/etc",
            )
        };
        (load("fifo.json"), load("link.json"))
    });
    assert!(
        f.error
            .unwrap_or_default()
            .contains("is not a regular file")
    );
    assert!(l.error.unwrap_or_default().contains("is a symlink"));
}

// ── fs ops never read, write or replace a FIFO, socket or device ─────────────────────────────────────────────

fn writable(tmp: &Path) -> Policy {
    let w = tmp.join("w");
    std::fs::create_dir_all(&w).unwrap();
    parse(
        &json!({"mode": "manage", "model_roots": [
            {"id": "w", "path": w.to_string_lossy(), "kind": "plain", "writable": true}]}),
        Some("/test/policy.json"),
        None,
    )
    .unwrap()
}

fn write_args(rel: &str, size: i64) -> Obj {
    obj(json!({"root": "w", "rel": rel, "size": size, "sha256": null, "offset": null}))
}

#[test]
fn test_fs_ops_refuse_special_files_as_source_or_destination() {
    let tmp = tempfile::tempdir().unwrap();
    let p = writable(tmp.path());
    let w = tmp.path().join("w");
    fifo(&w.join("pipe.bin"));
    let _sock = socket(&w.join("sock.bin"));
    fifo(&w.join("part.bin.incomplete"));
    let locks = WriteLocks::new();
    // destinations: an upload, a small put and a link never replace a FIFO or socket
    for rel in ["pipe.bin", "sock.bin"] {
        let e = op_err(fs_write(
            &p,
            &write_args(rel, 3),
            &FakeStream::new(vec![b"abc".to_vec()]),
            &locks,
        ));
        assert_eq!(e.code, CONFLICT, "{rel}: {e:?}");
        assert!(
            e.message.contains("is a FIFO, socket or device file"),
            "{e:?}"
        );
        let e = op_err(fs_put_small(
            &p,
            &obj(json!({"root": "w", "rel": rel, "content_b64": "YWJj"})),
        ));
        assert_eq!(e.code, CONFLICT, "{rel}: {e:?}");
        let e = op_err(fs_symlink(
            &p,
            &obj(json!({"root": "w", "link": rel, "target": "x"})),
        ));
        assert_eq!(e.code, CONFLICT, "{rel}: {e:?}");
    }
    assert!(
        nix::sys::stat::lstat(&w.join("pipe.bin"))
            .is_ok_and(|s| s.st_mode & nix::libc::S_IFMT == nix::libc::S_IFIFO)
    );
    // the upload's partial file: a FIFO there is refused (and never opened)
    let e = within_5s("fs.write over a FIFO partial", {
        let p = p.clone();
        move || {
            op_err(fs_write(
                &p,
                &write_args("part.bin", 3),
                &FakeStream::new(vec![b"abc".to_vec()]),
                &WriteLocks::new(),
            ))
        }
    });
    assert_eq!(
        (e.code.as_str(), e.message.as_str()),
        (CONFLICT, "the upload's partial file is not a regular file")
    );
    // sources: fs.read of a socket or FIFO is refused, a symlink is named as one
    symlink("pipe.bin", w.join("link.bin")).unwrap();
    for (rel, msg) in [
        ("sock.bin", "'sock.bin' is not a regular file"),
        ("pipe.bin", "'pipe.bin' is not a regular file"),
        (
            "link.bin",
            "'link.bin' is a symbolic link; read the file it points to (in an HF cache: the blob)",
        ),
    ] {
        let p2 = p.clone();
        let rel2 = rel.to_string();
        let e = within_5s("fs.read", move || {
            op_err(fs_read(
                &p2,
                &obj(json!({"root": "w", "rel": rel2, "offset": null, "length": null})),
                &FakeStream::empty(),
            ))
        });
        assert_eq!((e.code.as_str(), e.message.as_str()), (BAD_ARGS, msg));
    }
    // a regular file is still written and read as before
    let res = fs_write(
        &p,
        &write_args("ok.bin", 3),
        &FakeStream::new(vec![b"abc".to_vec()]),
        &locks,
    )
    .unwrap();
    assert_eq!(res["size"], json!(3));
    let s = FakeStream::empty();
    fs_read(
        &p,
        &obj(json!({"root": "w", "rel": "ok.bin", "offset": null, "length": null})),
        &s,
    )
    .unwrap();
    assert_eq!(s.data(), b"abc");
}

// ── panics and poisoned locks ───────────────────────────────────────────────────────────────────────────────

#[test]
fn test_poisoned_locks_keep_working() {
    let m = Arc::new(Mutex::new(vec![1]));
    let m2 = Arc::clone(&m);
    let _ = std::thread::spawn(move || {
        let _g = ecnode::util::lock(&m2);
        panic!("poison it");
    })
    .join();
    assert!(m.is_poisoned());
    ecnode::util::lock(&m).push(2);
    assert_eq!(*ecnode::util::lock(&m), vec![1, 2]);
}

#[test]
fn test_contain_reports_a_redacted_panic_and_carries_on() {
    let r: Option<()> =
        ecnode::util::panics::contain("t", || panic!("--api-key ZQ9VLMT4PR2SXW8KABCDEF"));
    assert!(r.is_none());
    let line = ecnode::util::panics::panic_line(
        Some("ecnode-req"),
        None,
        "token=ZQ9VLMT4PR2SXW8KABCDEF Authorization: Bearer ZQ9VLMT4PR2SXW8KABCDEF",
    );
    assert!(!line.contains("ZQ9VLMT4"), "{line}");
}

// ── no build-time paths in runtime code ─────────────────────────────────────────────────────────────────────

#[test]
fn test_runtime_code_has_no_build_time_paths() {
    // env!("CARGO_MANIFEST_DIR") would bake this checkout's path into the shipped binary
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut stack = vec![src];
    let mut seen = 0;
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                seen += 1;
                let text = std::fs::read_to_string(&p).unwrap();
                assert!(
                    !text.contains(concat!("CARGO_", "MANIFEST_DIR")),
                    "{} uses the build-time manifest dir",
                    p.display()
                );
            }
        }
    }
    assert!(seen > 20);
}
