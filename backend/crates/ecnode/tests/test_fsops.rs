//! node/tests/test_fsops.py: fs.* ops: path rules, symlink escapes, secret names, resumable uploads with whole-file
//! sha256, deletes.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

use ec_proto::{
    BAD_ARGS, CONFLICT, DOCKER_ERROR, HASH_MISMATCH, IN_USE, NO_SPACE, NOT_FOUND, Obj,
    POLICY_DENIED, SIZE_MISMATCH,
};
use ecnode::confine::read_inside;
use ecnode::errors::{HandlerError, OpError};
use ecnode::fsops::{
    self, WriteLocks, fs_delete, fs_put_small, fs_read, fs_stat, fs_symlink, fs_write,
    fs_write_with,
};
use ecnode::policy::{Policy, parse};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use wsd_kit::{FakeStream, obj};

struct Roots {
    tmp: TempDir,
    base: PathBuf,
    outside: PathBuf,
    data: Value,
}

/// the `roots` fixture: models/{w, ro, w/inner} and an outside/secret.txt next to them
fn roots() -> Roots {
    let tmp = tempfile::tempdir().unwrap();
    let base = tmp.path().join("models");
    for d in ["w", "ro", "w/inner"] {
        std::fs::create_dir_all(base.join(d)).unwrap();
    }
    let outside = tmp.path().join("outside");
    std::fs::create_dir(&outside).unwrap();
    std::fs::write(outside.join("secret.txt"), "outside the root").unwrap();
    let p = |d: &str| base.join(d).to_string_lossy().into_owned();
    let data = json!({"mode": "manage", "allow_delete": true, "model_roots": [
        {"id": "w", "path": p("w"), "kind": "gguf", "writable": true},
        {"id": "ro", "path": p("ro"), "kind": "gguf", "writable": false},
        {"id": "inner", "path": p("w/inner"), "kind": "gguf", "writable": true}]});
    Roots {
        tmp,
        base,
        outside,
        data,
    }
}

/// `parse({**data, **over}, "/test/policy.json")`
fn pol(data: &Value, over: Value) -> Policy {
    let mut d = data.clone();
    for (k, v) in obj(over) {
        d[k] = v;
    }
    parse(&d, Some("/test/policy.json"), None).unwrap()
}

fn code(r: Result<Obj, HandlerError>) -> String {
    match r {
        Err(HandlerError::Op(e)) => e.code,
        other => panic!("expected an OpError, got {other:?}"),
    }
}

fn op_err(r: Result<Obj, HandlerError>) -> OpError {
    match r {
        Err(HandlerError::Op(e)) => e,
        other => panic!("expected an OpError, got {other:?}"),
    }
}

fn a(v: Value) -> Obj {
    obj(v)
}

/// deterministic bytes (an os.urandom stand-in)
fn noise(n: usize, seed: u64) -> Vec<u8> {
    let mut x = seed | 1;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.to_le_bytes()[3]
        })
        .collect()
}

fn sha(b: &[u8]) -> String {
    hex::encode(Sha256::digest(b))
}

fn b64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4usize {
            if i <= c.len() {
                out.push(char::from(
                    T[usize::try_from((n >> (18 - 6 * i)) & 63).unwrap()],
                ));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn read_args(root: &str, rel: &str) -> Obj {
    a(json!({"root": root, "rel": rel, "offset": null, "length": null}))
}

fn write_args(rel: &str, size: i64, sha: Option<&str>, offset: Option<i64>) -> Obj {
    a(json!({"root": "w", "rel": rel, "size": size, "sha256": sha, "offset": offset}))
}

fn chunks(c: &[&[u8]]) -> Vec<Vec<u8>> {
    c.iter().map(|x| x.to_vec()).collect()
}

fn no_mounts() -> Result<Vec<String>, OpError> {
    Ok(Vec::new())
}

#[test]
fn test_path_rules_refuse_traversal_and_odd_paths() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    for rel in [
        "../x",
        "/abs/path",
        "a/../b",
        "a//b",
        "a/./b",
        "",
        ".",
        "a\\b",
        "a\u{0}b",
        "a/",
        "x\ny",
    ] {
        assert_eq!(
            code(fs_read(&p, &read_args("w", rel), &FakeStream::empty())),
            BAD_ARGS,
            "{rel:?}"
        );
        assert_eq!(
            code(fs_stat(&p, &a(json!({"root": "w", "paths": [rel]})))),
            BAD_ARGS,
            "{rel:?}"
        );
    }
}

#[test]
fn test_unknown_root_and_missing_root_dir() {
    let r = roots();
    assert_eq!(
        code(fs_stat(
            &pol(&r.data, json!({})),
            &a(json!({"root": "nope", "paths": ["a"]}))
        )),
        NOT_FOUND
    );
    let gone = pol(
        &r.data,
        json!({"model_roots": [{"id": "g", "path": r.tmp.path().join("unmounted").to_string_lossy()}]}),
    );
    assert_eq!(
        code(fs_stat(&gone, &a(json!({"root": "g", "paths": ["a"]})))),
        NOT_FOUND
    );
}

#[test]
fn test_symlinked_directory_escaping_the_root_is_refused() {
    let r = roots();
    symlink(&r.outside, r.base.join("w").join("escape")).unwrap();
    let p = pol(&r.data, json!({}));
    assert_eq!(
        code(fs_read(
            &p,
            &read_args("w", "escape/secret.txt"),
            &FakeStream::empty()
        )),
        POLICY_DENIED
    );
    let res = fs_write(
        &p,
        &write_args("escape/new.bin", 1, None, None),
        &FakeStream::new(chunks(&[b"x"])),
        &WriteLocks::new(),
    );
    assert_eq!(code(res), POLICY_DENIED);
    assert!(!r.outside.join("new.bin").exists() && !r.outside.join("new.bin.incomplete").exists());
}

#[test]
fn test_final_symlink_is_never_followed_by_fs_read() {
    let r = roots();
    symlink(
        r.outside.join("secret.txt"),
        r.base.join("w").join("link.gguf"),
    )
    .unwrap();
    assert_eq!(
        code(fs_read(
            &pol(&r.data, json!({})),
            &read_args("w", "link.gguf"),
            &FakeStream::empty()
        )),
        BAD_ARGS
    );
}

#[test]
fn test_secret_looking_names_are_never_read() {
    for rel in [
        "token",
        "sub/id_ed25519",
        ".env",
        "creds/credentials.json",
        ".ssh/model.gguf",
        "server.pem",
    ] {
        let r = roots();
        let f = r.base.join("w").join(rel);
        std::fs::create_dir_all(f.parent().unwrap()).unwrap();
        std::fs::write(&f, "do not read").unwrap();
        let s = FakeStream::empty();
        assert_eq!(
            code(fs_read(&pol(&r.data, json!({})), &read_args("w", rel), &s)),
            POLICY_DENIED,
            "{rel}"
        );
        assert!(s.data().is_empty());
    }
}

#[test]
fn test_symlink_inside_the_root_cannot_reach_a_secret_directory_inside_it() {
    let r = roots();
    let w = r.base.join("w");
    std::fs::create_dir(w.join(".ssh")).unwrap();
    std::fs::write(w.join(".ssh").join("config"), "Host *").unwrap();
    symlink(".ssh", w.join("innocent")).unwrap();
    let s = FakeStream::empty();
    assert_eq!(
        code(fs_read(
            &pol(&r.data, json!({})),
            &read_args("w", "innocent/config"),
            &s
        )),
        POLICY_DENIED
    );
    assert!(s.data().is_empty());
    symlink(".ssh/config", w.join("cfg")).unwrap(); // the scanner's reader refuses the same trick
    assert_eq!(read_inside(&w, &w.join("cfg"), 1000), None);
}

#[test]
fn test_stat_reports_files_dirs_links_and_partials() {
    let r = roots();
    let w = r.base.join("w");
    std::fs::write(w.join("a.bin"), b"12345").unwrap();
    std::fs::write(w.join("b.bin.incomplete"), b"123").unwrap();
    std::fs::create_dir(w.join("d")).unwrap();
    std::fs::create_dir(w.join("blobs")).unwrap();
    std::fs::write(w.join("blobs").join("x"), b"blob!").unwrap();
    std::fs::create_dir(w.join("snap")).unwrap();
    symlink("../blobs/x", w.join("snap").join("f")).unwrap();
    symlink(r.outside.join("secret.txt"), w.join("esc")).unwrap();
    let res = fs_stat(
        &pol(&r.data, json!({})),
        &a(json!({"root": "w", "paths": ["a.bin", "b.bin", "d", "snap/f", "esc", "nope/x"]})),
    )
    .unwrap();
    let items: std::collections::HashMap<String, Value> = res["items"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| (i["rel"].as_str().unwrap().to_string(), i.clone()))
        .collect();
    assert_eq!(
        items["a.bin"],
        json!({"rel": "a.bin", "exists": true, "is_dir": false, "size": 5, "partial": 0})
    );
    assert_eq!(items["b.bin"]["exists"], false);
    assert_eq!(items["b.bin"]["partial"], 3);
    assert_eq!(items["d"]["is_dir"], true);
    assert_eq!(items["snap/f"]["exists"], true);
    assert_eq!(items["snap/f"]["size"], 5);
    assert_eq!(items["esc"]["exists"], false); // a link leaving the root does not count
    assert_eq!(items["nope/x"]["exists"], false);
}

// ── fs.write ─────────────────────────────────────────────────────────────────────────────────────────────
fn write(
    p: &Policy,
    rel: &str,
    c: Vec<Vec<u8>>,
    size: i64,
    sha: Option<&str>,
    offset: Option<i64>,
) -> Result<Obj, HandlerError> {
    fs_write(
        p,
        &write_args(rel, size, sha, offset),
        &FakeStream::new(c),
        &WriteLocks::new(),
    )
}

#[test]
fn test_write_creates_dirs_checks_the_whole_file_hash_and_renames() {
    let r = roots();
    let payload = noise(300_000, 7);
    let digest = sha(&payload);
    let size = i64::try_from(payload.len()).unwrap();
    let res = write(
        &pol(&r.data, json!({})),
        "hub/models--org--m/blobs/abc",
        vec![payload[..100_000].to_vec(), payload[100_000..].to_vec()],
        size,
        Some(&digest),
        None,
    )
    .unwrap();
    assert_eq!(Value::Object(res), json!({"size": size, "sha256": digest}));
    let out = r.base.join("w/hub/models--org--m/blobs/abc");
    assert_eq!(std::fs::read(&out).unwrap(), payload);
    assert!(!out.with_file_name("abc.incomplete").exists());
}

#[test]
fn test_write_resumes_from_the_partial_and_hashes_the_whole_file() {
    let r = roots();
    let payload = noise(50_000, 11);
    std::fs::write(r.base.join("w/f.bin.incomplete"), &payload[..20_000]).unwrap();
    let digest = sha(&payload);
    let res = write(
        &pol(&r.data, json!({})),
        "f.bin",
        vec![payload[20_000..].to_vec()],
        50_000,
        Some(&digest),
        Some(20_000),
    )
    .unwrap();
    assert_eq!(res["sha256"], json!(digest));
    assert_eq!(std::fs::read(r.base.join("w/f.bin")).unwrap(), payload);
}

#[test]
fn test_write_offset_must_match_the_partial() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    std::fs::write(r.base.join("w/f.bin.incomplete"), b"xxxxxxxxxx").unwrap();
    let e = op_err(write(&p, "f.bin", chunks(&[b"y"]), 20, None, Some(5)));
    assert!(
        e.code == CONFLICT && e.message.contains("resume from 10"),
        "{e:?}"
    );
    write(&p, "f.bin", vec![b"z".repeat(20)], 20, None, Some(0)).unwrap(); // 0 truncates and starts over
    assert_eq!(
        std::fs::read(r.base.join("w/f.bin")).unwrap(),
        b"z".repeat(20)
    );
}

/// (chunks, size, sha256, expected code)
type MismatchCase<'a> = (&'a [&'a [u8]], i64, Option<&'a str>, &'a str);

#[test]
fn test_write_mismatch_deletes_the_partial() {
    let zeros = "0".repeat(64);
    let cases: [MismatchCase<'_>; 3] = [
        (&[b"abc"], 3, Some(&zeros), HASH_MISMATCH),
        (&[b"ab"], 3, None, SIZE_MISMATCH),
        (&[b"abcd"], 3, None, SIZE_MISMATCH),
    ];
    for (c, size, sha, expected) in cases {
        let r = roots();
        let e = op_err(write(
            &pol(&r.data, json!({})),
            "m.bin",
            chunks(c),
            size,
            sha,
            None,
        ));
        assert_eq!(e.code, expected);
        assert!(!r.base.join("w/m.bin").exists() && !r.base.join("w/m.bin.incomplete").exists());
    }
}

#[test]
fn test_cancelled_write_keeps_the_partial_for_a_resume() {
    let r = roots();
    let s = FakeStream::cancel_after(vec![b"a".repeat(1000), b"b".repeat(1000)], 1);
    let res = fs_write(
        &pol(&r.data, json!({})),
        &write_args("c.bin", 5000, None, None),
        &s,
        &WriteLocks::new(),
    );
    assert!(matches!(res, Err(HandlerError::Cancelled)), "{res:?}");
    assert_eq!(
        std::fs::read(r.base.join("w/c.bin.incomplete")).unwrap(),
        b"a".repeat(1000)
    );
}

#[test]
fn test_write_needs_a_writable_root_and_free_space() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    let ro = fs_write(
        &p,
        &a(json!({"root": "ro", "rel": "x", "size": 1, "sha256": null, "offset": null})),
        &FakeStream::new(chunks(&[b"x"])),
        &WriteLocks::new(),
    );
    assert_eq!(code(ro), POLICY_DENIED);
    // 400 KiB free
    let tiny = |_fd: std::os::fd::BorrowedFd<'_>| -> nix::Result<(u64, u64)> { Ok((100, 4096)) };
    let res = fs_write_with(
        &p,
        &write_args("big", 10, None, None),
        &FakeStream::new(vec![b"x".repeat(10)]),
        &WriteLocks::new(),
        &tiny,
    );
    assert_eq!(code(res), NO_SPACE);
}

#[test]
fn test_two_uploads_of_one_file_conflict() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    let locks = WriteLocks::new();
    let gate = Arc::new((Mutex::new(false), Condvar::new()));
    let (started_tx, started_rx) = mpsc::channel::<()>();
    let slow = FakeStream::new(chunks(&[b"x"]));
    {
        let gate = Arc::clone(&gate);
        let started = Mutex::new(started_tx);
        *slow.before_read.lock().unwrap() = Some(Box::new(move || {
            let _ = started.lock().unwrap().send(());
            let (m, cv) = &*gate;
            let g = m.lock().unwrap();
            let _ = cv
                .wait_timeout_while(g, Duration::from_secs(5), |open| !*open)
                .unwrap();
        }));
    }
    std::thread::scope(|sc| {
        let t = sc.spawn(|| fs_write(&p, &write_args("s.bin", 1, None, None), &slow, &locks));
        started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = fs_write(
            &p,
            &write_args("s.bin", 1, None, None),
            &FakeStream::new(chunks(&[b"y"])),
            &locks,
        );
        {
            let (m, cv) = &*gate;
            *m.lock().unwrap() = true;
            cv.notify_all();
        }
        assert_eq!(code(second), CONFLICT);
        t.join().unwrap().unwrap();
    });
}

// ── fs.read ──────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_read_streams_a_range() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    let payload = noise(fsops::CHUNK * 2 + 10, 3);
    std::fs::write(r.base.join("w/f.gguf"), &payload).unwrap();
    let s = FakeStream::empty();
    let res = fs_read(
        &p,
        &a(json!({"root": "w", "rel": "f.gguf", "offset": 5, "length": null})),
        &s,
    )
    .unwrap();
    assert_eq!(Value::Object(res), json!({"size": payload.len() - 5}));
    assert_eq!(s.data(), payload[5..]);
    assert!(s.chunk_sizes().into_iter().max().unwrap() <= fsops::CHUNK);
    let s2 = FakeStream::empty();
    fs_read(
        &p,
        &a(json!({"root": "w", "rel": "f.gguf", "offset": 0, "length": 7})),
        &s2,
    )
    .unwrap();
    assert_eq!(s2.data(), payload[..7]);
    let res = fs_read(
        &p,
        &a(json!({"root": "w", "rel": "f.gguf", "offset": payload.len() + 1, "length": null})),
        &FakeStream::empty(),
    );
    assert_eq!(code(res), BAD_ARGS);
}

// ── fs.symlink / fs.put_small ────────────────────────────────────────────────────────────────────────────
#[test]
fn test_symlink_inside_the_root_and_atomic_replace() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    let w = r.base.join("w");
    std::fs::create_dir_all(w.join("repo/blobs")).unwrap();
    std::fs::write(w.join("repo/blobs/h1"), "one").unwrap();
    let args =
        |t: &str| a(json!({"root": "w", "link": "repo/snapshots/rev/config.json", "target": t}));
    fs_symlink(&p, &args("../../blobs/h1")).unwrap();
    let link = w.join("repo/snapshots/rev/config.json");
    assert!(link.symlink_metadata().unwrap().file_type().is_symlink());
    assert_eq!(std::fs::read_to_string(&link).unwrap(), "one");
    std::fs::write(w.join("repo/blobs/h2"), "two").unwrap();
    fs_symlink(&p, &args("../../blobs/h2")).unwrap();
    assert_eq!(std::fs::read_to_string(&link).unwrap(), "two");
}

#[test]
fn test_symlink_targets_must_stay_inside() {
    for (target, expected) in [
        ("../../../../../../etc/passwd", POLICY_DENIED),
        ("/etc/passwd", BAD_ARGS),
        ("a//b", BAD_ARGS),
        ("a\\b", BAD_ARGS),
    ] {
        let r = roots();
        let res = fs_symlink(
            &pol(&r.data, json!({})),
            &a(json!({"root": "w", "link": "x/l", "target": target})),
        );
        assert_eq!(code(res), expected, "{target}");
    }
}

#[test]
fn test_symlink_never_replaces_a_directory_and_needs_a_writable_root() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    std::fs::create_dir(r.base.join("w/dir")).unwrap();
    assert_eq!(
        code(fs_symlink(
            &p,
            &a(json!({"root": "w", "link": "dir", "target": "x"}))
        )),
        CONFLICT
    );
    assert_eq!(
        code(fs_symlink(
            &p,
            &a(json!({"root": "ro", "link": "l", "target": "x"}))
        )),
        POLICY_DENIED
    );
}

#[test]
fn test_put_small() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    fs_put_small(
        &p,
        &a(
            json!({"root": "w", "rel": "hub/models--a--b/refs/main", "content_b64": b64(&b"0".repeat(40))}),
        ),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(r.base.join("w/hub/models--a--b/refs/main")).unwrap(),
        b"0".repeat(40)
    );
    assert_eq!(
        code(fs_put_small(
            &p,
            &a(json!({"root": "w", "rel": "x", "content_b64": "%%%"}))
        )),
        BAD_ARGS
    );
    let big = b64(&b"x".repeat(fsops::PUT_SMALL_MAX + 1));
    assert_eq!(
        code(fs_put_small(
            &p,
            &a(json!({"root": "w", "rel": "x", "content_b64": big}))
        )),
        BAD_ARGS
    );
}

// ── fs.delete ────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_delete_rules() {
    let r = roots();
    let m = r.base.join("w/models--org--m");
    std::fs::create_dir_all(m.join("blobs")).unwrap();
    std::fs::write(m.join("blobs/a"), b"x".repeat(1000)).unwrap();
    std::fs::write(m.join("blobs/b"), b"y".repeat(24)).unwrap();
    let p = pol(&r.data, json!({}));
    let del = |rel: &str| a(json!({"root": "w", "rel": rel}));
    assert_eq!(
        code(fs_delete(
            &pol(&r.data, json!({"allow_delete": false})),
            &del("models--org--m"),
            &no_mounts
        )),
        POLICY_DENIED
    );
    assert_eq!(
        code(fs_delete(
            &p,
            &a(json!({"root": "ro", "rel": "x"})),
            &no_mounts
        )),
        POLICY_DENIED
    );
    assert_eq!(code(fs_delete(&p, &del("nope"), &no_mounts)), NOT_FOUND);
    // a running container mounting the target (or something under it) blocks the delete
    let blobs = m.join("blobs").to_string_lossy().into_owned();
    assert_eq!(
        code(fs_delete(&p, &del("models--org--m"), &|| Ok(vec![
            blobs.clone()
        ]))),
        IN_USE
    );
    // a container mounting the PARENT (the whole cache) does not
    let parent = r.base.join("w").to_string_lossy().into_owned();
    let res = fs_delete(&p, &del("models--org--m"), &|| Ok(vec![parent.clone()])).unwrap();
    assert_eq!(Value::Object(res), json!({"freed": 1024}));
    assert!(!m.exists());
}

#[test]
fn test_delete_refuses_the_root_itself_and_other_roots() {
    let r = roots();
    let w = r.base.join("w");
    symlink(&w, w.join("self")).unwrap();
    let p = pol(&r.data, json!({}));
    let del = |rel: &str| a(json!({"root": "w", "rel": rel}));
    assert_eq!(
        code(fs_delete(&p, &del("self/inner"), &no_mounts)),
        POLICY_DENIED
    ); // holds root
    std::fs::remove_file(w.join("self")).unwrap();
    assert_eq!(
        code(fs_delete(&p, &del("inner"), &no_mounts)),
        POLICY_DENIED
    );
    symlink(".", w.join("loop")).unwrap();
    // deleting a symlink removes the link only, never what it points at
    assert_eq!(
        Value::Object(fs_delete(&p, &del("loop"), &no_mounts).unwrap()),
        json!({"freed": 0})
    );
    assert!(w.join("inner").is_dir());
}

#[test]
fn test_delete_when_docker_cannot_say_what_runs() {
    let r = roots();
    std::fs::write(r.base.join("w/f.gguf"), b"1").unwrap();
    let broken =
        || -> Result<Vec<String>, OpError> { Err(OpError::new(DOCKER_ERROR, "cannot ask docker")) };
    assert_eq!(
        code(fs_delete(
            &pol(&r.data, json!({})),
            &a(json!({"root": "w", "rel": "f.gguf"})),
            &broken
        )),
        DOCKER_ERROR
    );
    assert!(r.base.join("w/f.gguf").exists());
}

// ── security review 2026-10-04: regressions for H1 and H2 ─────────────────────────────────────────────────
#[test]
fn test_h1_a_link_through_a_link_cannot_reach_the_roots_parent() {
    // `y -> sub/..` (the root itself) then `z -> y/..`: lexically the root, really the root's parent.
    let r = roots();
    let p = pol(&r.data, json!({}));
    let w = r.base.join("w");
    std::fs::create_dir(w.join("sub")).unwrap();
    let link = |l: &str, t: &str| a(json!({"root": "w", "link": l, "target": t}));
    assert_eq!(code(fs_symlink(&p, &link("y", "sub/.."))), POLICY_DENIED);
    assert!(w.join("y").symlink_metadata().is_err());
    symlink("sub/..", w.join("y")).unwrap(); // y planted some other way: z must still be refused
    assert_eq!(code(fs_symlink(&p, &link("z", "y/.."))), POLICY_DENIED);
    assert!(w.join("z").symlink_metadata().is_err());
    // leading '..' segments are what HF snapshot links use, and they still work
    std::fs::create_dir_all(w.join("repo/blobs")).unwrap();
    fs_symlink(&p, &link("repo/snapshots/rev/f", "../../blobs/h")).unwrap();
    assert_eq!(
        std::fs::read_link(w.join("repo/snapshots/rev/f")).unwrap(),
        Path::new("../../blobs/h")
    );
}

#[test]
fn test_h2_delete_never_follows_a_path_swapped_during_the_in_use_check() {
    // fs.delete asks docker what runs between its checks and the removal. A parent swapped for an outward link in
    // that window must not redirect the delete to a same-named directory outside the root.
    for how in ["removed", "renamed"] {
        let r = roots();
        let p = pol(&r.data, json!({}));
        let w = r.base.join("w");
        let victim = r.base.join("precious"); // next to the root w, outside it
        std::fs::create_dir(&victim).unwrap();
        std::fs::write(victim.join("file.txt"), "user data").unwrap();
        std::fs::create_dir_all(w.join("in/precious")).unwrap();
        std::fs::write(w.join("in/precious/f"), b"z").unwrap();
        let w2 = w.clone();
        let swap = move || -> Result<Vec<String>, OpError> {
            if how == "removed" {
                std::fs::remove_dir_all(w2.join("in")).unwrap();
            } else {
                std::fs::rename(w2.join("in"), w2.join("in-old")).unwrap();
            }
            symlink("..", w2.join("in")).unwrap(); // w/in now leads to the root's parent
            Ok(Vec::new())
        };
        let res = fs_delete(&p, &a(json!({"root": "w", "rel": "in/precious"})), &swap);
        assert_eq!(code(res), CONFLICT, "{how}");
        assert_eq!(
            std::fs::read_to_string(victim.join("file.txt")).unwrap(),
            "user data"
        );
        if how == "renamed" {
            assert!(w.join("in-old/precious/f").exists()); // moved, so nothing at all was removed
        }
    }
}

#[test]
fn test_delete_removes_links_inside_the_tree_without_following_them() {
    let r = roots();
    let m = r.base.join("w/m");
    std::fs::create_dir_all(m.join("d")).unwrap();
    std::fs::write(m.join("d/w.bin"), b"x".repeat(10)).unwrap();
    std::fs::hard_link(m.join("d/w.bin"), m.join("d/hard")).unwrap(); // one inode: counted once
    symlink(&r.outside, m.join("esc")).unwrap();
    let res = fs_delete(
        &pol(&r.data, json!({})),
        &a(json!({"root": "w", "rel": "m"})),
        &no_mounts,
    )
    .unwrap();
    assert_eq!(Value::Object(res), json!({"freed": 10}));
    assert!(!m.exists() && r.outside.join("secret.txt").exists());
}

// ── explicit tests of each control (Rust-only extras) ──────────────────────────────────────────────────
#[test]
fn test_wsd_fs_write_refuses_a_symlinked_partial() {
    let r = roots();
    let w = r.base.join("w");
    symlink(r.outside.join("secret.txt"), w.join("v.bin.incomplete")).unwrap();
    let e = op_err(write(
        &pol(&r.data, json!({})),
        "v.bin",
        chunks(&[b"x"]),
        1,
        None,
        None,
    ));
    assert_eq!(e.code, POLICY_DENIED);
    assert_eq!(
        std::fs::read_to_string(r.outside.join("secret.txt")).unwrap(),
        "outside the root"
    );
}

#[test]
fn test_wsd_fs_write_refuses_a_name_too_long_for_its_partial() {
    let r = roots();
    let e = op_err(write(
        &pol(&r.data, json!({})),
        &"n".repeat(250),
        chunks(&[b"x"]),
        1,
        None,
        None,
    ));
    assert!(
        e.code == BAD_ARGS && e.message.contains("max 243 bytes"),
        "{e:?}"
    );
}

#[test]
fn test_wsd_fs_write_refuses_another_process_holding_the_partial() {
    use nix::fcntl::{Flock, FlockArg};
    let r = roots();
    let part = r.base.join("w/l.bin.incomplete");
    std::fs::write(&part, b"").unwrap();
    let held = Flock::lock(std::fs::File::open(&part).unwrap(), FlockArg::LockExclusive).unwrap();
    let e = op_err(write(
        &pol(&r.data, json!({})),
        "l.bin",
        chunks(&[b"x"]),
        1,
        None,
        None,
    ));
    assert!(
        e.code == CONFLICT && e.message.contains("another process"),
        "{e:?}"
    );
    drop(held);
}

#[test]
fn test_wsd_fs_write_size_and_sha_arguments_are_checked() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    for (size, sha, offset, frag) in [
        (-1, None, None, "whole file's size"),
        (1 << 51, None, None, "whole file's size"),
        (3, Some("abc"), None, "64 hex digits"),
        (3, None, Some(4), "outside the file (0-3)"),
    ] {
        let e = op_err(write(&p, "q.bin", chunks(&[b"abc"]), size, sha, offset));
        assert!(e.code == BAD_ARGS && e.message.contains(frag), "{e:?}");
    }
    // the sha256 is compared case-insensitively and reported lower-case
    let upper = sha(b"abc").to_uppercase();
    let res = write(&p, "q.bin", chunks(&[b"abc"]), 3, Some(&upper), None).unwrap();
    assert_eq!(res["sha256"], json!(sha(b"abc")));
}

#[test]
fn test_wsd_fs_read_refuses_non_regular_files_and_negative_lengths() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    std::fs::create_dir(r.base.join("w/dir.gguf")).unwrap();
    nix::unistd::mkfifo(
        &r.base.join("w/fifo.gguf"),
        nix::sys::stat::Mode::from_bits_truncate(0o644),
    )
    .unwrap();
    for rel in ["dir.gguf", "fifo.gguf"] {
        let e = op_err(fs_read(&p, &read_args("w", rel), &FakeStream::empty()));
        assert!(
            e.code == BAD_ARGS && e.message.contains("is not a regular file"),
            "{rel}: {e:?}"
        );
    }
    std::fs::write(r.base.join("w/f.gguf"), b"12345").unwrap();
    let e = op_err(fs_read(
        &p,
        &a(json!({"root": "w", "rel": "f.gguf", "offset": 0, "length": -1})),
        &FakeStream::empty(),
    ));
    assert_eq!(e.message, "length must not be negative");
    let e = op_err(fs_read(
        &p,
        &read_args("w", "missing.gguf"),
        &FakeStream::empty(),
    ));
    assert_eq!(
        (e.code.as_str(), e.message.as_str()),
        (NOT_FOUND, "'missing.gguf' does not exist in model root 'w'")
    );
}

#[test]
fn test_wsd_fs_read_stops_when_cancelled() {
    let r = roots();
    std::fs::write(r.base.join("w/c.gguf"), noise(fsops::CHUNK * 3, 5)).unwrap();
    let s = FakeStream::empty();
    s.set_cancelled();
    let res = fs_read(&pol(&r.data, json!({})), &read_args("w", "c.gguf"), &s);
    assert!(matches!(res, Err(HandlerError::Cancelled)), "{res:?}");
    assert!(s.data().is_empty());
}

#[test]
fn test_wsd_fs_delete_needs_a_writable_root_and_never_follows_a_link() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    std::fs::write(r.base.join("ro/m.gguf"), b"1").unwrap();
    assert_eq!(
        code(fs_delete(
            &p,
            &a(json!({"root": "ro", "rel": "m.gguf"})),
            &no_mounts
        )),
        POLICY_DENIED
    );
    assert!(r.base.join("ro/m.gguf").exists());
    // a link to the root's parent inside the root: the link itself goes, the parent stays
    symlink("..", r.base.join("w/up")).unwrap();
    let res = fs_delete(&p, &a(json!({"root": "w", "rel": "up"})), &no_mounts).unwrap();
    assert_eq!(Value::Object(res), json!({"freed": 0}));
    assert!(r.base.join("ro/m.gguf").exists() && r.outside.join("secret.txt").exists());
}

#[test]
fn test_wsd_fs_put_small_and_symlink_messages() {
    let r = roots();
    let p = pol(&r.data, json!({}));
    let e = op_err(fs_put_small(
        &p,
        &a(json!({"root": "w", "rel": "x", "content_b64": "YWJj=="})),
    ));
    assert_eq!(e.message, "content_b64 is not valid base64");
    let e = op_err(fs_symlink(
        &p,
        &a(json!({"root": "w", "link": "l", "target": "a/../b"})),
    ));
    assert_eq!(
        e.message,
        "the link target may use '..' only at its start, like ../../blobs/<hash>"
    );
    let e = op_err(fs_symlink(
        &p,
        &a(json!({"root": "w", "link": "l", "target": "./b"})),
    ));
    assert_eq!(
        e.message,
        "the link target must be a relative path like ../../blobs/<hash>"
    );
    // nothing temporary is left behind
    let left: Vec<_> = std::fs::read_dir(r.base.join("w"))
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert!(
        left.iter()
            .all(|n| !n.to_string_lossy().contains(".ecnode-")),
        "{left:?}"
    );
}
