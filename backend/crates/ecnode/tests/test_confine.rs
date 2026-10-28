//! The node's path boundary (src/confine.rs + the posixpath half of src/util/py.rs) against goldens made by the
//! Python reference (tests/gen/wsd_path_vectors.py -> tests/data/wsd_path_vectors.json), plus proptests on the
//! invariants the boundary rests on.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

use std::os::fd::AsFd;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use ecnode::confine::{
    Fail, RootHandle, fd_path, inside, open_dir, open_inside, open_root, read_inside,
    secret_inside, split_rel,
};
use ecnode::policy::{Policy, Root};
use ecnode::util::py;
use proptest::prelude::*;
use serde_json::{Value, json};

fn vectors() -> Value {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/wsd_path_vectors.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

fn s(v: &Value) -> &str {
    v.as_str().unwrap()
}

fn build_tree(tmp: &Path, tree: &Value) {
    for d in tree["dirs"].as_array().unwrap() {
        std::fs::create_dir_all(tmp.join(s(d))).unwrap();
    }
    for f in tree["files"].as_array().unwrap() {
        std::fs::write(tmp.join(s(f)), "x").unwrap();
    }
    for l in tree["links"].as_array().unwrap() {
        symlink(s(&l[1]), tmp.join(s(&l[0]))).unwrap();
    }
}

fn rel_to_tmp(p: &Path, tmp: &Path) -> String {
    let p = p.to_string_lossy().into_owned();
    let t = tmp.to_string_lossy().into_owned();
    if p == t || p.starts_with(&format!("{t}/")) {
        format!("<TMP>{}", &p[t.len()..])
    } else {
        p
    }
}

fn policy_for(root: &Path) -> Policy {
    Policy {
        path: Some("/x/policy.json".into()),
        model_roots: vec![Root {
            writable: true,
            ..Root::new("t", &root.to_string_lossy())
        }],
        ..Policy::default()
    }
}

#[test]
fn test_wsd_path_vectors_match_python() {
    let v = vectors();
    let mut checked = 0usize;
    for c in v["split_rel"].as_array().unwrap() {
        let got = split_rel(c["rel"].as_str(), s(&c["what"]));
        match (&got, c.get("ok")) {
            (Ok(parts), Some(ok)) => assert_eq!(&json!(parts), ok, "{c}"),
            (Err(e), None) => {
                assert_eq!(
                    (e.code.as_str(), e.message.as_str()),
                    (s(&c["code"]), s(&c["message"])),
                    "{c}"
                );
            }
            _ => panic!("split_rel {c}: got {got:?}"),
        }
        checked += 1;
    }
    for c in v["inside"].as_array().unwrap() {
        let got = inside(Path::new(s(&c["child"])), Path::new(s(&c["root"])));
        assert_eq!(json!(got), c["inside"], "{c}");
        checked += 1;
    }
    for c in v["secret_inside"].as_array().unwrap() {
        let got = secret_inside(Path::new(s(&c["real"])), Path::new(s(&c["root"])));
        assert_eq!(json!(got), c["secret"], "{c}");
        checked += 1;
    }
    for c in v["normpath"].as_array().unwrap() {
        assert_eq!(py::normpath_str(s(&c["in"])), s(&c["out"]), "{c}");
        checked += 1;
    }
    for c in v["join"].as_array().unwrap() {
        let got = py::join(Path::new(s(&c["a"])), &[s(&c["b"])]);
        assert_eq!(got.to_string_lossy(), s(&c["out"]), "{c}");
        checked += 1;
    }
    for c in v["relpath"].as_array().unwrap() {
        let got = py::relpath(Path::new(s(&c["path"])), Path::new(s(&c["start"])));
        assert_eq!(got.to_string_lossy(), s(&c["out"]), "{c}");
        checked += 1;
    }
    for c in v["splitext"].as_array().unwrap() {
        // splitext_ext works on one name; posixpath.splitext on a path: same rule on the last component
        assert_eq!(py::splitext_ext(s(&c["name"])), s(&c["ext"]), "{c}");
        checked += 1;
    }

    let dir = tempfile::tempdir().unwrap();
    let tmp = py::realpath(dir.path());
    build_tree(&tmp, &v["tree"]);
    let root = tmp.join("root");
    for c in v["realpath"].as_array().unwrap() {
        let q = s(&c["in"]);
        let p = if q.starts_with('/') {
            PathBuf::from(q)
        } else {
            py::join(&root, &[q])
        };
        assert_eq!(rel_to_tmp(&py::realpath(&p), &tmp), s(&c["out"]), "{c}");
        checked += 1;
    }
    let policy = policy_for(&root);
    let h = open_root(&policy, &json!("t"), false).unwrap();
    for c in v["open_dir"].as_array().unwrap() {
        let parts: Vec<String> = c["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| s(p).to_string())
            .collect();
        match (open_dir(&h, &parts, false), c.get("ok")) {
            (Ok(fd), Some(ok)) => {
                assert_eq!(
                    rel_to_tmp(&fd_path(fd.as_fd()).unwrap(), &tmp),
                    s(ok),
                    "{c}"
                );
            }
            (Err(Fail::Op(e)), None) => {
                assert_eq!(
                    (e.code.as_str(), e.message.as_str()),
                    (s(&c["code"]), s(&c["message"])),
                    "{c}"
                );
            }
            (got, _) => panic!("open_dir {c}: got {:?}", got.map(|_| "an fd")),
        }
        checked += 1;
    }
    for c in v["open_inside"].as_array().unwrap() {
        let p = py::join(&root, &[s(&c["in"])]);
        assert_eq!(json!(open_inside(&root, &p).is_some()), c["open"], "{c}");
        let data = read_inside(&root, &p, 1).map(|d| String::from_utf8(d).unwrap());
        assert_eq!(json!(data), c["read1"], "{c}");
        checked += 1;
    }
    assert!(checked > 1400, "only {checked} vectors");
}

// ── proptests ────────────────────────────────────────────────────────────────────────────────────────────
fn rel_strategy() -> impl Strategy<Value = String> {
    let piece = prop::sample::select(vec![
        "a", "b", ".", "..", "", "/", "\\", "\u{0}", "\n", "\u{7f}", "\u{e9}", " ", "x.gguf", "-",
    ]);
    prop::collection::vec(piece, 0..8).prop_map(|v| v.concat())
}

proptest! {
    #![proptest_config(ProptestConfig { failure_persistence: None, ..ProptestConfig::default() })]

    #[test]
    fn test_wsd_path_proptest_split_rel_accepts_only_clean_segments(rel in rel_strategy()) {
        if let Ok(parts) = split_rel(Some(&rel), "path") {
            prop_assert_eq!(parts.join("/"), rel.clone());
            for p in &parts {
                prop_assert!(!p.is_empty() && p != "." && p != "..");
                prop_assert!(p.len() <= 255);
                let bad = p.chars().any(|c| c == '\\' || c.is_ascii_control());
                prop_assert!(!bad, "segment {:?} holds a backslash or a control character", p);
            }
            prop_assert!(!rel.starts_with('/'));
        }
    }

    #[test]
    fn test_wsd_path_proptest_inside_matches_component_check(
        c in prop::collection::vec(prop::sample::select(vec!["a", "b", "ab", "a.b", "c"]), 0..5),
        r in prop::collection::vec(prop::sample::select(vec!["a", "b", "ab", "a.b", "c"]), 0..5),
        slash in any::<bool>(),
    ) {
        let child = format!("/{}", c.join("/"));
        let mut root = format!("/{}", r.join("/"));
        if slash && root != "/" {
            root.push('/');
        }
        let component_prefix = r.len() <= c.len() && c[..r.len()] == r[..];
        // Python's quirk, kept: a root written with a trailing slash does not contain itself without one
        let expected = component_prefix && !(slash && root != "/" && c.len() == r.len());
        prop_assert_eq!(inside(Path::new(&child), Path::new(&root)), expected);
    }

    #[test]
    fn test_wsd_path_proptest_realpath_without_links_is_normpath(
        segs in prop::collection::vec(prop::sample::select(vec!["d1", "d2", "f", "nope", ".", "..", ""]), 1..7),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let t = py::realpath(dir.path());
        std::fs::create_dir_all(t.join("d1/d2")).unwrap();
        std::fs::write(t.join("d1/f"), "x").unwrap();
        let p = format!("{}/{}", t.display(), segs.join("/"));
        prop_assert_eq!(py::realpath(Path::new(&p)), py::normpath(Path::new(&p)));
    }

    #[test]
    fn test_wsd_path_proptest_open_dir_never_escapes(
        rel in prop::collection::vec(
            prop::sample::select(vec!["d1", "d2", "out", "out2", "abs", "in", "loop", "up", "new", "f"]), 1..5)
            .prop_map(|v| v.join("/")),
        create in any::<bool>(),
    ) {
        let dir = tempfile::tempdir().unwrap();
        let t = py::realpath(dir.path());
        let root = t.join("root");
        let outside = t.join("outside");
        std::fs::create_dir_all(root.join("d1/d2")).unwrap();
        std::fs::create_dir_all(outside.join("d1")).unwrap();
        std::fs::write(root.join("f"), "x").unwrap();
        symlink("../outside", root.join("out")).unwrap();
        symlink("d1/../../outside", root.join("out2")).unwrap();
        symlink(&outside, root.join("abs")).unwrap();
        symlink("d1", root.join("in")).unwrap();
        symlink("loop", root.join("loop")).unwrap();
        symlink("..", root.join("d1/up")).unwrap();
        let before: Vec<_> = walk(&outside);
        let h: RootHandle = open_root(&policy_for(&root), &json!("t"), true).unwrap();
        let parts = split_rel(Some(&rel), "path").unwrap();
        if let Ok(fd) = open_dir(&h, &parts, create) {
            let real = fd_path(fd.as_fd()).unwrap();
            prop_assert!(inside(&real, &h.real), "{} escaped to {}", rel, real.display());
        }
        prop_assert_eq!(walk(&outside), before, "something was created outside the root");
    }
}

fn walk(p: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(p).unwrap() {
        let e = e.unwrap();
        out.push(e.path());
        if e.file_type().unwrap().is_dir() {
            out.extend(walk(&e.path()));
        }
    }
    out.sort();
    out
}
