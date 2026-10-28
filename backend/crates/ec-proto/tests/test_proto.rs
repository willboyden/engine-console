//! Port of node/tests/test_proto.py: framing, preamble, limits and the whitelist cleaners. Test names mirror the
//! Python ones (backend/TEST-MAP.md).
#![allow(clippy::panic, unreachable_pub)]

use std::io::{BufReader, Cursor};

use ec_proto::clean::{JSON_MAX_DEPTH, l, m, o, s};
use ec_proto::{
    Frame, KIND_DATA, KIND_JSON, MAGIC, MAX_DATA, MAX_JSON, Obj, ProtoError, ReadError, Spec,
    WINDOW, clean, clean_args, clean_json, decode_json, encode_data, encode_json, find_magic_sync,
    missing_args, msg, parse_header, read_frame_sync, sanitize_push, sanitize_res, specs,
};
use serde_json::{Value, json};

fn obj(v: Value) -> Obj {
    match v {
        Value::Object(o) => o,
        _ => panic!("not an object"),
    }
}

fn hdr(len: u32, kind: u8) -> [u8; 5] {
    let b = len.to_be_bytes();
    [b[0], b[1], b[2], b[3], kind]
}

fn is_proto_err<T: std::fmt::Debug>(r: &Result<T, ReadError>) -> bool {
    matches!(r, Err(ReadError::Proto(_)))
}

// ── framing ──────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_json_frame_round_trip_sync() {
    let msg = obj(
        json!({"type": "req", "id": 7, "op": "docker.images", "args": {}, "unicode": "Grüße ✓"}),
    );
    let mut f = Cursor::new(encode_json(&msg).unwrap());
    let frame = read_frame_sync(&mut f).unwrap();
    assert_eq!(frame, Some(Frame::Json(msg)));
    assert_eq!(read_frame_sync(&mut f).unwrap(), None); // clean end between frames
}

#[test]
fn test_data_frame_round_trip_and_binary_payload() {
    let payload: Vec<u8> = (0..10).flat_map(|_| 0u8..=255).collect();
    let mut f = Cursor::new(encode_data(42, &payload).unwrap());
    let frame = read_frame_sync(&mut f).unwrap();
    assert_eq!(
        frame,
        Some(Frame::Data {
            stream: 42,
            data: payload
        })
    );
}

#[cfg(feature = "tokio")]
#[tokio::test]
async fn test_async_reader_reads_the_same_frames() {
    let mut blob = b"Welcome to the box\nmotd line\n".to_vec();
    blob.extend_from_slice(MAGIC);
    blob.extend(encode_json(&obj(json!({"type": "hello", "proto": 1}))).unwrap());
    blob.extend(encode_data(3, b"abc").unwrap());
    let mut r = tokio::io::BufReader::new(Cursor::new(blob));
    let skipped = ec_proto_async::find_magic(&mut r).await.unwrap();
    assert_eq!(skipped, b"Welcome to the box\nmotd line\n".len());
    let a = ec_proto_async::read(&mut r).await.unwrap().map(|(f, _)| f);
    let b = ec_proto_async::read(&mut r).await.unwrap().map(|(f, _)| f);
    let c = ec_proto_async::read(&mut r).await.unwrap();
    assert_eq!(
        a,
        Some(Frame::Json(obj(json!({"type": "hello", "proto": 1}))))
    );
    assert_eq!(
        b,
        Some(Frame::Data {
            stream: 3,
            data: b"abc".to_vec()
        })
    );
    assert!(c.is_none());
}

/// The async readers live behind the `tokio` feature (the hub enables it, so a workspace `cargo test` has it on).
#[cfg(feature = "tokio")]
mod ec_proto_async {
    use ec_proto::{Frame, ReadError};

    pub async fn find_magic<R: tokio::io::AsyncBufRead + Unpin>(
        r: &mut R,
    ) -> Result<usize, ReadError> {
        ec_proto::find_magic_async(r, ec_proto::MAX_DATA + 4096).await
    }

    pub async fn read<R: tokio::io::AsyncRead + Unpin>(
        r: &mut R,
    ) -> Result<Option<(Frame, usize)>, ReadError> {
        ec_proto::read_frame_async(r).await
    }
}

#[test]
fn test_magic_after_noise_without_trailing_newline() {
    let mut blob = b"no newline here".to_vec();
    blob.extend_from_slice(MAGIC);
    blob.extend(encode_json(&obj(json!({"type": "x"}))).unwrap());
    let mut f = BufReader::new(Cursor::new(blob));
    assert_eq!(find_magic_sync(&mut f).unwrap(), b"no newline here".len());
    assert_eq!(
        read_frame_sync(&mut f).unwrap(),
        Some(Frame::Json(obj(json!({"type": "x"}))))
    );
}

#[tokio::test]
async fn test_preamble_limit_and_early_close() {
    let mut early = vec![b'x'; 100];
    early.push(b'\n');
    assert!(is_proto_err(&find_magic_sync(&mut BufReader::new(
        Cursor::new(early)
    ))));
    let mut noise = Vec::new();
    for _ in 0..70 {
        noise.extend(std::iter::repeat_n(b'y', 1000));
        noise.push(b'\n');
    }
    let mut with_magic = noise.clone();
    with_magic.extend_from_slice(MAGIC);
    assert!(is_proto_err(&find_magic_sync(&mut BufReader::new(
        Cursor::new(with_magic.clone())
    ))));
    #[cfg(feature = "tokio")]
    {
        let mut r = tokio::io::BufReader::new(Cursor::new(with_magic));
        assert!(is_proto_err(
            &ec_proto::find_magic_async(&mut r, MAX_DATA + 4096).await
        ));
    }
}

#[test]
fn test_header_limits() {
    for h in [
        hdr(1, KIND_JSON),                                    // empty JSON body
        hdr(u32::try_from(MAX_JSON + 2).unwrap(), KIND_JSON), // over MAX_JSON
        hdr(3, KIND_DATA),                                    // data frame without a stream id
        hdr(u32::try_from(MAX_DATA + 6).unwrap(), KIND_DATA), // over MAX_DATA
        hdr(10, 9),                                           // unknown kind
    ] {
        assert!(parse_header(&h).is_err(), "{h:?}");
    }
}

#[test]
fn test_truncated_frames_are_errors_not_eof() {
    let full = encode_json(&obj(json!({"type": "x", "pad": "z".repeat(50)}))).unwrap();
    let r = read_frame_sync(&mut Cursor::new(full[..3].to_vec()));
    assert!(matches!(r, Err(ReadError::Proto(ProtoError::Truncated(_)))));
    let r = read_frame_sync(&mut Cursor::new(full[..full.len() - 1].to_vec()));
    assert!(matches!(r, Err(ReadError::Proto(ProtoError::Truncated(_)))));
}

#[test]
fn test_encoders_refuse_what_the_reader_would_refuse() {
    assert!(encode_data(0, b"x").is_err());
    assert!(encode_data(1, &vec![b'x'; MAX_DATA + 1]).is_err());
    assert!(encode_json(&obj(json!({"big": "x".repeat(MAX_JSON + 1)}))).is_err());
    // NaN cannot even be put in a serde_json value: the builder refuses it at construction
    assert!(serde_json::Number::from_f64(f64::NAN).is_none());
}

#[test]
fn test_decode_json_rejects_hostile_bodies() {
    let deep = "{".repeat(20000);
    for body in [
        b"{\"n\": NaN}".as_slice(),
        b"{\"n\": Infinity}",
        b"{\"n\": 1e999}",
        b"[1, 2]",
        b"\xff\xfe",
        deep.as_bytes(),
    ] {
        assert!(
            decode_json(body).is_err(),
            "{}",
            String::from_utf8_lossy(&body[..body.len().min(20)])
        );
    }
}

#[test]
fn test_data_frame_with_zero_stream_id_is_rejected() {
    let mut raw = hdr(5 + 3, KIND_DATA).to_vec();
    raw.extend_from_slice(&0u32.to_be_bytes());
    raw.extend_from_slice(b"abc");
    assert!(is_proto_err(&read_frame_sync(&mut Cursor::new(raw))));
}

// ── cleaning ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_clean_types_caps_and_nul_stripping() {
    let spec = o(&[
        ("n", Spec::Int),
        ("x", Spec::Num),
        ("b", Spec::Bool),
        ("s", s(5)),
        ("l", l(Spec::Int, 3)),
        ("m", m(s(3), 2)),
    ]);
    let got = clean(
        &json!({"n": true, "x": 1u64 << 60, "b": 1, "s": "ab\u{0}cdefgh", "l": [1, "2", 3, 4, 5],
                "m": {"a": "xyzw", "b": 5, "c": "q"}, "extra": "dropped"}),
        &spec,
    );
    assert_eq!(
        got,
        json!({"n": null, "x": null, "b": null, "s": "abcd", "l": [1, 3], "m": {"a": "xyz"}})
    );
}

#[test]
fn test_clean_json_is_bounded() {
    let mut deep = json!("leaf");
    for _ in 0..50 {
        deep = json!([deep]);
    }
    let out = clean_json(&json!({"a": deep, "f": 1.5, "big": 1u64 << 60}));
    assert_eq!(out["big"], Value::Null);
    assert_eq!(out["f"], json!(1.5));
    let mut depth = 0;
    let mut node = &out["a"];
    while let Some(first) = node.as_array().and_then(|a| a.first()) {
        node = first;
        depth += 1;
    }
    assert!(depth <= JSON_MAX_DEPTH);
}

#[test]
fn test_sanitize_push_whitelists_and_rejects_hello_without_proto() {
    let s = sanitize_push(&obj(json!({"type": "sample", "seq": 1, "t": 1.5,
        "gpus": {"GPU-1": {"util_pct": 50, "evil": 1}}, "injected": "<script>"})))
    .unwrap()
    .unwrap();
    assert_eq!(s["type"], "sample");
    assert!(!s.contains_key("injected"));
    assert_eq!(s["gpus"]["GPU-1"]["util_pct"], 50);
    assert!(s["gpus"]["GPU-1"].get("evil").is_none());
    assert!(
        sanitize_push(&obj(json!({"type": "who-knows"})))
            .unwrap()
            .is_none()
    );
    assert!(sanitize_push(&obj(json!({"type": "hello"}))).is_err());
    let h = sanitize_push(&obj(json!({"type": "hello", "proto": 1,
        "gpus": [{"uuid": "GPU-a"}, {"name": "no uuid"}]})))
    .unwrap()
    .unwrap();
    let uuids: Vec<&Value> = h["gpus"]
        .as_array()
        .unwrap()
        .iter()
        .map(|g| &g["uuid"])
        .collect();
    assert_eq!(uuids, vec![&json!("GPU-a")]);
}

#[test]
fn test_sanitize_res_ok_and_error_shapes() {
    let ok = sanitize_res(
        &obj(json!({"type": "res", "id": 3, "ok": true, "result": {"present": true, "x": 1}})),
        "docker.image_present",
    );
    assert_eq!(
        Value::Object(ok),
        json!({"id": 3, "ok": true, "result": {"present": true, "id": null}})
    );
    let bad = sanitize_res(
        &obj(json!({"type": "res", "id": 4, "ok": "yes"})),
        "docker.run",
    );
    assert_eq!(bad["ok"], false);
    assert_eq!(bad["error"]["code"], ec_proto::INTERNAL);
    assert!(
        bad["error"]["message"]
            .as_str()
            .is_some_and(|m| !m.is_empty())
    );
}

#[test]
fn test_clean_args_and_required() {
    let a = clean_args(
        "fs.write",
        &json!({"root": "hf", "rel": "x", "size": "12", "evil": true}),
    )
    .unwrap();
    assert_eq!(
        Value::Object(a.clone()),
        json!({"root": "hf", "rel": "x", "size": null, "sha256": null, "offset": null})
    );
    assert_eq!(missing_args("fs.write", &a), vec!["size"]);
    assert!(clean_args("shell.exec", &json!({"cmd": "id"})).is_err());
}

#[test]
fn test_every_op_has_args_and_result_specs_and_monitor_ops_exist() {
    let mut a = specs::arg_ops();
    let mut r = specs::result_ops();
    a.sort_unstable();
    r.sort_unstable();
    assert_eq!(a, r);
    let ops: Vec<&str> = specs::OPS.to_vec();
    assert_eq!(specs::arg_ops(), ops);
    for op in specs::MONITOR_OPS {
        assert!(specs::is_op(op), "{op}");
    }
    for op in specs::event_ops() {
        assert!(specs::is_op(op), "{op}");
    }
    for op in specs::REQUIRED_OPS {
        assert!(specs::is_op(op), "{op}");
        assert!(!specs::required(op).is_empty(), "{op}");
    }
    for op in specs::UPLOAD_OPS {
        assert!(specs::is_op(op), "{op}");
    }
    for op in [
        "docker.run",
        "fs.write",
        "fs.delete",
        "scripts.run",
        "docker.pull",
        "gateway.start",
    ] {
        assert!(!specs::is_monitor_op(op), "{op}");
    }
}

#[test]
fn test_builders_produce_frames_the_reader_accepts() {
    let msgs = [
        msg::req(1, "node.info", None),
        msg::res_ok(1, None),
        msg::res_err(2, ec_proto::BUSY, &"x".repeat(5000)),
        msg::event(3, obj(json!({"line": "hi"}))),
        msg::credit(3, WINDOW),
        msg::end(3),
        msg::cancel(3),
        msg::ping(9),
        msg::pong(9),
    ];
    let mut blob = Vec::new();
    for mm in &msgs {
        blob.extend(encode_json(mm).unwrap());
    }
    let mut f = Cursor::new(blob);
    let mut got = Vec::new();
    while let Some(Frame::Json(fr)) = read_frame_sync(&mut f).unwrap() {
        got.push(fr);
    }
    assert_eq!(got[2]["error"]["message"], "x".repeat(1000));
    let types: Vec<&str> = got.iter().map(|g| g["type"].as_str().unwrap()).collect();
    assert_eq!(
        types,
        [
            "req", "res", "res", "event", "credit", "end", "cancel", "ping", "pong"
        ]
    );
}

#[test]
fn test_audit_f5_strict_args_are_refused_not_cut() {
    // security audit F5: what `clean` would cut or alter in a path, link, container name, image or argv is reported,
    // so the node refuses the request; `clean` itself still caps (the hub keeps cleaning pushed data with it)
    use ec_proto::overlong_args;
    let why = |op: &str, args: Value| overlong_args(op, &args);
    let at = |n: usize| "a".repeat(n);
    assert_eq!(
        why("fs.write", json!({"root": "r", "rel": at(1025), "size": 1})).as_deref(),
        Some(
            "fs.write: rel is longer than 1024 characters; the node refuses it rather than cut it short"
        )
    );
    assert_eq!(
        why("fs.write", json!({"root": "r", "rel": at(1024), "size": 1})),
        None
    );
    // characters, not bytes, as `clean` counts
    assert_eq!(
        why(
            "fs.read",
            json!({"root": "r", "rel": "\u{e9}".repeat(1024)})
        ),
        None
    );
    for (op, args, frag) in [
        (
            "fs.symlink",
            json!({"root": "r", "link": at(1025), "target": "t"}),
            "link is longer than 1024",
        ),
        (
            "fs.symlink",
            json!({"root": "r", "link": "l", "target": at(1025)}),
            "target is longer than 1024",
        ),
        (
            "fs.put_small",
            json!({"root": "r", "rel": "a\u{0}b", "content_b64": ""}),
            "rel contains a NUL",
        ),
        (
            "fs.delete",
            json!({"root": "r", "rel": at(2000)}),
            "rel is longer than 1024",
        ),
        (
            "models.config",
            json!({"root": "r", "rel": at(1025)}),
            "rel is longer than 1024",
        ),
        (
            "docker.rm",
            json!({"name": at(129)}),
            "name is longer than 128",
        ),
        (
            "docker.logs",
            json!({"name": "ec-x\u{0}"}),
            "name contains a NUL",
        ),
        (
            "docker.image_present",
            json!({"image": at(257)}),
            "image is longer than 256",
        ),
        (
            "scripts.run",
            json!({"name": at(33), "action": "up"}),
            "name is longer than 32",
        ),
        (
            "gateway.start",
            json!({"name": at(129)}),
            "name is longer than 128",
        ),
        (
            "docker.run",
            json!({"spec": {"name": at(65)}}),
            "spec.name is longer than 64",
        ),
        (
            "docker.run",
            json!({"spec": {"image": at(257)}}),
            "spec.image is longer than 256",
        ),
        (
            "docker.run",
            json!({"spec": {"argv": vec!["x"; 513]}}),
            "spec.argv has more than 512 items",
        ),
        (
            "docker.run",
            json!({"spec": {"argv": ["x", at(4097)]}}),
            "spec.argv[1] is longer than 4096",
        ),
        (
            "docker.run",
            json!({"spec": {"argv": ["x\u{0}"]}}),
            "spec.argv[0] contains a NUL",
        ),
        (
            "docker.render",
            json!({"spec": {"mounts": [{}, {"rel": at(1025)}]}}),
            "spec.mounts[1].rel is longer",
        ),
        (
            "docker.render",
            json!({"spec": {"volumes": [{"name": at(65)}]}}),
            "spec.volumes[0].name is longer",
        ),
    ] {
        let w = why(op, args.clone()).unwrap_or_else(|| panic!("{op} {args} passed"));
        assert!(w.starts_with(&format!("{op}: {frag}")), "{w}");
    }
    // other fields keep Python's cleaning; wrong types are left to clean (null) and the op's own checks
    for (op, args) in [
        (
            "http.request",
            json!({"port": 1, "method": "GET", "path": format!("/v1/{}", at(3000))}),
        ),
        (
            "fs.stat",
            json!({"root": "r", "paths": vec![at(2000); 300]}),
        ),
        (
            "docker.run",
            json!({"spec": {"env": {"A": at(5000)}, "labels": {"x": at(300)}}}),
        ),
        (
            "docker.run",
            json!({"spec": {"name": 5, "argv": "not a list"}}),
        ),
        ("fs.write", json!("not an object")),
        ("no.such.op", json!({"rel": at(5000)})),
    ] {
        assert_eq!(why(op, args.clone()), None, "{op} {args}");
    }
    // clean is unchanged: it still cuts (the parity vectors pin it against Python)
    let c = clean_args(
        "fs.write",
        &json!({"root": "r", "rel": at(1100), "size": 1}),
    )
    .unwrap();
    assert_eq!(c["rel"], Value::String(at(1024)));
}
