//! node/tests/test_gguf.py: the bounded GGUF header reader: real-shaped headers, every truncation, hostile counts
//! and lengths.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::cast_possible_truncation,
    unreachable_pub
)]

use std::time::{Duration, Instant};

use ecnode::gguf::{self, MAX_HEADER_BYTES};
use proptest::prelude::*;
use serde_json::{Value, json};

const U32: u32 = 4;
const STR: u32 = 8;
const ARR: u32 = 9;
const U64: u32 = 10;
const F32: u32 = 6;
const BOOL: u32 = 7;

fn s(text: &str) -> Vec<u8> {
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text.as_bytes());
    out
}

fn kv(key: &str, vtype: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = s(key);
    out.extend(vtype.to_le_bytes());
    out.extend(payload);
    out
}

fn u32b(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

/// test_gguf.build
fn build(
    arch: &str,
    tensors: Option<Vec<(&str, Vec<u64>)>>,
    extra: Vec<Vec<u8>>,
    version: u32,
    file_type: u32,
) -> Vec<u8> {
    let tensors = tensors.unwrap_or_else(|| {
        vec![
            ("tok_embd", vec![4096, 32000]),
            ("blk.0.attn_q", vec![4096, 4096]),
        ]
    });
    let mut arr_str = STR.to_le_bytes().to_vec();
    arr_str.extend(3u64.to_le_bytes());
    for t in ["a", "b", "c"] {
        arr_str.extend(s(t));
    }
    let mut arr_f32 = F32.to_le_bytes().to_vec();
    arr_f32.extend(3u64.to_le_bytes());
    for f in [0.0f32, 1.0, 2.0] {
        arr_f32.extend(f.to_le_bytes());
    }
    let mut kvs = vec![
        kv("general.architecture", STR, &s(arch)),
        kv("general.name", STR, &s("Test Model 8B")),
        kv("general.file_type", U32, &u32b(file_type)),
        kv(&format!("{arch}.context_length"), U32, &u32b(131_072)),
        kv(&format!("{arch}.block_count"), U32, &u32b(32)),
        kv(&format!("{arch}.embedding_length"), U32, &u32b(4096)),
        kv(&format!("{arch}.attention.head_count"), U32, &u32b(32)),
        kv(&format!("{arch}.attention.head_count_kv"), U32, &u32b(8)),
        kv("tokenizer.ggml.tokens", ARR, &arr_str),
        kv("tokenizer.ggml.scores", ARR, &arr_f32),
        kv("general.flag", BOOL, b"\x01"),
    ];
    kvs.extend(extra);
    let mut head = b"GGUF".to_vec();
    head.extend(version.to_le_bytes());
    head.extend((tensors.len() as u64).to_le_bytes());
    head.extend((kvs.len() as u64).to_le_bytes());
    for k in &kvs {
        head.extend(k);
    }
    for (name, dims) in &tensors {
        head.extend(s(name));
        head.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            head.extend(d.to_le_bytes());
        }
        head.extend(12u32.to_le_bytes());
        head.extend(0u64.to_le_bytes());
    }
    head
}

fn default_build() -> Vec<u8> {
    build("llama", None, Vec::new(), 3, 15)
}

fn read(blob: &[u8]) -> Option<ec_proto::Obj> {
    gguf::read_header(&mut &blob[..], MAX_HEADER_BYTES, true)
}

#[test]
fn test_reads_the_facts_a_library_needs() {
    let h = read(&default_build()).unwrap();
    assert_eq!(h["complete"], json!(true));
    assert_eq!(
        (
            &h["arch"],
            &h["name"],
            &h["quant"],
            &h["context"],
            &h["block_count"]
        ),
        (
            &json!("llama"),
            &json!("Test Model 8B"),
            &json!("Q4_K_M"),
            &json!(131_072),
            &json!(32)
        )
    );
    assert_eq!(
        (
            &h["head_count"],
            &h["head_count_kv"],
            &h["embedding_length"]
        ),
        (&json!(32), &json!(8), &json!(4096))
    );
    assert_eq!(h["params"], json!(4096 * 32000 + 4096 * 4096));
    assert_eq!(h["tensors"], json!(2));
}

#[test]
fn test_moe_keys_and_other_versions() {
    let extra = vec![
        kv("qwen3moe.expert_count", U32, &u32b(128)),
        kv("qwen3moe.expert_used_count", U32, &u32b(8)),
    ];
    let h = read(&build("qwen3moe", None, extra, 3, 7)).unwrap();
    assert_eq!(
        (&h["expert_count"], &h["expert_used_count"], &h["quant"]),
        (&json!(128), &json!(8), &json!("Q8_0"))
    );
    let mut nope = b"NOPE".to_vec();
    nope.extend([0u8; 40]);
    assert_eq!(read(&nope), None);
    let mut v99 = b"GGUF".to_vec();
    v99.extend(99u32.to_le_bytes());
    v99.extend([0u8; 40]);
    assert_eq!(
        Value::Object(read(&v99).unwrap()),
        json!({"version": 99, "complete": false})
    );
}

#[test]
fn test_every_truncation_is_handled_without_raising() {
    let blob = default_build();
    for n in 0..blob.len() {
        if let Some(h) = read(&blob[..n]) {
            assert!(h["complete"] == json!(false) || n == blob.len(), "{n}");
        }
    }
}

fn hostile_headers() -> Vec<Vec<u8>> {
    let huge: u64 = 1 << 63;
    let mut base = b"GGUF".to_vec();
    base.extend(3u32.to_le_bytes());
    let qq = |a: u64, b: u64| {
        let mut v = a.to_le_bytes().to_vec();
        v.extend(b.to_le_bytes());
        v
    };
    let iq = |a: u32, b: u64| {
        let mut v = a.to_le_bytes().to_vec();
        v.extend(b.to_le_bytes());
        v
    };
    let cat = |parts: &[&[u8]]| -> Vec<u8> {
        let mut v = base.clone();
        for p in parts {
            v.extend_from_slice(p);
        }
        v
    };
    let k = s("k");
    let t = s("t");
    let mut eight_huge = Vec::new();
    for _ in 0..8 {
        eight_huge.extend(huge.to_le_bytes());
    }
    vec![
        cat(&[&qq(1, huge)]),                                   // absurd KV count
        cat(&[&qq(huge, 0)]),                                   // absurd tensor count
        cat(&[&qq(0, 1), &huge.to_le_bytes()]),                 // absurd key length
        cat(&[&qq(0, 1), &k, &u32b(STR), &huge.to_le_bytes()]), // string length
        cat(&[&qq(0, 1), &k, &u32b(ARR), &iq(U64, huge)]),
        cat(&[&qq(0, 1), &k, &u32b(ARR), &iq(STR, huge)]),
        cat(&[&qq(0, 1), &k, &u32b(ARR), &iq(ARR, huge)]),
        cat(&[&qq(0, 1), &k, &u32b(77)]), // unknown value type
        cat(&[&qq(1, 0), &t, &u32b(1 << 31), &[0u8; 64]]), // absurd n_dims
        cat(&[&qq(1, 0), &t, &u32b(8), &eight_huge, &iq(0, 0)]), // huge dims: params None
        // nested arrays deeper than the reader follows
        cat(&[
            &qq(0, 1),
            &k,
            &u32b(ARR),
            &iq(ARR, 1),
            &iq(ARR, 1),
            &iq(ARR, 1),
            &iq(ARR, 1),
            &iq(U32, 1),
            &[0u8; 4],
        ]),
    ]
}

/// Python also measures the peak allocation (tracemalloc, < 8 MiB). That cannot be measured here without
/// `unsafe` (a counting global allocator); the bound holds by construction: the reader allocates only what it
/// keeps (a key of at most MAX_KEY bytes, a string of at most MAX_KEEP_STR) and skips everything else in a
/// streaming copy. Each header must be answered at once, without a panic.
#[test]
fn test_hostile_headers_never_raise_or_allocate_much() {
    for (i, blob) in hostile_headers().iter().enumerate() {
        let t0 = Instant::now();
        let h = read(blob);
        assert!(
            t0.elapsed() < Duration::from_secs(1),
            "header {i} took too long"
        );
        if let Some(h) = &h {
            assert!(h.contains_key("version"), "{i}");
        }
        if i == 9 {
            let h = h.unwrap();
            assert_eq!((&h["params"], &h["complete"]), (&Value::Null, &json!(true)));
        }
    }
}

#[test]
fn test_random_bytes_after_the_magic() {
    // a deterministic generator (Python's random.Random(1234) stream is not reproducible here)
    let mut x: u64 = 1234;
    let mut next = || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (x >> 33) as u32
    };
    for _ in 0..300 {
        let mut blob = b"GGUF".to_vec();
        blob.extend((1 + next() % 3).to_le_bytes());
        let n = next() % 401;
        for _ in 0..n {
            blob.push((next() & 0xff) as u8);
        }
        let h = gguf::read_header(&mut blob.as_slice(), 1 << 20, true);
        assert!(h.is_none_or(|h| h.contains_key("version")));
    }
}

proptest! {
    #[test]
    fn test_random_bytes_never_panic(version in 0u32..5, body in proptest::collection::vec(any::<u8>(), 0..600)) {
        let mut blob = b"GGUF".to_vec();
        blob.extend(version.to_le_bytes());
        blob.extend(body);
        let h = gguf::read_header(&mut blob.as_slice(), 1 << 20, true);
        prop_assert!(h.is_none_or(|h| h.contains_key("version")));
    }
}

#[test]
fn test_budget_stops_a_header_bigger_than_allowed() {
    let big: Vec<Vec<u8>> = (0..100)
        .map(|i| kv(&format!("pad.{i}"), STR, &s(&"x".repeat(1000))))
        .collect();
    let h = gguf::read_header(
        &mut build("llama", None, big, 3, 15).as_slice(),
        20_000,
        true,
    )
    .unwrap();
    assert_eq!(h["complete"], json!(false));
    assert_eq!(h["arch"], json!("llama")); // what was read is kept
}

#[test]
fn test_version_1_uses_32_bit_counts() {
    let s1 = |text: &str| {
        let mut v = (text.len() as u32).to_le_bytes().to_vec();
        v.extend(text.as_bytes());
        v
    };
    let mut blob = b"GGUF".to_vec();
    for v in [1u32, 0, 2] {
        blob.extend(v.to_le_bytes());
    }
    blob.extend(s1("general.architecture"));
    blob.extend(STR.to_le_bytes());
    blob.extend(s1("gpt2"));
    blob.extend(s1("gpt2.context_length"));
    blob.extend(U32.to_le_bytes());
    blob.extend(1024u32.to_le_bytes());
    let h = read(&blob).unwrap();
    assert_eq!(
        (&h["arch"], &h["context"], &h["complete"]),
        (&json!("gpt2"), &json!(1024), &json!(true))
    );
}
