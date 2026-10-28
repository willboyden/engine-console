//! Port of backend/tests/fleet/test_link_streams.py: streaming requests over a real subprocess: events, DATA both
//! ways, credit (flow control), cancel, and limits.

#![allow(
    clippy::type_complexity,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::double_ended_iterator_last,
    clippy::assertions_on_constants,
    clippy::needless_pass_by_value,
    clippy::float_cmp
)]

#[path = "common/wsb_fake_node.rs"]
mod wsb_fake_node;

#[path = "common/wsb_harness.rs"]
mod wsb_harness;

use std::sync::Arc;
use std::time::{Duration, Instant};

use ec_hub::domain::fleet::{NodeClient, NodeStream, StreamItem};
use ec_hub::services::fleet::{CREDIT_BATCH, LinkTiming};
use serde_json::{Value, json};
use sha2::Digest as _;
use wsb_fake_node::pattern;
use wsb_harness::{Harness, MIB, RigOpts, event, fast, items, limited, obj, wait_until};

fake_node_entry!();

const WINDOW: usize = 8 * 1024 * 1024;
const MAX_DATA: usize = 1024 * 1024;

fn timing(edit: impl FnOnce(&mut LinkTiming)) -> LinkTiming {
    let mut t = fast();
    edit(&mut t);
    t
}

fn sha_of_pattern(length: usize, offset: usize) -> String {
    let mut h = sha2::Sha256::new();
    let mut off = 0;
    while off < length {
        let n = MIB.min(length - off);
        h.update(pattern(offset + off, n));
        off += n;
    }
    hex::encode(h.finalize())
}

fn last_total(rig: &wsb_harness::Rig, ev: &str, key: &str) -> i64 {
    rig.events(ev)
        .last()
        .and_then(|e| e[key].as_i64())
        .unwrap_or(0)
}

// ── events ─────────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_events_arrive_in_order_and_the_stream_ends_with_its_result() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "docker.logs_follow",
            Some(obj(json!({"name": "ec-a", "tail": 5}))),
        );
        let got = items(&mut stream).await;
        let evs: Vec<Value> = got
            .iter()
            .map(|i| Value::Object(event(i).cloned().expect("an event")))
            .collect();
        assert_eq!(
            evs,
            (1..=5)
                .map(|n| json!({"line": format!("line {n}")}))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"lines": 5})
        );
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"lines": 5})
        ); // idempotent
        assert!(items(&mut stream).await.is_empty()); // a second iteration ends at once
        assert_eq!(rig.link.inflight(), 0);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_refused_stream_ends_and_result_raises_the_nodes_reason() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "docker.pull",
            Some(obj(json!({"image": "vllm/vllm-openai:v9"}))),
        );
        // the event with only unknown fields was dropped
        let lines: Vec<String> = items(&mut stream)
            .await
            .iter()
            .filter_map(event)
            .map(|e| e["line"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(lines, ["layer 0", "layer 1", "layer 2"]);
        let e = stream.result().await.unwrap_err();
        assert_eq!(
            (e.code.as_str(), e.message.as_str()),
            ("docker_error", "pull failed (scripted)")
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_script_output_events_carry_stream_and_line() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "scripts.run",
            Some(obj(json!({"name": "big-model", "action": "up"}))),
        );
        let evs: Vec<Value> = items(&mut stream)
            .await
            .iter()
            .filter_map(event)
            .map(|e| Value::Object(e.clone()))
            .collect();
        assert_eq!(
            evs,
            [
                json!({"stream": "out", "line": "step 0"}),
                json!({"stream": "out", "line": "step 1"})
            ]
        );
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"rc": 0})
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_events_a_node_sends_for_an_op_that_has_none_are_dropped() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"pushes": []})).await;
        rig.up().await;
        let mut stream = rig
            .link
            .open("docker.stop", Some(obj(json!({"name": "ec-a"}))));
        assert!(items(&mut stream).await.is_empty());
        assert!(stream.result().await.unwrap().is_empty());
        h.stop().await;
    })
    .await;
}

/// Python patches the module's MAX_STREAM_EVENTS; here it is the link's `max_stream_events`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_the_event_queue_of_a_stream_nobody_reads_is_capped() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({}),
                RigOpts {
                    timing: timing(|t| t.max_stream_events = 50),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let mut stream = rig.link.open(
            "docker.logs_follow",
            Some(obj(json!({"name": "ec-a", "tail": 400}))),
        );
        let l = Arc::clone(&rig.link);
        // it answered; nobody has read a thing
        wait_until(move || l.inflight() == 0, 8.0, "the node to finish").await;
        let got = items(&mut stream).await;
        assert_eq!(got.len(), 50); // the rest were dropped, not buffered
        assert_eq!(
            Value::Object(event(&got[0]).cloned().unwrap()),
            json!({"line": "line 1"})
        );
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"lines": 400})
        );
        assert_eq!(rig.st().state, "up");
        h.stop().await;
    })
    .await;
}

// ── cancel ─────────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cancel_stops_a_stream_and_the_node_is_told() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"follow_delay_s": 0.01})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "docker.logs_follow",
            Some(obj(json!({"name": "ec-a", "tail": 0}))),
        ); // until cancelled
        let mut seen = 0;
        while stream.next().await.is_some() {
            seen += 1;
            if seen == 3 {
                stream.cancel().await;
            }
        }
        assert!(seen >= 3); // events up to the cancel were delivered
        assert_eq!(stream.result().await.unwrap_err().code, "cancelled");
        assert!(!rig.events("cancel").is_empty() && rig.link.inflight() == 0);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_abandoning_an_iteration_cancels_the_request() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"follow_delay_s": 0.01})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "docker.logs_follow",
            Some(obj(json!({"name": "ec-a", "tail": 0}))),
        );
        assert!(stream.next().await.is_some());
        drop(stream); // stop early, no cancel() call
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "cancel")
            },
            8.0,
            "the node to be told to stop",
        )
        .await;
        let l = Arc::clone(&rig.link);
        wait_until(move || l.inflight() == 0, 8.0, "the request to settle").await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_events_after_a_cancel_are_not_delivered() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"follow_delay_s": 0.005, "ignore_cancel": true}))
            .await;
        rig.up().await;
        let mut stream = rig
            .link
            .open_stream(
                "docker.logs_follow",
                Some(obj(json!({"name": "ec-a", "tail": 0}))),
            )
            .unwrap();
        let first = stream.next().await.unwrap();
        assert!(matches!(first, StreamItem::Event(_)));
        stream.cancel().await;
        let queued_at_cancel = stream.queued();
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "cancel")
            },
            8.0,
            "the cancel to reach the node",
        )
        .await;
        let max_n = |r: &wsb_harness::Rig| {
            r.events("follow")
                .iter()
                .filter_map(|e| e["n"].as_i64())
                .max()
                .unwrap_or(0)
        };
        let sent_at_cancel = max_n(&rig);
        tokio::time::sleep(Duration::from_millis(300)).await;
        // the node ignored the cancel and kept talking ...
        assert!(max_n(&rig) > sent_at_cancel + 20);
        // ... and the hub kept none of it
        assert_eq!(stream.queued(), queued_at_cancel);
        assert_eq!(rig.st().state, "up");
        h.stop().await;
    })
    .await;
}

// ── DATA and flow control: node -> hub ─────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_download_larger_than_the_window_is_flow_controlled() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"data_chunk": 512 * 1024})).await;
        rig.up().await;
        let length = 24 * MIB;
        let mut stream = rig.link.open(
            "fs.read",
            Some(obj(
                json!({"root": "hf", "rel": "model.safetensors", "length": length}),
            )),
        );

        // Nobody reads: the node uses its initial credit and then stalls at exactly that boundary.
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "stalled")
            },
            8.0,
            "the node to run out of credit",
        )
        .await;
        let stalled = rig.events("stalled");
        assert_eq!(stalled[0]["sent"].as_u64().unwrap() as usize, WINDOW);
        assert!(stalled[0]["credit"].as_i64().unwrap() < 512 * 1024);
        assert_eq!(last_total(&rig, "sent", "sent") as usize, WINDOW);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert_eq!(last_total(&rig, "sent", "sent") as usize, WINDOW); // still stalled: no credit has been granted

        // The stalled stream blocks nothing else: pings are answered, other requests complete, samples keep coming.
        let pings_before = rig.events("ping").len();
        let seq_before = rig
            .link
            .latest("sample")
            .and_then(|s| s["seq"].as_i64())
            .unwrap_or(0);
        let t0 = Instant::now();
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        assert!(t0.elapsed() < Duration::from_secs(1));
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(rig.events("ping").len() > pings_before);
        assert!(
            rig.link
                .latest("sample")
                .and_then(|s| s["seq"].as_i64())
                .unwrap_or(0)
                > seq_before
        );
        assert!(rig.st().state == "up" && rig.st().rtt_ms.is_some());

        // A slow consumer: every byte arrives, in order, intact, and credit goes back in batches as it reads.
        let mut got = sha2::Sha256::new();
        let mut total = 0;
        while let Some(item) = stream.next().await {
            let StreamItem::Data(d) = item else {
                panic!("only DATA expected")
            };
            got.update(&d);
            total += d.len();
            if total % (4 * MIB) == 0 {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        }
        assert_eq!(total, length);
        assert_eq!(hex::encode(got.finalize()), sha_of_pattern(length, 0));
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"size": length})
        );
        let credits: Vec<u64> = rig
            .events("credit")
            .iter()
            .filter_map(|c| c["bytes"].as_u64())
            .collect();
        assert!(!credits.is_empty() && credits.iter().all(|c| *c >= CREDIT_BATCH));
        assert!(credits.iter().sum::<u64>() <= length as u64);
        assert_eq!(rig.link.inflight(), 0);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_two_streams_flow_independently() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"data_chunk": 512 * 1024})).await;
        rig.up().await;
        let mut slow = rig.link.open(
            "fs.read",
            Some(obj(json!({"root": "hf", "rel": "a", "length": 16 * MIB}))),
        );
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "stalled")
            },
            8.0,
            "the first stream to stall",
        )
        .await;
        let mut fast = rig.link.open(
            "fs.read",
            Some(obj(
                json!({"root": "hf", "rel": "b", "length": 6 * MIB, "offset": 4096}),
            )),
        );
        let mut got = Vec::new();
        while let Some(item) = fast.next().await {
            // completes while `slow` is stuck
            if let StreamItem::Data(d) = item {
                got.extend_from_slice(&d);
            }
        }
        assert!(got == pattern(4096, 6 * MIB));
        assert_eq!(
            Value::Object(fast.result().await.unwrap()),
            json!({"size": 6 * MIB})
        );
        let mut total = 0;
        while let Some(item) = slow.next().await {
            if let StreamItem::Data(d) = item {
                total += d.len();
            }
        }
        assert_eq!(total, 16 * MIB);
        h.stop().await;
    })
    .await;
}

/// A model copy moves far more than 20 MB in 10 s; only JSON messages count toward the flood limit.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_download_data_is_not_a_flood() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig_with(
                json!({"data_chunk": MIB}),
                RigOpts {
                    timing: timing(|t| {
                        t.flood_bytes = MIB as u64;
                        t.flood_frames = 500;
                    }),
                    ..RigOpts::default()
                },
            )
            .await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.read",
            Some(obj(json!({"root": "hf", "rel": "big", "length": 40 * MIB}))),
        );
        let mut total = 0;
        while let Some(item) = stream.next().await {
            if let StreamItem::Data(d) = item {
                total += d.len();
            }
        }
        assert_eq!(total, 40 * MIB);
        assert_eq!(rig.st().state, "up");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_that_sends_more_than_its_credit_is_dropped() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"ignore_credit": true, "data_chunk": MIB}))
            .await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.read",
            Some(obj(json!({"root": "hf", "rel": "x", "length": 12 * MIB}))),
        );
        rig.state("down").await; // nobody read, so no credit was ever granted
        let err = rig.error();
        assert!(
            err.contains("more data") && err.contains("than the credit this hub granted"),
            "{err}"
        );
        let mut got = 0;
        while let Some(item) = stream.next().await {
            if let StreamItem::Data(d) = item {
                got += d.len();
            }
        }
        assert!(got <= WINDOW);
        assert_eq!(stream.result().await.unwrap_err().code, "node_unavailable");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_dying_mid_stream_fails_the_stream_and_the_link_recovers() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"crash_after_bytes": MIB, "data_chunk": 512 * 1024}))
            .await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.read",
            Some(obj(json!({"root": "hf", "rel": "x", "length": 4 * MIB}))),
        );
        let mut got = Vec::new();
        while let Some(item) = stream.next().await {
            if let StreamItem::Data(d) = item {
                got.extend_from_slice(&d);
            }
        }
        assert!(got == pattern(0, got.len()) && got.len() <= 4 * MIB); // what arrived is a clean prefix
        assert_eq!(stream.result().await.unwrap_err().code, "node_unavailable");
        rig.state("down").await;
        assert!(
            rig.error().contains("the node exited (7)"),
            "{}",
            rig.error()
        );
        rig.up().await; // a new node, same host
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        h.stop().await;
    })
    .await;
}

// ── DATA and flow control: hub -> node ─────────────────────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_an_upload_larger_than_the_window_waits_for_credit() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"upload_hold_s": 0.8, "credit_batch": MIB}))
            .await;
        rig.up().await;
        let size = 20 * MIB;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(
                json!({"root": "hf", "rel": "model.safetensors", "size": size}),
            )),
        );
        let writer = tokio::spawn(async move {
            let r = stream.write(&pattern(0, size)).await;
            (stream, r)
        });

        // The node does not read or grant anything yet: the hub stops at exactly the window it was given.
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "data")
                    .last()
                    .and_then(|e| e["total"].as_u64())
                    == Some(WINDOW as u64)
            },
            8.0,
            "the window to fill",
        )
        .await;
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(!writer.is_finished());
        assert_eq!(last_total(&rig, "data", "total") as usize, WINDOW);
        // exactly eight full frames: a hub that kept "sending" while out of credit would show up as a flood of empty ones
        let totals: Vec<u64> = rig
            .events("data")
            .iter()
            .filter_map(|d| d["total"].as_u64())
            .collect();
        assert_eq!(
            totals,
            (1..=WINDOW / MAX_DATA)
                .map(|i| (i * MAX_DATA) as u64)
                .collect::<Vec<_>>()
        );

        // Blocked on credit, the hub still answers: pings flow, other requests complete.
        let pings_before = rig.events("ping").len();
        assert!(
            !rig.call("docker.images", json!({})).await.unwrap()["images"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(rig.events("ping").len() > pings_before && rig.st().state == "up");

        // the node starts consuming and granting credit
        let (mut stream, r) = tokio::time::timeout(Duration::from_secs(15), writer)
            .await
            .unwrap()
            .unwrap();
        r.unwrap();
        stream.end().await.unwrap();
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"size": size, "sha256": sha_of_pattern(size, 0)})
        );
        assert!(rig.events("violation").is_empty());
        assert_eq!(
            rig.events("data")
                .iter()
                .filter_map(|d| d["total"].as_u64())
                .max(),
            Some(size as u64)
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_small_writes_are_sent_as_they_are_and_in_order() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": 3000}))),
        );
        for i in 0..3u8 {
            stream.write(&[i; 1000]).await.unwrap();
        }
        stream.write(b"").await.unwrap(); // an empty write sends nothing
        stream.end().await.unwrap();
        stream.end().await.unwrap(); // and end() twice is harmless
        let mut all = vec![0u8; 1000];
        all.extend([1u8; 1000]);
        all.extend([2u8; 1000]);
        let expected = hex::encode(sha2::Sha256::digest(&all));
        assert_eq!(
            Value::Object(stream.result().await.unwrap()),
            json!({"size": 3000, "sha256": expected})
        );
        assert_eq!(rig.events("end").len(), 1);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_single_huge_write_is_split_into_frames_of_at_most_one_mib() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"credit_batch": 256 * 1024})).await;
        rig.up().await;
        let size = 3 * MIB + 5;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": size}))),
        );
        stream.write(&vec![b'z'; size]).await.unwrap();
        stream.end().await.unwrap();
        assert_eq!(stream.result().await.unwrap()["size"], json!(size));
        let totals: Vec<u64> = rig
            .events("data")
            .iter()
            .filter_map(|d| d["total"].as_u64())
            .collect();
        let mut prev = 0;
        let mut sum = 0;
        for t in totals {
            let step = t - prev;
            assert!(step > 0 && step as usize <= MAX_DATA);
            sum += step;
            prev = t;
        }
        assert_eq!(sum as usize, size);
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_cancelling_an_upload_that_waits_for_credit_unblocks_the_writer() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"upload_hold_s": 30})).await;
        rig.up().await;
        let mut stream = rig
            .link
            .open_stream(
                "fs.write",
                Some(obj(json!({"root": "hf", "rel": "x", "size": 20 * MIB}))),
            )
            .unwrap();
        let canceller = stream.canceller();
        let writer = tokio::spawn(async move { stream.write(&vec![b'q'; 20 * MIB]).await });
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .filter(|e| e["ev"] == "data")
                    .last()
                    .and_then(|e| e["total"].as_u64())
                    == Some(WINDOW as u64)
            },
            8.0,
            "the window to fill",
        )
        .await;
        canceller.cancel();
        let e = tokio::time::timeout(Duration::from_secs(5), writer)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err();
        assert_eq!(e.code, "cancelled");
        let l = Arc::clone(&rig.link);
        wait_until(move || l.inflight() == 0, 8.0, "the request to settle").await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_node_that_refuses_an_upload_early_fails_the_writer_with_its_reason() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig
            .link
            .open("docker.stop", Some(obj(json!({"name": "ec-a"})))); // answers at once, never wants DATA
        stream.result().await.unwrap();
        assert_eq!(
            stream.write(b"too late").await.unwrap_err().code,
            "conflict"
        );
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_write_after_end_is_a_bad_args_error() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"upload_hold_s": 5})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": 1}))),
        );
        stream.write(b"x").await.unwrap();
        stream.end().await.unwrap();
        assert_eq!(stream.write(b"y").await.unwrap_err().code, "bad_args");
        stream.cancel().await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_link_drop_fails_a_writer_waiting_for_credit() {
    limited(async {
        let mut h = Harness::new();
        let rig = h
            .rig(json!({"upload_hold_s": 30, "exit_after_s": 0.6}))
            .await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": 20 * MIB}))),
        );
        let e = tokio::time::timeout(Duration::from_secs(10), stream.write(&vec![b'q'; 20 * MIB]))
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(e.code, "node_unavailable");
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_dropping_an_unfinished_upload_cancels_it_on_the_node() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"upload_hold_s": 30})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": 100}))),
        );
        stream.write(b"abc").await.unwrap();
        assert_eq!(rig.link.inflight(), 1);
        drop(stream);
        let path = rig.log_path.clone();
        wait_until(
            move || {
                wsb_fake_node::read_log(&path)
                    .iter()
                    .any(|e| e["ev"] == "cancel")
            },
            8.0,
            "the abandoned upload to be cancelled",
        )
        .await;
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_a_finished_upload_that_is_dropped_before_its_answer_is_left_alone() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({"upload_hold_s": 0.5})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "fs.write",
            Some(obj(json!({"root": "hf", "rel": "x", "size": 3}))),
        );
        stream.write(b"abc").await.unwrap();
        stream.end().await.unwrap();
        drop(stream);
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(rig.events("cancel").is_empty()); // end() was sent: the request is still wanted
        let l = Arc::clone(&rig.link);
        wait_until(move || l.inflight() == 0, 8.0, "the node to answer").await;
        h.stop().await;
    })
    .await;
}

// ── http.request: events, then DATA, then the result ───────────────────────────────────────────────────
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_http_request_with_a_body_streams_the_response() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream: Box<dyn NodeStream> = rig.link.open(
            "http.request",
            Some(obj(json!({"port": 18000, "method": "POST", "path": "/v1/chat/completions", "body_len": 11,
                           "headers": {"content-type": "application/json"}}))),
        );
        stream.write(b"hello ").await.unwrap();
        stream.write(b"world").await.unwrap();
        stream.end().await.unwrap();
        let got = items(&mut stream).await;
        assert_eq!(event(&got[0]).map(|e| Value::Object(e.clone())), Some(json!({"status": 200, "headers": {"content-type": "text/plain"}})));
        let body: Vec<u8> = got[1..].iter().filter_map(|i| match i { StreamItem::Data(d) => Some(d.to_vec()), StreamItem::Event(_) => None }).flatten().collect();
        assert_eq!(body, b"engine answer, request body was 11 bytes");
        assert_eq!(Value::Object(stream.result().await.unwrap()), json!({"status": 200, "bytes": body.len()}));
        h.stop().await;
    })
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_http_request_without_a_body() {
    limited(async {
        let mut h = Harness::new();
        let rig = h.rig(json!({})).await;
        rig.up().await;
        let mut stream = rig.link.open(
            "http.request",
            Some(obj(
                json!({"port": 18000, "method": "GET", "path": "/health"}),
            )),
        );
        let got = items(&mut stream).await;
        assert_eq!(event(&got[0]).unwrap()["status"], 200);
        let body: Vec<u8> = got[1..]
            .iter()
            .filter_map(|i| match i {
                StreamItem::Data(d) => Some(d.to_vec()),
                StreamItem::Event(_) => None,
            })
            .flatten()
            .collect();
        assert_eq!(body, b"engine answer, request body was 0 bytes");
        h.stop().await;
    })
    .await;
}
