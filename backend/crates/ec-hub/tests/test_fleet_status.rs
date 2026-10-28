//! Port of backend/tests/fleet/test_fleet_status.py: the pure builders: node messages + link state -> HostSummary /
//! HostDetail, labels, and storage shared between hosts.

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

use std::sync::{Arc, Mutex};

use ec_hub::domain::fleet::{
    CountsProvider, HostCounts, HostDetail, HostSummary, ImageInfo, InstanceResolver,
};
use ec_hub::services::fleet_status::{
    self as fs, HostData, addresses_of, build_detail, build_summary, images_from_result,
    mount_label,
};
use serde_json::{Value, json};
use wsb_harness::{DataOpts, GIB, b_hello, data, disks, hello, mount, obj, root, sample};

fake_node_entry!();

const G: f64 = GIB as f64;

fn summary(d: &HostData) -> HostSummary {
    build_summary(d, None, None)
}

fn detail(id: &str, fleet: &[HostData]) -> HostDetail {
    build_detail(id, fleet, Vec::new(), None, None).unwrap()
}

fn d(h: Option<Value>, s: Option<Value>) -> DataOpts {
    DataOpts {
        h: h.map(obj),
        s: s.map(obj),
        ..DataOpts::default()
    }
}

fn hv(o: ec_proto::Obj) -> Option<Value> {
    Some(Value::Object(o))
}

// ── summary ────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_a_live_summary_converts_bytes_to_gib_and_orders_gpus_by_index() {
    let s = summary(&data(
        "gpu-box-2",
        DataOpts {
            skew_s: Some(2.0),
            ..d(hv(hello()), hv(sample(json!({}))))
        },
    ));
    assert!(s.id == "gpu-box-2" && s.label == "Gpu-Box-2" && s.remote && s.link.state == "up");
    let (a, b) = (&s.gpus[0], &s.gpus[1]);
    assert_eq!(
        (a.index, a.uuid.as_str(), a.name.as_str()),
        (0, "GPU-A", "RTX A")
    );
    assert_eq!((b.index, b.uuid.as_str()), (1, "GPU-B"));
    assert_eq!(
        (a.total_gib, a.used_gib, a.free_gib),
        (48.0, Some(40.0), Some(8.0))
    );
    assert_eq!(
        (a.util_pct, a.temp_c, a.power_w, a.power_limit_w, a.fan_pct),
        (Some(80.0), Some(61.0), Some(250.0), Some(300.0), Some(44.0))
    );
    assert!(a.compute_capability.as_deref() == Some("8.9") && !a.unified_memory);
    assert_eq!(
        (b.total_gib, b.used_gib, b.free_gib, b.power_w),
        (96.0, Some(1.0), Some(95.0), None)
    ); // absent stays None
    let m = s.memory.as_ref().unwrap();
    assert_eq!(
        (m.total_gib, m.available_gib, m.used_gib, m.free_gib),
        (256.0, 200.0, 56.0, 150.0)
    );
    assert_eq!(
        (m.cached_gib, m.shmem_gib, m.swap_total_gib, m.swap_used_gib),
        (40.0, 1.0, 8.0, 0.0)
    );
    assert_eq!(m.updated_at, 998.0); // the node's clock minus its measured skew: the hub's clock
    let c = s.cpu.as_ref().unwrap();
    assert_eq!((c.load1, c.util_pct), (Some(1.5), Some(12.0)));
    assert!(s.alerts.is_empty());
}

#[test]
fn test_gpu_processes_name_the_instance_behind_each_container() {
    let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::default();
    let c2 = Arc::clone(&calls);
    let resolver: InstanceResolver = Arc::new(move |host: &str, container: &str| {
        c2.lock().unwrap().push((host.into(), container.into()));
        (container == "ec-engine-1").then(|| "inst_abc".to_string())
    });
    let s = build_summary(
        &data("gpu-box-2", d(hv(hello()), hv(sample(json!({}))))),
        Some(&resolver),
        None,
    );
    let procs = &s.gpus[0].procs;
    let p = |pid: i64| procs.iter().find(|p| p.pid == pid).unwrap();
    assert!(
        p(11).instance_id.as_deref() == Some("inst_abc")
            && p(11).container.as_deref() == Some("ec-engine-1")
    );
    assert_eq!(p(11).used_gib, Some(30.0));
    assert!(p(12).instance_id.is_none() && p(13).container.is_none() && p(13).used_gib.is_none());
    assert_eq!(p(11).name.as_deref(), Some("python3"));
    let calls = calls.lock().unwrap();
    assert!(calls.contains(&("gpu-box-2".into(), "ec-engine-1".into())));
    assert!(calls.iter().all(|c| !c.1.is_empty())); // never asked about "no container"
}

#[test]
fn test_a_broken_resolver_or_counts_provider_does_not_take_the_summary_down() {
    let resolver: InstanceResolver = Arc::new(|_: &str, _: &str| panic!("bug in another service"));
    let counts: CountsProvider = Arc::new(|_: &str| panic!("bug in another service"));
    let s = build_summary(
        &data("gpu-box-2", d(hv(hello()), hv(sample(json!({}))))),
        Some(&resolver),
        Some(&counts),
    );
    assert!(s.gpus[0].procs[0].instance_id.is_none() && s.counts == HostCounts::default());
}

#[test]
fn test_counts_come_from_the_provider() {
    let counts: CountsProvider = Arc::new(|_: &str| HostCounts {
        engines_running: 2,
        engines_total: 3,
        models: 7,
        models_bytes: 9,
    });
    let s = build_summary(
        &data("gpu-box-2", d(hv(hello()), None)),
        None,
        Some(&counts),
    );
    let c = &s.counts;
    assert_eq!(
        (c.engines_running, c.engines_total, c.models, c.models_bytes),
        (2, 3, 7, 9)
    );
}

#[test]
fn test_node_docker_and_policy_facts() {
    let h = b_hello(
        &[],
        Vec::new(),
        "gpu-box-2",
        json!({"mode": "manage", "allow_pull": true, "external_control": "stop_start"}),
    );
    let s = summary(&data("gpu-box-2", d(hv(h), hv(sample(json!({}))))));
    let n = s.node.as_ref().unwrap();
    assert_eq!(
        (
            n.version.as_deref(),
            n.hostname.as_deref(),
            n.arch.as_deref(),
            n.os.as_deref(),
            n.kernel.as_deref()
        ),
        (
            Some("0.4.0"),
            Some("gpu-box-2"),
            Some("aarch64"),
            Some("Ubuntu 24.04"),
            Some("6.8")
        )
    );
    assert_eq!(
        (
            n.cpu_model.as_deref(),
            n.cpu_cores,
            n.python.as_deref(),
            n.driver.as_deref(),
            n.cuda.as_deref()
        ),
        (
            Some("Grace"),
            Some(20),
            Some("3.12.3"),
            Some("570.1"),
            Some("12.8")
        )
    );
    assert!(
        n.gpu_source.as_deref() == Some("nvml")
            && n.forced_cmd == Some(true)
            && n.via_ssh == Some(true)
    );
    let dk = s.docker.as_ref().unwrap();
    assert_eq!(
        (
            dk.available,
            dk.context.as_deref(),
            dk.rootless,
            dk.nvidia_runtime
        ),
        (true, Some("rootless"), Some(true), Some(true))
    );
    let p = s.policy.as_ref().unwrap();
    assert!(p.mode == "manage" && p.allow_pull && p.external_control == "stop_start");
    assert!(
        p.port_range == [18000, 18099]
            && p.engine_images == ["vllm/vllm-openai:*"]
            && p.max_shm_gib == Some(64.0)
    );
}

#[test]
fn test_policy_values_the_hub_does_not_recognise_fail_safe() {
    for (mode, ext, ipc) in [
        (json!("admin"), json!("everything"), Value::Null),
        (Value::Null, Value::Null, Value::Null),
        (json!("MANAGE"), json!("STOP_START"), json!(false)),
    ] {
        let h = b_hello(
            &[],
            Vec::new(),
            "gpu-box-2",
            json!({"mode": mode, "external_control": ext, "allow_ipc_host": ipc}),
        );
        let s = summary(&data("gpu-box-2", d(hv(h), None)));
        let p = s.policy.unwrap();
        // nothing may change on that machine
        assert!(p.mode == "monitor" && p.external_control == "none");
        assert_eq!(p.allow_ipc_host, ipc.as_bool().unwrap_or(true));
    }
}

#[test]
fn test_a_policy_error_and_a_bad_port_range_are_passed_on_safely() {
    let h = b_hello(
        &[],
        Vec::new(),
        "gpu-box-2",
        json!({"error": "policy.json is writable by group", "port_range": [18000]}),
    );
    let p = summary(&data("gpu-box-2", d(hv(h), None))).policy.unwrap();
    assert!(
        p.error.as_deref() == Some("policy.json is writable by group") && p.port_range.is_empty()
    );
}

#[test]
fn test_a_down_host_keeps_its_static_facts_and_shows_nothing_live() {
    let s = summary(&data(
        "gpu-box-2",
        DataOpts {
            state: "down".into(),
            error: Some("the node exited (3)".into()),
            ..d(hv(hello()), hv(sample(json!({}))))
        },
    ));
    assert_eq!(
        s.gpus.iter().map(|g| g.uuid.as_str()).collect::<Vec<_>>(),
        ["GPU-A", "GPU-B"]
    );
    assert_eq!(s.gpus[0].total_gib, 48.0);
    assert!(
        s.gpus[0].used_gib.is_none() && s.gpus[0].util_pct.is_none() && s.gpus[0].procs.is_empty()
    );
    assert!(s.memory.is_none() && s.cpu.is_none() && s.alerts.is_empty());
    assert_eq!(s.node.unwrap().version.as_deref(), Some("0.4.0"));
    assert_eq!(s.link.error.as_deref(), Some("the node exited (3)"));
}

#[test]
fn test_a_stale_host_still_shows_its_last_sample() {
    let s = summary(&data(
        "gpu-box-2",
        DataOpts {
            state: "stale".into(),
            ..d(hv(hello()), hv(sample(json!({}))))
        },
    ));
    assert!(s.gpus[0].used_gib == Some(40.0) && s.memory.is_some());
}

#[test]
fn test_a_host_that_never_connected_has_nothing_but_its_name_and_link() {
    let s = summary(&data(
        "gpu-box-2",
        DataOpts {
            state: "connecting".into(),
            ..DataOpts::default()
        },
    ));
    assert!(
        s.node.is_none()
            && s.docker.is_none()
            && s.policy.is_none()
            && s.gpus.is_empty()
            && s.memory.is_none()
    );
    assert!(
        s.id == "gpu-box-2" && s.link.state == "connecting" && s.counts == HostCounts::default()
    );
}

#[test]
fn test_the_local_host_is_not_remote() {
    assert!(!summary(&data("local", DataOpts::default())).remote);
}

#[test]
fn test_unified_memory_gpus_report_no_per_gpu_usage() {
    let mut h = hello();
    for g in h.get_mut("gpus").unwrap().as_array_mut().unwrap() {
        if g["uuid"] == "GPU-A" {
            g["unified_memory"] = json!(true);
        }
    }
    let s = summary(&data(
        "gpu-box-2",
        d(
            hv(h),
            hv(sample(
                json!({"gpus": {"GPU-A": {"util_pct": 5.0}, "GPU-B": {}}}),
            )),
        ),
    ));
    let g = s.gpus.iter().find(|g| g.uuid == "GPU-A").unwrap();
    assert!(
        g.unified_memory && g.used_gib.is_none() && g.free_gib.is_none() && g.util_pct == Some(5.0)
    );
}

#[test]
fn test_free_memory_is_never_negative_even_when_a_driver_reports_more_used_than_total() {
    let s = summary(&data(
        "gpu-box-2",
        d(
            hv(hello()),
            hv(sample(
                json!({"gpus": {"GPU-A": {"mem_used_bytes": 50 * GIB}, "GPU-B": {}}}),
            )),
        ),
    ));
    let g = s.gpus.iter().find(|g| g.uuid == "GPU-A").unwrap();
    assert_eq!(
        (g.total_gib, g.used_gib, g.free_gib),
        (48.0, Some(50.0), Some(0.0))
    );
}

#[test]
fn test_a_gpu_without_a_uuid_or_unknown_to_the_hello_is_not_listed() {
    let mut h = hello();
    h.get_mut("gpus")
        .unwrap()
        .as_array_mut()
        .unwrap()
        .push(json!({"index": 5, "name": "no uuid"}));
    let s = summary(&data(
        "gpu-box-2",
        d(
            hv(h),
            hv(sample(
                json!({"gpus": {"GPU-A": {"util_pct": 1.0}, "GPU-ROGUE": {"util_pct": 99.0}}}),
            )),
        ),
    ));
    assert_eq!(
        s.gpus.iter().map(|g| g.uuid.as_str()).collect::<Vec<_>>(),
        ["GPU-A", "GPU-B"]
    );
}

#[test]
fn test_memory_alerts_use_the_same_rules_as_the_local_report() {
    for (avail, total, code, level) in [
        (10, 256, "host_ram_low", "crit"),
        (20, 256, "host_ram_low", "warn"),
    ] {
        let s = summary(&data(
            "gpu-box-2",
            d(
                hv(hello()),
                hv(sample(
                    json!({"mem": {"total": total * GIB, "available": avail * GIB}}),
                )),
            ),
        ));
        let got: Vec<(&str, &str)> = s
            .alerts
            .iter()
            .map(|a| (a.code.as_str(), a.level.as_str()))
            .collect();
        assert_eq!(got, [(code, level)]);
    }
}

#[test]
fn test_shared_memory_and_swap_alerts() {
    let mem = json!({"total": 100 * GIB, "available": 60 * GIB, "shmem": 40 * GIB, "swap_total": 10 * GIB, "swap_free": 2 * GIB});
    let s = summary(&data(
        "gpu-box-2",
        d(hv(hello()), hv(sample(json!({"mem": mem})))),
    ));
    let mut codes: Vec<&str> = s.alerts.iter().map(|a| a.code.as_str()).collect();
    codes.sort_unstable();
    assert_eq!(codes, ["shmem_high", "swap_heavy"]);
}

#[test]
fn test_a_sample_without_memory_numbers_gives_no_memory_and_no_alerts() {
    let s = summary(&data(
        "gpu-box-2",
        d(
            hv(hello()),
            hv(sample(json!({"mem": {"total": null, "available": 5}}))),
        ),
    ));
    assert!(s.memory.is_none() && s.alerts.is_empty());
}

#[test]
fn test_hostile_garbage_never_raises() {
    let nasty_hello = ec_proto::sanitize_push(&obj(json!({"type": "hello", "proto": 1, "host": "text", "gpu": [],
        "gpus": [null, 5, {"uuid": 3}, {"uuid": "g"}], "docker": null, "addresses": [null, {"addr": 5}],
        "policy": {"model_roots": [{"id": null}, 7], "scripts": [{"name": null}, {"name": "ok", "actions": [null, {"name": 5}]}]}})))
    .unwrap()
    .unwrap();
    let nasty_sample = ec_proto::sanitize_push(&obj(json!({"type": "sample", "gpus": {"g": {"procs": [null, {"pid": "x"}, {"pid": 1, "used_bytes": "big"}]}},
        "mem": "lots", "cpu": [1]})))
    .unwrap()
    .unwrap();
    let dd = data(
        "gpu-box-2",
        DataOpts {
            h: Some(nasty_hello),
            s: Some(nasty_sample),
            d: Some(obj(json!({"mounts": [null, {"mount": 5}], "t": null}))),
            inv: Some(obj(json!({"scripts": [null, 1]}))),
            ..DataOpts::default()
        },
    );
    let s = summary(&dd);
    let det = detail("gpu-box-2", &[dd]);
    assert!(!s.gpus.is_empty() && s.gpus[0].uuid == "g");
    assert!(det.scripts[0].name == "ok" && det.scripts[0].actions.is_empty());
}

// ── labels ─────────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_mount_labels() {
    type Case<'a> = (
        &'a str,
        Option<&'a str>,
        Option<&'a str>,
        Option<&'a str>,
        bool,
        Option<&'a str>,
        Option<&'a str>,
        &'a str,
    );
    let cases: Vec<Case<'_>> = vec![
        (
            "/data/fast",
            Some("zfs"),
            Some("tank/models"),
            None,
            false,
            None,
            None,
            "tank (zfs)",
        ),
        (
            "/data/fast",
            Some("zfs"),
            Some("tank"),
            None,
            false,
            None,
            None,
            "tank (zfs)",
        ),
        (
            "/data/fast",
            Some("zfs"),
            Some("tank@snap"),
            None,
            false,
            None,
            None,
            "tank (zfs)",
        ),
        (
            "/data/fast",
            Some("zfs"),
            None,
            None,
            false,
            None,
            None,
            "/data/fast (zfs)",
        ),
        (
            "/",
            Some("ext4"),
            None,
            Some("Samsung 990"),
            false,
            None,
            None,
            "/ (ext4, Samsung 990)",
        ),
        (
            "/data",
            Some("xfs"),
            None,
            None,
            false,
            None,
            None,
            "/data (xfs)",
        ),
        (
            "/data",
            None,
            None,
            Some("Some Disk"),
            false,
            None,
            None,
            "/data (Some Disk)",
        ),
        ("/data", None, None, None, false, None, None, "/data"),
        (
            "/mnt/m",
            Some("nfs4"),
            None,
            None,
            true,
            Some("10.0.0.9"),
            None,
            "NFS from 10.0.0.9",
        ),
        (
            "/mnt/m",
            Some("nfs"),
            None,
            None,
            true,
            Some("10.0.0.9"),
            Some("GPU box 2"),
            "NFS from GPU box 2",
        ),
        (
            "/mnt/m",
            Some("cifs"),
            None,
            None,
            true,
            Some("nas"),
            None,
            "CIFS from nas",
        ),
        (
            "/mnt/m",
            Some("fuse.sshfs"),
            None,
            None,
            true,
            Some("box"),
            None,
            "FUSE.SSHFS from box",
        ),
        (
            "/mnt/m",
            None,
            Some("nas:/export"),
            None,
            true,
            None,
            None,
            "Network from nas:/export",
        ),
    ];
    for (m, ft, src, model, remote, server, label, want) in cases {
        assert_eq!(mount_label(m, ft, src, model, remote, server, label), want);
    }
}

// ── detail: disks, roots, shared storage ───────────────────────────────────────────────────────────────
fn hello_roots(roots: Vec<Value>) -> ec_proto::Obj {
    b_hello(&[], roots, "gpu-box-2", json!({}))
}

#[test]
fn test_disks_become_mountinfo_with_human_labels() {
    let h = hello_roots(vec![root("hf", "/data/hf", json!({}))]);
    let dk = disks(vec![
        mount(
            "/",
            json!({"model": "Samsung 990", "device": "nvme0n1", "source": "/dev/nvme0n1p2"}),
        ),
        mount(
            "/data",
            json!({"fstype": "zfs", "source": "tank/data", "readonly": true}),
        ),
        mount(
            "/mnt/n",
            json!({"fstype": "nfs4", "source": "10.0.0.9:/export", "remote": true, "remote_host": "10.0.0.9", "remote_path": "/export"}),
        ),
    ]);
    let det = detail(
        "gpu-box-2",
        &[data(
            "gpu-box-2",
            DataOpts {
                h: Some(h),
                d: Some(dk),
                ..DataOpts::default()
            },
        )],
    );
    let (r, dm, n) = (&det.disks[0], &det.disks[1], &det.disks[2]);
    assert!(
        r.label == "/ (ext4, Samsung 990)"
            && r.model.as_deref() == Some("Samsung 990")
            && r.device.as_deref() == Some("nvme0n1")
    );
    assert_eq!(
        (r.size_bytes, r.used_bytes, r.free_bytes, r.readonly),
        (Some(100 * GIB), Some(40 * GIB), Some(60 * GIB), false)
    );
    assert!(dm.label == "tank (zfs)" && dm.readonly);
    assert!(
        n.label == "NFS from 10.0.0.9"
            && n.remote
            && n.remote_path.as_deref() == Some("/export")
            && n.remote_host_id.is_none()
    );
    // the node's order is kept
    assert_eq!(
        det.disks
            .iter()
            .map(|m| m.mount.as_str())
            .collect::<Vec<_>>(),
        ["/", "/data", "/mnt/n"]
    );
}

#[test]
fn test_a_root_lives_on_the_mount_with_the_longest_matching_path() {
    let h = hello_roots(vec![
        root("a", "/data/models/hf", json!({})),
        root("b", "/data2/x", json!({})),
        root("c", "/other", json!({})),
        root("d", "/data/models", json!({})),
    ]);
    let dk = disks(vec![
        mount("/", json!({})),
        mount("/data", json!({})),
        mount("/data/models", json!({})),
    ]);
    let det = detail(
        "gpu-box-2",
        &[data(
            "gpu-box-2",
            DataOpts {
                h: Some(h),
                d: Some(dk),
                ..DataOpts::default()
            },
        )],
    );
    let by: Vec<(&str, &str)> = det
        .roots
        .iter()
        .map(|r| {
            (
                r.id.as_str(),
                r.mount.as_ref().map_or("", |m| m.mount.as_str()),
            )
        })
        .collect();
    // "/data2/x" is NOT under "/data"
    assert_eq!(
        by,
        [
            ("a", "/data/models"),
            ("b", "/"),
            ("c", "/"),
            ("d", "/data/models")
        ]
    );
    let mroots: Vec<(&str, Vec<String>)> = det
        .disks
        .iter()
        .map(|m| (m.mount.as_str(), m.roots.clone()))
        .collect();
    assert_eq!(
        mroots,
        [
            ("/", vec!["b".to_string(), "c".into()]),
            ("/data", vec![]),
            ("/data/models", vec!["a".into(), "d".into()])
        ]
    );
    assert!(
        det.roots
            .iter()
            .all(|r| r.visible_on == ["gpu-box-2"] && r.shared_from.is_none())
    );
}

#[test]
fn test_the_mount_the_node_names_wins_over_path_matching() {
    let h = hello_roots(vec![root("hf", "/data/hf", json!({}))]);
    let dk = disks(vec![
        mount("/", json!({})),
        mount("/data", json!({"roots": []})),
        mount("/bind/elsewhere", json!({"roots": ["hf"]})),
    ]);
    let det = detail(
        "gpu-box-2",
        &[data(
            "gpu-box-2",
            DataOpts {
                h: Some(h),
                d: Some(dk),
                ..DataOpts::default()
            },
        )],
    );
    assert_eq!(
        det.roots[0].mount.as_ref().unwrap().mount,
        "/bind/elsewhere"
    );
}

#[test]
fn test_roots_keep_their_policy_fields_and_survive_missing_disks() {
    let h = hello_roots(vec![
        root(
            "hf",
            "/data/hf",
            json!({"kind": "gguf", "writable": false, "label": null}),
        ),
        json!({"id": "bad", "path": "relative"}),
        json!({"path": "/x"}),
    ]);
    let det = detail(
        "gpu-box-2",
        &[data(
            "gpu-box-2",
            DataOpts {
                h: Some(h),
                ..DataOpts::default()
            },
        )],
    );
    assert_eq!(det.roots.len(), 1); // relative and id-less roots are not usable
    let r = &det.roots[0];
    assert_eq!(
        (
            r.id.as_str(),
            r.path.as_str(),
            r.kind.as_str(),
            r.writable,
            r.label.as_deref()
        ),
        ("hf", "/data/hf", "gguf", false, None)
    );
    assert!(r.mount.is_none());
}

fn two_hosts(
    b_root_path: &str,
    export: &str,
    a_root: &str,
    server: &str,
    fstype: &str,
) -> Vec<HostData> {
    let a = data(
        "gpu-box-2",
        DataOpts {
            label: Some("GPU box 2".into()),
            h: Some(b_hello(
                &["10.0.0.5", "172.17.0.1"],
                vec![root("hf", a_root, json!({}))],
                "gpu-box-2",
                json!({}),
            )),
            d: Some(disks(vec![
                mount("/", json!({})),
                mount("/data", json!({"fstype": "zfs", "source": "pool/data"})),
            ])),
            ..DataOpts::default()
        },
    );
    let b = data(
        "local",
        DataOpts {
            label: Some("Workstation".into()),
            h: Some(b_hello(
                &["10.0.0.7", "172.17.0.1"],
                vec![root("shared", b_root_path, json!({}))],
                "workstation",
                json!({}),
            )),
            d: Some(disks(vec![
                mount("/", json!({})),
                mount(
                    "/mnt/a-hf",
                    json!({"fstype": fstype, "source": format!("{server}:{export}"), "remote": true,
                                          "remote_host": server, "remote_path": export}),
                ),
            ])),
            ..DataOpts::default()
        },
    );
    vec![b, a]
}

fn two() -> Vec<HostData> {
    two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "10.0.0.5", "nfs4")
}

#[test]
fn test_an_nfs_root_is_the_other_hosts_storage() {
    let fleet = two();
    let b = detail("local", &fleet);
    let a = detail("gpu-box-2", &fleet);
    let br = &b.roots[0];
    assert!(
        br.shared_from.as_deref() == Some("gpu-box-2") && br.visible_on == ["local", "gpu-box-2"]
    );
    let bm = br.mount.as_ref().unwrap();
    assert!(bm.label == "NFS from GPU box 2" && bm.remote_host_id.as_deref() == Some("gpu-box-2"));
    let ar = &a.roots[0];
    assert!(ar.shared_from.is_none() && ar.visible_on == ["gpu-box-2", "local"]); // this host first
    assert_eq!(ar.mount.as_ref().unwrap().label, "pool (zfs)"); // /data/hf lives on the zfs mount
}

#[test]
fn test_shared_storage_is_found_whichever_way_the_paths_overlap() {
    for (b_root, export, a_root) in [
        ("/mnt/a-hf", "/data/hf", "/data/hf"), // the export is the root
        ("/mnt/a-hf/hf", "/data", "/data/hf"), // the export is a parent of the root, mounted somewhere else
        ("/mnt/a-hf/hub", "/data/hf", "/data/hf"), // the root is a folder inside the exported root
        ("/mnt/a-hf", "/data", "/data/hf"),    // the whole export holds the other host's root
    ] {
        let fleet = two_hosts(b_root, export, a_root, "10.0.0.5", "nfs4");
        let b = detail("local", &fleet);
        assert!(
            b.roots[0].shared_from.as_deref() == Some("gpu-box-2")
                && b.roots[0].visible_on == ["local", "gpu-box-2"],
            "{b_root} {export} {a_root}"
        );
        assert_eq!(
            detail("gpu-box-2", &fleet).roots[0].visible_on,
            ["gpu-box-2", "local"]
        );
    }
}

#[test]
fn test_overlap_is_by_whole_path_components() {
    let fleet = two_hosts("/mnt/a-hf", "/data/hf2", "/data/hf", "10.0.0.5", "nfs4"); // "/data/hf2" is not inside "/data/hf"
    let b = detail("local", &fleet);
    assert!(b.roots[0].shared_from.is_none() && b.roots[0].visible_on == ["local"]);
    assert_eq!(
        detail("gpu-box-2", &fleet).roots[0].visible_on,
        ["gpu-box-2"]
    );
}

#[test]
fn test_a_mount_of_an_export_that_holds_none_of_the_other_hosts_roots_is_not_shared() {
    let fleet = two_hosts("/mnt/a-hf", "/srv/backups", "/data/hf", "10.0.0.5", "nfs4");
    let b = detail("local", &fleet);
    assert!(b.roots[0].shared_from.is_none());
    let m = b.roots[0].mount.as_ref().unwrap();
    assert_eq!(m.remote_host_id.as_deref(), Some("gpu-box-2")); // still said to be from that machine
    assert_eq!(m.label, "NFS from GPU box 2");
}

#[test]
fn test_a_server_that_is_not_a_configured_host_is_just_an_address() {
    let fleet = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "10.0.0.99", "nfs4");
    let r = &detail("local", &fleet).roots[0];
    assert!(r.shared_from.is_none() && r.visible_on == ["local"]);
    let m = r.mount.as_ref().unwrap();
    assert!(m.label == "NFS from 10.0.0.99" && m.remote_host_id.is_none());
}

#[test]
fn test_only_nfs_is_matched_to_a_host_by_path() {
    let fleet = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "10.0.0.5", "cifs");
    assert!(detail("local", &fleet).roots[0].shared_from.is_none());
}

#[test]
fn test_a_name_instead_of_an_address_matches_by_hostname_case_insensitively() {
    let mut fleet = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "GPU-Box-2", "nfs4");
    fleet[1] = data(
        "gpu-box-2",
        DataOpts {
            label: Some("GPU box 2".into()),
            h: Some(b_hello(
                &["10.0.0.5"],
                vec![root("hf", "/data/hf", json!({}))],
                "gpu-box-2.lab.example",
                json!({}),
            )),
            d: Some(disks(vec![mount("/", json!({}))])),
            ..DataOpts::default()
        },
    );
    let b = detail("local", &fleet);
    assert_eq!(b.roots[0].shared_from.as_deref(), Some("gpu-box-2"));
    assert_eq!(
        b.roots[0].mount.as_ref().unwrap().remote_host_id.as_deref(),
        Some("gpu-box-2")
    );
}

#[test]
fn test_an_address_two_hosts_both_report_identifies_nobody() {
    let fleet = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "172.17.0.1", "nfs4"); // a docker bridge address, present on both
    let r = &detail("local", &fleet).roots[0];
    assert!(r.shared_from.is_none());
    assert!(r.mount.as_ref().unwrap().remote_host_id.is_none());
}

#[test]
fn test_loopback_servers_and_a_host_mounting_itself_are_never_matched() {
    let fleet = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "127.0.0.1", "nfs4");
    assert!(detail("local", &fleet).roots[0].shared_from.is_none());
    let own = two_hosts("/mnt/a-hf", "/data/hf", "/data/hf", "10.0.0.7", "nfs4"); // the workstation's own address
    let r = &detail("local", &own).roots[0];
    assert!(r.shared_from.is_none());
    let m = r.mount.as_ref().unwrap();
    assert!(m.remote_host_id.is_none() && m.label == "NFS from 10.0.0.7");
}

#[test]
fn test_three_hosts_sharing_one_disk_all_see_each_other() {
    let mut fleet = two();
    fleet.push(data(
        "edge",
        DataOpts {
            ssh: Some("ops@10.0.0.8".into()),
            label: Some("Edge".into()),
            h: Some(b_hello(&["10.0.0.8"], vec![root("models", "/mnt/models", json!({}))], "edge", json!({}))),
            d: Some(disks(vec![
                mount("/", json!({})),
                mount("/mnt/models", json!({"fstype": "nfs4", "remote": true, "remote_host": "10.0.0.5", "remote_path": "/data/hf", "source": "10.0.0.5:/data/hf"})),
            ])),
            ..DataOpts::default()
        },
    ));
    assert_eq!(
        detail("gpu-box-2", &fleet).roots[0].visible_on,
        ["gpu-box-2", "local", "edge"]
    );
    assert_eq!(
        detail("local", &fleet).roots[0].visible_on,
        ["local", "gpu-box-2", "edge"]
    );
    assert_eq!(
        detail("edge", &fleet).roots[0].visible_on,
        ["edge", "gpu-box-2", "local"]
    );
    assert_eq!(
        detail("edge", &fleet).roots[0].shared_from.as_deref(),
        Some("gpu-box-2")
    );
}

#[test]
fn test_a_down_host_still_takes_part_in_the_correlation_from_its_last_known_data() {
    let mut fleet = two();
    let (h, dk) = (fleet[1].hello.clone(), fleet[1].disks.clone());
    fleet[1] = data(
        "gpu-box-2",
        DataOpts {
            label: Some("GPU box 2".into()),
            state: "down".into(),
            h: h.map(|x| (*x).clone()),
            d: dk.map(|x| (*x).clone()),
            ..DataOpts::default()
        },
    );
    assert_eq!(
        detail("local", &fleet).roots[0].shared_from.as_deref(),
        Some("gpu-box-2")
    );
}

// ── addresses, scripts, images ─────────────────────────────────────────────────────────────────────────
#[test]
fn test_addresses_are_listed_without_loopback() {
    let mut h = hello();
    h.insert(
        "addresses".into(),
        json!([{"iface": "lo", "addr": "127.0.0.1", "prefix": 8}, {"iface": "eth0", "addr": "10.0.0.5", "prefix": 24},
               {"iface": "eth0", "addr": "fd00::5", "prefix": null}, {"addr": "::1", "iface": "lo"}, {"addr": "10.0.0.6"}, {"iface": "x"}]),
    );
    assert_eq!(
        addresses_of(Some(&h)),
        ["10.0.0.5/24 (eth0)", "fd00::5 (eth0)", "10.0.0.6"]
    );
    assert!(addresses_of(None).is_empty());
}

#[test]
fn test_scripts_combine_the_policy_with_the_last_reported_status() {
    let scripts = json!([{"name": "big-model", "label": "Big model (2 GPUs)", "engine": "sglang", "description": "Starts before the small one.",
                "containers": ["proj-sglang-*"], "status_action": "status",
                "actions": [{"name": "up", "label": "Start", "confirm": true}, {"name": "down", "label": "Stop", "confirm": true, "danger": true},
                            {"name": "switch", "label": "Switch", "args": ["model-a", "model-b"]}]},
               {"name": "other", "actions": []}]);
    let inv = ec_proto::sanitize_push(&obj(json!({"type": "inventory", "t": 1, "scripts": [
        {"name": "big-model", "status": "up", "detail": "running", "checked_at": 5.5}, {"name": "other", "status": "weird"}]})))
    .unwrap()
    .unwrap();
    let h = b_hello(&[], Vec::new(), "gpu-box-2", json!({"scripts": scripts}));
    let det = detail(
        "gpu-box-2",
        &[data(
            "gpu-box-2",
            DataOpts {
                h: Some(h.clone()),
                inv: Some(inv),
                ..DataOpts::default()
            },
        )],
    );
    let (big, other) = (&det.scripts[0], &det.scripts[1]);
    assert_eq!(
        (
            big.host_id.as_str(),
            big.name.as_str(),
            big.label.as_str(),
            big.engine.as_deref(),
            big.status.as_str(),
            big.detail.as_deref(),
            big.checked_at
        ),
        (
            "gpu-box-2",
            "big-model",
            "Big model (2 GPUs)",
            Some("sglang"),
            "up",
            Some("running"),
            Some(5.5)
        )
    );
    assert!(
        big.containers == ["proj-sglang-*"] && big.description == "Starts before the small one."
    );
    let (up, down, switch) = (&big.actions[0], &big.actions[1], &big.actions[2]);
    assert_eq!(
        (up.name.as_str(), up.confirm, up.danger),
        ("up", true, false)
    );
    assert_eq!((down.confirm, down.danger), (true, true));
    assert!(switch.args == ["model-a", "model-b"] && !switch.confirm);
    // unknown status words are not trusted
    assert!(other.label == "other" && other.status == "unknown" && other.description.is_empty());
    // nothing reported yet
    assert_eq!(
        detail(
            "gpu-box-2",
            &[data(
                "gpu-box-2",
                DataOpts {
                    h: Some(h),
                    ..DataOpts::default()
                }
            )]
        )
        .scripts[0]
            .status,
        "unknown"
    );
}

#[test]
fn test_images_from_a_docker_images_result() {
    let result = obj(
        json!({"images": [{"ref": "vllm/vllm-openai:v0.1", "id": "sha256:abc", "size": 10 * GIB, "created": 1.7e9, "arch": "amd64"},
                                       {"ref": null, "id": "x"}, {"ref": "bare", "size": null}]}),
    );
    let img =
        |r: &str, id: Option<&str>, size: Option<i64>, created: Option<f64>, arch: Option<&str>| {
            ImageInfo {
                ref_: r.into(),
                id: id.map(str::to_string),
                size_bytes: size,
                created,
                arch: arch.map(str::to_string),
            }
        };
    assert_eq!(
        images_from_result(&result),
        [
            img(
                "vllm/vllm-openai:v0.1",
                Some("sha256:abc"),
                Some(10 * GIB),
                Some(1.7e9),
                Some("amd64")
            ),
            img("bare", None, None, None, None)
        ]
    );
    assert!(images_from_result(&obj(json!({}))).is_empty());
}

#[test]
fn test_the_detail_carries_the_whole_summary_and_the_images() {
    let dd = data("gpu-box-2", d(hv(hello()), hv(sample(json!({})))));
    let counts: CountsProvider = Arc::new(|_: &str| HostCounts {
        models: 3,
        ..HostCounts::default()
    });
    let det = build_detail(
        "gpu-box-2",
        std::slice::from_ref(&dd),
        vec![ImageInfo {
            ref_: "x:1".into(),
            id: None,
            size_bytes: None,
            created: None,
            arch: None,
        }],
        None,
        Some(&counts),
    )
    .unwrap();
    let s = build_summary(&dd, None, Some(&counts));
    assert_eq!(det.summary, s);
    assert_eq!(
        det.images
            .iter()
            .map(|i| i.ref_.as_str())
            .collect::<Vec<_>>(),
        ["x:1"]
    );
    assert_eq!(det.addresses, ["10.0.0.5/24 (eth0)"]);
    // the JSON keeps HostDetail(HostSummary): every summary field at the top level
    let v = serde_json::to_value(&det).unwrap();
    for k in serde_json::to_value(&s)
        .unwrap()
        .as_object()
        .unwrap()
        .keys()
    {
        assert!(v.get(k).is_some(), "{k}");
    }
    let _ = (G, fs::LIVE_STATES);
}
