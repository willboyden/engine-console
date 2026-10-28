//! Port of backend/tests/core/test_hardware.py: per-host hardware from what the node pushed (hello = static, sample
//! = live), bytes -> GiB, never read from the hub itself; plus the pure host-memory helpers.

#![allow(
    clippy::needless_pass_by_value,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::double_ended_iterator_last,
    clippy::assertions_on_constants,
    clippy::too_many_lines
)]

#[path = "common/kit.rs"]
mod kit;

// the fake fleet is the kit's submodule (PORTING.md §A4): one copy of its types per test binary
use kit::fakefleet;

use ec_hub::domain::errors::AppError;
use ec_hub::domain::models::HostMemInfo;
use ec_hub::services::hostmem::{
    alerts_for, container_memory, meminfo_from_sample, parse_meminfo, sum_memory,
};
use fakefleet::{FleetEnv, GpuFree, HelloOpts, cgroup, cgroup_default};
use serde_json::{Value, json};

const P: &str = "/api/v1";
const KB_PER_GIB: i64 = 1024 * 1024;
const GIB: i64 = 1024 * 1024 * 1024;

fn meminfo() -> String {
    format!(
        "MemTotal: {} kB\nMemFree: {} kB\nMemAvailable: {} kB\nCached: {} kB\nShmem: {} kB\nSwapTotal: {} kB\nSwapFree: {} kB\nBogus line\nHugePages_Total: 0\n",
        246 * KB_PER_GIB,
        3 * KB_PER_GIB,
        54 * KB_PER_GIB,
        90 * KB_PER_GIB,
        66 * KB_PER_GIB,
        32 * KB_PER_GIB,
        8 * KB_PER_GIB
    )
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

fn node_err(e: &AppError) -> (String, String) {
    match e {
        AppError::Node(n) => (n.code.clone(), n.message.clone()),
        AppError::Problem(p) => (p.code.clone(), p.detail.clone()),
    }
}

// ── the report ─────────────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_report_shape_numbers_and_alerts_from_hello_and_sample() {
    let mut env = FleetEnv::new();
    env.node.set_gpu_free(GpuFree::Each(vec![90.0, 40.0]));
    let rep = env.container().hardware.report("local").unwrap();
    assert!(rep.host_id == "local" && rep.source == "nvml");
    assert_eq!(
        rep.gpus
            .iter()
            .map(|g| (g.index, g.uuid.as_str()))
            .collect::<Vec<_>>(),
        [(0, "GPU-uuid-0"), (1, "GPU-uuid-1")]
    );
    let (g0, g1) = (&rep.gpus[0], &rep.gpus[1]);
    assert_eq!((g0.total_gib, g0.free_gib, g0.used_gib), (96.0, 90.0, 6.0));
    assert_eq!(g1.free_gib, 40.0);
    assert!(
        g0.util_pct == Some(5.0)
            && g0.temp_c == Some(40.0)
            && g0.power_w == Some(90.0)
            && g0.fan_pct == Some(30.0)
    );
    assert!(
        g0.compute_capability.as_deref() == Some("12.0") && g0.name == "RTX PRO 6000 Blackwell"
    );
    assert!(rep.host_ram_gib == 246.0 && rep.host_ram_free_gib == 54.0);
    let m = rep.memory.as_ref().unwrap();
    assert_eq!(
        (m.used_gib, m.swap_used_gib, m.shmem_gib),
        (192.0, 24.0, 66.0)
    );
    let mut codes: Vec<&str> = rep.alerts.iter().map(|a| a.code.as_str()).collect();
    codes.sort_unstable();
    assert_eq!(codes, ["shmem_high", "swap_heavy"]);
}

#[test]
fn test_each_host_reports_its_own_numbers() {
    let mut env = FleetEnv::new();
    let spark = env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            gpus: 1,
            gpu_gib: 119.0,
            ram_gib: 128.0,
            source: "nvidia-smi".into(),
            ..HelloOpts::default()
        },
    );
    spark.set_mem(&[
        ("total", 128.0),
        ("available", 100.0),
        ("shmem", 1.0),
        ("swap_total", 0.0),
        ("swap_free", 0.0),
    ]);
    let rep = env.container().hardware.report("spark").unwrap();
    assert!(rep.gpus.len() == 1 && rep.gpus[0].total_gib == 119.0 && rep.source == "nvidia-smi");
    assert!(rep.host_ram_gib == 128.0 && rep.alerts.is_empty());
    assert_eq!(
        env.container().hardware.report("local").unwrap().gpus.len(),
        2
    );
}

#[test]
fn test_unified_memory_gpus_take_their_budget_from_host_ram() {
    let mut env = FleetEnv::new();
    let gb = env.add_host(
        "gb",
        "GB10",
        HelloOpts {
            gpus: 1,
            gpu_gib: 119.0,
            unified: true,
            ram_gib: 119.0,
            ..HelloOpts::default()
        },
    );
    gb.set_mem(&[("total", 119.0), ("available", 30.0)]);
    let g = env.container().hardware.report("gb").unwrap().gpus[0].clone();
    assert!(
        g.total_gib == 119.0 && approx(g.free_gib, 30.0, 1e-6) && approx(g.used_gib, 89.0, 1e-6)
    );
    let hw = env.container().hardware.hardware("gb", None).unwrap();
    assert!(
        hw.unified_memory && hw.gpu_free_gib.len() == 1 && approx(hw.gpu_free_gib[0], 30.0, 1e-6)
    );
}

#[test]
fn test_a_host_that_is_down_or_silent_has_no_numbers_rather_than_stale_ones() {
    let mut env = FleetEnv::new();
    let c = env.container();
    env.node.set_state("down", Some("no route to host"));
    let e = c.hardware.report("local").unwrap_err();
    let (code, msg) = node_err(&e);
    assert!(code == "node_unavailable" && msg.contains("no route to host"));
    assert!(c.hardware.host_memory("local").unwrap().is_none());
    env.node.set_state("stale", None);
    assert!(!c.hardware.report("local").unwrap().gpus.is_empty()); // stale: still the last sample
    let e2 = c.hardware.report("nope").unwrap_err();
    assert_eq!(e2.code(), "unknown_host");
    assert_eq!(e2.status(), 404);
}

#[test]
fn test_before_the_first_sample_the_hub_says_so() {
    let mut env = FleetEnv::new();
    env.node.st().hidden.insert("sample".into());
    let e = env.container().hardware.report("local").unwrap_err();
    let (code, msg) = node_err(&e);
    assert!(code == "node_unavailable" && msg.contains("first sample"));
}

// ── the adapter's view ─────────────────────────────────────────────────────────────────────────────────
#[test]
fn test_adapter_view_is_restricted_to_the_chosen_gpus_and_carries_host_facts() {
    let mut env = FleetEnv::new();
    env.node.set_gpu_free(GpuFree::Each(vec![90.0, 40.0]));
    env.add_host(
        "arm",
        "ARM",
        HelloOpts {
            arch: "aarch64".into(),
            cc: "12.1".into(),
            ..HelloOpts::default()
        },
    );
    let c = env.container();
    let hw = c.hardware.hardware("local", Some(&[1])).unwrap();
    assert_eq!(
        (
            hw.gpu_ids.clone(),
            hw.gpu_uuids.clone(),
            hw.gpu_free_gib.clone(),
            hw.compute_capability.as_str()
        ),
        (vec![1], vec!["GPU-uuid-1".to_string()], vec![40.0], "12.0")
    );
    assert_eq!(
        (
            hw.host_id.as_str(),
            hw.arch.as_str(),
            hw.unified_memory,
            hw.host_ram_gib
        ),
        ("local", "x86_64", false, 246.0)
    );
    assert_eq!(c.hardware.hardware("local", None).unwrap().gpu_ids, [0, 1]);
    assert_eq!(
        c.hardware.hardware("local", Some(&[])).unwrap().gpu_ids,
        [0, 1]
    );
    let arm = c.hardware.hardware("arm", None).unwrap();
    assert!(arm.arch == "aarch64" && arm.host_id == "arm" && arm.compute_capability == "12.1");
    assert!(
        c.hardware
            .hardware("local", Some(&[9]))
            .unwrap()
            .gpu_ids
            .is_empty()
    ); // an index the host does not have selects nothing
}

#[test]
fn test_a_host_without_gpus_reports_an_empty_but_valid_hardware() {
    let mut env = FleetEnv::new();
    env.add_host(
        "cpu",
        "CPU box",
        HelloOpts {
            gpus: 0,
            ..HelloOpts::default()
        },
    );
    let c = env.container();
    let hw = c.hardware.hardware("cpu", None).unwrap();
    assert!(hw.gpu_ids.is_empty() && hw.compute_capability == "unknown");
    assert!(c.hardware.report("cpu").unwrap().gpus.is_empty());
}

// ── the API ────────────────────────────────────────────────────────────────────────────────────────────
#[tokio::test]
async fn test_hardware_route_defaults_to_local_and_takes_a_host() {
    let mut env = FleetEnv::new();
    env.add_host(
        "spark",
        "Spark",
        HelloOpts {
            gpus: 1,
            ..HelloOpts::default()
        },
    );
    let body = env.get(&format!("{P}/hardware")).send().await.json();
    assert!(body["host_id"] == "local" && body["source"] == "nvml");
    assert_eq!(
        body["gpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["uuid"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["GPU-uuid-0", "GPU-uuid-1"]
    );
    assert_eq!(body["memory"]["total_gib"], 246.0);
    let mut codes: Vec<&str> = body["alerts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a["code"].as_str().unwrap())
        .collect();
    codes.sort_unstable();
    assert_eq!(codes, ["shmem_high", "swap_heavy"]);
    let sp = env
        .get(&format!("{P}/hardware?host=spark"))
        .send()
        .await
        .json();
    assert!(sp["host_id"] == "spark" && sp["gpus"].as_array().unwrap().len() == 1);
    assert_eq!(
        env.get(&format!("{P}/hardware?host=ghost"))
            .send()
            .await
            .code(),
        "unknown_host"
    );
    assert_eq!(
        env.get(&format!("{P}/hardware?host=Bad%20Host!"))
            .send()
            .await
            .status,
        422
    );
    env.node.set_state("down", None);
    let down = env.get(&format!("{P}/hardware")).send().await;
    assert!(down.status == 503 && down.code() == "node_unavailable");
}

// ── the pure helpers (hostmem) ─────────────────────────────────────────────────────────────────────────
#[test]
fn test_meminfo_text_parsing_is_unchanged() {
    let m = parse_meminfo(&meminfo(), 123.0).unwrap();
    assert_eq!(
        (
            m.total_gib,
            m.available_gib,
            m.free_gib,
            m.cached_gib,
            m.shmem_gib
        ),
        (246.0, 54.0, 3.0, 90.0, 66.0)
    );
    assert!(
        m.used_gib == 192.0
            && m.swap_total_gib == 32.0
            && m.swap_used_gib == 24.0
            && m.updated_at == 123.0
    );
    assert!(parse_meminfo("MemTotal: 1 kB\n", 0.0).is_none() && parse_meminfo("", 0.0).is_none());
    let no_swap = parse_meminfo("MemTotal: 1024 kB\nMemAvailable: 512 kB\n", 0.0).unwrap();
    assert!(no_swap.swap_total_gib == 0.0 && no_swap.swap_used_gib == 0.0);
}

/// Python also feeds `float("inf")`, which a serde_json value cannot hold; the other refusals are the same.
#[test]
fn test_a_samples_mem_block_converts_bytes_to_gib_and_rejects_nonsense() {
    let m = meminfo_from_sample(
        Some(&json!({"total": 246 * GIB, "available": 54 * GIB, "free": 3 * GIB, "cached": 90 * GIB, "shmem": 66 * GIB,
                     "swap_total": 32 * GIB, "swap_free": 8 * GIB})),
        Some(5.0),
    )
    .unwrap();
    assert_eq!(
        (m.total_gib, m.used_gib, m.swap_used_gib, m.updated_at),
        (246.0, 192.0, 24.0, 5.0)
    );
    assert!(
        meminfo_from_sample(None, Some(1.0)).is_none()
            && meminfo_from_sample(Some(&json!({})), Some(1.0)).is_none()
    );
    assert!(meminfo_from_sample(Some(&json!({"total": -1, "available": 5})), Some(1.0)).is_none());
    assert!(
        meminfo_from_sample(Some(&json!({"total": true, "available": 5})), Some(1.0)).is_none()
    );
    let swapless =
        meminfo_from_sample(Some(&json!({"total": 2 * GIB, "available": GIB})), None).unwrap();
    assert!(swapless.swap_used_gib == 0.0 && swapless.updated_at == 0.0);
}

#[test]
fn test_container_memory_cache_is_file_minus_shmem_and_never_negative() {
    let hm = container_memory(cgroup_default().get("mem")).unwrap();
    assert_eq!(hm.source, "cgroup");
    assert!(approx(hm.total_gib, 68.6, 1e-3) && approx(hm.anon_gib, 3.3, 1e-3));
    assert!(approx(hm.shmem_gib, 64.0, 1e-3) && approx(hm.kernel_gib, 0.6, 1e-3));
    assert!(approx(hm.cache_gib, 0.5, 1e-3)); // file 64.5 - shmem 64.0: nothing double counted
    let odd = container_memory(Some(&json!({"current": GIB, "anon": GIB, "file": GIB, "shmem": 3 * GIB, "kernel": 0, "source": "rss"}))).unwrap();
    assert!(odd.cache_gib == 0.0 && odd.source == "rss");
    assert!(
        container_memory(None).is_none() && container_memory(Some(&json!({"anon": GIB}))).is_none()
    );
}

#[test]
fn test_sum_memory_adds_parts_and_is_rss_if_any_part_is() {
    let a = container_memory(cgroup(10.0, 1.0, 9.0, 8.0, 0.1).get("mem")).unwrap();
    let mut bm = cgroup(5.0, 1.0, 4.0, 3.0, 0.1)["mem"].clone();
    bm["source"] = json!("rss");
    let b = container_memory(Some(&bm)).unwrap();
    let total = sum_memory(&[a, b]).unwrap();
    assert!(approx(total.total_gib, 15.0, 1e-6) && total.source == "rss");
    assert!(sum_memory(&[]).is_none());
}

#[test]
fn test_alert_thresholds() {
    let cases: Vec<(f64, f64, f64, Vec<(&str, &str)>)> = vec![
        (50.0, 0.0, 0.0, vec![]),
        (10.0, 0.0, 0.0, vec![]),
        (9.9, 0.0, 0.0, vec![("warn", "host_ram_low")]),
        (5.0, 0.0, 0.0, vec![("warn", "host_ram_low")]),
        (4.9, 0.0, 0.0, vec![("crit", "host_ram_low")]),
        (50.0, 25.0, 0.0, vec![]),
        (50.0, 25.1, 0.0, vec![("warn", "shmem_high")]),
        (50.0, 0.0, 16.0, vec![]),
        (50.0, 0.0, 16.1, vec![("warn", "swap_heavy")]),
        (
            3.0,
            40.0,
            9.0,
            vec![
                ("crit", "host_ram_low"),
                ("warn", "shmem_high"),
                ("warn", "swap_heavy"),
            ],
        ),
    ];
    for (avail, shmem, swap_used, expected) in cases {
        let swap_total = if swap_used > 10.0 {
            32.0
        } else if swap_used > 0.0 {
            10.0
        } else {
            0.0
        };
        let m = HostMemInfo {
            total_gib: 100.0,
            used_gib: 100.0 - avail,
            available_gib: avail,
            free_gib: 1.0,
            cached_gib: 1.0,
            shmem_gib: shmem,
            swap_total_gib: swap_total,
            swap_used_gib: swap_used,
            updated_at: 0.0,
        };
        let got: Vec<(String, String)> = alerts_for(&m)
            .into_iter()
            .map(|a| (a.level, a.code))
            .collect();
        let want: Vec<(String, String)> = expected
            .iter()
            .map(|(l, c)| ((*l).to_string(), (*c).to_string()))
            .collect();
        assert_eq!(got, want, "{avail} {shmem} {swap_used}");
    }
}

#[test]
fn test_the_reference_box_triggers_shmem_and_swap_alerts_with_an_explanation() {
    let m = parse_meminfo(&meminfo(), 0.0).unwrap();
    let got = alerts_for(&m);
    let mut codes: Vec<&str> = got.iter().map(|a| a.code.as_str()).collect();
    codes.sort_unstable();
    assert_eq!(codes, ["shmem_high", "swap_heavy"]);
    assert!(
        got.iter()
            .find(|a| a.code == "shmem_high")
            .unwrap()
            .message
            .contains("cannot be reclaimed")
    );
    let _: Value = json!(null);
}

/// test_api_smoke.py `test_health_hardware_engines`: health, the hardware report through the node, and the engine
/// catalogue (the health/engines half is also core's `test_health_and_engines`).
#[tokio::test]
async fn test_health_hardware_engines() {
    let mut env = FleetEnv::new();
    assert_eq!(
        env.get(&format!("{P}/health")).send().await.json()["status"],
        "ok"
    );
    let hwr = env.get(&format!("{P}/hardware")).send().await.json();
    assert_eq!(
        hwr["gpus"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["uuid"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["GPU-uuid-0", "GPU-uuid-1"]
    );
    assert!(hwr["source"] == "nvml" && hwr["host_id"] == "local");
    let eng = env.get(&format!("{P}/engines")).send().await.json();
    assert!(
        eng[0]["id"] == "fake"
            && eng[0]["presets"][0]["name"] == "balanced"
            && eng[0]["model_formats"] == json!(["hf"])
    );
    let params = env
        .get(&format!("{P}/engines/fake/params"))
        .send()
        .await
        .json();
    let mut keys: Vec<&str> = params
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p["key"].as_str().unwrap())
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "api_key",
            "image",
            "kv_dtype",
            "leaky_key",
            "max_model_len",
            "mem_fraction",
            "tp",
            "trust_remote_code"
        ]
    );
    let r = env.get(&format!("{P}/engines/nope/params")).send().await;
    assert!(
        r.header("content-type")
            .unwrap_or("")
            .starts_with("application/problem+json")
    );
    assert_eq!(r.code(), "unknown_engine");
}
