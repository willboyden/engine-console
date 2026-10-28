//! node/tests/test_machine.py: machine facts from fake /proc, /sys and /etc trees, fake nvidia-smi and fake ip.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::needless_pass_by_value,
    clippy::double_ended_iterator_last,
    clippy::redundant_clone,
    unreachable_pub
)]

#[path = "common/wsd_kit.rs"]
mod wsd_kit;

use std::sync::Arc;

use ecnode::disks::{Disks, mount_of, parse_mountinfo, split_remote};
use ecnode::gpus::Gpus;
use ecnode::hostinfo;
use ecnode::memory::Mem;
use ecnode::policy::Root;
use ecnode::util::{ProcRunner, Runner};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use wsd_kit::{CID_EXT, FakeBin, GPU_A, GPU_B, Machine, cid_engine, machine_statvfs};

fn runner(fb: &FakeBin) -> Arc<dyn Runner> {
    Arc::new(ProcRunner::new(fb.env()))
}

#[test]
fn test_host_facts_and_node_id() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    let fb = FakeBin::new(tmp.path());
    let h = hostinfo::host_facts(&m.paths, runner(&fb).as_ref());
    assert_eq!(h["os"], "Test Linux 1.0");
    assert_eq!(h["cpu_model"], "Test CPU 9000");
    assert_eq!(h["cpu_cores"], 2);
    assert_eq!(h["ram_total_bytes"], json!(263_000_000_i64 * 1024));
    assert_eq!(h["boot_time"], json!(1_700_000_000.0));
    assert_eq!(h["python"], Value::Null); // PORTING.md DV5
    let want = hex::encode(Sha256::digest(b"0123456789abcdef0123456789abcdef"));
    assert_eq!(hostinfo::node_id(&m.paths), want[..16]);
}

#[test]
fn test_addresses_skip_loopback_link_local_and_bridges() {
    let tmp = tempfile::tempdir().unwrap();
    let fb = FakeBin::new(tmp.path());
    let m = Machine::new(tmp.path());
    let addrs = hostinfo::addresses(runner(&fb).as_ref(), &m.paths);
    assert_eq!(
        addrs,
        vec![
            json!({"iface": "eth0", "addr": "10.0.0.5", "prefix": 24}),
            json!({"iface": "eth0", "addr": "fd00::5", "prefix": 64}),
        ]
    );
}

#[test]
fn test_meminfo_load_and_cpu() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    let mem = Mem::new(m.paths.clone());
    let info = mem.meminfo();
    assert_eq!(info["total"], json!(263_000_000_i64 * 1024));
    assert_eq!(info["available"], json!(200_000_000_i64 * 1024));
    assert_eq!(mem.loadavg(), (Some(0.5), Some(0.4), Some(0.3)));
    assert_eq!(mem.cpu_util(), None); // first call: no delta yet
    std::fs::write(
        m.paths.proc.join("stat"),
        "cpu  200 0 200 900 0 0 0 0 0 0\nbtime 1700000000\n",
    )
    .unwrap();
    assert_eq!(mem.cpu_util(), Some(66.7)); // 200 busy of 300 jiffies
}

#[test]
fn test_container_memory_from_cgroup_v2_and_rss_fallback() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    let mem = Mem::new(m.paths.clone());
    assert_eq!(mem.container_id_of(4242), Some(cid_engine()));
    assert_eq!(mem.container_id_of(4343), Some(CID_EXT.to_string()));
    assert_eq!(mem.comm(4242).as_deref(), Some("vllm"));
    assert_eq!(
        Value::Object(mem.container_mem(4242).unwrap()),
        json!({"current": 3_000_000_000_i64, "anon": 1_000_000_000_i64, "file": 1_500_000_000_i64,
               "shmem": 500_000_000_i64, "kernel": 100_000_000_i64, "source": "cgroup"})
    );
    assert_eq!(mem.container_cpu_usec(4242), Some(5_000_000));
    // no cgroup dir: VmRSS of the tree (4343 + child 4344)
    let rss = mem.container_mem(4343).unwrap();
    assert_eq!(rss["source"], "rss");
    assert_eq!(rss["current"], json!(3072 * 1024));
    assert!(mem.container_mem(0).is_none() && mem.container_mem(77777).is_none());
}

#[test]
fn test_container_id_from_every_cgroup_path_form() {
    let cases: Vec<(String, Option<String>)> = vec![
        (
            format!(
                "0::/user.slice/user-1000.slice/user@1000.service/user.slice/docker-{}.scope",
                "1".repeat(64)
            ),
            Some("1".repeat(64)),
        ),
        (
            format!("0::/system.slice/docker-{}.scope", "2".repeat(64)),
            Some("2".repeat(64)),
        ),
        (
            format!("0::/docker/{}", "3".repeat(64)),
            Some("3".repeat(64)),
        ),
        (
            format!(
                "0::/user.slice/user-1000.slice/user@1000.service/docker/{}",
                "4".repeat(64)
            ),
            Some("4".repeat(64)),
        ),
        (
            format!("12:memory:/docker/{}\n0::/", "5".repeat(64)),
            Some("5".repeat(64)),
        ),
        (
            format!(
                "0::/system.slice/docker-{}.scope/init.scope",
                "6".repeat(64)
            ),
            Some("6".repeat(64)),
        ),
        (
            format!("0::/docker/{}/docker/{}", "7".repeat(64), "8".repeat(64)),
            Some("7".repeat(64)),
        ),
        (
            format!("0::/machine.slice/libpod-{}.scope", "9".repeat(64)),
            None,
        ),
        (
            format!("0::/kubepods/besteffort/pod1/{}", "a".repeat(64)),
            None,
        ),
        (
            "0::/user.slice/user-1000.slice/session-2.scope".to_string(),
            None,
        ),
        (
            format!("0::/system.slice/docker-{}.scope", "b".repeat(63)),
            None,
        ),
    ];
    for (line, expected) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let m = Machine::new(tmp.path());
        let d = m.paths.proc.join("6000");
        std::fs::create_dir(&d).unwrap();
        std::fs::write(d.join("cgroup"), format!("{line}\n")).unwrap();
        assert_eq!(
            Mem::new(m.paths.clone()).container_id_of(6000),
            expected,
            "{line}"
        );
    }
}

#[test]
fn test_hostile_cgroup_paths_are_ignored() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    std::fs::create_dir(m.paths.proc.join("5000")).unwrap();
    std::fs::write(
        m.paths.proc.join("5000").join("cgroup"),
        "0::/../../../etc\n",
    )
    .unwrap();
    assert!(Mem::new(m.paths.clone()).cgroup_dir(5000).is_none());
}

#[test]
fn test_mountinfo_parsing_and_remote_split() {
    let mi = parse_mountinfo(
        "36 1 0:5 / /mnt/my\\040models rw - nfs4 [fd00::9]:/export/m rw\n\
         37 1 0:6 / /mnt/smb rw - cifs //nas/share/models rw\nbroken line\n",
    );
    assert_eq!(mi[0].mount, "/mnt/my models");
    assert_eq!(
        split_remote(&mi[0].fstype, &mi[0].source),
        (Some("fd00::9".into()), Some("/export/m".into()))
    );
    assert_eq!(
        split_remote("cifs", "//nas/share/models"),
        (Some("nas".into()), Some("/share/models".into()))
    );
    assert_eq!(
        split_remote("fuse.sshfs", "user@box:/srv/x"),
        (Some("box".into()), Some("/srv/x".into()))
    );
    assert_eq!(
        mount_of("/mnt/my models/a", &mi).unwrap().mount,
        "/mnt/my models"
    );
}

#[test]
fn test_disks_report_roots_drives_and_network_mounts() {
    let tmp = tempfile::tempdir().unwrap();
    let m = Machine::new(tmp.path());
    let roots = vec![
        Root::new("hf", m.data.join("hf").to_str().unwrap()),
        Root::new("shared", m.nfs.join("sub").to_str().unwrap()),
        Root::new("gone", "/nonexistent-for-test/x"),
    ];
    let rep = Disks::with_statvfs(m.paths.clone(), machine_statvfs()).report(&roots);
    let mounts = rep["mounts"].as_array().unwrap();
    let by = |mp: &str| mounts.iter().find(|x| x["mount"] == mp).cloned().unwrap();
    let mut names: Vec<&str> = mounts
        .iter()
        .map(|x| x["mount"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    let mut want = vec!["/", m.data.to_str().unwrap(), m.nfs.to_str().unwrap()];
    want.sort_unstable();
    assert_eq!(names, want); // no /proc, no squashfs
    let root = by("/");
    assert_eq!(
        (root["device"].clone(), root["model"].clone()),
        (json!("nvme0n1p2"), json!("Test SSD 4TB"))
    );
    assert_eq!(root["roots"], json!(["gone"]));
    let data = by(m.data.to_str().unwrap());
    assert_eq!(data["model"], "Data Drive 8TB");
    assert_eq!(data["roots"], json!(["hf"]));
    assert_eq!(data["size"], json!((1_u64 << 30) * 4096));
    assert_eq!(data["free"], json!((1_u64 << 28) * 4096));
    assert_eq!(data["used"], json!(((1_u64 << 30) - (1 << 28)) * 4096));
    let nfs = by(m.nfs.to_str().unwrap());
    assert_eq!(nfs["remote"], true);
    assert_eq!(nfs["remote_host"], "10.0.0.9");
    assert_eq!(nfs["remote_path"], "/export/models");
    assert_eq!(nfs["readonly"], true);
    assert_eq!(nfs["roots"], json!(["shared"]));
    assert_eq!(nfs["model"], Value::Null);
}

#[test]
fn test_gpus_from_nvidia_smi() {
    let tmp = tempfile::tempdir().unwrap();
    let fb = FakeBin::new(tmp.path());
    let g = Gpus::new(false, runner(&fb));
    let (statics, info) = g.static_info();
    let uuids: Vec<&str> = statics
        .iter()
        .map(|x| x["uuid"].as_str().unwrap())
        .collect();
    assert_eq!(uuids, vec![GPU_A, GPU_B]);
    assert_eq!(info["source"], "nvidia-smi");
    assert_eq!(
        (info["driver"].clone(), info["cuda"].clone()),
        (json!("595.0"), json!("13.0"))
    );
    assert_eq!(statics[0]["total_bytes"], json!(97887_i64 * 1024 * 1024));
    assert_eq!(statics[0]["compute_capability"], "12.0");
    let (stats, procs) = g.sample();
    let a = &stats.iter().find(|(u, _)| u == GPU_A).unwrap().1;
    assert_eq!(a["util_pct"], json!(7.0));
    assert_eq!(a["power_w"], json!(71.5));
    assert_eq!(a["fan_pct"], json!(30.0));
    assert!(procs.contains(&(GPU_A.to_string(), 4242, Some(2048 * 1024 * 1024))));
}

#[test]
fn test_unified_memory_gpu_reports_no_memory_numbers() {
    let tmp = tempfile::tempdir().unwrap();
    let fb = FakeBin::new(tmp.path());
    fb.smi(json!({
        "gpus": [{"index": 0, "uuid": GPU_A, "name": "NVIDIA GB10", "memory.total": "[N/A]", "compute_cap": "12.1",
                  "pci.bus_id": "0000000F:01:00.0", "driver_version": "580.0", "utilization.gpu": "3",
                  "memory.used": "[N/A]", "temperature.gpu": "50", "power.draw": "[N/A]", "power.limit": "[N/A]",
                  "fan.speed": "[N/A]"}],
        "apps": [{"pid": 4242, "used_memory": "[N/A]", "gpu_uuid": GPU_A}],
    }));
    let g = Gpus::new(false, runner(&fb));
    let (statics, _) = g.static_info();
    assert_eq!(statics[0]["unified_memory"], true);
    assert_eq!(statics[0]["total_bytes"], Value::Null);
    let (stats, procs) = g.sample();
    assert_eq!(stats[0].1["mem_used_bytes"], Value::Null);
    assert_eq!(stats[0].1["power_w"], Value::Null);
    assert_eq!(procs, vec![(GPU_A.to_string(), 4242, None)]);
}

#[test]
fn test_zero_gpus_and_no_driver() {
    let tmp = tempfile::tempdir().unwrap();
    let fb = FakeBin::new(tmp.path());
    fb.smi(json!({"gpus": []}));
    let (statics, info) = Gpus::new(false, runner(&fb)).static_info();
    assert!(statics.is_empty() && info["source"] == "none");
    assert!(info["error"].as_str().unwrap().contains("No devices"));
    // no nvidia-smi among the node's tools (ECNODE_TOOL_PATH: the node never looks at PATH)
    let empty = tmp.path().join("empty");
    std::fs::create_dir_all(&empty).unwrap();
    let env = fb.without_tools(&empty);
    let r: Arc<dyn Runner> = Arc::new(ProcRunner::new(env));
    let (statics, info) = Gpus::new(false, Arc::clone(&r)).static_info();
    assert!(statics.is_empty() && info["error"].as_str().unwrap().contains("not on PATH"));
    let (stats, procs) = Gpus::new(false, r).sample();
    assert!(stats.is_empty() && procs.is_empty());
}

/// PORTING.md §A2: ECNODE_NO_NVML (non-empty) never loads NVML; the GPUs come from nvidia-smi on PATH.
#[test]
fn test_ecnode_no_nvml_reads_nvidia_smi() {
    let tmp = tempfile::tempdir().unwrap();
    let fb = FakeBin::new(tmp.path());
    let mut env = fb.env();
    env.insert("ECNODE_NO_NVML".into(), "1".into());
    let g = Gpus::for_env(&env, Arc::new(ProcRunner::new(env.clone())));
    assert_eq!(g.source(), "nvidia-smi");
    let (statics, info) = g.static_info();
    assert_eq!(info["source"], "nvidia-smi");
    assert_eq!(statics.len(), 2);
    assert_eq!(statics[1]["uuid"], GPU_B);
}
