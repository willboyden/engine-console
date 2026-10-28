//! Python: node/src/ecnode/cli.py.
//!
//! Human-facing commands: `ecnode check` (what this machine would report) and `ecnode policy-template`. Both are
//! read-only. `check` never prints an environment value, a container's command line or a secret: only counts, names,
//! sizes and the policy summary. `policy-template` prints a starter MONITOR-mode policy; it writes nothing.
//!
//! Python prints both reports to stdout: `entry::main_with` hands these functions its stdout writer (`out`), so a
//! caller that captures `out` captures the report (`check_to` / `policy_template_to` take any writer, for tests).

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::agent::{Agent, VERSION};
use crate::disks::mount_of;
use crate::docker::Docker;
use crate::gpus::Gpus;
use crate::policy::{
    DEFAULT_DISCOVERY, DEFAULT_IMAGES, DEFAULT_NETWORK, DEFAULT_PORTS, PolicyStore, home_dir, parse,
};
use crate::util::{EnvMap, ProcRunner, Runner, node_env, py};

pub const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
/// Written next to engine_images in the template: tag patterns are a starting point, digest pins are the goal.
pub const IMAGES_NOTE: &str = "repository:* allows any tag of that repository, and a tag names whatever image was \
last pulled under it. Prefer exact references pinned by digest (repository@sha256:<digest>) for the images you have \
reviewed, and pin the same references for this machine in the console (host_image_pins). The console refers to \
llama.cpp by digest already.";

/// `f"{x:,.{nd}f}"`
fn fmt_commas(x: f64, nd: usize) -> String {
    let s = format!("{:.nd$}", x.abs());
    let (int, frac) = s.split_once('.').map_or((s.as_str(), ""), |(i, f)| (i, f));
    let mut grouped = String::new();
    for (i, c) in int.chars().enumerate() {
        if i > 0 && (int.len() - i) % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(c);
    }
    let sign = if x < 0.0 && s.chars().any(|c| c.is_ascii_digit() && c != '0') {
        "-"
    } else {
        ""
    };
    if frac.is_empty() {
        format!("{sign}{grouped}")
    } else {
        format!("{sign}{grouped}.{frac}")
    }
}

fn gib(n: &Value) -> String {
    let Some(v) = n.as_f64().filter(|_| !n.is_boolean()) else {
        return "?".into();
    };
    if v < 1024.0 * GIB {
        format!("{} GiB", fmt_commas(v / GIB, 1))
    } else {
        format!("{} TiB", fmt_commas(v / GIB / 1024.0, 2))
    }
}

/// f-string `{x}` of a JSON value (None for null).
fn f(v: &Value) -> String {
    py::str_of(v)
}

fn or<'a>(v: &'a Value, dflt: &'a str) -> String {
    if py::truthy(v) {
        f(v)
    } else {
        dflt.to_string()
    }
}

fn yes(v: &Value) -> &'static str {
    if py::truthy(v) { "yes" } else { "no" }
}

/// `f"{s:<n}"`
fn ljust(s: &str, n: usize) -> String {
    let len = py::len(s);
    if len >= n {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(n - len))
    }
}

/// `ecnode check [--policy PATH]`: exit code 0, or 1 when a policy file exists but was refused. The report goes
/// to stdout.
pub fn check(policy_path: Option<&str>, out: &mut dyn Write) -> i32 {
    let agent = Agent::new(PolicyStore::new(policy_path.map(str::to_string)));
    check_to(&agent, out)
}

/// Print what this machine would report to a hub. Exit 1 when a policy file exists but was refused.
pub fn check_to(agent: &Agent, out: &mut dyn Write) -> i32 {
    let hello = agent.hello(false);
    let get = |k: &str| hello.get(k).cloned().unwrap_or(Value::Null);
    let pol = get("policy");
    let host = get("host");
    let mut lines: Vec<String> = Vec::new();
    // Python: `ecnode <v> · protocol 1 · Python <x.y.z> · node id <id>`; this node has no Python (DV5), so the line
    // names the implementation instead (PORTING.md DV24: people read this line, nothing parses it)
    lines.push(format!(
        "ecnode {VERSION} (rust) · protocol {} · node id {}",
        ec_proto::PROTO,
        f(&get("node_id"))
    ));
    let cores = if py::truthy(&host["cpu_threads"]) {
        format!(
            "{} cores / {} threads",
            f(&host["cpu_cores"]),
            f(&host["cpu_threads"])
        )
    } else {
        String::new()
    };
    lines.push(format!(
        "host     {} ({}) · {} · kernel {}",
        f(&host["hostname"]),
        f(&host["arch"]),
        or(&host["os"], "unknown OS"),
        f(&host["kernel"])
    ));
    lines.push(format!(
        "         {} · {cores} · RAM {}",
        or(&host["cpu_model"], "unknown CPU"),
        gib(&host["ram_total_bytes"])
    ));
    let g = get("gpu");
    let gpus = get("gpus");
    let gpus = gpus.as_array().cloned().unwrap_or_default();
    let err = if py::truthy(&g["error"]) {
        format!(" · {}", f(&g["error"]))
    } else {
        String::new()
    };
    lines.push(format!(
        "GPUs     {} via {} · driver {} · CUDA {}{err}",
        gpus.len(),
        f(&g["source"]),
        or(&g["driver"], "-"),
        or(&g["cuda"], "-")
    ));
    for gpu in &gpus {
        let mem = if py::truthy(&gpu["unified_memory"]) {
            "unified memory".to_string()
        } else {
            gib(&gpu["total_bytes"])
        };
        lines.push(format!(
            "  {}  {} · {mem} · CC {} · {}",
            f(&gpu["index"]),
            f(&gpu["name"]),
            or(&gpu["compute_capability"], "?"),
            f(&gpu["uuid"])
        ));
    }
    let d = get("docker");
    if py::truthy(&d["available"]) {
        lines.push(format!(
            "docker   {} · context {} · rootless {} · nvidia runtime {}",
            f(&d["version"]),
            or(&d["context"], "default"),
            yes(&d["rootless"]),
            yes(&d["nvidia_runtime"])
        ));
    } else {
        lines.push(format!(
            "docker   not available: {}",
            or(&d["error"], "unknown reason")
        ));
    }
    lines.push(format!(
        "policy   {} · mode {}",
        or(&pol["path"], "(none)"),
        f(&pol["mode"]).to_uppercase()
    ));
    if py::truthy(&pol["error"]) {
        lines.push(format!("  ! {}", f(&pol["error"])));
    }
    let strs = |v: &Value| -> Vec<String> {
        v.as_array()
            .map(|a| a.iter().map(f).collect())
            .unwrap_or_default()
    };
    let images = strs(&pol["engine_images"]).join(", ");
    lines.push(format!(
        "  engine images: {}",
        if images.is_empty() {
            "(none: nothing can be launched)".to_string()
        } else {
            images
        }
    ));
    lines.push(format!(
        "  gateway image: {} · network {} · ports {}-{} · discovery ports {}",
        or(&pol["gateway_image"], "(none)"),
        f(&pol["engine_network"]),
        f(&pol["port_range"][0]),
        f(&pol["port_range"][1]),
        strs(&pol["discovery_ports"]).join(", ")
    ));
    lines.push(format!(
        "  pull {} · delete {} · ipc host {} · external control {} · max shm {} GiB",
        yes(&pol["allow_pull"]),
        yes(&pol["allow_delete"]),
        yes(&pol["allow_ipc_host"]),
        f(&pol["external_control"]),
        ec_proto::pyfmt::fmt_g(pol["max_shm_gib"].as_f64().unwrap_or(0.0))
    ));
    let disks = agent.disk.report(&agent.policy().model_roots);
    let disks = disks
        .get("mounts")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let roots = pol["model_roots"].as_array().cloned().unwrap_or_default();
    lines.push(format!(
        "roots{}",
        if roots.is_empty() { "    (none)" } else { "" }
    ));
    let mounts = agent.disk.mounts();
    for r in &roots {
        let real = agent.disk.resolve(&f(&r["path"]));
        let exists = Path::new(&real).is_dir();
        let m = mount_of(&real, &mounts);
        let info = disks
            .iter()
            .find(|x| m.is_some_and(|m| x["mount"].as_str() == Some(m.mount.as_str())));
        let where_ = if !exists {
            "missing on this machine".to_string()
        } else if let Some(info) = info {
            let remote = if py::truthy(&info["remote"]) {
                format!(
                    " from {}:{}",
                    f(&info["remote_host"]),
                    f(&info["remote_path"])
                )
            } else {
                String::new()
            };
            let model = if py::truthy(&info["model"]) {
                format!(", {}", f(&info["model"]))
            } else {
                String::new()
            };
            format!(
                "on {} ({}{remote}{model}) · {} free of {}",
                f(&info["mount"]),
                f(&info["fstype"]),
                gib(&info["free"]),
                gib(&info["size"])
            )
        } else {
            "mount unknown".to_string()
        };
        lines.push(format!(
            "  {} {} {}  ({})  {where_}",
            ljust(&f(&r["id"]), 10),
            ljust(&f(&r["kind"]), 8),
            f(&r["path"]),
            if py::truthy(&r["writable"]) {
                "writable"
            } else {
                "read-only"
            }
        ));
    }
    let scripts = pol["scripts"].as_array().cloned().unwrap_or_default();
    if !scripts.is_empty() {
        let parts: Vec<String> = scripts
            .iter()
            .map(|s| {
                let acts: Vec<String> = s["actions"]
                    .as_array()
                    .map(|a| a.iter().map(|x| f(&x["name"])).collect())
                    .unwrap_or_default();
                format!("{} ({})", f(&s["name"]), acts.join(", "))
            })
            .collect();
        lines.push(format!("scripts  {}", parts.join("; ")));
    }
    let inv = agent.refresh();
    let containers = inv
        .msg
        .get("containers")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let engines = containers
        .iter()
        .filter(|c| py::truthy(&c["engine"]) && c["role"].as_str() != Some("gateway"))
        .count();
    let owned = containers
        .iter()
        .filter(|c| py::truthy(&c["managed"]))
        .count();
    let eps = inv
        .msg
        .get("endpoints")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let running = containers
        .iter()
        .filter(|c| c["state"].as_str() == Some("running"))
        .count();
    let ports = if eps.is_empty() {
        String::new()
    } else {
        let p: Vec<String> = eps.iter().map(|e| f(&e["port"])).collect();
        format!(" on ports {}", p.join(", "))
    };
    let derr = inv
        .error
        .as_ref()
        .map(|e| format!(" · docker: {e}"))
        .unwrap_or_default();
    lines.push(format!(
        "running  {running} containers ({engines} engines, {owned} console-owned) · {} endpoint(s){ports}{derr}",
        eps.len()
    ));
    for l in lines {
        let _ = writeln!(out, "{l}");
    }
    i32::from(py::truthy(&pol["error"]) && py::truthy(&pol["path"]))
}

// ── policy-template ──────────────────────────────────────────────────────────────────────────────────────
/// Model directories that exist on this machine. HF caches are listed as their hub/ directory and Ollama homes as
/// their models/ directory, so files next to them (tokens, keys, history) are outside every root.
pub fn candidate_roots(
    env: &EnvMap,
    home: Option<&str>,
    isdir: &dyn Fn(&str) -> bool,
) -> Vec<Value> {
    let home = home
        .filter(|h| !h.is_empty())
        .map_or_else(home_dir, str::to_string);
    let mut out: Vec<Value> = Vec::new();
    let add = |out: &mut Vec<Value>, rid: &str, path: &str, kind: &str, label: &str| {
        let path = py::normpath_str(path);
        if out
            .iter()
            .any(|r| r["path"] == json!(path) || r["id"] == json!(rid))
        {
            return;
        }
        let probe = json!({"model_roots": [{"id": rid, "path": path, "kind": kind}]});
        if parse(&probe, None, Some(&home)).is_err() {
            return;
        }
        out.push(json!({"id": rid, "path": path, "kind": kind, "writable": false, "label": label}));
    };
    let get = |k: &str| env.get(k).filter(|v| !v.is_empty()).cloned();
    let join = |a: &str, parts: &[&str]| py::path_str(&py::join(Path::new(a), parts));
    let hf_home = get("HF_HOME").unwrap_or_else(|| join(&home, &[".cache", "huggingface"]));
    let hub = get("HF_HUB_CACHE")
        .or_else(|| get("HUGGINGFACE_HUB_CACHE"))
        .unwrap_or_else(|| join(&hf_home, &["hub"]));
    if isdir(&hub) {
        add(&mut out, "hf", &hub, "hf-hub", "Hugging Face cache");
    }
    let ollama = [
        get("OLLAMA_MODELS"),
        Some(join(&home, &[".ollama", "models"])),
        Some("/usr/share/ollama/.ollama/models".to_string()),
        Some("/var/lib/ollama/models".to_string()),
    ];
    for (i, base) in ollama.iter().enumerate() {
        let Some(base) = base else { continue };
        if isdir(&join(base, &["manifests"])) {
            let rid = if i == 0 || !out.iter().any(|r| r["kind"] == "ollama") {
                "ollama".to_string()
            } else {
                format!("ollama{i}")
            };
            add(&mut out, &rid, base, "ollama", "Ollama models");
        }
    }
    let plain = [
        join(&home, &["models"]),
        "/models".into(),
        "/data/models".into(),
        "/data/hf".into(),
        "/data/gguf".into(),
        "/srv/models".into(),
        "/mnt/models".into(),
        "/opt/models".into(),
    ];
    for path in plain {
        if isdir(&path) {
            let base = py::basename_str(&path).to_lowercase().replace('.', "-");
            let rid = py::take(&base, 32).to_string();
            let rid = if rid.is_empty() {
                "models".to_string()
            } else {
                rid
            };
            let rid = if out.iter().any(|r| r["id"] == json!(rid)) {
                format!("{rid}{}", out.len())
            } else {
                rid
            };
            add(&mut out, &rid, &path, "auto", &path);
        }
    }
    out
}

/// A local nginx image pinned by digest, if there is one (the operator still has to read and accept it).
fn gateway_digest(d: &Docker) -> Option<String> {
    let res = d.call(
        &[
            "images",
            "--digests",
            "--format",
            "{{.Repository}}@{{.Digest}}",
            "nginx",
        ],
        10.0,
        None,
    );
    if !res.ok() {
        return None;
    }
    for line in py::splitlines(&res.out) {
        let (repo, digest) = py::strip(line)
            .split_once('@')
            .unwrap_or((py::strip(line), ""));
        if repo == "nginx" && digest.starts_with("sha256:") && digest.len() == 71 {
            return Some(format!("nginx@{digest}"));
        }
    }
    None
}

/// The docker context the node should use. A `rootless` context wins over "default".
pub fn detect_context(runner: &dyn Runner) -> Option<String> {
    let s = |a: &[&str]| -> Vec<String> { a.iter().map(|x| (*x).to_string()).collect() };
    let shown = runner.run(&s(&["docker", "context", "show"]), 5.0, None);
    let current = if shown.ok() {
        py::strip(&shown.out).to_string()
    } else {
        String::new()
    };
    if !current.is_empty() && current != "default" {
        return Some(current);
    }
    let listed = runner.run(
        &s(&["docker", "context", "ls", "--format", "{{.Name}}"]),
        5.0,
        None,
    );
    let has = listed.ok()
        && py::splitlines(&listed.out)
            .into_iter()
            .any(|l| py::rstrip_chars(py::strip(l), " *") == "rootless");
    has.then(|| "rootless".to_string())
}

/// `date.today().isoformat()` (the node's local date, through `date`; UTC when that fails).
fn today(runner: &dyn Runner, env: &EnvMap) -> String {
    let mut child = crate::util::child_env(None, env);
    if let Some(tz) = env.get("TZ") {
        child.insert("TZ".into(), tz.clone());
    }
    let res = runner.run(
        &["date".to_string(), "+%Y-%m-%d".to_string()],
        5.0,
        Some(&child),
    );
    let t = py::strip(&res.out);
    if res.ok() && t.len() == 10 {
        return t.to_string();
    }
    let days = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() / 86400);
    civil_from_days(i64::try_from(days).unwrap_or(0))
}

fn civil_from_days(z: i64) -> String {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{:04}-{m:02}-{d:02}", if m <= 2 { y + 1 } else { y })
}

/// `ecnode policy-template`: prints a starter monitor-mode policy to stdout.
pub fn policy_template(out: &mut dyn Write) -> i32 {
    let env = node_env();
    let runner: Arc<dyn Runner> = Arc::new(ProcRunner::new(env.clone()));
    let gpus = Gpus::for_env(&env, Arc::clone(&runner));
    policy_template_to(out, &runner, &gpus, &env, None)
}

pub fn policy_template_to(
    out: &mut dyn Write,
    runner: &Arc<dyn Runner>,
    gpus: &Gpus,
    env: &EnvMap,
    home: Option<&str>,
) -> i32 {
    let context = detect_context(runner.as_ref());
    let (statics, _info) = gpus.static_info();
    let detected: Vec<Value> = statics
        .iter()
        .map(|g| {
            json!(format!(
                "{}: {} ({})",
                f(&g["index"]),
                f(&g["name"]),
                f(&g["uuid"])
            ))
        })
        .collect();
    let gateway = gateway_digest(&Docker::new("docker", context.clone(), Arc::clone(runner)));
    let isdir = |p: &str| Path::new(p).is_dir();
    let template = json!({
        "_comment": format!(
            "Starter policy from `ecnode policy-template` ({}, ecnode {VERSION}). MONITOR mode: Engine Console can \
             watch this machine but change nothing. Check every root, then set \"mode\": \"manage\" and \
             \"writable\": true where downloads may land. Reference: docs/FLEET-PROTOCOL.md section 6. Keys starting \
             with _ are comments.", today(runner.as_ref(), env)),
        "_detected_gpus": detected,
        "version": 1,
        "mode": "monitor",
        "docker": {"binary": "docker", "context": context},
        "_engine_images": IMAGES_NOTE,
        "engine_images": DEFAULT_IMAGES,
        "gateway_image": gateway,
        "engine_network": DEFAULT_NETWORK,
        "port_range": [DEFAULT_PORTS.0, DEFAULT_PORTS.1],
        "discovery_ports": DEFAULT_DISCOVERY,
        "model_roots": candidate_roots(env, home, &isdir),
        "scripts": [],
        "allow_pull": false,
        "allow_delete": false,
        "allow_ipc_host": true,
        "external_control": "none",
        "max_shm_gib": 64,
    });
    // the template must load as written
    if let Err(e) = parse(&template, None, home) {
        let _ = writeln!(
            std::io::stderr(),
            "ecnode: the policy template does not load: {e}"
        );
        return 1;
    }
    let _ = writeln!(out, "{}", py::dumps_indent(&template, 2));
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_formats_match_python() {
        assert_eq!(fmt_commas(250.8, 1), "250.8");
        assert_eq!(fmt_commas(1234.56, 1), "1,234.6");
        assert_eq!(fmt_commas(1_234_567.0, 2), "1,234,567.00");
        assert_eq!(gib(&json!(263_000_000_i64 * 1024)), "250.8 GiB");
        assert_eq!(gib(&json!(null)), "?");
        assert_eq!(civil_from_days(0), "1970-01-01");
        assert_eq!(civil_from_days(20_753), "2026-10-27");
    }
}
