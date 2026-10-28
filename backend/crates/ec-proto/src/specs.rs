//! Every message, argument, result and event shape of protocol v1, transcribed from proto.py in the same field
//! order. Additive changes only (backend/PORTING.md "Wire protocol"): a new op, a new optional field, a new type.

use std::sync::LazyLock;

use crate::MAX_FILE_CONTENT;
use crate::MAX_LOG_TEXT;
use crate::MAX_METRICS_TEXT;
use crate::clean::{Spec, l, m, o, s};

use Spec::{Bool as BOOL, Int as INT, Json as JSONV, Num as NUM};

// ── node -> hub: pushed messages ────────────────────────────────────────────────────────────────────────────
pub fn gpu_static() -> Spec {
    o(&[
        ("index", INT),
        ("uuid", s(96)),
        ("name", s(128)),
        ("total_bytes", INT),
        ("compute_capability", s(16)),
        ("unified_memory", BOOL),
        ("pci_bus_id", s(32)),
    ])
}

pub fn root() -> Spec {
    o(&[
        ("id", s(32)),
        ("path", s(512)),
        ("kind", s(16)),
        ("writable", BOOL),
        ("label", s(64)),
    ])
}

pub fn script_action() -> Spec {
    o(&[
        ("name", s(32)),
        ("label", s(64)),
        ("args", l(s(64), 32)),
        ("confirm", BOOL),
        ("danger", BOOL),
    ])
}

pub fn script() -> Spec {
    o(&[
        ("name", s(32)),
        ("label", s(64)),
        ("engine", s(32)),
        ("description", s(500)),
        ("actions", l(script_action(), 16)),
        ("containers", l(s(128), 16)),
        ("status_action", s(32)),
    ])
}

pub fn policy() -> Spec {
    o(&[
        ("path", s(512)),
        ("mode", s(16)),
        ("error", s(500)),
        ("engine_images", l(s(256), 64)),
        ("gateway_image", s(256)),
        ("engine_network", s(64)),
        ("port_range", l(INT, 2)),
        ("discovery_ports", l(INT, 128)),
        ("model_roots", l(root(), 32)),
        ("scripts", l(script(), 32)),
        ("allow_pull", BOOL),
        ("allow_delete", BOOL),
        ("allow_ipc_host", BOOL),
        ("external_control", s(16)),
        ("max_shm_gib", NUM),
    ])
}

pub fn hello() -> Spec {
    o(&[
        ("proto", INT),
        ("version", s(32)),
        ("node_id", s(64)),
        (
            "host",
            o(&[
                ("hostname", s(253)),
                ("arch", s(32)),
                ("os", s(128)),
                ("kernel", s(128)),
                ("cpu_model", s(128)),
                ("cpu_cores", INT),
                ("cpu_threads", INT),
                ("ram_total_bytes", INT),
                ("boot_time", NUM),
                ("python", s(32)),
            ]),
        ),
        (
            "gpu",
            o(&[
                ("driver", s(64)),
                ("cuda", s(32)),
                ("source", s(16)),
                ("error", s(500)),
            ]),
        ),
        ("gpus", l(gpu_static(), 16)),
        (
            "docker",
            o(&[
                ("available", BOOL),
                ("context", s(64)),
                ("rootless", BOOL),
                ("version", s(64)),
                ("nvidia_runtime", BOOL),
                ("error", s(500)),
            ]),
        ),
        (
            "addresses",
            l(o(&[("iface", s(32)), ("addr", s(64)), ("prefix", INT)]), 64),
        ),
        ("policy", policy()),
        ("forced_cmd", BOOL),
        ("via_ssh", BOOL),
    ])
}

pub fn gpu_proc() -> Spec {
    o(&[
        ("pid", INT),
        ("used_bytes", INT),
        ("container", s(128)),
        ("name", s(64)),
    ])
}

pub fn gpu_sample() -> Spec {
    o(&[
        ("util_pct", NUM),
        ("mem_used_bytes", INT),
        ("mem_total_bytes", INT),
        ("temp_c", NUM),
        ("power_w", NUM),
        ("power_limit_w", NUM),
        ("fan_pct", NUM),
        ("procs", l(gpu_proc(), 32)),
    ])
}

pub fn mem() -> Spec {
    o(&[
        ("total", INT),
        ("available", INT),
        ("free", INT),
        ("cached", INT),
        ("shmem", INT),
        ("swap_total", INT),
        ("swap_free", INT),
    ])
}

pub fn cg_mem() -> Spec {
    o(&[
        ("current", INT),
        ("anon", INT),
        ("file", INT),
        ("shmem", INT),
        ("kernel", INT),
        ("source", s(8)),
    ])
}

pub fn sample() -> Spec {
    o(&[
        ("seq", INT),
        ("t", NUM),
        ("gpus", m(gpu_sample(), 16)),
        ("mem", mem()),
        (
            "cpu",
            o(&[
                ("load1", NUM),
                ("load5", NUM),
                ("load15", NUM),
                ("util_pct", NUM),
            ]),
        ),
        (
            "containers",
            m(o(&[("mem", cg_mem()), ("cpu_pct", NUM)]), 256),
        ),
    ])
}

pub fn mount() -> Spec {
    o(&[
        ("mount", s(512)),
        ("fstype", s(32)),
        ("source", s(512)),
        ("device", s(64)),
        ("model", s(128)),
        ("size", INT),
        ("used", INT),
        ("free", INT),
        ("readonly", BOOL),
        ("remote", BOOL),
        ("remote_host", s(253)),
        ("remote_path", s(512)),
        ("roots", l(s(32), 32)),
    ])
}

pub fn disks() -> Spec {
    o(&[("t", NUM), ("mounts", l(mount(), 64))])
}

pub fn port_map() -> Spec {
    o(&[
        ("host_ip", s(64)),
        ("host_port", INT),
        ("container_port", INT),
    ])
}

pub fn container_mount() -> Spec {
    o(&[
        ("source", s(512)),
        ("dest", s(512)),
        ("ro", BOOL),
        ("root", s(32)),
        ("rel", s(1024)),
    ])
}

pub fn container() -> Spec {
    o(&[
        ("name", s(128)),
        ("id", s(64)),
        ("image", s(256)),
        ("image_id", s(80)),
        ("state", s(16)),
        ("status", s(128)),
        ("created", NUM),
        ("started_at", NUM),
        ("pid", INT),
        ("exit_code", INT),
        ("labels", m(s(256), 48)),
        ("engine", s(32)),
        ("managed", BOOL),
        ("role", s(16)),
        ("instance", s(64)),
        ("ports", l(port_map(), 20)),
        ("argv", l(s(1024), 400)),
        ("params", m(s(200), 60)),
        ("gpus", l(s(96), 16)),
        ("all_gpus", BOOL),
        ("mounts", l(container_mount(), 32)),
        ("network_mode", s(64)),
        ("compose_project", s(128)),
        ("compose_service", s(128)),
    ])
}

pub fn endpoint() -> Spec {
    o(&[
        ("port", INT),
        ("container", s(128)),
        ("engine", s(32)),
        ("models", l(s(200), 20)),
        ("health_status", INT),
        ("models_status", INT),
        ("metrics_status", INT),
        ("auth_required", BOOL),
    ])
}

pub fn script_status() -> Spec {
    o(&[
        ("name", s(32)),
        ("status", s(16)),
        ("detail", s(500)),
        ("checked_at", NUM),
    ])
}

pub fn inventory() -> Spec {
    o(&[
        ("t", NUM),
        ("containers", l(container(), 256)),
        ("endpoints", l(endpoint(), 64)),
        ("scripts", l(script_status(), 32)),
    ])
}

pub fn metrics() -> Spec {
    o(&[
        ("t", NUM),
        ("port", INT),
        ("container", s(128)),
        ("engine", s(32)),
        ("status", INT),
        ("text", s(MAX_METRICS_TEXT)),
    ])
}

pub fn models_changed() -> Spec {
    o(&[("generation", INT), ("t", NUM)])
}

pub fn log() -> Spec {
    o(&[("level", s(8)), ("msg", s(1000))])
}

pub fn fatal() -> Spec {
    o(&[("error", s(1000))])
}

pub fn pong() -> Spec {
    o(&[("id", INT)])
}

pub fn credit() -> Spec {
    o(&[("id", INT), ("bytes", INT)])
}

pub fn error() -> Spec {
    o(&[("code", s(64)), ("message", s(1000))])
}

/// Pushed message types the hub knows, in proto.py's PUSH order.
pub const PUSH_TYPES: &[&str] = &[
    "hello",
    "sample",
    "disks",
    "inventory",
    "metrics",
    "models_changed",
    "log",
    "fatal",
    "pong",
    "credit",
];

static PUSH: LazyLock<Vec<(&'static str, Spec)>> = LazyLock::new(|| {
    vec![
        ("hello", hello()),
        ("sample", sample()),
        ("disks", disks()),
        ("inventory", inventory()),
        ("metrics", metrics()),
        ("models_changed", models_changed()),
        ("log", log()),
        ("fatal", fatal()),
        ("pong", pong()),
        ("credit", credit()),
    ]
});

/// `PUSH.get(t)`
pub fn push(t: &str) -> Option<&'static Spec> {
    PUSH.iter().find(|(k, _)| *k == t).map(|(_, s)| s)
}

// ── hub -> node: request arguments ──────────────────────────────────────────────────────────────────────────
pub fn run_mount() -> Spec {
    o(&[
        ("root", s(32)),
        ("rel", s(1024)),
        ("dest", s(512)),
        ("ro", BOOL),
    ])
}

pub fn run_volume() -> Spec {
    o(&[("name", s(64)), ("dest", s(512))])
}

pub fn run_spec() -> Spec {
    o(&[
        ("name", s(64)),
        ("instance", s(64)),
        ("image", s(256)),
        ("argv", l(s(4096), 512)),
        ("container_port", INT),
        ("gpu_uuids", l(s(96), 16)),
        ("network_alias", s(64)),
        ("labels", m(s(256), 32)),
        ("env", m(s(4096), 128)),
        ("secret_env", m(s(4096), 16)),
        ("mounts", l(run_mount(), 16)),
        ("volumes", l(run_volume(), 8)),
        ("shm_size", s(16)),
        ("ipc_host", BOOL),
        ("files", m(s(MAX_FILE_CONTENT), 4)),
        ("ulimits", m(s(32), 4)),
    ])
}

/// Every op, in proto.py's ARGS order.
pub const OPS: &[&str] = &[
    "node.info",
    "metrics.configure",
    "docker.render",
    "docker.run",
    "docker.stop",
    "docker.start",
    "docker.rm",
    "docker.inspect",
    "docker.logs",
    "docker.logs_follow",
    "docker.image_present",
    "docker.images",
    "docker.pull",
    "docker.external",
    "net.ensure_internal",
    "gateway.start",
    "http.request",
    "fs.stat",
    "fs.write",
    "fs.read",
    "fs.symlink",
    "fs.put_small",
    "fs.delete",
    "models.scan",
    "models.list",
    "models.config",
    "scripts.run",
];

/// Allowed while the node's policy says mode "monitor": nothing here changes the machine.
pub const MONITOR_OPS: &[&str] = &[
    "node.info",
    "metrics.configure",
    "docker.render",
    "docker.inspect",
    "docker.logs",
    "docker.logs_follow",
    "docker.image_present",
    "docker.images",
    "http.request",
    "fs.stat",
    "fs.read",
    "models.scan",
    "models.list",
    "models.config",
];

/// May receive DATA from the hub (http.request only if body_len > 0).
pub const UPLOAD_OPS: &[&str] = &["fs.write", "http.request"];

pub fn is_op(op: &str) -> bool {
    OPS.contains(&op)
}

pub fn is_monitor_op(op: &str) -> bool {
    MONITOR_OPS.contains(&op)
}

static ARGS: LazyLock<Vec<(&'static str, Spec)>> = LazyLock::new(|| {
    vec![
        ("node.info", o(&[])),
        (
            "metrics.configure",
            o(&[("interval_s", NUM), ("families", m(l(s(64), 32), 16))]),
        ),
        ("docker.render", o(&[("spec", run_spec())])),
        ("docker.run", o(&[("spec", run_spec())])),
        (
            "docker.stop",
            o(&[
                ("name", s(128)),
                ("timeout_s", INT),
                ("role", s(16)),
                ("instance", s(64)),
            ]),
        ),
        (
            "docker.start",
            o(&[("name", s(128)), ("role", s(16)), ("instance", s(64))]),
        ),
        (
            "docker.rm",
            o(&[("name", s(128)), ("role", s(16)), ("instance", s(64))]),
        ),
        ("docker.inspect", o(&[("name", s(128))])),
        ("docker.logs", o(&[("name", s(128)), ("tail", INT)])),
        ("docker.logs_follow", o(&[("name", s(128)), ("tail", INT)])),
        ("docker.image_present", o(&[("image", s(256))])),
        ("docker.images", o(&[])),
        ("docker.pull", o(&[("image", s(256))])),
        ("docker.external", o(&[("name", s(128)), ("action", s(16))])),
        ("net.ensure_internal", o(&[])),
        (
            "gateway.start",
            o(&[
                ("name", s(128)),
                ("instance", s(64)),
                ("host_port", INT),
                ("engine_host", s(64)),
                ("engine_port", INT),
            ]),
        ),
        (
            "http.request",
            o(&[
                ("port", INT),
                ("method", s(8)),
                ("path", s(2048)),
                ("headers", m(s(4096), 32)),
                ("body_len", INT),
                ("timeout_s", NUM),
            ]),
        ),
        ("fs.stat", o(&[("root", s(32)), ("paths", l(s(1024), 256))])),
        (
            "fs.write",
            o(&[
                ("root", s(32)),
                ("rel", s(1024)),
                ("size", INT),
                ("sha256", s(64)),
                ("offset", INT),
            ]),
        ),
        (
            "fs.read",
            o(&[
                ("root", s(32)),
                ("rel", s(1024)),
                ("offset", INT),
                ("length", INT),
            ]),
        ),
        (
            "fs.symlink",
            o(&[("root", s(32)), ("link", s(1024)), ("target", s(1024))]),
        ),
        (
            "fs.put_small",
            o(&[
                ("root", s(32)),
                ("rel", s(1024)),
                ("content_b64", s(128 * 1024)),
            ]),
        ),
        ("fs.delete", o(&[("root", s(32)), ("rel", s(1024))])),
        (
            "models.scan",
            o(&[("roots", l(s(32), 32)), ("force", BOOL)]),
        ),
        ("models.list", o(&[("cursor", INT), ("limit", INT)])),
        ("models.config", o(&[("root", s(32)), ("rel", s(1024))])),
        (
            "scripts.run",
            o(&[("name", s(32)), ("action", s(32)), ("arg", s(64))]),
        ),
    ]
});

/// `ARGS.get(op)`
pub fn args(op: &str) -> Option<&'static Spec> {
    ARGS.iter().find(|(k, _)| *k == op).map(|(_, s)| s)
}

/// `REQUIRED.get(op, ())`
pub fn required(op: &str) -> &'static [&'static str] {
    match op {
        "metrics.configure" => &["families"],
        "docker.render" | "docker.run" => &["spec"],
        "docker.stop" | "docker.start" | "docker.rm" | "docker.inspect" | "docker.logs"
        | "docker.logs_follow" => &["name"],
        "docker.image_present" | "docker.pull" => &["image"],
        "docker.external" | "scripts.run" => &["name", "action"],
        "gateway.start" => &[
            "name",
            "instance",
            "host_port",
            "engine_host",
            "engine_port",
        ],
        "http.request" => &["port", "method", "path"],
        "fs.stat" => &["root", "paths"],
        "fs.write" => &["root", "rel", "size"],
        "fs.read" | "fs.delete" | "models.config" => &["root", "rel"],
        "fs.symlink" => &["root", "link", "target"],
        "fs.put_small" => &["root", "rel", "content_b64"],
        _ => &[],
    }
}

/// Ops that have a REQUIRED entry (for the contract tests).
pub const REQUIRED_OPS: &[&str] = &[
    "metrics.configure",
    "docker.render",
    "docker.run",
    "docker.stop",
    "docker.start",
    "docker.rm",
    "docker.inspect",
    "docker.logs",
    "docker.logs_follow",
    "docker.image_present",
    "docker.pull",
    "docker.external",
    "gateway.start",
    "http.request",
    "fs.stat",
    "fs.write",
    "fs.read",
    "fs.symlink",
    "fs.put_small",
    "fs.delete",
    "models.config",
    "scripts.run",
];

// ── node -> hub: results and events per op ──────────────────────────────────────────────────────────────────
pub fn image() -> Spec {
    o(&[
        ("ref", s(256)),
        ("id", s(80)),
        ("size", INT),
        ("created", NUM),
        ("arch", s(16)),
    ])
}

pub fn model_entry() -> Spec {
    o(&[
        ("root", s(32)),
        ("rel", s(1024)),
        ("format", s(16)),
        ("name", s(256)),
        ("revision", s(64)),
        ("size", INT),
        ("files", INT),
        ("mtime", NUM),
        ("atime", NUM),
        ("quant", s(32)),
        ("params", INT),
        ("arch", s(64)),
        ("context", INT),
        ("shards", INT),
        ("meta", JSONV),
        ("warnings", l(s(300), 8)),
    ])
}

pub fn root_scan() -> Spec {
    o(&[
        ("root", s(32)),
        ("scanned_at", NUM),
        ("complete", BOOL),
        ("entries", INT),
        ("error", s(500)),
        ("duration_s", NUM),
    ])
}

static RESULTS: LazyLock<Vec<(&'static str, Spec)>> = LazyLock::new(|| {
    vec![
        ("node.info", hello()),
        ("metrics.configure", o(&[])),
        ("docker.render", o(&[("argv", l(s(4096), 1024))])),
        ("docker.run", o(&[("id", s(64))])),
        ("docker.stop", o(&[])),
        ("docker.start", o(&[])),
        ("docker.rm", o(&[])),
        ("docker.inspect", o(&[("info", container())])),
        (
            "docker.logs",
            o(&[("text", s(MAX_LOG_TEXT)), ("truncated", BOOL)]),
        ),
        ("docker.logs_follow", o(&[("lines", INT)])),
        (
            "docker.image_present",
            o(&[("present", BOOL), ("id", s(80))]),
        ),
        ("docker.images", o(&[("images", l(image(), 512))])),
        ("docker.pull", o(&[("id", s(80))])),
        ("docker.external", o(&[])),
        ("net.ensure_internal", o(&[("network", s(64))])),
        ("gateway.start", o(&[("id", s(64))])),
        ("http.request", o(&[("status", INT), ("bytes", INT)])),
        (
            "fs.stat",
            o(&[(
                "items",
                l(
                    o(&[
                        ("rel", s(1024)),
                        ("exists", BOOL),
                        ("is_dir", BOOL),
                        ("size", INT),
                        ("partial", INT),
                    ]),
                    256,
                ),
            )]),
        ),
        ("fs.write", o(&[("size", INT), ("sha256", s(64))])),
        ("fs.read", o(&[("size", INT)])),
        ("fs.symlink", o(&[])),
        ("fs.put_small", o(&[])),
        ("fs.delete", o(&[("freed", INT)])),
        ("models.scan", o(&[("generation", INT)])),
        (
            "models.list",
            o(&[
                ("generation", INT),
                ("entries", l(model_entry(), 200)),
                ("next_cursor", INT),
                ("roots", l(root_scan(), 32)),
            ]),
        ),
        ("models.config", o(&[("config", JSONV)])),
        ("scripts.run", o(&[("rc", INT)])),
    ]
});

/// `RESULTS.get(op, {})`
pub fn results(op: &str) -> &'static Spec {
    static EMPTY: LazyLock<Spec> = LazyLock::new(|| o(&[]));
    RESULTS
        .iter()
        .find(|(k, _)| *k == op)
        .map_or(&*EMPTY, |(_, s)| s)
}

/// Ops with a RESULTS entry (for the contract tests).
pub fn result_ops() -> Vec<&'static str> {
    RESULTS.iter().map(|(k, _)| *k).collect()
}

static EVENTS: LazyLock<Vec<(&'static str, Spec)>> = LazyLock::new(|| {
    vec![
        (
            "http.request",
            o(&[("status", INT), ("headers", m(s(4096), 64))]),
        ),
        ("docker.logs_follow", o(&[("line", s(8192))])),
        ("docker.pull", o(&[("line", s(1024))])),
        ("scripts.run", o(&[("stream", s(8)), ("line", s(8192))])),
    ]
});

/// `EVENTS.get(op, {})`
pub fn events(op: &str) -> &'static Spec {
    static EMPTY: LazyLock<Spec> = LazyLock::new(|| o(&[]));
    EVENTS
        .iter()
        .find(|(k, _)| *k == op)
        .map_or(&*EMPTY, |(_, s)| s)
}

pub fn event_ops() -> Vec<&'static str> {
    EVENTS.iter().map(|(k, _)| *k).collect()
}

pub fn arg_ops() -> Vec<&'static str> {
    ARGS.iter().map(|(k, _)| *k).collect()
}
