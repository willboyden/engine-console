//! Python: backend/src/engine_console/services/runspec.py.
//!
//! The node's `docker.run` spec (ecnode.proto.RUN_SPEC) for one engine launch, and the model mounts it needs.
//!
//! Pure functions: no I/O. A spec built here is validated against the same rules the node applies
//! (docs/FLEET-PROTOCOL.md §5) BEFORE it is sent, so a mistake is a clear 422 here instead of the node silently
//! dropping a field (proto.clean() drops what it cannot keep) or refusing with a generic message.
//!
//! Mounts name a policy model root and a path inside it (`root` + `rel`), never a host path: the node resolves them.
//! An empty `rel` means the root itself. Every mount is read-only (security review H3).

use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Value, json};

use crate::adapters::base::{LaunchSpec, MODEL_MOUNT};
use crate::domain::errors::{Obj, Problem, unprocessable};
use crate::domain::refs::ModelRef;
use crate::domain::repo_id::check_repo_id;

fn re(p: &str) -> Regex {
    Regex::new(p).unwrap_or_else(|_| unreachable!())
}

static CONTAINER_NAME: LazyLock<Regex> = LazyLock::new(|| re(r"^ec-[a-z0-9][a-z0-9-]{0,58}$"));
static INSTANCE_ID: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z0-9_-]{1,64}$"));
static ENV_NAME: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$"));
static VOLUME_NAME: LazyLock<Regex> = LazyLock::new(|| re(r"^ec-[a-z0-9][a-z0-9_.-]{0,60}$"));
static SHM_SIZE: LazyLock<Regex> = LazyLock::new(|| re(r"^([0-9]{1,4})([kmg])$"));
static OLLAMA_NAME: LazyLock<Regex> =
    LazyLock::new(|| re(r"^[A-Za-z0-9][A-Za-z0-9_.:/@\-]{0,127}$"));
/// A variable whose NAME looks secret travels in `secret_env` (docker only ever sees its name on the command line).
pub static SECRET_ENV: LazyLock<Regex> = LazyLock::new(|| re(r"(?i)(key|token|secret|password)"));
static SHARD_SUFFIX: LazyLock<Regex> = LazyLock::new(|| re(r"-0*1-of-0*[0-9]+$"));
static ROOT_ID: LazyLock<Regex> = LazyLock::new(|| re(r"^[A-Za-z0-9_.\-]{1,32}$"));
static ULIMIT_VALUE: LazyLock<Regex> =
    LazyLock::new(|| re(r"^(?:-1|[0-9]{1,15})(?::(?:-1|[0-9]{1,15}))?$"));

pub const OLLAMA_MODELS_DEST: &str = "/root/.ollama/models";
pub const MAX_FILES: usize = 4;
pub const MAX_FILE_BYTES: usize = 128 * 1024;
/// mirrors the node's allowlist (ecnode/docker.py), sorted
pub const ULIMITS: &[&str] = &["nofile", "stack"];

/// `str(v)` of a JSON-decoded value.
fn py_str(v: &Value) -> String {
    ec_proto::pyfmt::str_value(v)
}

/// A regex `fullmatch` that also refuses a trailing newline (Rust's `$` is end of text: it already does).
fn full(re: &Regex, s: &str) -> bool {
    re.is_match(s)
}

// ---- the model reference ------------------------------------------------------------------------------------------
/// The node's path rule for `rel`: relative, no backslash or NUL, no empty / `.` / `..` segment.
pub fn rel_ok(rel: &str) -> bool {
    if rel.is_empty() || rel.starts_with('/') || rel.contains('\\') || rel.contains('\0') {
        return false;
    }
    rel.split('/').all(|seg| !matches!(seg, "" | "." | ".."))
}

/// Validate a ModelRef's fields for its format (422 `invalid_model`). Identifiers are matched with fullmatch.
pub fn check_ref(r: &ModelRef) -> Result<(), Problem> {
    let mut bad = String::new();
    match r.format.as_str() {
        "hf" => {
            if let Err(e) = check_repo_id(r.repo_id.as_deref().unwrap_or("")) {
                bad = e.to_string();
            }
        }
        fmt @ ("gguf" | "plain") => {
            let rel = r.rel.as_deref().unwrap_or("");
            if !r.root_id.as_deref().is_some_and(|id| full(&ROOT_ID, id)) {
                bad = format!("a {fmt} model needs the id of the model root it lives in");
            } else if fmt == "gguf"
                && !(!rel.is_empty() && rel_ok(rel) && rel.to_lowercase().ends_with(".gguf"))
            {
                bad = "a gguf model needs `rel`: the path of the .gguf file (or its first shard) inside the root"
                    .into();
            } else if fmt == "plain" && !rel.is_empty() && !rel_ok(rel) {
                bad = "`rel` must be a relative path without '..' segments".into();
            }
        }
        "ollama" => {
            if let Some(name) = r.name.as_deref()
                && !name.is_empty()
                && !full(&OLLAMA_NAME, name)
            {
                bad = "an ollama model name looks like 'llama3:8b'".into();
            }
        }
        _ => {}
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(unprocessable(bad).code("invalid_model"))
    }
}

/// `posixpath.basename`.
fn basename(p: &str) -> &str {
    p.rsplit('/').next().unwrap_or(p)
}

/// `posixpath.dirname`.
fn dirname(p: &str) -> String {
    match p.rfind('/') {
        None => String::new(),
        Some(i) => {
            let head = &p[..=i];
            let stripped = head.trim_end_matches('/');
            if stripped.is_empty() {
                head.to_string()
            } else {
                stripped.to_string()
            }
        }
    }
}

/// The name an engine serves the model under and the UI shows: what chat sends as `model`.
pub fn display_of(r: &ModelRef) -> String {
    match r.format.as_str() {
        "hf" => r
            .repo_id
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "model".into()),
        "ollama" => r
            .name
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "ollama".into()),
        fmt => {
            let rel = r.rel.as_deref().unwrap_or("");
            let mut base = basename(rel.trim_end_matches('/')).to_string();
            if base.is_empty() {
                base = r
                    .root_id
                    .clone()
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "model".into());
            }
            if fmt == "gguf" {
                if base.to_lowercase().ends_with(".gguf") {
                    let n = base.chars().count();
                    base = base.chars().take(n.saturating_sub(5)).collect();
                }
                base = SHARD_SUFFIX.replace(&base, "").into_owned();
            }
            if base.is_empty() {
                "model".into()
            } else {
                base
            }
        }
    }
}

// ---- mounts -------------------------------------------------------------------------------------------------------
/// "home": the root is an HF_HOME (models under `<root>/hub/`); "hub": the root is the hub cache itself.
pub fn hf_layout(root_kind: Option<&str>, copy_rel: &str) -> &'static str {
    match root_kind {
        Some("hf-hub") => "hub",
        Some("hf-home") => "home",
        _ if copy_rel.starts_with("models--") => "hub",
        _ => "home",
    }
}

fn mount(root: Option<&str>, rel: &str, dest: &str) -> Obj {
    let mut m = Obj::new();
    m.insert("root".into(), root.map_or(Value::Null, |r| json!(r)));
    m.insert("rel".into(), json!(rel));
    m.insert("dest".into(), json!(dest));
    m.insert("ro".into(), json!(true));
    m
}

/// The host's HF hub cache, read-only, at `<HF_HOME>/hub` in the engine. Only the `hub/` folder of an HF_HOME root
/// is mounted: engines run offline and read nothing else, and the HF_HOME itself holds `token` and
/// `stored_tokens`, which the node refuses to mount (security review M5). Every mount is read-only (H3).
pub fn hf_mounts(
    root_id: &str,
    root_kind: Option<&str>,
    copy_rel: &str,
    hf_path: &str,
) -> Vec<Obj> {
    let rel = if hf_layout(root_kind, copy_rel) == "home" {
        "hub"
    } else {
        ""
    };
    vec![mount(
        Some(root_id),
        rel,
        &format!("{}/hub", hf_path.trim_end_matches('/')),
    )]
}

/// (mounts, model_path) for the non-HF formats (HF is mounted by `hf_mounts`).
///
/// gguf: the directory holding the file, so every shard of a split set is visible; model_path is the file.
/// plain: the model directory at MODEL_MOUNT. ollama: the models dir of an Ollama home, or the home's `models/`
/// (`ollama_root` is (root id, root path)).
pub fn model_mounts(
    r: &ModelRef,
    copy_rel: Option<&str>,
    ollama_root: Option<(&str, &str)>,
) -> (Vec<Obj>, Option<String>) {
    match r.format.as_str() {
        "gguf" => {
            let rel = r.rel.as_deref().unwrap_or("");
            if let Some((repo, inner)) = rel.split_once("/snapshots/")
                && !repo.is_empty()
                && !inner.is_empty()
            {
                // A quant inside an HF cache repo: its snapshot file is a relative link into the repo's `blobs/`, so
                // the whole repo directory is mounted and the model path keeps the snapshot path below it.
                return (
                    vec![mount(r.root_id.as_deref(), repo, MODEL_MOUNT)],
                    Some(format!("{MODEL_MOUNT}/snapshots/{inner}")),
                );
            }
            (
                vec![mount(r.root_id.as_deref(), &dirname(rel), MODEL_MOUNT)],
                Some(format!("{MODEL_MOUNT}/{}", basename(rel))),
            )
        }
        "plain" => (
            vec![mount(
                r.root_id.as_deref(),
                r.rel.as_deref().unwrap_or(""),
                MODEL_MOUNT,
            )],
            Some(MODEL_MOUNT.to_string()),
        ),
        "ollama" => {
            let Some((root_id, root_path)) = ollama_root else {
                return (Vec::new(), None);
            };
            let cr = copy_rel.unwrap_or("");
            let first = cr.split('/').next().unwrap_or("");
            let rel = if !cr.is_empty() && first == "manifests" {
                ""
            } else if !cr.is_empty() && first == "models" {
                "models"
            } else if basename(root_path.trim_end_matches('/')) == "models" {
                ""
            } else {
                "models"
            };
            (vec![mount(Some(root_id), rel, OLLAMA_MODELS_DEST)], None)
        }
        _ => (Vec::new(), None),
    }
}

// ---- the spec -----------------------------------------------------------------------------------------------------
/// (plain env, secret env). A variable whose NAME looks secret travels in `secret_env` (docker only ever sees its
/// name on the command line); everything else is plain. `HF_HUB_OFFLINE=1` unless an adapter set it.
pub fn split_env(launch_env: &Obj, offline: bool) -> (Obj, Vec<(String, String)>) {
    let mut env = Obj::new();
    let mut secret: Vec<(String, String)> = Vec::new();
    for (k, v) in launch_env {
        let v = py_str(v);
        if SECRET_ENV.is_match(k) {
            if let Some(slot) = secret.iter_mut().find(|(kk, _)| kk == k) {
                slot.1 = v;
            } else {
                secret.push((k.clone(), v));
            }
        } else {
            env.insert(k.clone(), Value::String(v));
        }
    }
    if offline && !env.contains_key("HF_HUB_OFFLINE") {
        env.insert("HF_HUB_OFFLINE".into(), json!("1"));
    }
    (env, secret)
}

/// What `engine_labels` is built from.
#[derive(Debug, Clone, Copy)]
pub struct LabelArgs<'a> {
    pub instance: &'a str,
    pub engine: &'a str,
    pub display: &'a str,
    pub r: &'a ModelRef,
    pub port: i64,
    pub name: &'a str,
    pub gpu_ids: &'a [i64],
    pub gpu_uuids: &'a [String],
}

/// `engine-console.*` labels (the node adds `engine-console=1`). Values are capped at 256 characters by the node,
/// so a value that would not fit is left out rather than truncated into something misleading.
pub fn engine_labels(a: &LabelArgs<'_>) -> Obj {
    let ids: Vec<String> = a.gpu_ids.iter().map(ToString::to_string).collect();
    let mut labels: Vec<(&str, String)> = vec![
        ("engine-console.role", "engine".into()),
        ("engine-console.instance", a.instance.into()),
        ("engine-console.engine", a.engine.into()),
        ("engine-console.model", a.display.into()),
        ("engine-console.port", a.port.to_string()),
        ("engine-console.name", a.name.into()),
        ("engine-console.format", a.r.format.clone()),
        ("engine-console.gpu_ids", ids.join(",")),
    ];
    for (key, val) in [
        ("engine-console.gpus", a.gpu_uuids.join(",")),
        (
            "engine-console.root",
            a.r.root_id.clone().unwrap_or_default(),
        ),
        ("engine-console.rel", a.r.rel.clone().unwrap_or_default()),
    ] {
        if !val.is_empty() && val.chars().count() <= 256 {
            labels.push((key, val));
        }
    }
    labels
        .into_iter()
        .filter(|(_, v)| !v.is_empty() && v.chars().count() <= 256)
        .map(|(k, v)| (k.to_string(), Value::String(v)))
        .collect()
}

/// What `build_run_spec` is built from.
#[derive(Debug, Clone, Copy)]
pub struct RunArgs<'a> {
    pub name: &'a str,
    pub instance: &'a str,
    pub image: &'a str,
    pub launch: &'a LaunchSpec,
    pub gpu_uuids: &'a [String],
    pub labels: &'a Obj,
    pub env: &'a Obj,
    pub secret_env: &'a Obj,
    pub mounts: &'a [Obj],
    pub volumes: &'a [(String, String)],
    pub ipc_host: bool,
}

/// The `docker.run` spec, keys in Python's order (`files` / `ulimits` only when the adapter set them).
pub fn build_run_spec(a: &RunArgs<'_>) -> Obj {
    let mut spec = Obj::new();
    spec.insert("name".into(), json!(a.name));
    spec.insert("instance".into(), json!(a.instance));
    spec.insert("image".into(), json!(a.image));
    spec.insert("argv".into(), json!(a.launch.argv));
    spec.insert("container_port".into(), json!(a.launch.container_port));
    spec.insert("gpu_uuids".into(), json!(a.gpu_uuids));
    spec.insert("network_alias".into(), json!(a.name));
    spec.insert("labels".into(), Value::Object(a.labels.clone()));
    spec.insert("env".into(), Value::Object(a.env.clone()));
    spec.insert("secret_env".into(), Value::Object(a.secret_env.clone()));
    spec.insert(
        "mounts".into(),
        Value::Array(a.mounts.iter().cloned().map(Value::Object).collect()),
    );
    spec.insert(
        "volumes".into(),
        Value::Array(
            a.volumes
                .iter()
                .map(|(n, d)| json!({"name": n, "dest": d}))
                .collect(),
        ),
    );
    spec.insert("shm_size".into(), json!(a.launch.shm_size));
    spec.insert("ipc_host".into(), json!(a.ipc_host));
    if !a.launch.files.is_empty() {
        spec.insert("files".into(), Value::Object(a.launch.files.clone()));
    }
    if !a.launch.ulimits.is_empty() {
        spec.insert("ulimits".into(), Value::Object(a.launch.ulimits.clone()));
    }
    spec
}

fn obj_or_empty(v: Option<&Value>) -> Obj {
    match v {
        Some(Value::Object(o)) => o.clone(),
        _ => Obj::new(),
    }
}

fn arr_or_empty(v: Option<&Value>) -> Vec<Value> {
    match v {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    }
}

fn str_of(o: &Obj, k: &str) -> String {
    o.get(k).map_or_else(String::new, py_str)
}

/// The node's validation rules (docs/FLEET-PROTOCOL.md §5), evaluated hub-side. Empty = acceptable.
#[allow(clippy::too_many_lines)]
pub fn validate_run_spec(spec: &Obj, max_shm_gib: Option<f64>) -> Vec<String> {
    let mut errs: Vec<String> = Vec::new();
    if !full(&CONTAINER_NAME, &str_of(spec, "name")) {
        errs.push("container name must look like ec-<letters, digits, hyphens>".into());
    }
    if !full(&INSTANCE_ID, &str_of(spec, "instance")) {
        errs.push("instance id has characters the node does not accept".into());
    }
    let argv = arr_or_empty(spec.get("argv"));
    if argv.len() > 512
        || argv.iter().any(|a| {
            let s = py_str(a);
            s.chars().count() > 4096 || s.contains('\0')
        })
    {
        errs.push(
            "the engine command line is too long (over 512 arguments, or an argument over 4096 characters)"
                .into(),
        );
    }
    let port_ok = matches!(spec.get("container_port"), Some(Value::Number(n)) if !n.is_f64()
        && n.as_i64().is_some_and(|p| (1..=65535).contains(&p)));
    if !port_ok {
        errs.push("container port must be 1-65535".into());
    }
    if arr_or_empty(spec.get("gpu_uuids")).len() > 16 {
        errs.push("at most 16 GPUs per engine".into());
    }
    let labels = obj_or_empty(spec.get("labels"));
    if labels.len() > 32
        || labels
            .iter()
            .any(|(k, v)| !k.starts_with("engine-console.") || py_str(v).chars().count() > 256)
    {
        errs.push(
            "labels must start with 'engine-console.', be at most 32, and each value at most 256 characters"
                .into(),
        );
    }
    for (field, cap) in [("env", 128), ("secret_env", 16)] {
        let vals = obj_or_empty(spec.get(field));
        if vals.len() > cap
            || vals
                .iter()
                .any(|(k, v)| !full(&ENV_NAME, k) || py_str(v).chars().count() > 4096)
        {
            errs.push(format!(
                "{field}: too many variables, or a bad name or value (names: letters, digits, underscore)"
            ));
        }
    }
    let mounts = arr_or_empty(spec.get("mounts"));
    if mounts.len() > 16 {
        errs.push("at most 16 mounts".into());
    }
    for m in &mounts {
        let mo = m.as_object().cloned().unwrap_or_default();
        let rel = mo.get("rel").map_or_else(String::new, py_str);
        let dest = mo.get("dest").map_or_else(String::new, py_str);
        if (!rel.is_empty() && !rel_ok(&rel))
            || !dest.starts_with('/')
            || dest.split('/').any(|s| s == "..")
        {
            let root = mo
                .get("root")
                .map_or_else(|| "None".to_string(), ec_proto::pyfmt::repr_value);
            errs.push(format!(
                "mount of root {root}: bad path ({} -> {})",
                ec_proto::pyfmt::repr_str(&rel),
                ec_proto::pyfmt::repr_str(&dest)
            ));
        }
    }
    let volumes = arr_or_empty(spec.get("volumes"));
    if volumes.len() > 8
        || volumes.iter().any(|v| {
            let vo = v.as_object().cloned().unwrap_or_default();
            !full(&VOLUME_NAME, &str_of(&vo, "name")) || !str_of(&vo, "dest").starts_with('/')
        })
    {
        errs.push("named volumes must be called ec-<name> and have an absolute destination".into());
    }
    let shm_text = str_of(spec, "shm_size");
    match SHM_SIZE.captures(&shm_text) {
        None => errs.push("shm_size must look like 16g".into()),
        Some(c) => {
            if let Some(max) = max_shm_gib {
                let n: f64 = c[1].parse::<u32>().map(f64::from).unwrap_or(0.0);
                let gib = n * match &c[2] {
                    "k" => 1.0 / 1_048_576.0,
                    "m" => 1.0 / 1024.0,
                    _ => 1.0,
                };
                if gib > max {
                    errs.push(format!(
                        "shm_size {shm_text} is above this machine's policy limit of {} GiB",
                        ec_proto::pyfmt::fmt_g(max)
                    ));
                }
            }
        }
    }
    let files = obj_or_empty(spec.get("files"));
    if files.len() > MAX_FILES
        || files
            .iter()
            .any(|(p, c)| !p.starts_with('/') || py_str(c).len() > MAX_FILE_BYTES)
    {
        errs.push("at most 4 generated files, absolute paths, 128 KiB each".into());
    }
    for (k, v) in obj_or_empty(spec.get("ulimits")) {
        let vs = py_str(&v);
        if !ULIMITS.contains(&k.as_str()) || !full(&ULIMIT_VALUE, &vs) {
            errs.push(format!(
                "ulimit {k}={vs}: only {} with -1, a number or soft:hard numbers",
                ULIMITS.join(", ")
            ));
        }
    }
    errs
}

#[cfg(test)]
mod tests {
    use super::*;

    const HF: &str = "/root/.cache/huggingface";

    #[allow(clippy::needless_pass_by_value)]
    fn obj(v: Value) -> Obj {
        v.as_object().cloned().unwrap_or_default()
    }

    fn gguf(rel: &str) -> ModelRef {
        gguf_in(rel, "gguf")
    }

    fn gguf_in(rel: &str, root: &str) -> ModelRef {
        ModelRef {
            format: "gguf".into(),
            repo_id: None,
            root_id: Some(root.into()),
            rel: Some(rel.into()),
            name: None,
        }
    }

    fn mr(
        format: &str,
        repo: Option<&str>,
        root: Option<&str>,
        rel: Option<&str>,
        name: Option<&str>,
    ) -> ModelRef {
        ModelRef {
            format: format.into(),
            repo_id: repo.map(String::from),
            root_id: root.map(String::from),
            rel: rel.map(String::from),
            name: name.map(String::from),
        }
    }

    // ---- tests/core/test_model_mounts.py, pure planning ----
    #[test]
    fn test_hf_layouts() {
        assert_eq!(hf_layout(Some("hf-home"), "models--a--b"), "home");
        assert_eq!(hf_layout(Some("hf-hub"), "hub/x"), "hub");
        assert_eq!(hf_layout(Some("auto"), "hub/models--a--b"), "home");
        assert_eq!(hf_layout(Some("auto"), "models--a--b"), "hub");
        assert_eq!(hf_layout(None, "anything"), "home");
        assert_eq!(
            hf_mounts("hf", Some("hf-home"), "hub/models--a--b", HF),
            vec![obj(
                json!({"root": "hf", "rel": "hub", "dest": format!("{HF}/hub"), "ro": true})
            )]
        );
        assert_eq!(
            hf_mounts("cache", Some("hf-hub"), "models--a--b", &format!("{HF}/")),
            vec![obj(
                json!({"root": "cache", "rel": "", "dest": format!("{HF}/hub"), "ro": true})
            )]
        );
        assert_eq!(
            hf_mounts("cache", Some("auto"), "models--a--b", HF)[0]["ro"],
            json!(true)
        );
    }

    #[test]
    fn test_gguf_mounts_the_directory_of_the_file_so_split_sets_are_whole() {
        let (mounts, path) =
            model_mounts(&gguf("unsloth/m/m-Q4_K_M-00001-of-00003.gguf"), None, None);
        assert_eq!(
            mounts,
            vec![obj(
                json!({"root": "gguf", "rel": "unsloth/m", "dest": "/models", "ro": true})
            )]
        );
        assert_eq!(
            path.as_deref(),
            Some("/models/m-Q4_K_M-00001-of-00003.gguf")
        );
        let (top, p2) = model_mounts(&gguf("model.gguf"), None, None);
        assert_eq!(
            top,
            vec![obj(
                json!({"root": "gguf", "rel": "", "dest": "/models", "ro": true})
            )]
        );
        assert_eq!(p2.as_deref(), Some("/models/model.gguf"));
        // a quant inside an HF cache repo mounts the whole repo
        let (snap, p3) = model_mounts(&gguf("models--a--b/snapshots/abc/m.gguf"), None, None);
        assert_eq!(snap[0]["rel"], json!("models--a--b"));
        assert_eq!(p3.as_deref(), Some("/models/snapshots/abc/m.gguf"));
    }

    #[test]
    fn test_plain_mounts_the_model_directory_at_models() {
        let r = mr("plain", None, Some("plain"), Some("org/model"), None);
        assert_eq!(
            model_mounts(&r, None, None),
            (
                vec![obj(
                    json!({"root": "plain", "rel": "org/model", "dest": "/models", "ro": true})
                )],
                Some("/models".to_string())
            )
        );
        let root_is_model = mr("plain", None, Some("plain"), Some(""), None);
        assert_eq!(
            model_mounts(&root_is_model, None, None).0[0]["rel"],
            json!("")
        );
    }

    // the 4 parametrised ids of test_ollama_mounts_the_models_dir_read_only
    #[test]
    fn test_ollama_mounts_the_models_dir_read_only() {
        let r = mr("ollama", None, None, None, Some("llama3:8b"));
        for (copy_rel, root_path, expected) in [
            (
                Some("manifests/registry.ollama.ai/library/llama3/8b"),
                "/data/ollama/models",
                "",
            ),
            (
                Some("models/manifests/registry.ollama.ai/library/llama3/8b"),
                "/data/ollama",
                "models",
            ),
            (None, "/data/ollama/models", ""),
            (None, "/data/ollama", "models"),
        ] {
            let (mounts, path) = model_mounts(&r, copy_rel, Some(("ollama", root_path)));
            assert_eq!(
                mounts,
                vec![obj(
                    json!({"root": "ollama", "rel": expected, "dest": "/root/.ollama/models", "ro": true})
                )]
            );
            assert_eq!(path, None);
            assert_eq!(model_mounts(&r, None, None), (Vec::new(), None));
        }
    }

    // the 15 parametrised ids of test_hostile_or_incomplete_refs_are_422_invalid_model
    #[test]
    fn test_hostile_or_incomplete_refs_are_422_invalid_model() {
        let long = "x".repeat(200);
        let refs = [
            mr("hf", Some("a--b/c"), None, None, None),
            mr("hf", None, None, None, None),
            mr("hf", Some("../x/y"), None, None, None),
            mr("gguf", None, Some("g"), Some("/abs/m.gguf"), None),
            mr("gguf", None, Some("g"), Some("../m.gguf"), None),
            mr("gguf", None, Some("g"), Some("a//m.gguf"), None),
            mr("gguf", None, Some("g"), Some("m.bin"), None),
            mr("gguf", None, None, Some("m.gguf"), None),
            mr("gguf", None, Some("a b"), Some("m.gguf"), None),
            mr("gguf", None, Some("g"), None, None),
            mr("plain", None, Some("g"), Some("a/../b"), None),
            mr("plain", None, None, Some("a"), None),
            mr("ollama", None, None, None, Some("llama3; rm -rf /")),
            mr("ollama", None, None, None, Some(&long)),
            mr("gguf", None, Some("g"), Some("a\\b.gguf"), None),
        ];
        for r in refs {
            let e = check_ref(&r).expect_err("refused");
            assert_eq!((e.status, e.code.as_str()), (422, "invalid_model"), "{r:?}");
        }
    }

    // the 8 parametrised ids of test_valid_refs_pass
    #[test]
    fn test_valid_refs_pass() {
        for r in [
            mr("hf", Some("Qwen/Qwen3-32B"), None, None, None),
            gguf("a/b/model-Q4.gguf"),
            mr("plain", None, Some("p"), Some("org/m"), None),
            mr("plain", None, Some("p"), Some(""), None),
            mr("ollama", None, None, None, Some("llama3:8b")),
            mr("ollama", None, None, None, Some("hf.co/org/repo:Q4_K_M")),
            mr("ollama", None, None, None, None),
            mr("ollama", None, None, None, Some("")),
        ] {
            assert_eq!(check_ref(&r), Ok(()), "{r:?}");
        }
    }

    #[test]
    fn test_display_names_are_what_chat_sends_as_model() {
        assert_eq!(
            display_of(&ModelRef::hf("Qwen/Qwen3-32B")),
            "Qwen/Qwen3-32B"
        );
        assert_eq!(
            display_of(&gguf("unsloth/m/Model-Q4_K_M.gguf")),
            "Model-Q4_K_M"
        );
        assert_eq!(
            display_of(&gguf("m/Model-Q4_K_M-00001-of-00004.gguf")),
            "Model-Q4_K_M"
        );
        assert_eq!(
            display_of(&mr("plain", None, Some("p"), Some("org/my-model"), None)),
            "my-model"
        );
        assert_eq!(
            display_of(&mr("plain", None, Some("p"), Some(""), None)),
            "p"
        );
        assert_eq!(
            display_of(&mr("ollama", None, None, None, Some("llama3:8b"))),
            "llama3:8b"
        );
        assert_eq!(display_of(&mr("ollama", None, None, None, None)), "ollama");
    }

    // security control: secret-looking variables never travel as plain env
    #[test]
    fn test_secret_looking_variables_travel_as_secret_env_and_offline_is_default() {
        let input = obj(
            json!({"HF_HOME": HF, "VLLM_API_KEY": "s", "MY_TOKEN": "t", "PASSWORD_FILE": "p", "X": "1"}),
        );
        let (env, secret) = split_env(&input, true);
        assert_eq!(
            env,
            obj(json!({"HF_HOME": HF, "X": "1", "HF_HUB_OFFLINE": "1"}))
        );
        assert_eq!(
            secret,
            vec![
                ("VLLM_API_KEY".to_string(), "s".to_string()),
                ("MY_TOKEN".to_string(), "t".to_string()),
                ("PASSWORD_FILE".to_string(), "p".to_string())
            ]
        );
        let (env2, _) = split_env(&obj(json!({"HF_HUB_OFFLINE": "0"})), true);
        assert_eq!(env2["HF_HUB_OFFLINE"], json!("0"));
        assert!(
            !split_env(&Obj::new(), false)
                .0
                .contains_key("HF_HUB_OFFLINE")
        );
    }

    #[test]
    fn test_labels_leave_out_what_would_not_fit_instead_of_truncating() {
        let r = gguf(&format!("a/{}.gguf", "d".repeat(300)));
        let uuids: Vec<String> = vec![format!("GPU-{}", "u".repeat(40)); 8];
        let labels = engine_labels(&LabelArgs {
            instance: "inst_1",
            engine: "fakecpp",
            display: "m",
            r: &r,
            port: 18000,
            name: "n",
            gpu_ids: &[0, 1],
            gpu_uuids: &uuids,
        });
        assert!(!labels.contains_key("engine-console.rel"));
        assert_eq!(labels["engine-console.root"], json!("gguf"));
        assert_eq!(labels["engine-console.format"], json!("gguf"));
        assert!(!labels.contains_key("engine-console.gpus"));
        assert_eq!(labels["engine-console.gpu_ids"], json!("0,1"));
        assert!(
            labels
                .values()
                .all(|v| v.as_str().is_some_and(|s| s.chars().count() <= 256))
        );
        assert_eq!(labels["engine-console.port"], json!("18000"));
    }

    // ---- tests/core/test_runspec_ulimits.py ----
    fn launch_with(ulimits: Obj) -> LaunchSpec {
        LaunchSpec {
            image: "img:1".into(),
            argv: vec!["x".into()],
            env: Obj::new(),
            container_port: 8000,
            health_path: "/health".into(),
            metrics_path: "/metrics".into(),
            models_path: "/v1/models".into(),
            shm_size: "16g".into(),
            ipc_host: true,
            files: Obj::new(),
            needs_hf_cache: true,
            ulimits,
        }
    }

    fn spec_with(ulimits: Obj) -> Obj {
        let launch = launch_with(ulimits);
        build_run_spec(&RunArgs {
            name: "ec-t-1",
            instance: "inst_1",
            image: "img:1",
            launch: &launch,
            gpu_uuids: &[],
            labels: &Obj::new(),
            env: &Obj::new(),
            secret_env: &Obj::new(),
            mounts: &[],
            volumes: &[],
            ipc_host: false,
        })
    }

    #[test]
    fn test_ulimits_pass_through_and_are_validated() {
        let ok = spec_with(obj(json!({"stack": "67108864", "nofile": "65536:65536"})));
        assert_eq!(
            ok["ulimits"],
            json!({"stack": "67108864", "nofile": "65536:65536"})
        );
        assert!(validate_run_spec(&ok, None).is_empty());
        assert!(!spec_with(Obj::new()).contains_key("ulimits"));
        assert!(
            validate_run_spec(&spec_with(obj(json!({"memlock": "-1"}))), None)
                .iter()
                .any(|e| e.contains("ulimit memlock"))
        );
        assert!(
            validate_run_spec(&spec_with(obj(json!({"stack": "lots"}))), None)
                .iter()
                .any(|e| e.contains("ulimit stack"))
        );
    }

    #[test]
    fn test_run_spec_keys_are_the_nodes_and_in_pythons_order() {
        let s = spec_with(obj(json!({"stack": "1"})));
        let keys: Vec<&str> = s.keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "name",
                "instance",
                "image",
                "argv",
                "container_port",
                "gpu_uuids",
                "network_alias",
                "labels",
                "env",
                "secret_env",
                "mounts",
                "volumes",
                "shm_size",
                "ipc_host",
                "ulimits"
            ]
        );
        // every key is one the node's RUN_SPEC keeps (nothing it would have to drop)
        let allowed: Vec<&str> = ec_proto::specs::run_spec()
            .fields()
            .iter()
            .map(|(k, _)| *k)
            .collect();
        assert!(keys.iter().all(|k| allowed.contains(k)));
    }

    // Port of tests/core/test_lifecycle_fleet.py::test_spec_validation_helpers_are_not_reachable_with_hostile_names
    // (security control: the hub refuses what the node would refuse, before sending anything).
    #[test]
    fn test_spec_validation_helpers_are_not_reachable_with_hostile_names() {
        let spec = obj(
            json!({"name": "ec-x", "instance": "inst_1", "image": "i:1", "argv": [], "container_port": 8000,
            "labels": {}, "env": {}, "secret_env": {}, "mounts": [], "volumes": [], "shm_size": "16g"}),
        );
        assert!(validate_run_spec(&spec, None).is_empty());
        for bad in [
            json!({"name": "../x"}),
            json!({"name": "ec-A"}),
            json!({"instance": "a b"}),
            json!({"container_port": 0}),
            json!({"shm_size": "16 g"}),
            json!({"env": {"1BAD": "x"}}),
            json!({"labels": {"x": "y"}}),
            json!({"mounts": [{"root": "hf", "rel": "../x", "dest": "/m"}]}),
            json!({"mounts": [{"root": "hf", "rel": "a", "dest": "rel/path"}]}),
            json!({"volumes": [{"name": "evil", "dest": "/x"}]}),
            json!({"argv": ["a\u{0}b"]}),
            json!({"files": {"relative": "x"}}),
        ] {
            let mut s = spec.clone();
            for (k, v) in obj(bad.clone()) {
                s.insert(k, v);
            }
            assert!(!validate_run_spec(&s, None).is_empty(), "{bad}");
        }
        let mut big = spec.clone();
        big.insert("shm_size".into(), json!("100g"));
        let errs = validate_run_spec(&big, Some(64.0));
        assert_eq!(
            errs,
            vec!["shm_size 100g is above this machine's policy limit of 64 GiB"]
        );
        let mut m = spec;
        m.insert(
            "mounts".into(),
            json!([{"root": "hf", "rel": "../x", "dest": "/m"}]),
        );
        assert_eq!(
            validate_run_spec(&m, None),
            vec!["mount of root 'hf': bad path ('../x' -> '/m')"]
        );
    }

    proptest::proptest! {
        // rel_ok never accepts an absolute path, a backslash, NUL or a `..` segment
        #[test]
        fn prop_rel_ok_refuses_every_escape(s in "[a-z./\\\\\u{0}]{0,12}") {
            if rel_ok(&s) {
                proptest::prop_assert!(!s.starts_with('/') && !s.contains('\\') && !s.contains('\0'));
                proptest::prop_assert!(s.split('/').all(|seg| !seg.is_empty() && seg != "." && seg != ".."));
            }
        }
    }
}
