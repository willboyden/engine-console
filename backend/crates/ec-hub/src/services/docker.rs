//! Python: backend/src/engine_console/services/docker.py.
//!
//! Hub-side docker helpers: names, labels and the "copy as command" snippets. Pure functions.
//!
//! The hub never runs docker (contract v2, §10.1 decision 14): every container action is a node operation
//! (`docker.run`, `gateway.start`, `docker.stop`, ...). The node renders the exact `docker run` argv it would use
//! (`docker.render`); this module wraps it, and builds the compose file and the bare engine command from the same run
//! spec. Secrets are never in a snippet: a secret variable appears as a name only.
//!
//! The compose text is PyYAML's `yaml.safe_dump(compose, sort_keys=False)`, byte for byte ([`yaml`], pinned by
//! golden vectors generated with PyYAML 6.0.3: `tests/data/wse_yaml_vectors.json`).

use serde_json::Value;

use crate::domain::errors::Obj;
use crate::domain::models::CommandSnippets;

pub const LABEL: &str = "engine-console";
pub const ROLE_LABEL: &str = "engine-console.role";
pub const INSTANCE_LABEL: &str = "engine-console.instance";

/// The per-instance gateway sidecar (the only container with a published port) is `<engine container>-gw`.
pub fn gateway_name(container_name: &str) -> String {
    format!("{container_name}-gw")
}

/// `shlex.quote(s)`.
pub fn shlex_quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = s
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

/// `shlex.join(argv)`.
pub fn shlex_join<S: AsRef<str>>(argv: &[S]) -> String {
    argv.iter()
        .map(|a| shlex_quote(a.as_ref()))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The node's rendered argv as one copyable `docker run ...` line.
///
/// Accepts either `["run", ...]` or `["docker", "--context", "x", "run", ...]` (the node's choice): the binary is
/// normalised to `docker` and a `--context` pair is dropped, because the context belongs to that machine.
pub fn clean_render<S: AsRef<str>>(argv: &[S]) -> String {
    let mut toks: Vec<&str> = argv.iter().map(AsRef::as_ref).collect();
    if toks
        .first()
        .is_some_and(|t| t.rsplit('/').next() == Some("docker"))
    {
        toks.remove(0);
    }
    while toks.len() >= 2 && toks[0] == "--context" {
        toks.drain(..2);
    }
    let mut all = vec!["docker"];
    all.extend(toks);
    shlex_join(&all)
}

/// `str(x)` of a JSON value (a str is itself).
fn py_str(v: &Value) -> String {
    ec_proto::pyfmt::str_value(v)
}

fn truthy(v: &Value) -> bool {
    ec_proto::sanitize::truthy(v)
}

/// `docker_run`, `compose_yaml` and `engine_cli` for a run spec (secret_env values must already be absent).
///
/// `host_paths` maps a mount's root id to that root's absolute path on the host, so a compose volume can name the
/// real directory; a mount whose root is unknown is left out of the compose file rather than guessed.
pub fn command_snippets(
    spec: &Obj,
    rendered: Option<&[String]>,
    host_paths: &[(String, String)],
    network: Option<&str>,
) -> CommandSnippets {
    use yaml::Node;
    let obj = |k: &str| match spec.get(k) {
        Some(Value::Object(o)) => o.clone(),
        _ => Obj::new(),
    };
    let arr = |k: &str| match spec.get(k) {
        Some(Value::Array(a)) => a.clone(),
        _ => Vec::new(),
    };
    // environment: plain values, then each secret as `${NAME}`
    let mut env: Vec<(String, Node)> = obj("env")
        .iter()
        .map(|(k, v)| (k.clone(), Node::from_json(v)))
        .collect();
    for k in obj("secret_env").keys() {
        let v = Node::Str(format!("${{{k}}}"));
        if let Some(slot) = env.iter_mut().find(|(kk, _)| kk == k) {
            slot.1 = v;
        } else {
            env.push((k.clone(), v));
        }
    }
    let mut volumes: Vec<Node> = Vec::new();
    for m in arr("mounts") {
        let Value::Object(m) = m else { continue };
        let root_key = m.get("root").map_or_else(|| "None".to_string(), py_str);
        let Some((_, root)) = host_paths.iter().find(|(k, _)| *k == root_key) else {
            continue;
        };
        let rel = m.get("rel").filter(|v| truthy(v));
        let host = format!(
            "{}{}",
            root.trim_end_matches('/'),
            rel.map(|r| format!("/{}", py_str(r))).unwrap_or_default()
        );
        let ro = m.get("ro").is_none_or(truthy);
        let dest = m.get("dest").map_or_else(|| "None".to_string(), py_str);
        volumes.push(Node::Str(format!(
            "{host}:{dest}{}",
            if ro { ":ro" } else { "" }
        )));
    }
    for v in arr("volumes") {
        let name = v.get("name").map_or_else(|| "None".to_string(), py_str);
        let dest = v.get("dest").map_or_else(|| "None".to_string(), py_str);
        volumes.push(Node::Str(format!("{name}:{dest}")));
    }
    let net = network.unwrap_or("engine-console-engines").to_string();
    let mut labels: Vec<(String, Node)> = vec![(LABEL.to_string(), Node::Str("1".into()))];
    for (k, v) in obj("labels") {
        let n = Node::from_json(&v);
        if let Some(slot) = labels.iter_mut().find(|(kk, _)| *kk == k) {
            slot.1 = n;
        } else {
            labels.push((k, n));
        }
    }
    let argv = arr("argv");
    let shm = spec
        .get("shm_size")
        .filter(|v| truthy(v))
        .map_or_else(|| Node::Str("16g".into()), Node::from_json);
    let ipc = if spec.get("ipc_host").is_some_and(truthy) {
        "host"
    } else {
        "private"
    };
    let device = Node::Map(vec![
        ("driver".into(), Node::Str("nvidia".into())),
        (
            "device_ids".into(),
            Node::Seq(arr("gpu_uuids").iter().map(Node::from_json).collect()),
        ),
        (
            "capabilities".into(),
            Node::Seq(vec![Node::Str("gpu".into())]),
        ),
    ]);
    let service = Node::Map(vec![
        (
            "image".into(),
            spec.get("image").map_or(Node::Null, Node::from_json),
        ),
        (
            "command".into(),
            Node::Seq(argv.iter().map(Node::from_json).collect()),
        ),
        ("restart".into(), Node::Str("no".into())),
        ("ipc".into(), Node::Str(ipc.into())),
        ("shm_size".into(), shm),
        ("environment".into(), Node::Map(env)),
        ("volumes".into(), Node::Seq(volumes)),
        ("labels".into(), Node::Map(labels)),
        ("networks".into(), Node::Seq(vec![Node::Str(net.clone())])),
        (
            "security_opt".into(),
            Node::Seq(vec![Node::Str("no-new-privileges:true".into())]),
        ),
        ("cap_drop".into(), Node::Seq(vec![Node::Str("ALL".into())])),
        ("pids_limit".into(), Node::int(4096)),
        (
            "deploy".into(),
            Node::Map(vec![(
                "resources".into(),
                Node::Map(vec![(
                    "reservations".into(),
                    Node::Map(vec![("devices".into(), Node::Seq(vec![device]))]),
                )]),
            )]),
        ),
    ]);
    let svc_name = spec
        .get("name")
        .filter(|v| truthy(v))
        .map_or_else(|| "engine".to_string(), py_str);
    let compose = Node::Map(vec![
        ("services".into(), Node::Map(vec![(svc_name, service)])),
        (
            "networks".into(),
            Node::Map(vec![(
                net,
                Node::Map(vec![("external".into(), Node::Bool(true))]),
            )]),
        ),
    ]);
    let mut text = yaml::safe_dump(&compose, false);
    let mut files: Vec<String> = obj("files").keys().cloned().collect();
    files.sort();
    if !files.is_empty() {
        // the node writes these next to the instance and mounts them read-only; compose cannot know their content
        text = format!(
            "# generated config files the console mounts: {}\n{text}",
            files.join(", ")
        );
    }
    let docker_run = match rendered {
        Some(r) if !r.is_empty() => clean_render(r),
        _ => String::new(),
    };
    let cli: Vec<String> = argv.iter().map(py_str).collect();
    CommandSnippets {
        docker_run,
        compose_yaml: text,
        engine_cli: shlex_join(&cli),
    }
}

/// The shared PyYAML emitter (PORTING.md §A9), under the name this module's code uses.
pub use crate::pyyaml as yaml;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[allow(clippy::needless_pass_by_value)]
    fn obj(v: Value) -> Obj {
        v.as_object().cloned().unwrap_or_default()
    }

    // Port of tests/core/test_helpers.py::test_gateway_name_is_the_engine_name_plus_gw.
    #[test]
    fn test_gateway_name_is_the_engine_name_plus_gw() {
        assert_eq!(gateway_name("ec-qwen-abc123"), "ec-qwen-abc123-gw");
    }

    // Port of the 4 parametrised ids of tests/core/test_helpers.py::test_the_nodes_rendering_is_normalised_into_one_copyable_line.
    #[test]
    fn test_the_nodes_rendering_is_normalised_into_one_copyable_line() {
        let cases: &[(&[&str], &str)] = &[
            (
                &["run", "-d", "--name", "ec-x", "img:1"],
                "docker run -d --name ec-x img:1",
            ),
            (&["docker", "run", "-d", "img:1"], "docker run -d img:1"),
            (
                &[
                    "/usr/bin/docker",
                    "--context",
                    "rootless",
                    "run",
                    "-d",
                    "img:1",
                ],
                "docker run -d img:1",
            ),
            (
                &["docker", "--context", "a", "--context", "b", "run", "x y"],
                "docker run 'x y'",
            ),
        ];
        for (argv, want) in cases {
            assert_eq!(clean_render(argv), *want);
        }
    }

    fn spec() -> Obj {
        obj(
            json!({"name": "ec-x", "image": "img:1", "argv": ["--model", "a b", "--port", "8000"],
            "gpu_uuids": ["GPU-1", "GPU-2"], "labels": {"engine-console.instance": "i1"}, "env": {"A": "b"},
            "secret_env": {"VLLM_API_KEY": "SHOULD-NEVER-APPEAR"},
            "mounts": [{"root": "hf", "rel": "", "dest": "/root/.cache/huggingface", "ro": true},
                       {"root": "gguf", "rel": "a/b", "dest": "/models", "ro": false},
                       {"root": "unknown", "rel": "", "dest": "/x", "ro": true}],
            "volumes": [{"name": "ec-vllm-cache", "dest": "/root/.cache/vllm"}], "ipc_host": true, "shm_size": "16g",
            "files": {"/etc/x.json": "{}"}}),
        )
    }

    // Port of tests/core/test_helpers.py::test_command_snippets_have_no_secret_value_and_no_unknown_mount (security
    // control: a secret value never reaches a snippet; an unknown root is never guessed).
    #[test]
    fn test_command_snippets_have_no_secret_value_and_no_unknown_mount() {
        let paths = vec![
            ("hf".to_string(), "/data/hf".to_string()),
            ("gguf".to_string(), "/data/gguf".to_string()),
        ];
        let rendered: Vec<String> = ["docker", "run", "-d", "img:1"].map(String::from).to_vec();
        let out = command_snippets(&spec(), Some(&rendered), &paths, Some("engines"));
        let blob = format!("{out:?}");
        assert!(!blob.contains("SHOULD-NEVER-APPEAR"));
        assert_eq!(out.docker_run, "docker run -d img:1");
        assert_eq!(out.engine_cli, "--model 'a b' --port 8000");
        let want = "# generated config files the console mounts: /etc/x.json\nservices:\n  ec-x:\n    image: img:1\n    command:\n    - --model\n    - a b\n    - --port\n    - '8000'\n    restart: 'no'\n    ipc: host\n    shm_size: 16g\n    environment:\n      A: b\n      VLLM_API_KEY: ${VLLM_API_KEY}\n    volumes:\n    - /data/hf:/root/.cache/huggingface:ro\n    - /data/gguf/a/b:/models\n    - ec-vllm-cache:/root/.cache/vllm\n    labels:\n      engine-console: '1'\n      engine-console.instance: i1\n    networks:\n    - engines\n    security_opt:\n    - no-new-privileges:true\n    cap_drop:\n    - ALL\n    pids_limit: 4096\n    deploy:\n      resources:\n        reservations:\n          devices:\n          - driver: nvidia\n            device_ids:\n            - GPU-1\n            - GPU-2\n            capabilities:\n            - gpu\nnetworks:\n  engines:\n    external: true\n";
        assert_eq!(out.compose_yaml, want);
    }

    // Port of tests/core/test_helpers.py::test_command_snippets_without_a_rendering_leave_docker_run_empty_and_default_the_network.
    #[test]
    fn test_command_snippets_without_a_rendering_leave_docker_run_empty_and_default_the_network() {
        let mut s = spec();
        s.insert("files".into(), json!({}));
        s.insert("ipc_host".into(), json!(false));
        let out = command_snippets(&s, None, &[], None);
        assert_eq!(out.docker_run, "");
        assert!(out.compose_yaml.contains("engine-console-engines"));
        assert!(!out.compose_yaml.starts_with('#'));
        assert!(out.compose_yaml.contains("\n    ipc: private\n"));
        assert!(
            out.compose_yaml
                .contains("    volumes:\n    - ec-vllm-cache:/root/.cache/vllm\n")
        );
    }

    #[test]
    fn test_shlex_quote_matches_python() {
        for (s, q) in [
            ("", "''"),
            ("abc", "abc"),
            ("a b", "'a b'"),
            ("it's", "'it'\"'\"'s'"),
            ("x=1,y:2/@%+.-_", "x=1,y:2/@%+.-_"),
            ("$HOME", "'$HOME'"),
            ("é", "'é'"),
        ] {
            assert_eq!(shlex_quote(s), q, "{s}");
        }
    }

    /// The golden vectors from PyYAML (tests/data/wse_yaml_vectors.json, made by backend/parity/gen/wse_yaml_vectors.py).
    #[test]
    fn test_yaml_safe_dump_matches_pyyaml_vectors() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/wse_yaml_vectors.json"
        );
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let cases: Vec<Value> = serde_json::from_str(&text).unwrap_or_default();
        assert!(cases.len() > 20, "no vectors at {path}");
        for c in cases {
            let doc = yaml::Node::from_json(&c["doc"]);
            assert_eq!(
                yaml::safe_dump(&doc, false),
                c["yaml"].as_str().unwrap_or(""),
                "{}",
                c["doc"]
            );
        }
    }

    /// The whole snippet against Python's `command_snippets` on the same inputs (same vector file).
    #[test]
    fn test_command_snippets_match_python_vectors() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/data/wse_snippet_vectors.json"
        );
        let text = std::fs::read_to_string(path).unwrap_or_default();
        let cases: Vec<Value> = serde_json::from_str(&text).unwrap_or_default();
        assert!(!cases.is_empty(), "no vectors at {path}");
        for c in cases {
            let spec = c["spec"].as_object().cloned().unwrap_or_default();
            let rendered: Option<Vec<String>> = c["rendered"].as_array().map(|a| {
                a.iter()
                    .map(|x| x.as_str().unwrap_or("").to_string())
                    .collect()
            });
            let paths: Vec<(String, String)> = c["host_paths"]
                .as_object()
                .map(|o| {
                    o.iter()
                        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                        .collect()
                })
                .unwrap_or_default();
            let out = command_snippets(&spec, rendered.as_deref(), &paths, c["network"].as_str());
            assert_eq!(
                serde_json::to_value(&out).unwrap_or_default(),
                c["out"],
                "{}",
                c["spec"]
            );
        }
    }
}
