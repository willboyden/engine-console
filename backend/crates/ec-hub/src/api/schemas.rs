//! Validation schemas of every request body (pydantic models of domain/models.py, domain/fleet.py and the routers),
//! in their field order. A change here changes the API.

use std::sync::LazyLock;

use serde_json::{Map, Value, json};

use crate::api::validate::{Field, Model, T};
use crate::domain::models::{HOST_ID_PATTERN, INSTANCE_NAME_PATTERN};
use crate::domain::repo_id::{check_repo_id, check_revision};

pub const MODEL_FORMATS: &[&str] = &["hf", "gguf", "ollama", "plain"];

fn host_id() -> T {
    T::Str {
        min: None,
        max: None,
        pattern: Some(HOST_ID_PATTERN),
    }
}

fn repo_id() -> T {
    T::After(Box::new(T::str()), check_repo_id)
}

fn revision() -> T {
    T::After(Box::new(T::str()), check_revision)
}

fn instance_name() -> T {
    T::Str {
        min: None,
        max: None,
        pattern: Some(INSTANCE_NAME_PATTERN),
    }
}

fn give_repo_id_or_model(m: &Map<String, Value>) -> Result<(), String> {
    if m.get("repo_id").is_none_or(Value::is_null) && m.get("model").is_none_or(Value::is_null) {
        return Err("give repo_id or model".into());
    }
    Ok(())
}

pub static MODEL_REF: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ModelRef",
    fields: vec![
        Field::opt("format", T::Literal(MODEL_FORMATS), json!("hf")),
        Field::opt("repo_id", T::opt(T::str()), Value::Null),
        Field::opt("root_id", T::opt(T::str()), Value::Null),
        Field::opt("rel", T::opt(T::str()), Value::Null),
        Field::opt("name", T::opt(T::str()), Value::Null),
    ],
    after: None,
});

pub static FIT_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "FitRequest",
    fields: vec![
        Field::req("engine", T::str()),
        Field::opt("repo_id", T::opt(repo_id()), Value::Null),
        Field::opt("revision", T::opt(revision()), Value::Null),
        Field::opt("params", T::AnyDict, json!({})),
        Field::opt("gpu_ids", T::list(T::int()), json!([])),
        Field::opt(
            "concurrency",
            T::Int {
                ge: Some(1),
                le: Some(4096),
            },
            json!(1),
        ),
        Field::opt("host_id", host_id(), json!("local")),
        Field::opt("model", T::opt(T::Model(&MODEL_REF)), Value::Null),
    ],
    after: Some(give_repo_id_or_model),
});

pub static DOWNLOAD_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "DownloadRequest",
    fields: vec![
        Field::req("repo_id", repo_id()),
        Field::opt("revision", T::opt(revision()), Value::Null),
        Field::opt("allow_patterns", T::opt(T::list(T::str())), Value::Null),
        Field::opt("host_id", host_id(), json!("local")),
        Field::opt("root_id", T::opt(T::str()), Value::Null),
    ],
    after: None,
});

pub static INSTANCE_CREATE: LazyLock<Model> = LazyLock::new(|| Model {
    name: "InstanceCreate",
    fields: vec![
        Field::req("engine", T::str()),
        Field::opt("repo_id", T::opt(repo_id()), Value::Null),
        Field::opt("params", T::AnyDict, json!({})),
        Field::opt("gpu_ids", T::list(T::int()), json!([])),
        Field::opt("profile_id", T::opt(T::str()), Value::Null),
        Field::opt("name", T::opt(instance_name()), Value::Null),
        Field::opt("ttl_idle_s", T::opt(T::int()), Value::Null),
        Field::opt("host_id", host_id(), json!("local")),
        Field::opt("model", T::opt(T::Model(&MODEL_REF)), Value::Null),
        Field::opt("image", T::opt(T::str()), Value::Null),
    ],
    after: Some(give_repo_id_or_model),
});

pub static INSTANCE_PATCH: LazyLock<Model> = LazyLock::new(|| Model {
    name: "InstancePatch",
    fields: vec![
        Field::opt("pinned", T::opt(T::Bool), Value::Null),
        Field::opt("ttl_idle_s", T::opt(T::int()), Value::Null),
        Field::opt("name", T::opt(instance_name()), Value::Null),
    ],
    after: None,
});

pub static PROFILE_IN: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ProfileIn",
    fields: vec![
        Field::req(
            "name",
            T::Str {
                min: Some(1),
                max: Some(120),
                pattern: None,
            },
        ),
        Field::req("engine", T::str()),
        Field::opt("repo_id", T::opt(repo_id()), Value::Null),
        Field::opt("preset", T::opt(T::str()), Value::Null),
        Field::opt("params", T::AnyDict, json!({})),
        Field::opt("description", T::str(), json!("")),
    ],
    after: None,
});

pub static BENCH_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "BenchRequest",
    fields: vec![
        Field::req("instance_id", T::str()),
        Field::opt(
            "suite",
            T::Union(vec![("str", T::str()), ("dict[str,any]", T::AnyDict)]),
            json!("quick"),
        ),
        Field::opt("confirm_external", T::Bool, json!(false)),
    ],
    after: None,
});

pub static CONVERSATION_IN: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ConversationIn",
    fields: vec![
        Field::opt("title", T::str(), json!("New chat")),
        Field::opt("system_prompt", T::opt(T::str()), Value::Null),
        Field::opt("instance_id", T::opt(T::str()), Value::Null),
    ],
    after: None,
});

pub static PROMPT_IN: LazyLock<Model> = LazyLock::new(|| Model {
    name: "PromptIn",
    fields: vec![
        Field::req(
            "title",
            T::Str {
                min: Some(1),
                max: Some(200),
                pattern: None,
            },
        ),
        Field::req("content", T::str()),
        Field::opt("tags", T::list(T::str()), json!([])),
    ],
    after: None,
});

pub static ARENA_MATCH_IN: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ArenaMatchIn",
    fields: vec![
        Field::req("prompt", T::str()),
        Field::req("instance_a", T::str()),
        Field::req("instance_b", T::str()),
        Field::opt("system_prompt", T::opt(T::str()), Value::Null),
        Field::opt("blind", T::Bool, json!(true)),
        Field::opt(
            "max_tokens",
            T::Int {
                ge: Some(1),
                le: Some(32768),
            },
            json!(512),
        ),
    ],
    after: None,
});

pub static ARENA_VOTE: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ArenaVote",
    fields: vec![Field::req("winner", T::Literal(&["a", "b", "tie"]))],
    after: None,
});

pub static KEY_IN: LazyLock<Model> = LazyLock::new(|| Model {
    name: "KeyIn",
    fields: vec![
        Field::req(
            "name",
            T::Str {
                min: Some(1),
                max: Some(80),
                pattern: None,
            },
        ),
        Field::opt("role", T::Literal(&["admin", "viewer"]), json!("viewer")),
    ],
    after: None,
});

pub static SETTINGS_PATCH: LazyLock<Model> = LazyLock::new(|| Model {
    name: "SettingsPatch",
    fields: vec![
        Field::opt("default_gpu_ids", T::opt(T::list(T::int())), Value::Null),
        Field::opt(
            "idle_ttl_s",
            T::opt(T::Int {
                ge: Some(0),
                le: None,
            }),
            Value::Null,
        ),
        Field::opt("image_pins", T::opt(T::dict(T::str())), Value::Null),
        Field::opt(
            "host_image_pins",
            T::opt(T::dict(T::dict(T::str()))),
            Value::Null,
        ),
    ],
    after: None,
});

pub static COPY_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "CopyRequest",
    fields: vec![
        Field::req("host_id", host_id()),
        Field::opt("root_id", T::opt(T::str()), Value::Null),
    ],
    after: None,
});

pub static SCRIPT_RUN_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "ScriptRunRequest",
    fields: vec![Field::opt("arg", T::opt(T::str()), Value::Null)],
    after: None,
});

pub static RESCAN_REQUEST: LazyLock<Model> = LazyLock::new(|| Model {
    name: "RescanRequest",
    fields: vec![Field::opt("host_id", T::opt(host_id()), Value::Null)],
    after: None,
});
