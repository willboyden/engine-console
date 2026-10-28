//! Secret engine parameters (api keys etc.): never in the DB, API responses, profiles or audit (port of
//! services/secrets_store.py).
//!
//! The DB/API only ever carry MARKER. The real value lives in a 0600 file per instance (`<data>/secrets/<iid>.json`,
//! directory 0700), read only at container launch. Hardening over Python: the file is written and read only as a
//! regular file reached without a symlink (`O_NOFOLLOW | O_NONBLOCK`, checked on the descriptor), and read capped.

use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Map, Value};

use crate::services::common::is_sensitive_key;
use crate::services::store::create_dir_0700;

pub const MARKER: &str = "[set]";
/// The largest secrets file read (one instance's few secret parameters).
pub const MAX_SECRETS_FILE: usize = 1024 * 1024;
static IID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$").unwrap_or_else(|_| unreachable!())
});

fn iid_ok(iid: &str) -> bool {
    IID.is_match(iid) && !iid.ends_with('\n')
}

/// (params with real secret values replaced by MARKER, {key: real value}), both in the params' order.
pub fn split_secrets(params: &Map<String, Value>) -> (Map<String, Value>, Vec<(String, String)>) {
    let mut public = Map::new();
    let mut secrets = Vec::new();
    for (k, v) in params {
        match v {
            Value::String(s) if is_sensitive_key(k) && !s.is_empty() && s != MARKER => {
                secrets.push((k.clone(), s.clone()));
                public.insert(k.clone(), Value::String(MARKER.into()));
            }
            _ => {
                public.insert(k.clone(), v.clone());
            }
        }
    }
    (public, secrets)
}

#[derive(Debug, Clone)]
pub struct SecretStore {
    dir: PathBuf,
}

impl SecretStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    fn path(&self, iid: &str) -> Option<PathBuf> {
        iid_ok(iid).then(|| self.dir.join(format!("{iid}.json")))
    }

    /// Store the secrets of one instance (no file when there are none). Errors on a bad instance id.
    pub fn put(&self, iid: &str, secrets: &[(String, String)]) -> std::io::Result<()> {
        if secrets.is_empty() {
            return Ok(());
        }
        let p = self
            .path(iid)
            .ok_or_else(|| std::io::Error::other("bad instance id"))?;
        create_dir_0700(&self.dir)?;
        std::fs::set_permissions(&self.dir, std::fs::Permissions::from_mode(0o700))?;
        let mut f = crate::fsutil::create_private(&p, 0o600)?;
        let obj: Map<String, Value> = secrets
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        std::io::Write::write_all(&mut f, crate::pyjson::dumps(&Value::Object(obj)).as_bytes())
    }

    /// {key: value} of the string entries; empty when missing, unreadable or not a JSON object.
    pub fn get(&self, iid: &str) -> Vec<(String, String)> {
        let Some(p) = self.path(iid) else {
            return Vec::new();
        };
        let Ok(text) = crate::fsutil::read_text(&p, crate::fsutil::Follow::No, MAX_SECRETS_FILE)
        else {
            return Vec::new();
        };
        match serde_json::from_str::<Value>(&text) {
            Ok(Value::Object(o)) => o
                .into_iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                .collect(),
            _ => Vec::new(),
        }
    }

    /// Never fails: an id that could never have been stored is simply ignored.
    pub fn delete(&self, iid: &str) {
        if let Some(p) = self.path(iid) {
            let _ = std::fs::remove_file(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_and_round_trip_0600() {
        let params: Map<String, Value> = serde_json::from_str(
            r#"{"api_key": "s3cret", "max_len": 5, "hf_token": "[set]", "x_password": ""}"#,
        )
        .unwrap();
        let (public, secrets) = split_secrets(&params);
        assert_eq!(public["api_key"], MARKER);
        assert_eq!(public["hf_token"], MARKER);
        assert_eq!(public["x_password"], "");
        assert_eq!(secrets, vec![("api_key".to_string(), "s3cret".to_string())]);
        let dir = tempfile::tempdir().unwrap();
        let store = SecretStore::new(dir.path().join("secrets"));
        store.put("inst_ab12", &secrets).unwrap();
        let mode = std::fs::metadata(dir.path().join("secrets/inst_ab12.json"))
            .unwrap()
            .permissions()
            .mode();
        assert_eq!(mode & 0o777, 0o600);
        assert_eq!(store.get("inst_ab12"), secrets);
        assert!(store.put("../x", &secrets).is_err());
        assert!(store.get("../x").is_empty());
        store.delete("inst_ab12");
        assert!(store.get("inst_ab12").is_empty());
    }
}
