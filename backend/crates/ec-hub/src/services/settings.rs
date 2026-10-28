//! Global settings, HF token handling, and API keys / roles (port of services/settings.py; ADR 7 and 9).

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock};

use regex::Regex;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::config::Settings;
use crate::domain::errors::{AppResult, Problem, bad_request, not_found};
use crate::domain::models::{KeyInfo, SettingsPatch};
use crate::pyjson;
use crate::services::common::{new_id, now, token_urlsafe};
use crate::services::store::{Store, create_dir_0700};

pub const BOOTSTRAP_KEY_FILE: &str = "bootstrap-admin.key";
pub const MAX_PIN_HOSTS: usize = 32;
pub const MAX_PINS_PER_HOST: usize = 16;

static HOST_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[a-z0-9][a-z0-9-]{0,31}$").unwrap_or_else(|_| unreachable!()));
static ENGINE_ID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$").unwrap_or_else(|_| unreachable!())
});
static IMAGE_REF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,255}$").unwrap_or_else(|_| unreachable!())
});

/// The caller of a request, set by the security middleware.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub name: String,
    /// "admin" | "viewer" | "public"
    pub role: String,
    pub key_id: Option<String>,
}

impl Principal {
    pub fn new(name: &str, role: &str, key_id: Option<String>) -> Self {
        Self {
            name: name.into(),
            role: role.into(),
            key_id,
        }
    }
    pub fn public() -> Self {
        Self::new("public", "public", None)
    }
}

/// Exact match against full image references (a prefix match would let `vllm/vllm-openai:evil` through).
pub fn image_allowed(image: &str, allowlist: &[String]) -> bool {
    allowlist.iter().any(|a| a == image)
}

/// `repr(s[:40])` as the error messages show it.
fn short_repr(s: &str) -> String {
    ec_proto::sanitize::py_repr(&s.chars().take(40).collect::<String>())
}

/// Validate PUT /settings `host_image_pins` ({host id: {engine id: image}}) and drop empty hosts.
pub fn clean_host_image_pins(raw: &Map<String, Value>) -> Result<Map<String, Value>, Problem> {
    let bad = |m: String| bad_request(m).code("invalid_setting");
    if raw.len() > MAX_PIN_HOSTS {
        return Err(bad(format!(
            "at most {MAX_PIN_HOSTS} hosts can carry image pins"
        )));
    }
    let mut out = Map::new();
    for (host, pins) in raw {
        if !HOST_ID.is_match(host) || host.ends_with('\n') {
            return Err(bad(format!("{} is not a valid host id", short_repr(host))));
        }
        let pins = pins.as_object().cloned().unwrap_or_default();
        if pins.len() > MAX_PINS_PER_HOST {
            return Err(bad(format!(
                "at most {MAX_PINS_PER_HOST} image pins per host"
            )));
        }
        let mut clean = Map::new();
        for (engine, image) in &pins {
            if !ENGINE_ID.is_match(engine) || engine.ends_with('\n') {
                return Err(bad(format!(
                    "{} is not a valid engine id",
                    short_repr(engine)
                )));
            }
            let image = image.as_str().unwrap_or_default();
            if !IMAGE_REF.is_match(image)
                || image.ends_with('\n')
                || !image.contains(':')
                || image.ends_with(":latest")
            {
                return Err(bad(format!(
                    "image pin for {engine} on {host} must be a pinned tag or digest (not :latest)"
                )));
            }
            clean.insert(engine.clone(), Value::String(image.to_string()));
        }
        if !clean.is_empty() {
            out.insert(host.clone(), Value::Object(clean));
        }
    }
    Ok(out)
}

/// keys are 256-bit random, so a fast unsalted hash is fine (no dictionary to attack)
pub fn key_hash(secret: &str) -> String {
    hex::encode(Sha256::digest(secret.as_bytes()))
}

/// `public_proxy`: the proxy URL without any userinfo, safe to show (`scheme://host[:port]`, host lower-cased).
pub fn public_proxy(url: Option<&str>) -> Option<String> {
    let url = url.filter(|u| !u.is_empty())?;
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let netloc = rest.split(['/', '?', '#']).next().unwrap_or("");
    let hostport = netloc.rsplit_once('@').map_or(netloc, |(_, h)| h);
    let (host, port) = if let Some(stripped) = hostport.strip_prefix('[') {
        let (h, after) = stripped.split_once(']').unwrap_or((stripped, ""));
        (h.to_string(), after.strip_prefix(':').map(str::to_string))
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), Some(p.to_string())),
            None => (hostport.to_string(), None),
        }
    };
    let host = if host.is_empty() {
        "None".to_string()
    } else {
        host.to_lowercase()
    };
    let port = port
        .filter(|p| !p.is_empty())
        .and_then(|p| p.parse::<u16>().ok())
        .filter(|p| *p != 0);
    Some(format!(
        "{}://{host}{}",
        scheme.to_lowercase(),
        port.map(|p| format!(":{p}")).unwrap_or_default()
    ))
}

pub struct SettingsService {
    db: Arc<Store>,
    cfg: Arc<Settings>,
    adapter_images: Box<dyn Fn() -> Vec<String> + Send + Sync>,
}

impl std::fmt::Debug for SettingsService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("SettingsService")
    }
}

impl SettingsService {
    pub fn new(
        db: Arc<Store>,
        cfg: Arc<Settings>,
        adapter_images: Box<dyn Fn() -> Vec<String> + Send + Sync>,
    ) -> Self {
        Self {
            db,
            cfg,
            adapter_images,
        }
    }

    pub fn engine_image_allowed(&self, image: &str) -> bool {
        let mut allow = self.cfg.engine_image_allowlist.clone();
        allow.extend((self.adapter_images)());
        image_allowed(image, &allow)
    }

    // -- HF token ------------------------------------------------------------------------------
    /// The token: HF_TOKEN, else the token file (refused when group/other bits are set; at most 64 KiB, read only
    /// as a regular file).
    pub fn hf_token(&self) -> Option<String> {
        /// a Hugging Face token is ~40 characters
        const MAX_TOKEN_FILE: usize = 64 * 1024;
        if let Some(t) = &self.cfg.hf_token
            && !t.is_empty()
        {
            let t = t.trim();
            return (!t.is_empty()).then(|| t.to_string());
        }
        // followed like Python's `read_text` (a link to the HF CLI's own token file is fine), but opened without
        // blocking and checked on the descriptor: a FIFO or device is no token, and the read is capped
        let mut file =
            crate::fsutil::open_regular(&self.cfg.hf_token_file, crate::fsutil::Follow::Yes)
                .ok()?;
        let meta = file.metadata().ok()?;
        if meta.permissions().mode() & 0o077 != 0 {
            return None; // a group/world-accessible token file is refused; status reports why
        }
        let raw = crate::fsutil::read_capped(&mut file, MAX_TOKEN_FILE).ok()?;
        let t = String::from_utf8(raw).ok()?;
        let t = t.trim();
        (!t.is_empty()).then(|| t.to_string())
    }

    pub fn token_status(&self) -> String {
        if let Some(t) = &self.cfg.hf_token
            && !t.is_empty()
        {
            return format!("[set, {} chars] (env)", t.trim().chars().count());
        }
        let f = &self.cfg.hf_token_file;
        let Ok(meta) = std::fs::metadata(f) else {
            return "[unset]".into();
        };
        if meta.permissions().mode() & 0o077 != 0 {
            return format!("[unusable: {} must be chmod 600]", f.display());
        }
        match self.hf_token() {
            Some(t) => format!("[set, {} chars] (file)", t.chars().count()),
            None => "[unset]".into(),
        }
    }

    // -- persisted overrides ---------------------------------------------------------------------
    fn get(&self, key: &str) -> AppResult<Option<Value>> {
        let row = self
            .db
            .one("SELECT value FROM settings WHERE key=?", &[&key])?;
        Ok(row.map(|r| r.json_text("value")))
    }

    pub fn hf_cache_dir(&self) -> PathBuf {
        self.cfg.hf_cache_dir.clone()
    }

    pub fn hub_dir(&self) -> PathBuf {
        self.hf_cache_dir().join("hub")
    }

    pub fn idle_ttl_s(&self) -> AppResult<i64> {
        Ok(self
            .get("idle_ttl_s")?
            .and_then(|v| v.as_i64().or_else(|| v.as_f64().map(f64_trunc)))
            .unwrap_or(0))
    }

    pub fn default_gpu_ids(&self) -> AppResult<Vec<i64>> {
        Ok(match self.get("default_gpu_ids")? {
            Some(Value::Array(a)) => a
                .iter()
                .filter_map(|x| x.as_i64().or_else(|| x.as_f64().map(f64_trunc)))
                .collect(),
            _ => Vec::new(),
        })
    }

    /// {host id: {engine id: image}}; anything in the table that is not that shape is ignored, not trusted.
    pub fn host_image_pins(&self) -> AppResult<Map<String, Value>> {
        let Some(Value::Object(raw)) = self.get("host_image_pins")? else {
            return Ok(Map::new());
        };
        let mut out = Map::new();
        for (h, pins) in raw {
            if let Value::Object(p) = pins {
                let kept: Map<String, Value> =
                    p.into_iter().filter(|(_, i)| i.is_string()).collect();
                out.insert(h, Value::Object(kept));
            }
        }
        Ok(out)
    }

    /// The image to run `engine` with on `host_id`: that host's pin, else the engine's global pin, else None.
    pub fn image_pin(&self, engine: &str, host_id: &str) -> AppResult<Option<String>> {
        let host = self.host_image_pins()?;
        if let Some(Value::String(p)) = host.get(host_id).and_then(|h| h.get(engine))
            && !p.is_empty()
        {
            return Ok(Some(p.clone()));
        }
        Ok(match self.get("image_pins")? {
            Some(Value::Object(pins)) => {
                pins.get(engine).and_then(Value::as_str).map(str::to_string)
            }
            _ => None,
        })
    }

    pub fn effective(&self) -> AppResult<Value> {
        let mut m = Map::new();
        m.insert(
            "hf_cache_dir".into(),
            Value::String(py_path_str(&self.hf_cache_dir())),
        );
        m.insert(
            "default_gpu_ids".into(),
            Value::from(self.default_gpu_ids()?),
        );
        m.insert("idle_ttl_s".into(), Value::from(self.idle_ttl_s()?));
        m.insert(
            "image_pins".into(),
            self.get("image_pins")?
                .unwrap_or_else(|| Value::Object(Map::new())),
        );
        m.insert(
            "host_image_pins".into(),
            Value::Object(self.host_image_pins()?),
        );
        m.insert("hf_token".into(), Value::String(self.token_status()));
        m.insert(
            "docker_context".into(),
            Value::String(self.cfg.docker_context.clone()),
        );
        m.insert(
            "engine_network".into(),
            Value::String(self.cfg.engine_network.clone()),
        );
        let proxied = self
            .cfg
            .egress_proxy
            .as_deref()
            .is_some_and(|p| !p.is_empty());
        m.insert(
            "egress_mode".into(),
            Value::String(if proxied { "proxied" } else { "direct" }.into()),
        );
        m.insert(
            "egress_proxy".into(),
            public_proxy(self.cfg.egress_proxy.as_deref()).map_or(Value::Null, Value::String),
        );
        m.insert(
            "require_egress_proxy".into(),
            Value::Bool(self.cfg.require_egress_proxy),
        );
        m.insert(
            "port_range".into(),
            Value::from(vec![self.cfg.port_range_start, self.cfg.port_range_end]),
        );
        m.insert(
            "hf_allowed_hosts".into(),
            Value::from(self.cfg.hf_allowed_hosts.clone()),
        );
        Ok(Value::Object(m))
    }

    pub fn update(&self, patch: &SettingsPatch) -> AppResult<Value> {
        let mut data: Vec<(&str, Value)> = Vec::new();
        if let Some(v) = &patch.default_gpu_ids {
            data.push(("default_gpu_ids", Value::from(v.clone())));
        }
        if let Some(v) = patch.idle_ttl_s {
            data.push(("idle_ttl_s", Value::from(v)));
        }
        if let Some(pins) = &patch.image_pins {
            for (eng, img) in pins {
                let img = img.as_str().unwrap_or_default();
                if !img.contains(':') || img.ends_with(":latest") {
                    return Err(bad_request(format!(
                        "image pin for {eng} must be a pinned tag (not :latest)"
                    ))
                    .code("invalid_setting")
                    .into());
                }
                if !self.engine_image_allowed(img) {
                    return Err(bad_request(format!(
                        "image {img} is not on ENGINE_IMAGE_ALLOWLIST"
                    ))
                    .code("image_not_allowed")
                    .into());
                }
            }
            data.push(("image_pins", Value::Object(pins.clone())));
        }
        if let Some(h) = &patch.host_image_pins {
            data.push(("host_image_pins", Value::Object(clean_host_image_pins(h)?)));
        }
        for (k, v) in data {
            self.db.execute(
                "INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                &[&k, &pyjson::dumps(&v)],
            )?;
        }
        self.effective()
    }

    // -- API keys ------------------------------------------------------------------------------
    pub fn create_key(&self, name: &str, role: &str) -> AppResult<KeyInfo> {
        let secret = format!(
            "ec_{}",
            token_urlsafe(32)
                .map_err(|_| crate::domain::errors::internal("unexpected server error"))?
        );
        let kid = new_id("key_");
        let ts = now();
        let prefix: String = secret.chars().take(8).collect();
        self.db.execute(
            "INSERT INTO api_keys(id,name,role,prefix,hash,created_at) VALUES(?,?,?,?,?,?)",
            &[&kid, &name, &role, &prefix, &key_hash(&secret), &ts],
        )?;
        Ok(KeyInfo {
            id: kid,
            name: name.into(),
            role: role.into(),
            prefix,
            created_at: ts,
            last_used_at: None,
            secret: Some(secret),
        })
    }

    pub fn list_keys(&self) -> AppResult<Vec<KeyInfo>> {
        let rows = self.db.all(
            "SELECT * FROM api_keys WHERE revoked_at IS NULL ORDER BY created_at",
            &[],
        )?;
        Ok(rows
            .iter()
            .map(|r| KeyInfo {
                id: r.str_or("id", ""),
                name: r.str_or("name", ""),
                role: r.str_or("role", ""),
                prefix: r.str_or("prefix", ""),
                created_at: r.f64("created_at").unwrap_or(0.0),
                last_used_at: r.f64("last_used_at"),
                secret: None,
            })
            .collect())
    }

    pub fn revoke_key(&self, key_id: &str) -> AppResult<()> {
        let n = self.db.execute(
            "UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL",
            &[&now(), &key_id],
        )?;
        if n == 0 {
            return Err(not_found(format!("no such key: {key_id}")).into());
        }
        Ok(())
    }

    /// The principal of a bearer key, or None. The stored hash is compared in constant time. The key's
    /// `last_used_at` is written on its first use and then at most once every [`LAST_USED_EVERY_S`] (Python writes
    /// it on every request: one database write per API call).
    pub fn authenticate(&self, secret: &str) -> AppResult<Option<Principal>> {
        let h = key_hash(secret);
        let Some(row) = self.db.one(
            "SELECT id, name, role, hash, last_used_at FROM api_keys WHERE hash=? AND revoked_at IS NULL",
            &[&h],
        )?
        else {
            return Ok(None);
        };
        let stored = row.str_or("hash", "");
        if !bool::from(stored.as_bytes().ct_eq(h.as_bytes())) {
            return Ok(None);
        }
        let ts = now();
        if last_used_due(row.f64("last_used_at"), ts) {
            self.db.execute(
                "UPDATE api_keys SET last_used_at=? WHERE hash=?",
                &[&ts, &h],
            )?;
        }
        Ok(Some(Principal {
            name: row.str_or("name", ""),
            role: row.str_or("role", ""),
            key_id: row.text("id"),
        }))
    }

    /// First start: mint an admin key and write it 0600. Returns the file path if one was created.
    pub fn ensure_bootstrap_key(&self, data_dir: &Path) -> AppResult<Option<PathBuf>> {
        if self
            .db
            .one("SELECT 1 FROM api_keys LIMIT 1", &[])?
            .is_some()
        {
            return Ok(None);
        }
        let info = self.create_key("bootstrap-admin", "admin")?;
        let path = data_dir.join(BOOTSTRAP_KEY_FILE);
        let io = |_e: std::io::Error| {
            crate::domain::errors::AppError::Problem(crate::domain::errors::internal(
                "unexpected server error",
            ))
        };
        create_dir_0700(data_dir).map_err(io)?;
        // never through a symlink or into a FIFO; O_CREAT's mode is ignored for a pre-existing, possibly looser file,
        // so the mode is forced on the descriptor
        let mut f = crate::fsutil::create_private(&path, 0o600).map_err(io)?;
        std::io::Write::write_all(
            &mut f,
            format!("{}\n", info.secret.unwrap_or_default()).as_bytes(),
        )
        .map_err(io)?;
        Ok(Some(path))
    }
}

/// A key's `last_used_at` is refreshed at most this often (seconds).
pub const LAST_USED_EVERY_S: f64 = 60.0;

/// Whether a use at `now` writes `last_used_at`: the first use (none recorded), a use at least
/// [`LAST_USED_EVERY_S`] after the recorded one, or a recorded use ahead of the clock (the clock went back).
fn last_used_due(recorded: Option<f64>, now: f64) -> bool {
    recorded.is_none_or(|r| !(0.0..LAST_USED_EVERY_S).contains(&(now - r)))
}

#[allow(clippy::cast_possible_truncation)]
fn f64_trunc(x: f64) -> i64 {
    x as i64
}

/// `str(pathlib.Path)`
pub fn py_path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn svc() -> (tempfile::TempDir, SettingsService) {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = Settings::defaults();
        cfg.data_dir = dir.path().join("data");
        cfg.hf_token_file = dir.path().join("hf_token");
        let store = Arc::new(Store::open(&cfg.db_path()).unwrap());
        let s = SettingsService::new(
            store,
            Arc::new(cfg),
            Box::new(|| vec!["vllm/vllm-openai:v1".to_string()]),
        );
        (dir, s)
    }

    #[test]
    fn test_keys_create_authenticate_revoke() {
        let (_d, s) = svc();
        let k = s.create_key("ci", "viewer").unwrap();
        let secret = k.secret.clone().unwrap();
        assert!(secret.starts_with("ec_") && secret.len() == 46 && k.prefix == secret[..8]);
        let p = s.authenticate(&secret).unwrap().unwrap();
        assert_eq!(
            (p.role.as_str(), p.key_id.as_deref()),
            ("viewer", Some(k.id.as_str()))
        );
        assert!(s.authenticate("ec_wrong").unwrap().is_none());
        assert!(s.list_keys().unwrap()[0].last_used_at.is_some());
        s.revoke_key(&k.id).unwrap();
        assert!(s.authenticate(&secret).unwrap().is_none());
        assert_eq!(s.revoke_key(&k.id).unwrap_err().status(), 404);
    }

    /// Audit F7: a key's `last_used_at` is written on its first use, then at most once a minute (Python writes it on
    /// every request), per key; a value ahead of the clock (the clock went back) is replaced at once.
    #[test]
    fn test_last_used_at_is_written_at_most_once_a_minute() {
        let (_d, s) = svc();
        let a = s.create_key("a", "viewer").unwrap();
        let b = s.create_key("b", "viewer").unwrap();
        let used = |s: &SettingsService, id: &str| {
            s.list_keys()
                .unwrap()
                .into_iter()
                .find(|k| k.id == id)
                .and_then(|k| k.last_used_at)
        };
        let set = |s: &SettingsService, id: &str, v: f64| {
            s.db.execute("UPDATE api_keys SET last_used_at=? WHERE id=?", &[&v, &id])
                .unwrap();
        };
        let auth = |s: &SettingsService, k: &KeyInfo| {
            s.authenticate(k.secret.as_deref().unwrap())
                .unwrap()
                .unwrap();
        };
        assert_eq!(used(&s, &a.id), None);
        auth(&s, &a);
        let first = used(&s, &a.id).unwrap(); // the first use writes
        assert!((now() - first).abs() < 5.0);
        assert_eq!(used(&s, &b.id), None, "per key");
        // used again 30 s after the recorded use: not written
        let recent = now() - 30.0;
        set(&s, &a.id, recent);
        auth(&s, &a);
        assert_eq!(used(&s, &a.id), Some(recent));
        // another key's first use still writes, whatever the first key did
        auth(&s, &b);
        assert!(used(&s, &b.id).is_some());
        // a minute after the recorded use: written again
        set(&s, &a.id, now() - 61.0);
        auth(&s, &a);
        assert!((now() - used(&s, &a.id).unwrap()).abs() < 5.0);
        // a recorded use in the future (the clock went back): replaced
        set(&s, &a.id, now() + 3600.0);
        auth(&s, &a);
        assert!(used(&s, &a.id).unwrap() <= now());
    }

    #[test]
    fn test_bootstrap_key_written_once_0600() {
        let (d, s) = svc();
        let p = s
            .ensure_bootstrap_key(&d.path().join("data"))
            .unwrap()
            .unwrap();
        let mode = std::fs::metadata(&p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        assert!(
            s.ensure_bootstrap_key(&d.path().join("data"))
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn test_pins_validation() {
        let (_d, s) = svc();
        let mut bad = SettingsPatch {
            image_pins: Some(
                serde_json::from_str(r#"{"vllm": "vllm/vllm-openai:latest"}"#).unwrap(),
            ),
            ..SettingsPatch::default()
        };
        assert_eq!(s.update(&bad).unwrap_err().code(), "invalid_setting");
        bad.image_pins = Some(serde_json::from_str(r#"{"vllm": "evil/x:1"}"#).unwrap());
        assert_eq!(s.update(&bad).unwrap_err().code(), "image_not_allowed");
        let ok = SettingsPatch {
            image_pins: Some(serde_json::from_str(r#"{"vllm": "vllm/vllm-openai:v1"}"#).unwrap()),
            host_image_pins: Some(
                serde_json::from_str(r#"{"gpu-a": {"vllm": "x/y:2"}, "gpu-b": {}}"#).unwrap(),
            ),
            ..SettingsPatch::default()
        };
        let eff = s.update(&ok).unwrap();
        assert_eq!(
            eff["host_image_pins"],
            serde_json::json!({"gpu-a": {"vllm": "x/y:2"}})
        );
        assert_eq!(
            s.image_pin("vllm", "gpu-a").unwrap().as_deref(),
            Some("x/y:2")
        );
        assert_eq!(
            s.image_pin("vllm", "local").unwrap().as_deref(),
            Some("vllm/vllm-openai:v1")
        );
        let r =
            clean_host_image_pins(&serde_json::from_str(r#"{"BAD": {}}"#).unwrap()).unwrap_err();
        assert_eq!(r.detail, "'BAD' is not a valid host id");
    }

    #[test]
    fn test_token_status_refuses_open_file() {
        let (d, s) = svc();
        let f = d.path().join("hf_token");
        std::fs::write(&f, "hf_abc\n").unwrap();
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(s.token_status().starts_with("[unusable: "));
        assert!(s.hf_token().is_none());
        std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert_eq!(s.token_status(), "[set, 6 chars] (file)");
    }

    #[test]
    fn test_public_proxy_drops_userinfo() {
        assert_eq!(
            public_proxy(Some("http://u:p@Proxy.Example:8082/x")).as_deref(),
            Some("http://proxy.example:8082")
        );
        assert_eq!(
            public_proxy(Some("http://127.0.0.1")).as_deref(),
            Some("http://127.0.0.1")
        );
        assert_eq!(public_proxy(None), None);
    }
}
