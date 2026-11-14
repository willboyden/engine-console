//! Process configuration (port of config.py).
//!
//! Environment variables, with an optional per-user env file (dotenv syntax) so settings survive restarts. Real
//! environment variables win over the file. Names are case-insensitive and carry no prefix (HF_TOKEN, EGRESS_PROXY,
//! ...), except the generic ones in [`PREFIXED`], which are read ONLY as `ENGINE_CONSOLE_<NAME>` (review finding L5).
//! Value parsing follows pydantic-settings: list fields are JSON, booleans accept 1/0/true/false/yes/no/on/off/t/f/
//! y/n, numbers are trimmed. A value that does not parse is a startup error naming the field (never echoing it).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

// The env file's reader, the settings table and the value parsers live in `ec-engines` (ADR 0023: the browser's
// Config files page checks an env draft with the same code); re-exported so every `config::...` path keeps working.
pub use ec_engines::envfile::{
    DEFAULT_HOST, EnvSummary, ExpansionTooLarge, MAX_ENV_EXPANDED, MAX_ENV_FILE, MAX_FLOAT_SETTING,
    PREFIX, PREFIXED, SETTINGS, SettingsError, ValueNote, check_env_draft, env_value_notes,
    env_var_name, parse_dotenv, try_parse_dotenv,
};
use ec_engines::envfile::{
    check_hf_cache_dir, check_host, err, p_bool, p_float, p_int, p_int_list, p_opt_path, p_opt_str,
    p_opt_str_list, p_path, p_port, p_str, p_str_list, p_u64, prefix_only,
};

pub const GATEWAY_IMAGE: &str =
    "nginx@sha256:62ff2089abf5a9ed33bd232895bef5e22f7bb4b200675cec49a5ebc48e3d4ac8";
pub const DEFAULT_REMOTE_COMMAND: &str = "/usr/local/lib/ecnode/ecnode --stdio";

#[derive(Debug, Clone, PartialEq)]
pub struct Settings {
    pub host: String,
    pub port: u16,
    pub data_dir: PathBuf,
    pub hf_cache_dir: PathBuf,
    pub hf_cache_container_path: String,
    pub hf_endpoint: String,
    /// ADR 7: never logged or returned
    pub hf_token: Option<String>,
    pub hf_token_file: PathBuf,
    pub hf_allowed_hosts: Vec<String>,
    pub docker_bin: String,
    pub docker_context: String,
    pub engine_network: String,
    pub gateway_image: String,
    pub egress_proxy: Option<String>,
    pub egress_ca_bundle: Option<PathBuf>,
    pub require_egress_proxy: bool,
    pub discovery_enabled: bool,
    pub discovery_interval_s: f64,
    pub discovery_ports: Vec<i64>,
    pub port_range_start: i64,
    pub port_range_end: i64,
    pub engine_image_allowlist: Vec<String>,
    pub gateway_image_allowlist: Vec<String>,
    pub enable_api_docs: bool,
    pub allowed_hosts: Vec<String>,
    pub body_cap_bytes: u64,
    pub chat_body_cap_bytes: u64,
    pub import_body_cap_bytes: u64,
    pub max_sse_connections: i64,
    pub max_sse_per_key: i64,
    pub max_chat_streams: i64,
    pub max_chat_streams_per_key: i64,
    pub sse_idle_timeout_s: f64,
    pub max_download_queue: i64,
    pub startup_timeout_s: i64,
    pub engine_hf_offline: bool,
    pub otlp_endpoint: String,
    pub otlp_enabled: bool,
    pub frontend_dir: Option<PathBuf>,
    pub background_tasks: bool,
    pub scrape_interval_s: f64,
    pub supervisor_interval_s: f64,
    pub download_concurrency: i64,
    pub trust_loopback: bool,
    pub hosts_file: PathBuf,
    pub hub_key_file: PathBuf,
    pub node_policy: Option<PathBuf>,
    pub node_command: Option<Vec<String>>,
    pub remote_node_command: String,
    pub fleet_enabled: bool,
    pub restart_settle_s: f64,
}

/// The user's home (`Path.home()`: $HOME, else the password database).
pub fn home() -> PathBuf {
    if let Some(h) = std::env::var_os("HOME").filter(|h| !h.is_empty()) {
        return PathBuf::from(h);
    }
    nix::unistd::User::from_uid(nix::unistd::getuid())
        .ok()
        .flatten()
        .map_or_else(|| PathBuf::from("/"), |u| u.dir)
}

/// `ENV_FILE`: $ENGINE_CONSOLE_ENV_FILE (when set and non-empty), else ~/.config/engine-console/env.
pub fn env_file() -> PathBuf {
    match std::env::var("ENGINE_CONSOLE_ENV_FILE") {
        Ok(v) if !v.is_empty() => PathBuf::from(v),
        _ => home().join(".config").join("engine-console").join("env"),
    }
}

/// ~/.cache/huggingface, or its target when it is a symlink (the validator refuses symlinked cache dirs).
fn default_cache_dir(home: &Path) -> PathBuf {
    let p = home.join(".cache").join("huggingface");
    if p.is_symlink() {
        return std::fs::canonicalize(&p).unwrap_or(p);
    }
    p
}

impl Settings {
    /// All defaults (as `Settings()` with an empty environment and no env file).
    pub fn defaults() -> Self {
        let h = home();
        let cfg = h.join(".config").join("engine-console");
        Self {
            host: DEFAULT_HOST.into(),
            port: 8791,
            data_dir: h.join(".local").join("share").join("engine-console"),
            hf_cache_dir: default_cache_dir(&h),
            hf_cache_container_path: "/root/.cache/huggingface".into(),
            hf_endpoint: "https://huggingface.co".into(),
            hf_token: None,
            hf_token_file: cfg.join("hf_token"),
            hf_allowed_hosts: [
                "huggingface.co",
                "cdn-lfs*.huggingface.co",
                "cas-bridge.xethub.hf.co",
                "*.hf.co",
            ]
            .map(String::from)
            .to_vec(),
            docker_bin: "docker".into(),
            docker_context: "rootless".into(),
            engine_network: "engine-console-engines".into(),
            gateway_image: GATEWAY_IMAGE.into(),
            egress_proxy: None,
            egress_ca_bundle: None,
            require_egress_proxy: false,
            discovery_enabled: true,
            discovery_interval_s: 10.0,
            discovery_ports: [8000, 30000, 11434, 8001]
                .into_iter()
                .chain(18000..18100)
                .collect(),
            port_range_start: 18000,
            port_range_end: 18099,
            engine_image_allowlist: Vec::new(),
            gateway_image_allowlist: vec![GATEWAY_IMAGE.into()],
            enable_api_docs: false,
            allowed_hosts: Vec::new(),
            body_cap_bytes: 1024 * 1024,
            chat_body_cap_bytes: 32 * 1024 * 1024,
            import_body_cap_bytes: 256 * 1024,
            max_sse_connections: 8,
            max_sse_per_key: 4,
            max_chat_streams: 16,
            max_chat_streams_per_key: 4,
            sse_idle_timeout_s: 300.0,
            max_download_queue: 50,
            startup_timeout_s: 1800,
            engine_hf_offline: true,
            otlp_endpoint: "localhost:4317".into(),
            otlp_enabled: true,
            frontend_dir: None,
            background_tasks: true,
            scrape_interval_s: 2.0,
            supervisor_interval_s: 2.0,
            download_concurrency: 2,
            trust_loopback: false,
            hosts_file: cfg.join("hosts.json"),
            hub_key_file: cfg.join("ssh").join("id_ed25519"),
            node_policy: None,
            node_command: None,
            remote_node_command: DEFAULT_REMOTE_COMMAND.into(),
            fleet_enabled: true,
            restart_settle_s: 4.0,
        }
    }

    /// `Settings()`: the process environment over the env file over the defaults.
    pub fn from_env() -> Result<Self, SettingsError> {
        let environ: Vec<(String, String)> = std::env::vars_os()
            .filter_map(|(k, v)| Some((k.into_string().ok()?, v.into_string().ok()?)))
            .collect();
        Self::load(&environ, Some(&env_file()))
    }

    /// The same, from an explicit environment and env file (tests).
    pub fn load(
        environ: &[(String, String)],
        env_file: Option<&Path>,
    ) -> Result<Self, SettingsError> {
        let file_vars = env_file
            .map(|p| read_env_file(p, environ))
            .unwrap_or_default();
        // env-file first, then the environment over it (env wins); names lower-cased, prefix rule applied to each
        let mut vars: HashMap<String, String> = HashMap::new();
        for (k, v) in prefix_only(&file_vars) {
            vars.insert(k, v);
        }
        for (k, v) in prefix_only(environ) {
            vars.insert(k, v);
        }
        let mut s = Self::defaults();
        s.apply(&vars)?;
        s.validate()?;
        Ok(s)
    }

    fn apply(&mut self, vars: &HashMap<String, String>) -> Result<(), SettingsError> {
        let get = |name: &str| vars.get(name).map(String::as_str);
        macro_rules! set {
            ($field:ident, $parse:expr) => {
                if let Some(v) = get(stringify!($field)) {
                    self.$field = $parse(stringify!($field), v)?;
                }
            };
        }
        set!(host, p_str);
        set!(port, p_port);
        set!(data_dir, p_path);
        set!(hf_cache_dir, p_path);
        set!(hf_cache_container_path, p_str);
        set!(hf_endpoint, p_str);
        set!(hf_token, p_opt_str);
        set!(hf_token_file, p_path);
        set!(hf_allowed_hosts, p_str_list);
        set!(docker_bin, p_str);
        set!(docker_context, p_str);
        set!(engine_network, p_str);
        set!(gateway_image, p_str);
        set!(egress_proxy, p_opt_str);
        set!(egress_ca_bundle, p_opt_path);
        set!(require_egress_proxy, p_bool);
        set!(discovery_enabled, p_bool);
        set!(discovery_interval_s, p_float);
        set!(discovery_ports, p_int_list);
        set!(port_range_start, p_int);
        set!(port_range_end, p_int);
        set!(engine_image_allowlist, p_str_list);
        set!(gateway_image_allowlist, p_str_list);
        set!(enable_api_docs, p_bool);
        set!(allowed_hosts, p_str_list);
        set!(body_cap_bytes, p_u64);
        set!(chat_body_cap_bytes, p_u64);
        set!(import_body_cap_bytes, p_u64);
        set!(max_sse_connections, p_int);
        set!(max_sse_per_key, p_int);
        set!(max_chat_streams, p_int);
        set!(max_chat_streams_per_key, p_int);
        set!(sse_idle_timeout_s, p_float);
        set!(max_download_queue, p_int);
        set!(startup_timeout_s, p_int);
        set!(engine_hf_offline, p_bool);
        set!(otlp_endpoint, p_str);
        set!(otlp_enabled, p_bool);
        set!(frontend_dir, p_opt_path);
        set!(background_tasks, p_bool);
        set!(scrape_interval_s, p_float);
        set!(supervisor_interval_s, p_float);
        set!(download_concurrency, p_int);
        set!(trust_loopback, p_bool);
        set!(hosts_file, p_path);
        set!(hub_key_file, p_path);
        set!(node_policy, p_opt_path);
        set!(node_command, p_opt_str_list);
        set!(remote_node_command, p_str);
        set!(fleet_enabled, p_bool);
        set!(restart_settle_s, p_float);
        Ok(())
    }

    /// The field validators of config.py.
    pub fn validate(&self) -> Result<(), SettingsError> {
        self.validate_values()?;
        if self.hf_cache_dir.is_symlink() {
            return Err(err("hf_cache_dir", "HF_CACHE_DIR must not be a symlink"));
        }
        Ok(())
    }

    /// [`Self::validate`] without its one look at the disk (whether HF_CACHE_DIR is a symlink): the Config files
    /// page's check of a draft looks up no path the draft names (second audit F6); start-up still checks it.
    fn validate_values(&self) -> Result<(), SettingsError> {
        check_host(&self.host)?;
        check_hf_cache_dir(&self.hf_cache_dir.to_string_lossy())
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("console.db")
    }

    pub fn hub_dir(&self) -> PathBuf {
        self.hf_cache_dir.join("hub")
    }
}

/// The bare PREFIXED names (HOST, PORT, ...) set in `environ` or in the env file, which are therefore ignored.
/// Names only: their values are never returned.
pub fn legacy_env_names(environ: &[(String, String)], env_file: Option<&Path>) -> Vec<String> {
    let mut found: Vec<String> = environ
        .iter()
        .map(|(k, _)| k.to_ascii_lowercase())
        .collect();
    if let Some(p) = env_file {
        found.extend(
            read_env_file(p, environ)
                .into_iter()
                .map(|(k, _)| k.to_ascii_lowercase()),
        );
    }
    PREFIXED
        .iter()
        .filter(|n| found.iter().any(|f| f == *n))
        .map(|n| n.to_ascii_uppercase())
        .collect()
}

/// main.py `warn_legacy_env`: the one start-up warning when an old unprefixed name is still set (it is ignored now, so
/// the console may not be listening where the operator expects), or None when nothing is stale. Names only, never
/// values.
pub fn legacy_env_warning(environ: &[(String, String)], env_file: Option<&Path>) -> Option<String> {
    let old = legacy_env_names(environ, env_file);
    if old.is_empty() {
        return None;
    }
    let new: Vec<String> = old.iter().map(|n| format!("{PREFIX}{n}")).collect();
    let file = env_file.map_or_else(|| "None".to_string(), |p| p.display().to_string());
    Some(format!(
        "ignoring {} (set in the environment or {file}): these settings are read only with the {PREFIX} prefix now; use {}",
        old.join(", "),
        new.join(", ")
    ))
}

/// The env-file editor's check (Config files page): `text` read exactly as the hub reads its env file (dotenv, the
/// prefix rule, every value parser and validator) over the defaults, with an EMPTY environment, so a `${VAR}`
/// expands to its `:-default` (or nothing) and no real environment variable takes part. No path the draft names is
/// looked up on disk: the start-up check that HF_CACHE_DIR is not a symlink is left out. Returns the names the file
/// sets, as written and in order, and the first error. Values never leave this function; nothing is written.
pub fn check_env_text(text: &str) -> (Vec<String>, Result<Settings, SettingsError>) {
    let vars = parse_dotenv(text, &[]);
    let names = vars.iter().map(|(k, _)| k.clone()).collect();
    let mut map: HashMap<String, String> = HashMap::new();
    for (k, v) in prefix_only(&vars) {
        map.insert(k, v);
    }
    let mut s = Settings::defaults();
    let r = s.apply(&map).and_then(|()| s.validate_values()).map(|()| s);
    (names, r)
}

// ── the env file (python-dotenv's dotenv_values, the subset pydantic-settings relies on) ─────────────────────
/// KEY=VALUE lines; `export ` prefix; `#` comments (whole-line, and after an unquoted value when preceded by a
/// space); single quotes literal; double quotes with \n \t \" \\ escapes; `${VAR}` / `${VAR:-default}` expansion
/// in unquoted and double-quoted values (from earlier keys of the file, then the environment). A missing or
/// unreadable file is empty. Key order kept. Hardening: the file is opened without blocking and must be a regular
/// file (a FIFO there would hang start-up) of at most `MAX_ENV_FILE` bytes; otherwise it counts as unreadable.
pub fn read_env_file(path: &Path, environ: &[(String, String)]) -> Vec<(String, String)> {
    let Ok(text) = crate::fsutil::read_text(path, crate::fsutil::Follow::Yes, MAX_ENV_FILE) else {
        return Vec::new();
    };
    parse_dotenv(&text, environ)
}

/// Why the hub ignores its env file (reads it as empty), or None when the file is read or does not exist. A reason
/// and never a value or a fragment of the file: `engine-console` logs it once at start-up, with the path (second
/// audit F4: an ignored file used to be dropped silently, so its settings quietly did not apply).
pub fn env_file_problem(path: &Path, environ: &[(String, String)]) -> Option<String> {
    use std::io::ErrorKind;
    let bytes = match crate::fsutil::read_file(path, crate::fsutil::Follow::Yes, MAX_ENV_FILE) {
        Ok(b) => b,
        Err(e) if e.kind() == ErrorKind::NotFound => return None,
        Err(e) if e.kind() == ErrorKind::InvalidInput => {
            return Some("it is not a regular file".into());
        }
        Err(e) if e.kind() == ErrorKind::InvalidData => {
            return Some(format!(
                "it is larger than {} MiB",
                MAX_ENV_FILE / (1024 * 1024)
            ));
        }
        Err(e) => return Some(format!("it cannot be read ({})", e.kind())),
    };
    let Ok(text) = String::from_utf8(bytes) else {
        return Some("it is not valid UTF-8".into());
    };
    if try_parse_dotenv(&text, environ).is_err() {
        return Some(format!(
            "its ${{VAR}} expansions add up to more than {} MiB",
            MAX_ENV_EXPANDED / (1024 * 1024)
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
            .collect()
    }

    // Port of tests/core/test_env_prefix.py (names mirror the Python tests).
    const BARE: &[(&str, &str)] = &[
        ("HOST", "0.0.0.0"),
        ("PORT", "9"),
        ("ALLOWED_HOSTS", "[\"evil.example\"]"),
        ("TRUST_LOOPBACK", "true"),
        ("DATA_DIR", "/tmp/somewhere-else"),
    ];

    fn defaults(s: &Settings) -> bool {
        let d = Settings::defaults();
        (
            s.host.as_str(),
            s.port,
            s.allowed_hosts.is_empty(),
            s.trust_loopback,
            &s.data_dir,
        ) == ("127.0.0.1", 8791, true, false, &d.data_dir)
    }

    #[test]
    fn test_bare_generic_names_in_the_environment_are_ignored() {
        assert!(defaults(&Settings::load(&env(BARE), None).unwrap()));
    }

    #[test]
    fn test_bare_generic_names_in_the_env_file_are_ignored() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        let text: String = BARE.iter().map(|(k, v)| format!("{k}={v}\n")).collect();
        std::fs::write(&f, text).unwrap();
        assert!(defaults(&Settings::load(&env(&[]), Some(&f)).unwrap()));
    }

    #[test]
    fn test_prefixed_names_work_from_the_env_file_and_the_environment_wins() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        let data = dir.path().join("data");
        std::fs::write(
            &f,
            format!(
                "ENGINE_CONSOLE_HOST=10.0.0.1\nENGINE_CONSOLE_PORT=9000\nENGINE_CONSOLE_ALLOWED_HOSTS=[\"console.example\"]\n\
                 ENGINE_CONSOLE_TRUST_LOOPBACK=true\nENGINE_CONSOLE_DATA_DIR={}\nHOST=0.0.0.0\n",
                data.display()
            ),
        )
        .unwrap();
        let s = Settings::load(&env(&[]), Some(&f)).unwrap();
        assert_eq!(
            (
                s.host.as_str(),
                s.port,
                s.allowed_hosts.clone(),
                s.trust_loopback,
                &s.data_dir
            ),
            (
                "10.0.0.1",
                9000,
                vec!["console.example".to_string()],
                true,
                &data
            )
        );
        // bare in the environment: still ignored
        let s = Settings::load(
            &env(&[("ENGINE_CONSOLE_PORT", "9100"), ("HOST", "0.0.0.0")]),
            Some(&f),
        )
        .unwrap();
        assert_eq!((s.port, s.host.as_str()), (9100, "10.0.0.1"));
    }

    #[test]
    fn test_every_other_setting_keeps_its_documented_name() {
        let s = Settings::load(
            &env(&[
                ("EGRESS_PROXY", "http://127.0.0.1:8082"),
                ("MAX_SSE_PER_KEY", "3"),
                ("MAX_CHAT_STREAMS", "5"),
            ]),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                s.egress_proxy.as_deref(),
                s.max_sse_per_key,
                s.max_chat_streams
            ),
            (Some("http://127.0.0.1:8082"), 3, 5)
        );
        // only the five generic names take the prefix
        let s = Settings::load(
            &env(&[("ENGINE_CONSOLE_EGRESS_PROXY", "http://127.0.0.1:9")]),
            None,
        )
        .unwrap();
        assert_eq!(s.egress_proxy, None);
    }

    #[test]
    fn test_an_empty_host_is_refused_rather_than_listening_everywhere() {
        let e = Settings::load(&env(&[("ENGINE_CONSOLE_HOST", "")]), None).unwrap_err();
        assert_eq!(e.field, "host");
        assert!(
            e.message.contains("ENGINE_CONSOLE_HOST must not be empty"),
            "{e}"
        );
    }

    #[test]
    fn test_a_stale_name_gives_one_warning_with_the_new_names_and_no_values() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("env");
        std::fs::write(
            &f,
            "PORT=4242\nDATA_DIR=/srv/private-place\nHF_ENDPOINT=https://huggingface.co\n",
        )
        .unwrap();
        let environ = env(&[
            ("HOST", "203.0.113.9"),
            ("PATH", "/usr/bin"),
            ("ENGINE_CONSOLE_PORT", "8791"),
        ]);
        assert_eq!(
            legacy_env_names(&environ, Some(&f)),
            vec!["HOST", "PORT", "DATA_DIR"]
        );
        assert!(
            legacy_env_names(
                &env(&[("PATH", "/usr/bin")]),
                Some(&dir.path().join("missing"))
            )
            .is_empty()
        );
        assert_eq!(
            legacy_env_warning(&env(&[("PATH", "/usr/bin")]), None),
            None
        ); // nothing stale: no warning
        let msg = legacy_env_warning(&environ, Some(&f)).unwrap();
        for name in [
            "ENGINE_CONSOLE_HOST",
            "ENGINE_CONSOLE_PORT",
            "ENGINE_CONSOLE_DATA_DIR",
        ] {
            assert!(msg.contains(name), "{msg}");
        }
        for value in ["203.0.113.9", "4242", "private-place"] {
            assert!(!msg.contains(value), "{msg}");
        }
    }

    #[test]
    fn test_a_stray_exported_host_no_longer_widens_the_bind() {
        // The finding end to end at the settings layer (the server itself is never started on the console's port):
        // `engine-console` started with HOST=0.0.0.0 and PORT=80 in its environment still binds 127.0.0.1:8791, and the
        // start-up warning names ENGINE_CONSOLE_HOST.
        let dir = tempfile::tempdir().unwrap();
        let environ = env(&[("HOST", "0.0.0.0"), ("PORT", "80")]);
        let no_file = dir.path().join("no-env-file");
        let s = Settings::load(&environ, Some(&no_file)).unwrap();
        assert_eq!((s.host.as_str(), s.port), ("127.0.0.1", 8791));
        assert!(
            legacy_env_warning(&environ, Some(&no_file))
                .unwrap()
                .contains("ENGINE_CONSOLE_HOST")
        );
    }

    #[test]
    fn test_validators() {
        assert_eq!(
            Settings::load(&env(&[("ENGINE_CONSOLE_HOST", "  ")]), None)
                .unwrap_err()
                .field,
            "host"
        );
        assert_eq!(
            Settings::load(&env(&[("HF_CACHE_DIR", "rel/x")]), None)
                .unwrap_err()
                .field,
            "hf_cache_dir"
        );
        assert_eq!(
            Settings::load(&env(&[("HF_CACHE_DIR", "/a:b")]), None)
                .unwrap_err()
                .field,
            "hf_cache_dir"
        );
    }

    #[test]
    fn test_values_parse_like_pydantic_settings() {
        let s = Settings::load(
            &env(&[
                ("otlp_enabled", "no"),
                ("FLEET_ENABLED", "0"),
                ("NODE_COMMAND", "[\"/bin/ecnode\", \"--stdio\"]"),
                ("DISCOVERY_PORTS", "[1, 2]"),
                ("SCRAPE_INTERVAL_S", " 0.5 "),
            ]),
            None,
        )
        .unwrap();
        assert!(!s.otlp_enabled && !s.fleet_enabled);
        assert_eq!(
            s.node_command,
            Some(vec!["/bin/ecnode".to_string(), "--stdio".to_string()])
        );
        assert_eq!(s.discovery_ports, vec![1, 2]);
        assert!((s.scrape_interval_s - 0.5).abs() < f64::EPSILON);
        assert!(Settings::load(&env(&[("PORT_RANGE_START", "x")]), None).is_err());
        assert!(Settings::load(&env(&[("ALLOWED_HOSTS", "not json")]), None).is_ok()); // bare name: ignored
        assert!(
            Settings::load(&env(&[("ENGINE_CONSOLE_ALLOWED_HOSTS", "not json")]), None).is_err()
        );
    }

    #[test]
    fn test_dotenv_quotes_comments_and_expansion() {
        let got = parse_dotenv(
            "export A=1\nB='x # y'\nC=\"q\\n\" \nD=${A}-${MISSING:-z} # trailing\n#E=2\n",
            &env(&[]),
        );
        assert_eq!(
            got,
            env(&[("A", "1"), ("B", "x # y"), ("C", "q\n"), ("D", "1-z")])
                .into_iter()
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_dotenv_expansion_that_keeps_doubling_is_unreadable() {
        let mut text = format!("A0={}\n", "x".repeat(1024));
        for i in 1..40 {
            text.push_str(&format!("A{i}=${{A{}}}${{A{}}}\n", i - 1, i - 1));
        }
        assert_eq!(try_parse_dotenv(&text, &[]), Err(ExpansionTooLarge));
        assert!(parse_dotenv(&text, &[]).is_empty());
        // a few doublings stay far below the cap and expand as before
        let small: String = text.lines().take(4).map(|l| format!("{l}\n")).collect();
        let got = try_parse_dotenv(&small, &[]).unwrap();
        assert_eq!(got[3].1.len(), 8 * 1024);
        // the environment wins over the file, its first entry over a later one
        let got = parse_dotenv("A=f\nB=${A}\n", &env(&[("A", "e1"), ("A", "e2")]));
        assert_eq!(got[1].1, "e1");
    }

    #[test]
    fn test_a_draft_check_does_not_look_up_hf_cache_dir() {
        // second audit F6: the draft check names no file of the hub; only start-up refuses a symlinked HF_CACHE_DIR
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("cache");
        std::fs::create_dir(&target).unwrap();
        let link = dir.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let text = format!("HF_CACHE_DIR={}\n", link.display());
        let (names, r) = check_env_text(&text);
        assert_eq!(names, vec!["HF_CACHE_DIR".to_string()]);
        assert!(r.is_ok(), "{:?}", r.err());
        let file = dir.path().join("env");
        std::fs::write(&file, &text).unwrap();
        let e = Settings::load(&[], Some(&file)).err().unwrap();
        assert_eq!(
            (e.field.as_str(), e.message.as_str()),
            ("hf_cache_dir", "HF_CACHE_DIR must not be a symlink")
        );
        // the value checks still apply to a draft
        let (_, r) = check_env_text("HF_CACHE_DIR=relative/path\n");
        assert_eq!(r.err().unwrap().field, "hf_cache_dir");
    }

    #[test]
    fn test_an_ignored_env_file_says_why_and_never_what() {
        // second audit F4: the start-up warning's reason, for each way the file is ignored; no value leaks into it
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("env");
        assert_eq!(env_file_problem(&p, &[]), None); // missing: nothing to say
        std::fs::write(&p, "ENGINE_CONSOLE_PORT=8792\nHF_TOKEN=hf_SECRETVALUE123\n").unwrap();
        assert_eq!(env_file_problem(&p, &[]), None);
        let secret = "hf_SECRETVALUE123";
        let mut big = format!("HF_TOKEN={secret}\n");
        big.push_str(&"#".repeat(MAX_ENV_FILE));
        std::fs::write(&p, &big).unwrap();
        let why = env_file_problem(&p, &[]).unwrap();
        assert_eq!(why, "it is larger than 1 MiB");
        std::fs::write(&p, b"HF_TOKEN=hf_SECRETVALUE123\xff\n").unwrap();
        assert_eq!(env_file_problem(&p, &[]).unwrap(), "it is not valid UTF-8");
        let mut bomb = format!("HF_TOKEN={secret}\nA0={}\n", "x".repeat(1024));
        for i in 1..40 {
            bomb.push_str(&format!("A{i}=${{A{}}}${{A{}}}\n", i - 1, i - 1));
        }
        std::fs::write(&p, &bomb).unwrap();
        assert_eq!(
            env_file_problem(&p, &[]).unwrap(),
            "its ${VAR} expansions add up to more than 8 MiB"
        );
        std::fs::remove_file(&p).unwrap();
        std::fs::create_dir(&p).unwrap();
        assert_eq!(
            env_file_problem(&p, &[]).unwrap(),
            "it is not a regular file"
        );
    }

    #[test]
    fn test_float_settings_must_be_finite() {
        for v in ["inf", "-inf", "Infinity", "nan", "NaN", "1e999", "2e15"] {
            let e = Settings::load(&env(&[("SSE_IDLE_TIMEOUT_S", v)]), None).unwrap_err();
            assert_eq!(e.field, "sse_idle_timeout_s", "{v}");
            assert!(e.message.starts_with("Input should be"), "{v}: {e}");
        }
        let s = Settings::load(
            &env(&[("SSE_IDLE_TIMEOUT_S", "1e6"), ("RESTART_SETTLE_S", "-1")]),
            None,
        )
        .unwrap();
        assert!((s.sse_idle_timeout_s - 1e6).abs() < f64::EPSILON);
        // the uses turn it into a Duration without panicking
        let _ = std::time::Duration::from_secs_f64(MAX_FLOAT_SETTING);
    }

    /// `check_env_text` (the hub's loader) as the browser's draft check answers.
    fn loader(text: &str) -> (Vec<String>, Result<EnvSummary, SettingsError>) {
        let (names, r) = check_env_text(text);
        (
            names,
            r.map(|s| EnvSummary {
                host: s.host,
                trust_loopback: s.trust_loopback,
            }),
        )
    }

    const VALUES: &[&str] = &[
        "",
        " ",
        "x",
        "0",
        "1",
        "-1",
        "08",
        "1_000",
        "0.5",
        "2.0",
        "-3.5",
        "1e3",
        "1e999",
        "nan",
        "inf",
        "70000",
        "65535",
        "true",
        "no",
        "maybe",
        "[]",
        "[\"a\"]",
        "[1, 2]",
        "[1.5]",
        "[\"7\"]",
        "[true]",
        "[null]",
        "null",
        "{}",
        "/abs/dir",
        "rel/dir",
        "/a:b",
        "/a,b",
        "~/x",
        "'quoted value'",
        "\"dq\\n\"",
        "${MISSING:-5}",
        "${MISSING}",
    ];

    /// ADR 0023: the browser checks an env draft with `ec_engines::envfile::check_env_draft`, which has no `Settings`.
    /// For every setting and a spread of good and bad values (alone, prefixed or bare, and two at a time, so the
    /// first error must be the same one) it answers exactly what the hub's loader answers.
    #[test]
    fn test_env_draft_check_matches_the_loader() {
        let mut drafts: Vec<String> = vec![
            String::new(),
            "# only a comment\n".into(),
            "ENGINE_CONSOLE_HOST=\n".into(),
            "ENGINE_CONSOLE_HOST=0.0.0.0\nENGINE_CONSOLE_TRUST_LOOPBACK=yes\n".into(),
            "HOST=\nTRUST_LOOPBACK=1\n".into(),
            "PORT=x\nSSE_IDLE_TIMEOUT_S=y\n".into(),
            "HF_CACHE_DIR=relative\nENGINE_CONSOLE_HOST= \n".into(),
            "export A=1\nPORT_RANGE_START=${A}\n".into(),
        ];
        for (field, _) in SETTINGS {
            for v in VALUES {
                drafts.push(format!("{}={v}\n", env_var_name(field)));
                drafts.push(format!("{}={v}\n", field.to_ascii_uppercase()));
                drafts.push(format!("export {}={v}\nPORT=x\n", env_var_name(field)));
            }
        }
        for d in &drafts {
            assert_eq!(check_env_draft(d), loader(d), "draft {d:?}");
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(3000))]
        #[test]
        fn test_env_draft_check_matches_the_loader_on_random_drafts(
            lines in proptest::collection::vec((0..SETTINGS.len(), 0..VALUES.len(), proptest::bool::ANY), 0..6)
        ) {
            let text: String = lines
                .iter()
                .map(|&(f, v, bare)| {
                    let name = if bare { SETTINGS[f].0.to_ascii_uppercase() } else { env_var_name(SETTINGS[f].0) };
                    format!("{name}={}\n", VALUES[v])
                })
                .collect();
            proptest::prop_assert_eq!(check_env_draft(&text), loader(&text));
        }
    }

    #[test]
    fn test_settings_table_matches_apply() {
        let src = include_str!("config.rs");
        let re = regex::Regex::new(r"set!\((\w+), p_(\w+)\);").unwrap();
        let found: Vec<(String, String)> = re
            .captures_iter(src)
            .map(|c| (c[1].to_string(), c[2].to_string()))
            .collect();
        let table: Vec<(String, String)> = SETTINGS
            .iter()
            .map(|(n, k)| ((*n).to_string(), (*k).to_string()))
            .collect();
        assert_eq!(found, table);
        assert_eq!(env_var_name("port"), "ENGINE_CONSOLE_PORT");
        assert_eq!(env_var_name("hf_token_file"), "HF_TOKEN_FILE");
    }
}
