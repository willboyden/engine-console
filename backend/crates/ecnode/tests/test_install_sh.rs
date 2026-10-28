//! `node/install.sh --binary FILE`: the checks (sha256 against FILE.sha256, an ELF executable for this machine's
//! `uname -m`), the stage-and-rename swap of the whole prefix, the policy written once, and `--uninstall`.
//!
//! The script needs root and writes to /usr/local/lib and /etc, so the test runs a copy with these edits only, each
//! asserted to apply: the two paths point into a temp dir, the root check is skipped, `-o root -g root` and
//! `chown root:root` are dropped, the parent directory must belong to the test's user instead of root, and the final
//! `ecnode check` reads a hermetic policy (no discovery probes of this machine's loopback ports). Everything else
//! (argument parsing, the checks, the swap, the messages) is the script as shipped. The node's tools are the parity
//! fakes (ECNODE_TOOL_PATH), NVML is off and HOME is a temp dir, so no real docker, GPU or home directory is read.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    unreachable_pub
)]

use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::Command;

use ecnode::agent::VERSION;
use sha2::{Digest, Sha256};

const SCRIPT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../node/install.sh");
const FAKES: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../parity/fakes/bin");

struct Sandbox {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    script: PathBuf,
    home: PathBuf,
    check_policy: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        let home = tmp.path().join("home");
        let bin = tmp.path().join("script");
        for d in [&root, &home, &bin] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::set_permissions(d, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut text = std::fs::read_to_string(SCRIPT).unwrap();
        for (from, to) in [
            (
                "PREFIX=/usr/local/lib/ecnode\n",
                "PREFIX=\"$EC_TEST_ROOT/usr/local/lib/ecnode\"\n",
            ),
            (
                "POLICY_DIR=/etc/ecnode\n",
                "POLICY_DIR=\"$EC_TEST_ROOT/etc/ecnode\"\n",
            ),
            ("if [ \"$(id -u)\" -ne 0 ]; then", "if false; then"),
            (" -o root -g root", ""),
            ("chown root:root \"$STAGE\" ||", "true ||"),
            ("chown root:root \"$TMP_POLICY\" ||", "true ||"),
            (
                "if [ \"$parent_owner\" != \"0\" ]",
                "if [ \"$parent_owner\" != \"$(id -u)\" ]",
            ),
            (
                "as_user \"$BIN\" check --policy \"$POLICY\" ||",
                "as_user \"$BIN\" check --policy \"$EC_TEST_CHECK_POLICY\" ||",
            ),
        ] {
            assert!(
                text.contains(from),
                "install.sh no longer contains {from:?}: update this test"
            );
            text = text.replace(from, to);
        }
        let script = bin.join("install.sh");
        std::fs::write(&script, text).unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
        let check_policy = tmp.path().join("check-policy.json");
        std::fs::write(
            &check_policy,
            r#"{"version": 1, "mode": "monitor", "discovery_ports": [], "docker": {"binary": "docker"}}"#,
        )
        .unwrap();
        std::fs::set_permissions(&check_policy, std::fs::Permissions::from_mode(0o644)).unwrap();
        Self {
            _tmp: tmp,
            root,
            script,
            home,
            check_policy,
        }
    }

    fn prefix(&self) -> PathBuf {
        self.root.join("usr/local/lib/ecnode")
    }

    fn policy(&self) -> PathBuf {
        self.root.join("etc/ecnode/policy.json")
    }

    /// Run the script with exactly these arguments and a hermetic environment: (exit code, stdout, stderr).
    fn run(&self, args: &[&str]) -> (i32, String, String) {
        let out = Command::new("/bin/bash")
            .arg(&self.script)
            .args(args)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("HOME", &self.home)
            .env("LANG", "C.UTF-8")
            .env("ECNODE_NO_NVML", "1")
            .env("ECNODE_TOOL_PATH", format!("{FAKES}:/usr/bin:/bin"))
            .env("EC_TEST_ROOT", &self.root)
            .env("EC_TEST_CHECK_POLICY", &self.check_policy)
            .output()
            .unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    /// A copy of `bytes` as a downloaded release asset, with its `.sha256` (sha256sum's format) unless `sum` says
    /// otherwise: `Some(text)` writes that text instead.
    fn asset(&self, name: &str, bytes: &[u8], sum: Option<&str>) -> PathBuf {
        let dir = self._tmp.path().join("dl");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        let text = sum.map_or_else(
            || format!("{}  {name}\n", hex::encode(Sha256::digest(bytes))),
            str::to_string,
        );
        std::fs::write(dir.join(format!("{name}.sha256")), text).unwrap();
        p
    }

    /// No staging or old-copy directory is left next to the prefix.
    fn no_leftovers(&self) {
        let parent = self.root.join("usr/local/lib");
        if let Ok(rd) = std::fs::read_dir(&parent) {
            let names: Vec<String> = rd
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .collect();
            assert!(names.iter().all(|n| n == "ecnode"), "{names:?}");
        }
    }
}

fn ecnode_bytes() -> Vec<u8> {
    std::fs::read(env!("CARGO_BIN_EXE_ecnode")).unwrap()
}

fn mode(p: &Path) -> u32 {
    std::fs::metadata(p).unwrap().permissions().mode() & 0o7777
}

#[test]
fn test_installs_the_binary_writes_the_policy_once_and_replaces_the_whole_prefix() {
    let s = Sandbox::new();
    let file = s.asset("ecnode-x-arch", &ecnode_bytes(), None);
    let (code, out, err) = s.run(&["--binary", file.to_str().unwrap()]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert!(
        out.contains(&format!("installed ecnode {VERSION} in ")),
        "{out}"
    );
    assert!(
        out.contains("wrote ") && out.contains("MONITOR mode"),
        "{out}"
    );
    let bin = s.prefix().join("ecnode");
    assert_eq!(std::fs::read(&bin).unwrap(), ecnode_bytes());
    assert_eq!(mode(&bin), 0o755);
    assert_eq!(mode(&s.prefix()), 0o755);
    let policy: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(s.policy()).unwrap()).unwrap();
    assert_eq!(policy["mode"], "monitor");
    assert_eq!(mode(&s.policy()), 0o644);
    s.no_leftovers();

    // a re-run over an older install (the Python node's lib/ and launcher): the old prefix goes as a whole, the
    // policy stays exactly as it is
    std::fs::create_dir_all(s.prefix().join("lib/ecnode")).unwrap(); // under the prefix: its mode is not checked
    std::fs::write(s.prefix().join("lib/ecnode/__init__.py"), "").unwrap();
    std::fs::write(s.policy(), "{\"version\": 1, \"_comment\": \"mine\"}").unwrap();
    let (code, out, err) = s.run(&[&format!("--binary={}", file.display())]);
    assert_eq!(code, 0, "{out}\n{err}");
    assert!(out.contains("kept the existing policy"), "{out}");
    assert_eq!(
        std::fs::read_to_string(s.policy()).unwrap(),
        "{\"version\": 1, \"_comment\": \"mine\"}"
    );
    let entries: Vec<String> = std::fs::read_dir(s.prefix())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(entries, ["ecnode"]);
    s.no_leftovers();

    // --uninstall removes the code and keeps the policy
    let (code, out, _) = s.run(&["--uninstall"]);
    assert_eq!(code, 0, "{out}");
    assert!(!s.prefix().exists() && s.policy().exists(), "{out}");
}

/// Every refusal leaves an existing install exactly as it was and no staging directory behind.
fn assert_refused(s: &Sandbox, args: &[&str], why: &str) {
    let mut d = s.root.clone();
    for part in ["usr", "local", "lib", "ecnode"] {
        d.push(part);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    std::fs::write(s.prefix().join("marker"), "old").unwrap();
    let (code, out, err) = s.run(args);
    assert_eq!(code, 1, "{args:?}\n{out}\n{err}");
    assert!(err.contains(why), "{args:?}: wanted {why:?}\n{err}");
    assert_eq!(
        std::fs::read_to_string(s.prefix().join("marker")).unwrap(),
        "old"
    );
    assert!(!s.prefix().join("ecnode").exists() && !s.policy().exists());
    s.no_leftovers();
}

#[test]
fn test_refuses_a_binary_whose_sha256_does_not_match() {
    let s = Sandbox::new();
    let good = ecnode_bytes();
    let wrong = format!("{}  ecnode\n", "0".repeat(64));
    let f = s.asset("ecnode-bad-sum", &good, Some(&wrong));
    assert_refused(&s, &["--binary", f.to_str().unwrap()], "does not match");
    let f = s.asset("ecnode-no-hex", &good, Some("not a digest  ecnode\n"));
    assert_refused(
        &s,
        &["--binary", f.to_str().unwrap()],
        "does not start with a sha256 hex digest",
    );
    let f = s.asset("ecnode-short", &good, Some("abc123  ecnode\n"));
    assert_refused(
        &s,
        &["--binary", f.to_str().unwrap()],
        "does not start with a sha256 hex digest",
    );
    let f = s.asset("ecnode-no-sum", &good, None);
    std::fs::remove_file(f.with_file_name("ecnode-no-sum.sha256")).unwrap();
    assert_refused(&s, &["--binary", f.to_str().unwrap()], ".sha256 not found");
}

#[test]
fn test_refuses_a_file_that_is_not_an_executable_for_this_machine() {
    let s = Sandbox::new();
    // the right sha256, but built for the other architecture (ELF e_machine, bytes 18-19)
    let mut other = ecnode_bytes();
    let here = u16::from_le_bytes([other[18], other[19]]);
    let there: u16 = if here == 62 { 183 } else { 62 };
    other[18..20].copy_from_slice(&there.to_le_bytes());
    let f = s.asset("ecnode-other-arch", &other, None);
    assert_refused(
        &s,
        &["--binary", f.to_str().unwrap()],
        &format!("built for ELF machine {there}"),
    );
    // not an ELF file at all (a script with a matching sha256)
    let f = s.asset("ecnode-script", b"#!/bin/sh\necho ecnode 9.9.9\n", None);
    assert_refused(
        &s,
        &["--binary", f.to_str().unwrap()],
        "is not an ELF executable",
    );
    // a 32-bit ELF header
    let mut elf32 = ecnode_bytes();
    elf32[4] = 1;
    let f = s.asset("ecnode-elf32", &elf32, None);
    assert_refused(
        &s,
        &["--binary", f.to_str().unwrap()],
        "not a 64-bit little-endian ELF file",
    );
    // a symlink to a good binary, and a directory
    let good = s.asset("ecnode-good", &ecnode_bytes(), None);
    let link = good.with_file_name("ecnode-link");
    symlink(&good, &link).unwrap();
    std::fs::copy(
        good.with_file_name("ecnode-good.sha256"),
        link.with_file_name("ecnode-link.sha256"),
    )
    .unwrap();
    assert_refused(&s, &["--binary", link.to_str().unwrap()], "is a symlink");
    assert_refused(
        &s,
        &["--binary", s.home.to_str().unwrap()],
        "is not a regular file",
    );
}

#[test]
fn test_arguments() {
    let s = Sandbox::new();
    let (code, _, err) = s.run(&[]);
    assert_eq!(code, 1);
    assert!(
        err.contains("--binary") && err.contains("the tag python-final"),
        "{err}"
    );
    for (args, why) in [
        (vec!["--binary"], "--binary needs a file"),
        (vec!["--frobnicate"], "unknown option"),
        (vec!["--uninstall", "--binary", "x"], "not both"),
    ] {
        let (code, _, err) = s.run(&args);
        assert_eq!(code, 1, "{args:?}");
        assert!(err.contains(why), "{args:?}: {err}");
    }
    let (code, out, _) = s.run(&["--help"]);
    assert_eq!(code, 0);
    assert!(
        out.contains("--binary FILE") && out.contains("--uninstall"),
        "{out}"
    );
}
