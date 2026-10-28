//! External programs the hub starts (`ssh`, `ssh-keygen`, `xdg-open`, `notify-send`, a relative node command):
//! found on a fixed, safe search path, never on the caller's PATH.
//!
//! Python resolves them through the inherited `PATH` and passes that `PATH` on (hosts.py `ssh_env`/`_local_env`). A
//! console started from a shell whose PATH begins with a writable directory (`~/bin`, `.`) would then run whatever
//! `ssh` sits there, with the hub's key. The hub instead searches [`SAFE_PATH`] and gives its children that same
//! PATH (PORTING.md deviation, see the hardening report). `ENGINE_CONSOLE_TOOL_PATH` replaces the search path in debug
//! builds only: it exists for tests and development (fake `ssh`, `xdg-open`, `docker` ...; every test and harness runs
//! the debug binaries), is read from the process environment only (never from the env file), and keeps only absolute
//! entries. A release build does not contain it (`cfg(debug_assertions)`), so the fixed path holds whatever the
//! environment says.

use std::os::unix::fs::PermissionsExt as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};

/// The search path for every program the hub starts, and the PATH its children get.
pub const SAFE_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
/// Replaces [`SAFE_PATH`] in debug builds (tests and development); a release build never reads it.
pub const TOOL_PATH_ENV: &str = "ENGINE_CONSOLE_TOOL_PATH";

/// The tool search path of this process: [`SAFE_PATH`], or in a debug build `ENGINE_CONSOLE_TOOL_PATH` (absolute
/// entries only) when it is set.
pub fn tool_path() -> String {
    tool_path_from(&|k| std::env::var(k).ok())
}

/// `tool_path()` over a given environment (`env(name)`), for tests.
pub fn tool_path_from(env: &dyn Fn(&str) -> Option<String>) -> String {
    #[cfg(debug_assertions)]
    if let Some(dirs) = tool_path_override(env) {
        return dirs;
    }
    #[cfg(not(debug_assertions))]
    let _ = env;
    SAFE_PATH.to_string()
}

/// The absolute entries of `ENGINE_CONSOLE_TOOL_PATH`, if it is set and has any. Compiled into debug builds only.
#[cfg(debug_assertions)]
fn tool_path_override(env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    let v = env(TOOL_PATH_ENV)?;
    let dirs: Vec<&str> = v.split(':').filter(|d| d.starts_with('/')).collect();
    (!dirs.is_empty()).then(|| dirs.join(":"))
}

fn is_executable_file(p: &Path) -> bool {
    std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// The executable `name` on `search` (colon-separated; relative entries ignored). A name holding '/' is a path and
/// is returned as given (Python's and execvp's rule); an empty name is None.
pub fn find_in(name: &str, search: &str) -> Option<PathBuf> {
    if name.is_empty() {
        return None;
    }
    if name.contains('/') {
        return Some(PathBuf::from(name));
    }
    search
        .split(':')
        .filter(|d| d.starts_with('/'))
        .map(|d| Path::new(d).join(name))
        .find(|p| is_executable_file(p))
}

/// `name` on this process's tool path.
pub fn resolve(name: &str) -> Option<PathBuf> {
    find_in(name, &tool_path())
}

/// What a short-lived tool printed, kept up to a cap while it was read (never collected whole after it exited).
#[derive(Debug)]
pub struct Captured {
    /// None: the tool did not finish within its timeout and was killed
    pub status: Option<ExitStatus>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// more was printed than kept (the rest was read and dropped, so the tool never blocks on a full pipe)
    pub truncated: bool,
}

/// Read `r` to its end, keeping at most `cap` bytes; returns (kept, whether more came).
fn drain_capped(mut r: impl std::io::Read, cap: usize) -> (Vec<u8>, bool) {
    let mut kept = Vec::new();
    let mut more = false;
    let mut buf = [0u8; 8192];
    loop {
        match r.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let room = cap.saturating_sub(kept.len());
                kept.extend_from_slice(&buf[..n.min(room)]);
                more |= n > room;
            }
        }
    }
    (kept, more)
}

/// Run `prog args` (argv only, `env` as its whole environment, stdin /dev/null) for at most `timeout`, keeping at most
/// `cap` bytes of each output stream while it is read. A tool still running at the deadline is killed.
pub fn run_capped(
    prog: &Path,
    args: &[&str],
    env: &[(String, String)],
    timeout: Duration,
    cap: usize,
) -> std::io::Result<Captured> {
    let mut child = Command::new(prog)
        .args(args)
        .env_clear()
        .envs(env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, pipe) in [
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    ]
    .into_iter()
    .enumerate()
    {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let got = pipe.map_or_else(|| (Vec::new(), false), |p| drain_capped(p, cap));
            let _ = tx.send((i, got));
        });
    }
    drop(tx);
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break Some(s);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    // the pipes close when the tool exits; a grandchild still holding one is given 2 s, then abandoned
    let mut out = Captured {
        status,
        stdout: Vec::new(),
        stderr: Vec::new(),
        truncated: false,
    };
    let until = Instant::now() + Duration::from_secs(2);
    for _ in 0..2 {
        let left = until.saturating_duration_since(Instant::now());
        let Ok((i, (kept, more))) = rx.recv_timeout(left) else {
            break;
        };
        out.truncated |= more;
        if i == 0 {
            out.stdout = kept;
        } else {
            out.stderr = kept;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_tool_path_ignores_path_and_keeps_absolute_override_entries() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |k: &str| {
                pairs
                    .iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| (*v).to_string())
            }
        };
        assert_eq!(
            tool_path_from(&env(&[("PATH", "/home/x/bin:.")])),
            SAFE_PATH
        );
        // the override exists in debug builds only (see the next test)
        let want = if cfg!(debug_assertions) {
            "/opt/fakes:/usr/bin"
        } else {
            SAFE_PATH
        };
        assert_eq!(
            tool_path_from(&env(&[(TOOL_PATH_ENV, "relative:/opt/fakes:./x:/usr/bin")])),
            want
        );
        assert_eq!(
            tool_path_from(&env(&[(TOOL_PATH_ENV, "only-relative")])),
            SAFE_PATH
        );
    }

    /// Audit F10: the override is compiled into debug builds only (every test and harness runs debug binaries); a
    /// release build always searches [`SAFE_PATH`], whatever `ENGINE_CONSOLE_TOOL_PATH` says. This test runs in both
    /// build modes and checks the mode it was built in (`cargo test --release` runs the release half).
    #[test]
    fn test_the_tool_path_override_exists_only_in_debug_builds() {
        let fakes = |k: &str| (k == TOOL_PATH_ENV).then(|| "/opt/fakes".to_string());
        let got = tool_path_from(&fakes);
        if cfg!(debug_assertions) {
            assert_eq!(got, "/opt/fakes");
        } else {
            assert_eq!(got, SAFE_PATH, "a release build honoured {TOOL_PATH_ENV}");
        }
    }

    #[test]
    fn test_find_in_needs_an_executable_file() {
        let d = tempfile::tempdir().unwrap();
        let tool = d.path().join("tool");
        std::fs::write(&tool, "#!/bin/sh\n").unwrap();
        let search = format!("relative:{}", d.path().display());
        assert_eq!(find_in("tool", &search), None); // not executable yet
        std::fs::set_permissions(&tool, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(find_in("tool", &search), Some(tool));
        std::fs::create_dir(d.path().join("adir")).unwrap();
        assert_eq!(find_in("adir", &search), None);
        assert_eq!(find_in("/abs/x", "/nowhere"), Some(PathBuf::from("/abs/x")));
        assert_eq!(find_in("", &search), None);
    }

    /// A tool that floods its output is capped while it runs, never buffered whole (hardening finding 5). The tools
    /// are `/bin/sh -c` bodies, so no freshly written file is exec'd while other tests fork (ETXTBSY).
    #[test]
    fn test_run_capped_bounds_a_flooding_tool_and_kills_one_that_never_ends() {
        let sh = Path::new("/bin/sh");
        let env = vec![("PATH".to_string(), "/usr/bin:/bin".to_string())];
        // 16 MiB on each stream, then an exit status
        let flood = "head -c 16777216 /dev/zero; head -c 16777216 /dev/zero >&2; exit 3";
        let c = run_capped(sh, &["-c", flood], &env, Duration::from_secs(30), 4096).unwrap();
        assert_eq!(c.status.and_then(|s| s.code()), Some(3));
        assert_eq!(
            (c.stdout.len(), c.stderr.len(), c.truncated),
            (4096, 4096, true)
        );
        // an endless writer: killed at the deadline, and what was kept is still only the cap
        let t = Instant::now();
        let c = run_capped(
            sh,
            &["-c", "exec yes"],
            &env,
            Duration::from_millis(500),
            1000,
        )
        .unwrap();
        assert!(c.status.is_none() && c.stdout.len() == 1000 && c.truncated);
        assert!(t.elapsed() < Duration::from_secs(10));
        let c = run_capped(
            sh,
            &["-c", "printf ok"],
            &env,
            Duration::from_secs(10),
            1000,
        )
        .unwrap();
        assert_eq!(
            (c.stdout.as_slice(), c.truncated),
            (b"ok".as_slice(), false)
        );
    }
}
