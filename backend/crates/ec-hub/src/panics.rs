//! Panics in the hub: one process hook that never writes a secret, and the guard every background iteration runs
//! under.
//!
//! Python's equivalents: an uncaught exception in a request is FastAPI's generic 500 (app.rs catches the panic); in a
//! background loop it is logged (`log.exception`) and the loop goes on (main.py `_loop`, the pollers' own `except`).
//! In Rust a panic in a spawned task ends that task for good, so every loop iteration, tick, poll and per-message
//! handler runs inside [`guarded`] / [`guarded_sync`]: a panic is logged (redacted) and the loop continues.
//!
//! The hook replaces the default one (which prints the panic message verbatim, and a message can carry a token, a URL
//! with credentials or a request value). It keeps `adapters::pycompat::install_quiet_panic_hook`'s behaviour for the
//! adapters' Python exceptions (`PyExc` panics are expected, caught, and only debug-logged) and writes every other
//! panic to stderr with secrets masked by `services::common::scrub`.

use std::any::Any;
use std::future::Future;
use std::io::Write as _;
use std::panic::AssertUnwindSafe;

use futures::FutureExt as _;

use crate::adapters::pycompat::PyExc;
use crate::services::common::scrub;

/// The longest panic message the hook or a loop guard writes (characters).
pub const MAX_MESSAGE: usize = 2000;

/// The text of a panic payload: a `&str` / `String` message, a `PyExc` as "<kind>: <msg>", else a placeholder.
pub fn payload_text(payload: &(dyn Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(e) = payload.downcast_ref::<PyExc>() {
        e.to_string()
    } else {
        "Box<dyn Any>".to_string()
    }
}

/// `text` with secret-looking values masked and cut to [`MAX_MESSAGE`] characters.
pub fn redacted(text: &str) -> String {
    let s = scrub(text);
    if s.chars().count() <= MAX_MESSAGE {
        s
    } else {
        let head: String = s.chars().take(MAX_MESSAGE).collect();
        format!("{head}...")
    }
}

/// What the hook writes for one panic: the thread, the location and the redacted message (the default hook's shape).
pub fn report(
    thread: Option<&str>,
    location: Option<String>,
    payload: &(dyn Any + Send),
) -> String {
    let at = location.unwrap_or_else(|| "<unknown>".into());
    format!(
        "thread '{}' panicked at {at}:\n{}",
        thread.unwrap_or("<unnamed>"),
        redacted(&payload_text(payload))
    )
}

/// Install the hub's panic hook (once per process; later calls do nothing).
pub fn install_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            if let Some(e) = info.payload().downcast_ref::<PyExc>() {
                // an adapter's Python exception: caught by `pycompat::catch` or answered as a 500 by app.rs
                tracing::debug!(target: "engine_console.adapters", "raised {}", redacted(&e.to_string()));
                return;
            }
            let location = info
                .location()
                .map(|l| format!("{}:{}:{}", l.file(), l.line(), l.column()));
            let text = report(std::thread::current().name(), location, info.payload());
            let _ = writeln!(std::io::stderr(), "{text}");
        }));
    });
}

/// Run one iteration of a background job: its output, or None after a panic, which is logged (redacted) as
/// "<what>: panic: <message>" and swallowed so the caller's loop goes on.
pub async fn guarded<F: Future>(what: &str, fut: F) -> Option<F::Output> {
    match AssertUnwindSafe(fut).catch_unwind().await {
        Ok(v) => Some(v),
        Err(p) => {
            log_panic(what, &*p);
            None
        }
    }
}

/// `guarded` for synchronous work (a per-message handler, a callback).
pub fn guarded_sync<T>(what: &str, f: impl FnOnce() -> T) -> Option<T> {
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Some(v),
        Err(p) => {
            log_panic(what, &*p);
            None
        }
    }
}

fn log_panic(what: &str, payload: &(dyn Any + Send)) {
    tracing::error!(target: "engine_console", "{what}: panic: {}", redacted(&payload_text(payload)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_report_masks_secrets_in_the_panic_message() {
        let msg = String::from("upstream said no for Bearer abcdefgh12345678 and hf_abcdefghijkl");
        let text = report(Some("worker"), Some("src/x.rs:1:2".into()), &msg);
        assert!(text.starts_with("thread 'worker' panicked at src/x.rs:1:2:\n"));
        assert!(!text.contains("abcdefgh12345678") && !text.contains("hf_abcdefghijkl"));
        assert!(text.contains("[redacted]"));
        let long = "x".repeat(MAX_MESSAGE * 2);
        assert!(redacted(&long).chars().count() <= MAX_MESSAGE + 3);
        assert_eq!(payload_text(&42_u8), "Box<dyn Any>");
    }

    #[tokio::test]
    async fn test_guarded_swallows_a_panic_and_the_next_iteration_runs() {
        let mut runs = 0;
        for i in 0..3 {
            let out = guarded("tick", async move {
                if i == 1 {
                    #[allow(clippy::panic)]
                    {
                        panic!("tick {i} failed with token ec_aaaaaaaaaaaaaaaaaaaa");
                    }
                }
                i
            })
            .await;
            if out.is_some() {
                runs += 1;
            }
        }
        assert_eq!(runs, 2);
        assert_eq!(guarded_sync("cb", || 5), Some(5));
        #[allow(clippy::panic)]
        let none: Option<()> = guarded_sync("cb", || panic!("boom"));
        assert_eq!(none, None);
    }
}
