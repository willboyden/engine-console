//! Panics are contained and never print a secret.
//!
//! A panic in the node is a bug, but it must cost one round of a loop or one request, never the node: every loop
//! and handler thread runs its body through `contain` (or the server's own catch_unwind), which logs one redacted
//! line on stderr and carries on. The process-wide hook (`install_hook`, set by `main`) replaces Rust's default
//! one, which would print the raw panic message: that message can hold an argv, a header or an environment value.
//! Everything goes to stderr only; in `--stdio` mode stdout is the protocol stream (entry.rs points fd 1 at stderr
//! anyway, so nothing printed by accident can reach the hub).

use std::any::Any;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};

/// Characters of a panic message a log line keeps.
pub const PANIC_TEXT_MAX: usize = 300;

/// The message a panic carries (`panic!` gives a `&str` or a `String`), or "" for any other payload.
pub fn payload_text(p: &(dyn Any + Send)) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        String::new()
    }
}

/// What a log line may show of a panic message: secret-looking values masked (`scrub`, which fails closed),
/// control characters removed, capped at PANIC_TEXT_MAX characters.
pub fn redact(text: &str) -> String {
    super::clean_text(&super::scrub(text), PANIC_TEXT_MAX)
}

/// The hook's stderr line: `ecnode: panic in thread 'NAME' at FILE:LINE: <redacted message>`.
pub fn panic_line(thread: Option<&str>, location: Option<String>, message: &str) -> String {
    let mut line = format!(
        "ecnode: panic in thread '{}'",
        super::clean_text(thread.unwrap_or("<unnamed>"), 64)
    );
    if let Some(l) = location {
        line.push_str(" at ");
        line.push_str(&super::clean_text(&l, 256));
    }
    let msg = redact(message);
    if !msg.is_empty() {
        line.push_str(": ");
        line.push_str(&msg);
    }
    line
}

/// Install the node's panic hook: one redacted line on stderr per panic, no raw message, no backtrace.
pub fn install_hook() {
    std::panic::set_hook(Box::new(|info| {
        let thread = std::thread::current();
        let location = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()));
        let message = info.payload_as_str().unwrap_or("");
        let line = panic_line(thread.name(), location, message);
        let mut err = std::io::stderr().lock();
        let _ = writeln!(err, "{line}");
        let _ = err.flush();
    }));
}

/// One stderr line about a contained panic: `ecnode: <what>: panic: <redacted message>`.
pub fn report(what: &str, payload: &(dyn Any + Send)) {
    let msg = redact(&payload_text(payload));
    let mut err = std::io::stderr().lock();
    let _ = if msg.is_empty() {
        writeln!(err, "ecnode: {what}: panic")
    } else {
        writeln!(err, "ecnode: {what}: panic: {msg}")
    };
}

/// Run `f`; a panic is reported (`report`) and becomes None. For loop rounds and helper threads: the caller goes on.
pub fn contain<T>(what: &str, f: impl FnOnce() -> T) -> Option<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Some(v),
        Err(p) => {
            report(what, p.as_ref());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_lines_are_redacted_and_one_line() {
        let line = panic_line(
            Some("ecnode-sampler"),
            Some("src/x.rs:1".into()),
            "bad argv: vllm serve --api-key sk-abcdefghijklmnopqrstuvwx\nHF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz",
        );
        assert!(!line.contains("sk-abc"), "{line}");
        assert!(!line.contains("hf_abc"), "{line}");
        assert!(!line.contains('\n'));
        assert!(
            line.starts_with("ecnode: panic in thread 'ecnode-sampler' at src/x.rs:1: bad argv")
        );
        assert!(line.contains("[redacted]"));
        let long = "x".repeat(10_000);
        assert!(panic_line(None, None, &long).chars().count() < PANIC_TEXT_MAX + 64);
    }

    #[test]
    fn contain_turns_a_panic_into_none() {
        assert_eq!(contain("t", || 5), Some(5));
        #[allow(clippy::panic)]
        let r: Option<i32> = contain("t", || panic!("Authorization: Bearer abcdefghijklmnop"));
        assert_eq!(r, None);
        assert_eq!(payload_text(&"s"), "s");
        assert_eq!(payload_text(&String::from("t")), "t");
        assert_eq!(payload_text(&5u8), "");
    }
}
