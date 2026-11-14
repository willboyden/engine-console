//! Python text helpers the adapters share with the hub, and the secret-value scrub the hub's messages and the
//! browser's config checks share.

use std::sync::LazyLock;

use regex::Regex;

static SECRET_VALUE: LazyLock<Option<Regex>> = LazyLock::new(|| {
    Regex::new(r"\b(hf_[A-Za-z0-9]{8,}|ec_[A-Za-z0-9_\-]{16,}|sk-[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._\-]{8,})")
        .ok()
});
/// What `scrub` leaves of a text it cannot check (its pattern did not compile): nothing of the text.
pub const REDACTED: &str = "[redacted]";

/// Mask secret-looking tokens in free text (logs). No truncation, unlike `redact`.
pub fn scrub(text: &str) -> String {
    scrub_with(SECRET_VALUE.as_ref(), text)
}

/// `scrub` with the pattern given; `None` (a pattern that did not compile) masks the whole text (fail closed).
pub fn scrub_with(re: Option<&Regex>, text: &str) -> String {
    match re {
        Some(r) => r.replace_all(text, REDACTED).into_owned(),
        None if text.is_empty() => String::new(),
        None => REDACTED.to_string(),
    }
}

/// `str.splitlines()` (the line boundaries Python recognises, without keeping them).
pub fn py_splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let is_break = matches!(
            c,
            '\n' | '\r'
                | '\u{b}'
                | '\u{c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        );
        if is_break {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = chars.peek()
            {
                chars.next();
                end = j + 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}
