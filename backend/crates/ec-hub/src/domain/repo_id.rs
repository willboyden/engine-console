//! Validated identifier types shared by every route (port of domain/repo_id.py).

use std::sync::LazyLock;

use regex::Regex;

use crate::domain::errors::{Problem, bad_request};

static REPO: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"^[A-Za-z0-9][\w.\-]{0,95}/[A-Za-z0-9][\w.\-]{0,95}$")
        .unwrap_or_else(|_| unreachable!())
});
static REV: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^[A-Za-z0-9._-]{1,100}$").unwrap_or_else(|_| unreachable!()));

pub const REPO_ID_MSG: &str = "repo_id must look like 'org/name' (letters, digits, . _ -)";
pub const REVISION_MSG: &str = "revision may only contain letters, digits, . _ -";

/// Python's `re` `\w` is Unicode-aware for str patterns, and so is the regex crate's: both accept letters and
/// digits of any script. `$` in Python also matches before a trailing newline: fullmatch() is used there, and the
/// anchors here are explicit, so a trailing newline is refused by both.
fn full(re: &Regex, v: &str) -> bool {
    !v.ends_with('\n') && re.is_match(v)
}

/// `check_repo_id`: "--" is how the HF cache flattens "/", so "a--b/c" could alias "a/b--c".
pub fn check_repo_id(v: &str) -> Result<(), &'static str> {
    if !full(&REPO, v) || v.contains("--") || v.contains("..") {
        return Err(REPO_ID_MSG);
    }
    Ok(())
}

pub fn check_revision(v: &str) -> Result<(), &'static str> {
    if !full(&REV, v) || v.contains("..") {
        return Err(REVISION_MSG);
    }
    Ok(())
}

/// For path parameters and internal callers: the same rule, as the API's problem error.
pub fn require_repo_id(v: &str) -> Result<(), Problem> {
    check_repo_id(v).map_err(|m| bad_request(m).code("invalid_repo_id"))
}
