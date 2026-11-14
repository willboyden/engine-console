//! The node policy's `engine_images` rule: which image references a node runs. One copy for the node (which
//! enforces it), the hub (which says no before a round trip) and the browser (the image badge, through
//! ec-engines-wasm). Moved here from `ecnode::policy` unchanged.
//!
//! References follow docker's grammar, slightly stricter: `[domain[:port]/]path[:tag][@sha256:digest]`, at most 256
//! characters. A pattern is an exact reference, `repo:*` (any tag of exactly that repository, no digest) or
//! `repo@sha256:*` (any digest of it, no tag). The repository is compared literally.

use std::sync::LazyLock;

use regex::Regex;
use serde_json::Value;

const COMP: &str = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*";
const DOMAIN: &str = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*(?::[0-9]{1,5})?/)";
const TAG: &str = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}";
const DIGEST: &str = r"sha256:[a-f0-9]{64}";
/// The longest reference accepted, in characters (Python's `len`).
pub const MAX_IMAGE_REF: usize = 256;

fn repo_re() -> String {
    format!("{DOMAIN}?{COMP}(?:/{COMP})*")
}

/// A Python `re.fullmatch` pattern. None only if it does not compile, which then matches nothing (fail closed).
fn full(p: &str) -> Option<Regex> {
    Regex::new(&format!(r"\A(?:{p})\z")).ok()
}

static IMAGE_REF: LazyLock<Option<Regex>> = LazyLock::new(|| {
    full(&format!(
        r"(?P<repo>{})(?::(?P<tag>{TAG}))?(?:@(?P<digest>{DIGEST}))?",
        repo_re()
    ))
});
static PAT_TAG: LazyLock<Option<Regex>> =
    LazyLock::new(|| full(&format!(r"(?P<repo>{}):\*", repo_re())));
static PAT_DIGEST: LazyLock<Option<Regex>> =
    LazyLock::new(|| full(&format!(r"(?P<repo>{})@sha256:\*", repo_re())));

fn is_match(re: &Option<Regex>, s: &str) -> bool {
    re.as_ref().is_some_and(|r| r.is_match(s))
}

/// (repository, tag, digest) of a valid image reference, else None.
pub fn split_image(r: &str) -> Option<(String, Option<String>, Option<String>)> {
    if r.chars().count() > MAX_IMAGE_REF {
        return None;
    }
    let c = IMAGE_REF.as_ref()?.captures(r)?;
    Some((
        c.name("repo")?.as_str().to_string(),
        c.name("tag").map(|m| m.as_str().to_string()),
        c.name("digest").map(|m| m.as_str().to_string()),
    ))
}

pub fn valid_image(r: &str) -> bool {
    split_image(r).is_some()
}

/// `valid_image(v)` for a JSON value.
pub fn valid_image_value(v: &Value) -> bool {
    v.as_str().is_some_and(valid_image)
}

/// Exact reference, `repo:*` (any tag of exactly that repository) or `repo@sha256:*` (any digest of it). The
/// repository is compared literally.
pub fn image_matches(pattern: &str, image: &str) -> bool {
    let Some((repo, tag, digest)) = split_image(image) else {
        return false;
    };
    if let Some(m) = PAT_TAG.as_ref().and_then(|r| r.captures(pattern)) {
        return m.name("repo").is_some_and(|r| r.as_str() == repo)
            && tag.is_some()
            && digest.is_none();
    }
    if let Some(m) = PAT_DIGEST.as_ref().and_then(|r| r.captures(pattern)) {
        return m.name("repo").is_some_and(|r| r.as_str() == repo)
            && digest.is_some()
            && tag.is_none();
    }
    valid_image(pattern) && image == pattern
}

pub fn valid_image_pattern(p: &str) -> bool {
    is_match(&PAT_TAG, p) || is_match(&PAT_DIGEST, p) || valid_image(p)
}

/// The policy's decision: `image` matches one of `patterns` (`engine_images`).
pub fn image_allowed<S: AsRef<str>>(patterns: &[S], image: &str) -> bool {
    patterns.iter().any(|p| image_matches(p.as_ref(), image))
}

#[cfg(test)]
mod tests {
    use super::*;

    const D: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    #[test]
    fn test_the_three_pattern_forms() {
        assert!(image_allowed(
            &["vllm/vllm-openai:v0.9.0"],
            "vllm/vllm-openai:v0.9.0"
        ));
        assert!(!image_allowed(
            &["vllm/vllm-openai:v0.9.0"],
            "vllm/vllm-openai:v0.9.1"
        ));
        assert!(image_allowed(
            &["vllm/vllm-openai:*"],
            "vllm/vllm-openai:nightly"
        ));
        assert!(!image_allowed(
            &["vllm/vllm-openai:*"],
            "vllm/vllm-openai-x:v1"
        ));
        let pinned = format!("ghcr.io/ggml-org/llama.cpp@sha256:{D}");
        assert!(image_allowed(
            &["ghcr.io/ggml-org/llama.cpp@sha256:*"],
            &pinned
        ));
        assert!(!image_allowed(&["ghcr.io/ggml-org/llama.cpp:*"], &pinned));
    }

    /// A tag and a digest together match neither wildcard form (the node refuses it; the old browser glob did not).
    #[test]
    fn test_tag_plus_digest_matches_no_wildcard() {
        let both = format!("repo/x:v1@sha256:{D}");
        assert!(valid_image(&both));
        assert!(!image_matches("repo/x:*", &both));
        assert!(!image_matches("repo/x@sha256:*", &both));
        assert!(image_matches(&both, &both));
    }

    #[test]
    fn test_patterns_are_literal_and_invalid_references_never_match() {
        assert!(!image_matches("a.b/c:*", "aXb/c:1"));
        assert!(
            !image_matches("a/b:*-arm", "a/b:1-arm"),
            "a * only replaces the whole tag"
        );
        assert!(!image_matches("Bad Ref", "Bad Ref"));
        assert!(!valid_image(&format!("a/{}:1", "b".repeat(300))));
        assert!(!image_allowed::<&str>(&[], "a/b:1"));
        assert!(
            valid_image_pattern("a/b:*")
                && valid_image_pattern("a/b@sha256:*")
                && !valid_image_pattern("a/*")
        );
    }
}
