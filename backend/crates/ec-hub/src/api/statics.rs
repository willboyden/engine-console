//! The frontend (port of main.py `_mount_frontend` / `_file_response`).
//!
//! Serves ONLY the app shell and its asset dirs (css, js, i18n); a path that resolves outside its asset dir is a
//! 404. Every file carries `Cache-Control: no-cache` and Starlette's ETag (`"md5(str(st_mtime) + "-" + size)"`) and
//! Last-Modified; a request already holding that ETag (If-None-Match, weak or strong) gets a bodiless 304.
//! Content-Type follows Python's mimetypes: its built-in table, overridden by the system mime.types files.
//!
//! Hardening over Python (FileResponse opens whatever the path names): a file is opened once, without blocking
//! (`O_NONBLOCK`) and, for an asset whose path was already resolved, without following a symlink (`O_NOFOLLOW`);
//! its ETag and size come from that descriptor; anything but a regular file, or a file over `MAX_STATIC_BYTES`, is the
//! same 404 as a missing one. The mime.types files are read with a size and line cap.

use std::collections::HashMap;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use http::{HeaderValue, Response, StatusCode, header};
use md5::{Digest, Md5};

use crate::api::http::{Resp, not_found_resp, with_ct};
use crate::fsutil::{Follow, bounded_lines, open_regular, read_capped, read_text};
use crate::pyjson::float_repr;

pub const STATIC_DIRS: &[&str] = &["css", "js", "i18n"];
pub const STATIC_CACHE: &str = "no-cache";
/// The largest frontend file served (the shell and its assets are a few hundred KiB in all).
pub const MAX_STATIC_BYTES: usize = 32 * 1024 * 1024;
/// The largest mime.types file read, and the longest line of it used (the system's is ~25 KiB, lines < 200 bytes).
pub const MIME_TYPES_CAP: usize = 4 * 1024 * 1024;
pub const MIME_LINE_CAP: usize = 4096;

/// Python's `mimetypes.knownfiles`, read in this order (later files override earlier ones).
const KNOWN_FILES: &[&str] = &[
    "/etc/mime.types",
    "/etc/httpd/mime.types",
    "/etc/httpd/conf/mime.types",
    "/etc/apache/mime.types",
    "/etc/apache2/mime.types",
    "/usr/local/etc/httpd/conf/mime.types",
    "/usr/local/lib/netscape/mime.types",
    "/usr/local/etc/httpd/conf/mime.types",
    "/usr/local/etc/mime.types",
];

/// The web types of Python 3.13's built-in table (`mimetypes._types_map_default`).
const BUILTIN: &[(&str, &str)] = &[
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".html", "text/html"),
    (".htm", "text/html"),
    (".svg", "image/svg+xml"),
    (".png", "image/png"),
    (".jpg", "image/jpeg"),
    (".jpeg", "image/jpeg"),
    (".gif", "image/gif"),
    (".ico", "image/vnd.microsoft.icon"),
    (".webp", "image/webp"),
    (".woff", "font/woff"),
    (".woff2", "font/woff2"),
    (".ttf", "font/ttf"),
    (".otf", "font/otf"),
    (".txt", "text/plain"),
    (".md", "text/markdown"),
    (".map", "application/json"),
    (".webmanifest", "application/manifest+json"),
    (".wasm", "application/wasm"),
    (".xml", "text/xml"),
    (".csv", "text/csv"),
    (".pdf", "application/pdf"),
];

fn types_map() -> &'static HashMap<String, String> {
    static MAP: OnceLock<HashMap<String, String>> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut m: HashMap<String, String> = BUILTIN
            .iter()
            .filter(|(e, _)| *e != ".map")
            .map(|(e, t)| ((*e).to_string(), (*t).to_string()))
            .collect();
        for f in KNOWN_FILES {
            let Ok(text) = read_text(Path::new(f), Follow::Yes, MIME_TYPES_CAP) else {
                continue;
            };
            for line in bounded_lines(&text, MIME_LINE_CAP) {
                let line = line.split('#').next().unwrap_or("");
                let mut words = line.split_whitespace();
                let Some(t) = words.next() else { continue };
                for ext in words {
                    m.insert(format!(".{ext}"), t.to_string());
                }
            }
        }
        m
    })
}

/// `mimetypes.guess_type(path)[0]` (exact-case extension first, then lower-case).
pub fn guess_type(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().into_owned();
    let dot = name.rfind('.').filter(|i| *i > 0)?;
    let ext = &name[dot..];
    let m = types_map();
    m.get(ext).or_else(|| m.get(&ext.to_lowercase())).cloned()
}

/// Starlette's FileResponse content type: the guessed type (text/plain when unknown), `; charset=utf-8` for text/*.
pub fn content_type(path: &Path) -> String {
    let t = guess_type(path).unwrap_or_else(|| "text/plain".into());
    if t.starts_with("text/") && !t.contains("charset=") {
        format!("{t}; charset=utf-8")
    } else {
        t
    }
}

/// `str(st_mtime)` as CPython computes it (sec + nsec * 1e-9, as a double).
pub fn st_mtime_str(meta: &std::fs::Metadata) -> String {
    #[allow(clippy::cast_precision_loss)]
    let t = meta.mtime() as f64 + meta.mtime_nsec() as f64 * 1e-9;
    float_repr(t)
}

pub fn etag_of(meta: &std::fs::Metadata) -> String {
    let base = format!("{}-{}", st_mtime_str(meta), meta.len());
    format!("\"{}\"", hex::encode(Md5::digest(base.as_bytes())))
}

fn problem_404(detail: &str) -> Resp {
    not_found_resp(detail)
}

/// The file with `Cache-Control: no-cache`, its ETag and Last-Modified; a matching If-None-Match gets a 304.
/// `follow`: whether a symlink in the last component is followed (the shell's index.html, as Python; an asset path
/// was canonicalised already, so a symlink there now means it was swapped in: refused).
pub fn file_response(if_none_match: Option<&str>, target: &Path, follow: Follow) -> Resp {
    let Ok(mut file) = open_regular(target, follow) else {
        return problem_404("no such file");
    };
    let Ok(meta) = file.metadata() else {
        return problem_404("no such file");
    };
    let etag = etag_of(&meta);
    let sent: Vec<String> = if_none_match
        .unwrap_or("")
        .split(',')
        .map(|t| t.trim().trim_start_matches("W/").to_string())
        .collect();
    if sent.contains(&etag) {
        let mut r = Response::new(axum::body::Body::empty());
        *r.status_mut() = StatusCode::NOT_MODIFIED;
        if let Ok(v) = HeaderValue::from_str(&etag) {
            r.headers_mut().insert(header::ETAG, v);
        }
        r.headers_mut().insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static(STATIC_CACHE),
        );
        return r;
    }
    let Ok(bytes) = read_capped(&mut file, MAX_STATIC_BYTES) else {
        return problem_404("no such file");
    };
    let mut r = with_ct(200, &content_type(target), bytes);
    let h = r.headers_mut();
    let mtime = std::time::UNIX_EPOCH
        + std::time::Duration::from_secs(u64::try_from(meta.mtime()).unwrap_or(0));
    if let Ok(v) = HeaderValue::from_str(&httpdate::fmt_http_date(mtime)) {
        h.insert(header::LAST_MODIFIED, v);
    }
    if let Ok(v) = HeaderValue::from_str(&etag) {
        h.insert(header::ETAG, v);
    }
    h.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(STATIC_CACHE),
    );
    r
}

/// The SPA route's body: `path` is everything after the leading "/".
pub fn spa(root: &Path, path: &str, if_none_match: Option<&str>) -> Resp {
    if path.is_empty() || path == "index.html" {
        let index = root.join("index.html");
        if index.is_file() {
            return file_response(if_none_match, &index, Follow::Yes);
        }
        return problem_404("frontend not built");
    }
    let top = path.split('/').next().unwrap_or("");
    if STATIC_DIRS.contains(&top) && path.contains('/') {
        let base = std::fs::canonicalize(root.join(top));
        let target = std::fs::canonicalize(root.join(path));
        if let (Ok(base), Ok(target)) = (base, target)
            && target.is_file()
            && target.starts_with(&base)
            && target != base
        {
            return file_response(if_none_match, &target, Follow::No);
        }
    }
    problem_404("no such file")
}

/// The default frontend dir (FRONTEND_DIR / ENGINE_CONSOLE_FRONTEND_DIR wins over it, container.rs): found from the
/// running executable, never from a path compiled into the binary: the checkout the binary was built in,
/// `<checkout>/backend/target/[<triple>/]<profile>/engine-console` -> `<checkout>/frontend` (Python 0.4.x used the
/// source-relative `<checkout>/backend/src/engine_console/main.py` -> `<checkout>/frontend`).
/// An executable outside a checkout looks for `frontend/` next to itself (a missing dir means no UI, as in Python).
pub fn default_frontend_dir() -> PathBuf {
    std::env::current_exe()
        .map(|exe| frontend_dir_for(&exe))
        .unwrap_or_else(|_| PathBuf::from("frontend"))
}

/// `default_frontend_dir` for an executable path: the `frontend/` of the checkout whose `backend/target/` holds it, else
/// `frontend/` beside it.
pub fn frontend_dir_for(exe: &Path) -> PathBuf {
    let dir = exe.parent().unwrap_or_else(|| Path::new("."));
    for a in dir.ancestors() {
        if a.file_name().is_some_and(|n| n == "target")
            && let Some(workspace) = a.parent()
            && workspace.file_name().is_some_and(|n| n == "backend")
            && let Some(checkout) = workspace.parent()
        {
            return checkout.join("frontend");
        }
    }
    dir.join("frontend")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spa_serves_only_asset_dirs() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path();
        std::fs::create_dir_all(root.join("js")).unwrap();
        std::fs::write(root.join("index.html"), "<html>").unwrap();
        std::fs::write(root.join("js/app.js"), "x").unwrap();
        std::fs::write(root.join("package.json"), "{}").unwrap();
        assert_eq!(spa(root, "", None).status(), 200);
        assert_eq!(spa(root, "js/app.js", None).status(), 200);
        assert_eq!(spa(root, "js/../package.json", None).status(), 404);
        assert_eq!(spa(root, "package.json", None).status(), 404);
        assert_eq!(spa(root, "js", None).status(), 404);
        let r = spa(root, "js/app.js", None);
        assert_eq!(
            r.headers()["content-type"],
            "text/javascript; charset=utf-8"
        );
        assert_eq!(r.headers()["cache-control"], "no-cache");
        let etag = r.headers()["etag"].to_str().unwrap().to_string();
        let again = spa(root, "js/app.js", Some(&format!("W/{etag}, \"other\"")));
        assert_eq!(again.status(), 304);
    }

    /// A FIFO or a swapped-in symlink in an asset dir is a 404 at once, never a hung request (hardening finding 4).
    #[test]
    fn test_spa_refuses_fifos_and_swapped_symlinks_without_hanging() {
        let d = tempfile::tempdir().unwrap();
        let root = d.path();
        std::fs::create_dir_all(root.join("js")).unwrap();
        std::fs::write(root.join("index.html"), "<html>").unwrap();
        nix::unistd::mkfifo(
            &root.join("js/pipe.js"),
            nix::sys::stat::Mode::from_bits_truncate(0o600),
        )
        .unwrap();
        let t = std::time::Instant::now();
        assert_eq!(spa(root, "js/pipe.js", None).status(), 404);
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
        // canonicalised paths never end in a symlink, so file_response refuses one outright
        std::fs::write(root.join("js/real.js"), "x").unwrap();
        std::os::unix::fs::symlink(root.join("js/real.js"), root.join("js/link.js")).unwrap();
        assert_eq!(
            file_response(None, &root.join("js/link.js"), Follow::No).status(),
            404
        );
        assert_eq!(
            file_response(None, &root.join("js/link.js"), Follow::Yes).status(),
            200
        );
        // a symlinked asset inside its dir is still served (canonicalised first, as Python resolves it)
        assert_eq!(spa(root, "js/link.js", None).status(), 200);
        // index.html may be a symlink (Python follows it)
        std::fs::rename(root.join("index.html"), root.join("shell.html")).unwrap();
        std::os::unix::fs::symlink(root.join("shell.html"), root.join("index.html")).unwrap();
        assert_eq!(spa(root, "", None).status(), 200);
    }

    /// The default frontend dir comes from the executable's location, not from a compile-time path (finding 9).
    #[test]
    fn test_the_default_frontend_dir_follows_the_executable() {
        assert_eq!(
            frontend_dir_for(Path::new("/srv/ec/backend/target/release/engine-console")),
            PathBuf::from("/srv/ec/frontend")
        );
        assert_eq!(
            frontend_dir_for(Path::new(
                "/srv/ec/backend/target/x86_64-unknown-linux-gnu/release/engine-console"
            )),
            PathBuf::from("/srv/ec/frontend")
        );
        assert_eq!(
            frontend_dir_for(Path::new("/opt/engine-console/bin/engine-console")),
            PathBuf::from("/opt/engine-console/bin/frontend")
        );
        // this test binary lives in <checkout>/backend/target/<profile>/deps: the checkout's frontend/
        assert!(default_frontend_dir().join("index.html").is_file());
    }
}
