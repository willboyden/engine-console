//! Python semantics the node's text and path handling depends on, ported exactly where the result
//! reaches the hub, a file name or a security check: `str` whitespace/strip/splitlines/split, `float()`/`int()` of
//! a string, `round()`, posixpath (`normpath`, `realpath`, `join`, `split`, `relpath`, `splitext`), `json.dumps`
//! (default separators and `indent=2`, `ensure_ascii`), `fnmatch.fnmatchcase`, `shlex.split` (POSIX) and the
//! `OSError` subclass names and `strerror` texts. Every function names the Python it mirrors.

use std::collections::HashMap;
use std::ffi::{OsStr, OsString};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use serde_json::Value;

// ── str ──────────────────────────────────────────────────────────────────────────────────────────────────
/// `str.isspace()` for one character: Unicode White_Space plus the ASCII separators \x1c-\x1f.
pub fn isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `s.strip()`
pub fn strip(s: &str) -> &str {
    s.trim_matches(isspace)
}

/// `s.lstrip()`
pub fn lstrip(s: &str) -> &str {
    s.trim_start_matches(isspace)
}

/// `s.rstrip()`
pub fn rstrip(s: &str) -> &str {
    s.trim_end_matches(isspace)
}

/// `s.strip(chars)`
pub fn strip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_matches(|c| chars.contains(c))
}

/// `s.rstrip(chars)`
pub fn rstrip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_end_matches(|c| chars.contains(c))
}

/// `s.lstrip(chars)`
pub fn lstrip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_start_matches(|c| chars.contains(c))
}

fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\u{0b}'
            | '\u{0c}'
            | '\u{1c}'
            | '\u{1d}'
            | '\u{1e}'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()` (no line ends kept; `\r\n` is one break).
pub fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if is_line_break(c) {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r'
                && let Some(&(j, '\n')) = it.peek()
            {
                it.next();
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

/// `s.split()` (runs of whitespace, no empty strings).
pub fn split_ws(s: &str) -> Vec<&str> {
    s.split(isspace).filter(|p| !p.is_empty()).collect()
}

/// `s[:n]` in characters.
pub fn take(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// `s[-n:]` in characters.
pub fn tail(s: &str, n: usize) -> &str {
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    match s.char_indices().nth(count - n) {
        Some((i, _)) => &s[i..],
        None => "",
    }
}

/// `len(s)` (characters).
pub fn len(s: &str) -> usize {
    s.chars().count()
}

/// `repr(s)`
pub fn repr(s: &str) -> String {
    ec_proto::pyfmt::repr_str(s)
}

/// `repr(s[:n])`
pub fn repr_take(s: &str, n: usize) -> String {
    repr(take(s, n))
}

/// `str(v)` of a JSON-decoded value.
pub fn str_of(v: &Value) -> String {
    ec_proto::pyfmt::str_value(v)
}

/// `repr(v)` of a JSON-decoded value.
pub fn repr_of(v: &Value) -> String {
    ec_proto::pyfmt::repr_value(v)
}

/// `bool(v)` of a JSON-decoded value.
pub fn truthy(v: &Value) -> bool {
    ec_proto::sanitize::truthy(v)
}

/// `s.isdigit()` on ASCII text (every character a digit, at least one).
pub fn isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

fn digits_with_underscores(s: &str, radix: u32) -> bool {
    // Python numeric literals in int()/float(): single underscores only between digits
    if s.is_empty() || s.starts_with('_') || s.ends_with('_') || s.contains("__") {
        return false;
    }
    s.chars().all(|c| c == '_' || c.is_digit(radix))
}

/// `int(s)` (base 10): surrounding whitespace, a sign, digits with single underscores. None where Python raises
/// ValueError (and for values beyond i64).
pub fn parse_int(s: &str) -> Option<i64> {
    let t = strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if !digits_with_underscores(body, 10) {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    let v: i64 = digits.parse().ok()?;
    Some(if neg { -v } else { v })
}

/// `int(s, 16)` (as the chunk-size reader of http.client uses it): whitespace, optional `0x`, hex digits.
pub fn parse_int_hex(s: &str) -> Option<u64> {
    let t = strip(s);
    let body = t
        .strip_prefix("0x")
        .or_else(|| t.strip_prefix("0X"))
        .unwrap_or(t);
    if !digits_with_underscores(body, 16) {
        return None;
    }
    let digits: String = body.chars().filter(|c| *c != '_').collect();
    u64::from_str_radix(&digits, 16).ok()
}

/// `float(s)`: whitespace, sign, decimal or exponent form with single underscores between digits, `inf`,
/// `infinity`, `nan` in any case.
pub fn parse_float(s: &str) -> Option<f64> {
    let t = strip(s);
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let lower = body.to_ascii_lowercase();
    let v = if lower == "inf" || lower == "infinity" {
        f64::INFINITY
    } else if lower == "nan" {
        f64::NAN
    } else {
        // mantissa [. fraction] [e[+-]exponent], each digit run with single underscores
        let (mant, exp) = match lower.find('e') {
            Some(i) => (&lower[..i], Some(&lower[i + 1..])),
            None => (lower.as_str(), None),
        };
        let (int_part, frac_part) = match mant.find('.') {
            Some(i) => (&mant[..i], Some(&mant[i + 1..])),
            None => (mant, None),
        };
        let ok_run = |r: &str| r.is_empty() || digits_with_underscores(r, 10);
        if !ok_run(int_part) || !frac_part.is_none_or(ok_run) {
            return None;
        }
        if int_part.is_empty() && frac_part.is_none_or(str::is_empty) {
            return None;
        }
        let mut text: String = int_part.chars().filter(|c| *c != '_').collect();
        if let Some(f) = frac_part {
            text.push('.');
            text.extend(f.chars().filter(|c| *c != '_'));
        }
        if let Some(e) = exp {
            let eb = e.strip_prefix(['+', '-']).unwrap_or(e);
            if !digits_with_underscores(eb, 10) {
                return None;
            }
            text.push('e');
            text.extend(e.chars().filter(|c| *c != '_'));
        }
        if text.starts_with('.') {
            text.insert(0, '0');
        }
        text.parse::<f64>().ok()?
    };
    Some(if neg { -v } else { v })
}

/// `int(f)` of a float: truncated toward zero (saturating where Python would grow a big int).
#[allow(clippy::cast_possible_truncation)]
pub fn int_of_float(f: f64) -> i64 {
    f.trunc() as i64
}

/// `round(x, nd)` for a finite float: the exact binary value rounded to `nd` decimals, ties to even.
pub fn round(x: f64, nd: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.nd$}").parse().unwrap_or(x)
}

// ── OSError ──────────────────────────────────────────────────────────────────────────────────────────────
/// The Python exception class an `OSError` with this errno is raised as (`type(e).__name__`).
pub fn os_error_name(errno: i32) -> &'static str {
    use nix::errno::Errno as E;
    match E::from_raw(errno) {
        E::ENOENT => "FileNotFoundError",
        E::EEXIST => "FileExistsError",
        E::ENOTDIR => "NotADirectoryError",
        E::EISDIR => "IsADirectoryError",
        E::EACCES | E::EPERM => "PermissionError",
        E::EINTR => "InterruptedError",
        E::ECHILD => "ChildProcessError",
        E::EAGAIN | E::EALREADY | E::EINPROGRESS => "BlockingIOError",
        E::EPIPE | E::ESHUTDOWN => "BrokenPipeError",
        E::ECONNABORTED => "ConnectionAbortedError",
        E::ECONNREFUSED => "ConnectionRefusedError",
        E::ECONNRESET => "ConnectionResetError",
        E::ESRCH => "ProcessLookupError",
        E::ETIMEDOUT => "TimeoutError",
        _ => "OSError",
    }
}

/// `os.strerror(errno)`: the C library's message (std asks libc's strerror_r; nix's own table words some errors
/// differently, e.g. ELOOP).
pub fn strerror(errno: i32) -> String {
    let t = std::io::Error::from_raw_os_error(errno).to_string();
    t.strip_suffix(&format!(" (os error {errno})"))
        .map_or_else(|| t.clone(), str::to_string)
}

/// errno of an io::Error (EIO when it has none).
pub fn errno_of(e: &std::io::Error) -> i32 {
    e.raw_os_error().unwrap_or(nix::libc::EIO)
}

/// `e.strerror` of an io::Error.
pub fn io_strerror(e: &std::io::Error) -> String {
    match e.raw_os_error() {
        Some(n) => strerror(n),
        None => e.to_string(),
    }
}

// ── posixpath (bytes) ────────────────────────────────────────────────────────────────────────────────────
fn b(p: &Path) -> &[u8] {
    p.as_os_str().as_bytes()
}

fn pb(v: Vec<u8>) -> PathBuf {
    PathBuf::from(OsString::from_vec(v))
}

/// `posixpath.normpath`
pub fn normpath_bytes(path: &[u8]) -> Vec<u8> {
    if path.is_empty() {
        return b".".to_vec();
    }
    let initial: &[u8] = if path.starts_with(b"/") {
        if path.starts_with(b"//") && !path.starts_with(b"///") {
            b"//"
        } else {
            b"/"
        }
    } else {
        b""
    };
    let mut comps: Vec<&[u8]> = Vec::new();
    for comp in path.split(|c| *c == b'/') {
        if comp.is_empty() || comp == b"." {
            continue;
        }
        if comp != b".."
            || (initial.is_empty() && comps.is_empty())
            || comps.last() == Some(&&b".."[..])
        {
            comps.push(comp);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let mut out = initial.to_vec();
    out.extend(comps.join(&b'/'));
    if out.is_empty() { b".".to_vec() } else { out }
}

/// `os.path.normpath`
pub fn normpath(p: &Path) -> PathBuf {
    pb(normpath_bytes(b(p)))
}

/// `os.path.normpath` of text.
pub fn normpath_str(p: &str) -> String {
    String::from_utf8_lossy(&normpath_bytes(p.as_bytes())).into_owned()
}

/// `posixpath.join(a, b)`
pub fn join_bytes(a: &[u8], rest: &[u8]) -> Vec<u8> {
    if rest.starts_with(b"/") {
        return rest.to_vec();
    }
    let mut out = a.to_vec();
    if out.is_empty() || out.ends_with(b"/") {
        out.extend_from_slice(rest);
    } else {
        out.push(b'/');
        out.extend_from_slice(rest);
    }
    out
}

/// `os.path.join(a, *parts)`
pub fn join<S: AsRef<OsStr>>(a: &Path, parts: &[S]) -> PathBuf {
    let mut out = b(a).to_vec();
    for p in parts {
        out = join_bytes(&out, p.as_ref().as_bytes());
    }
    pb(out)
}

/// `posixpath.split(p)`
pub fn split_bytes(p: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let i = p.iter().rposition(|c| *c == b'/').map_or(0, |i| i + 1);
    let (head, tail) = (&p[..i], &p[i..]);
    let mut head = head.to_vec();
    if !head.is_empty() && head.iter().any(|c| *c != b'/') {
        while head.ends_with(b"/") {
            head.pop();
        }
    }
    (head, tail.to_vec())
}

/// `os.path.dirname`
pub fn dirname(p: &Path) -> PathBuf {
    pb(split_bytes(b(p)).0)
}

/// `os.path.basename`
pub fn basename(p: &Path) -> OsString {
    OsString::from_vec(split_bytes(b(p)).1)
}

/// `os.path.basename` as text.
pub fn basename_str(p: &str) -> &str {
    match p.rfind('/') {
        Some(i) => &p[i + 1..],
        None => p,
    }
}

/// `os.path.abspath`
pub fn abspath(p: &Path) -> PathBuf {
    if b(p).starts_with(b"/") {
        return normpath(p);
    }
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
    pb(normpath_bytes(&join_bytes(b(&cwd), b(p))))
}

/// `os.path.realpath(p)` (strict=False, Python 3.12's algorithm): symlinks resolved as far as the components
/// exist, the rest kept lexically, a symlink loop leaves the rest unresolved.
pub fn realpath(p: &Path) -> PathBuf {
    let mut seen: HashMap<Vec<u8>, Option<Vec<u8>>> = HashMap::new();
    let (path, _) = join_realpath(Vec::new(), b(p), &mut seen, 0);
    abspath(&pb(path))
}

fn join_realpath(
    mut path: Vec<u8>,
    rest: &[u8],
    seen: &mut HashMap<Vec<u8>, Option<Vec<u8>>>,
    depth: usize,
) -> (Vec<u8>, bool) {
    let mut rest: Vec<u8> = rest.to_vec();
    if rest.starts_with(b"/") {
        rest.remove(0);
        path = b"/".to_vec();
    }
    if depth > 256 {
        // Python would hit its recursion limit; treat as an unresolvable link
        return (join_bytes(&path, &rest), false);
    }
    while !rest.is_empty() {
        let (name, remaining) = match rest.iter().position(|c| *c == b'/') {
            Some(i) => (rest[..i].to_vec(), rest[i + 1..].to_vec()),
            None => (rest.clone(), Vec::new()),
        };
        rest = remaining;
        if name.is_empty() || name == b"." {
            continue;
        }
        if name == b".." {
            if path.is_empty() {
                path = b"..".to_vec();
            } else {
                let (head, tail) = split_bytes(&path);
                path = head;
                if tail == b".." {
                    path = join_bytes(&join_bytes(&path, b".."), b"..");
                }
            }
            continue;
        }
        let newpath = join_bytes(&path, &name);
        let is_link = std::fs::symlink_metadata(pb(newpath.clone()))
            .map(|m| m.file_type().is_symlink())
            .unwrap_or(false);
        if !is_link {
            path = newpath;
            continue;
        }
        if let Some(cached) = seen.get(&newpath) {
            if let Some(resolved) = cached {
                path = resolved.clone();
                continue;
            }
            return (join_bytes(&newpath, &rest), false);
        }
        seen.insert(newpath.clone(), None);
        let target = match std::fs::read_link(pb(newpath.clone())) {
            Ok(t) => t.into_os_string().into_vec(),
            Err(_) => {
                // vanished between lstat and readlink: keep it lexically
                path = newpath;
                continue;
            }
        };
        let (resolved, ok) = join_realpath(path, &target, seen, depth + 1);
        if !ok {
            return (join_bytes(&resolved, &rest), false);
        }
        seen.insert(newpath, Some(resolved.clone()));
        path = resolved;
    }
    (path, true)
}

/// `os.path.relpath(path, start)` for paths made absolute first.
pub fn relpath(path: &Path, start: &Path) -> PathBuf {
    let a = abspath(start);
    let p = abspath(path);
    let sl: Vec<&[u8]> = b(&a)
        .split(|c| *c == b'/')
        .filter(|x| !x.is_empty())
        .collect();
    let pl: Vec<&[u8]> = b(&p)
        .split(|c| *c == b'/')
        .filter(|x| !x.is_empty())
        .collect();
    let common = sl.iter().zip(pl.iter()).take_while(|(x, y)| x == y).count();
    let mut parts: Vec<&[u8]> = vec![b".."; sl.len() - common];
    parts.extend(&pl[common..]);
    if parts.is_empty() {
        return PathBuf::from(".");
    }
    pb(parts.join(&b'/'))
}

/// `os.path.splitext(name)[1]`
pub fn splitext_ext(name: &str) -> &str {
    let Some(dot) = name.rfind('.') else {
        return "";
    };
    let sep = name.rfind('/').map_or(0, |i| i + 1);
    if dot < sep {
        return "";
    }
    // the dot must not be part of the leading dots of the last component
    if name[sep..dot].chars().all(|c| c == '.') {
        return "";
    }
    &name[dot..]
}

/// A path as text (Python's surrogate-escaped str shown with replacement characters).
pub fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

// ── json.dumps ───────────────────────────────────────────────────────────────────────────────────────────
fn dumps_str(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            ' '..='~' => out.push(c),
            _ => {
                let mut buf = [0u16; 2];
                for u in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{u:04x}"));
                }
            }
        }
    }
    out.push('"');
}

fn dumps_num(n: &serde_json::Number) -> String {
    if n.is_f64() {
        let f = n.as_f64().unwrap_or(0.0);
        if f.is_nan() {
            "NaN".into()
        } else if f.is_infinite() {
            if f > 0.0 {
                "Infinity".into()
            } else {
                "-Infinity".into()
            }
        } else {
            ec_proto::pyfmt::repr_float(f)
        }
    } else {
        n.to_string()
    }
}

fn dumps_into(v: &Value, out: &mut String, indent: Option<usize>, level: usize, sort_keys: bool) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => out.push_str(&dumps_num(n)),
        Value::String(s) => dumps_str(s, out),
        Value::Array(a) => {
            if a.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { "," } else { ", " });
                }
                if let Some(n) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(n * (level + 1)));
                }
                dumps_into(x, out, indent, level + 1, sort_keys);
            }
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * level));
            }
            out.push(']');
        }
        Value::Object(m) => {
            if m.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            let mut items: Vec<(&String, &Value)> = m.iter().collect();
            if sort_keys {
                items.sort_by(|a, b| a.0.cmp(b.0));
            }
            for (i, (k, x)) in items.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(if indent.is_some() { "," } else { ", " });
                }
                if let Some(n) = indent {
                    out.push('\n');
                    out.push_str(&" ".repeat(n * (level + 1)));
                }
                dumps_str(k, out);
                out.push_str(": ");
                dumps_into(x, out, indent, level + 1, sort_keys);
            }
            if let Some(n) = indent {
                out.push('\n');
                out.push_str(&" ".repeat(n * level));
            }
            out.push('}');
        }
    }
}

/// `json.dumps(v)` (separators ", " and ": ", ensure_ascii).
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    dumps_into(v, &mut out, None, 0, false);
    out
}

/// `json.dumps(v, sort_keys=True)`
pub fn dumps_sorted(v: &Value) -> String {
    let mut out = String::new();
    dumps_into(v, &mut out, None, 0, true);
    out
}

/// `json.dumps(v, indent=n)`
pub fn dumps_indent(v: &Value, n: usize) -> String {
    let mut out = String::new();
    dumps_into(v, &mut out, Some(n), 0, false);
    out
}

// ── fnmatch ──────────────────────────────────────────────────────────────────────────────────────────────
enum Part {
    Star,
    Re(String),
}

/// `fnmatch._translate` + `_join_translated_parts` (Python 3.12), as a pattern for the `regex` crate: the
/// atomic groups Python uses for interior stars are plain lazy stars here, which match the same names (every
/// fixed piece between stars has a fixed width, so the earliest occurrence is always a valid choice).
pub fn fnmatch_translate(pat: &str) -> String {
    let p: Vec<char> = pat.chars().collect();
    let n = p.len();
    let mut res: Vec<Part> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = p[i];
        i += 1;
        if c == '*' {
            if !matches!(res.last(), Some(Part::Star)) {
                res.push(Part::Star);
            }
        } else if c == '?' {
            res.push(Part::Re(".".into()));
        } else if c == '[' {
            let mut j = i;
            if j < n && p[j] == '!' {
                j += 1;
            }
            if j < n && p[j] == ']' {
                j += 1;
            }
            while j < n && p[j] != ']' {
                j += 1;
            }
            if j >= n {
                res.push(Part::Re("\\[".into()));
            } else {
                let mut stuff: String;
                let seg: String = p[i..j].iter().collect();
                if !seg.contains('-') {
                    stuff = seg.replace('\\', "\\\\");
                } else {
                    let mut chunks: Vec<String> = Vec::new();
                    let mut k = if p[i] == '!' { i + 2 } else { i + 1 };
                    let mut ii = i;
                    loop {
                        let found = (k..j).find(|&x| p[x] == '-');
                        let Some(kk) = found else { break };
                        chunks.push(p[ii..kk].iter().collect());
                        ii = kk + 1;
                        k = kk + 3;
                    }
                    let chunk: String = p[ii..j].iter().collect();
                    if chunk.is_empty() {
                        if let Some(last) = chunks.last_mut() {
                            last.push('-');
                        }
                    } else {
                        chunks.push(chunk);
                    }
                    // remove empty ranges (invalid in RE)
                    let mut k = chunks.len().saturating_sub(1);
                    while k > 0 {
                        let prev_last = chunks[k - 1].chars().last();
                        let cur_first = chunks[k].chars().next();
                        if let (Some(a), Some(bb)) = (prev_last, cur_first)
                            && a > bb
                        {
                            let mut merged: String = chunks[k - 1]
                                .chars()
                                .take(chunks[k - 1].chars().count() - 1)
                                .collect();
                            merged.extend(chunks[k].chars().skip(1));
                            chunks[k - 1] = merged;
                            chunks.remove(k);
                        }
                        k -= 1;
                    }
                    stuff = chunks
                        .iter()
                        .map(|s| s.replace('\\', "\\\\").replace('-', "\\-"))
                        .collect::<Vec<_>>()
                        .join("-");
                }
                // escape set operations
                let mut esc = String::new();
                for ch in stuff.chars() {
                    if matches!(ch, '&' | '~' | '|') {
                        esc.push('\\');
                    }
                    esc.push(ch);
                }
                stuff = esc;
                i = j + 1;
                if stuff.is_empty() {
                    res.push(Part::Re("[^\\s\\S]".into()));
                } else if stuff == "!" {
                    res.push(Part::Re(".".into()));
                } else {
                    let body = if let Some(rest) = stuff.strip_prefix('!') {
                        format!("^{rest}")
                    } else if stuff.starts_with('^') || stuff.starts_with('[') {
                        format!("\\{stuff}")
                    } else {
                        stuff
                    };
                    // a `[` inside a Python class is literal; the regex crate would open a nested class
                    let body = escape_inner_brackets(&body);
                    res.push(Part::Re(format!("[{body}]")));
                }
            }
        } else {
            res.push(Part::Re(regex::escape(&c.to_string())));
        }
    }
    let mut out = String::new();
    for part in &res {
        match part {
            Part::Star => out.push_str(".*?"),
            Part::Re(s) => out.push_str(s),
        }
    }
    format!("(?s)\\A(?:{out})\\z")
}

fn escape_inner_brackets(body: &str) -> String {
    // keep escapes as they are; escape an unescaped `[` (it is literal in Python's class syntax)
    let mut out = String::new();
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            out.push(c);
            if let Some(nx) = chars.next() {
                out.push(nx);
            }
        } else if c == '[' {
            out.push_str("\\[");
        } else {
            out.push(c);
        }
    }
    out
}

/// `fnmatch.fnmatchcase(name, pat)`
pub fn fnmatchcase(name: &str, pat: &str) -> bool {
    match regex::Regex::new(&fnmatch_translate(pat)) {
        Ok(re) => re.is_match(name),
        Err(_) => false,
    }
}

// ── shlex ────────────────────────────────────────────────────────────────────────────────────────────────
/// `shlex.split(s)` (POSIX mode, whitespace_split, no comments). Err where Python raises ValueError ("No closing
/// quotation", "No escaped character").
pub fn shlex_split(s: &str) -> Result<Vec<String>, &'static str> {
    const WS: &str = " \t\r\n";
    let chars: Vec<char> = s.chars().collect();
    let mut pos = 0;
    let mut out = Vec::new();
    loop {
        // read_token
        let mut state: Option<char> = Some(' ');
        let mut token = String::new();
        let mut quoted = false;
        let mut escapedstate = ' ';
        loop {
            let next = chars.get(pos).copied();
            if next.is_some() {
                pos += 1;
            }
            match state {
                None => break,
                Some(' ') => match next {
                    None => {
                        state = None;
                        break;
                    }
                    Some(c) if WS.contains(c) => {
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some('\\') => {
                        escapedstate = 'a';
                        state = Some('\\');
                    }
                    Some(c) if c == '\'' || c == '"' => state = Some(c),
                    Some(c) => {
                        token.push(c);
                        state = Some('a');
                    }
                },
                Some(q) if q == '\'' || q == '"' => {
                    quoted = true;
                    match next {
                        None => return Err("No closing quotation"),
                        Some(c) if c == q => state = Some('a'),
                        Some('\\') if q == '"' => {
                            escapedstate = q;
                            state = Some('\\');
                        }
                        Some(c) => token.push(c),
                    }
                }
                Some('\\') => {
                    let Some(c) = next else {
                        return Err("No escaped character");
                    };
                    if (escapedstate == '\'' || escapedstate == '"')
                        && c != '\\'
                        && c != escapedstate
                    {
                        token.push('\\');
                    }
                    token.push(c);
                    state = Some(escapedstate);
                }
                Some(_) => match next {
                    // 'a'
                    None => {
                        state = None;
                        break;
                    }
                    Some(c) if WS.contains(c) => {
                        state = Some(' ');
                        if !token.is_empty() || quoted {
                            break;
                        }
                    }
                    Some(c) if c == '\'' || c == '"' => state = Some(c),
                    Some('\\') => {
                        escapedstate = 'a';
                        state = Some('\\');
                    }
                    Some(c) => token.push(c),
                },
            }
        }
        if state.is_none() && token.is_empty() && !quoted {
            return Ok(out);
        }
        out.push(token);
        if state.is_none() {
            return Ok(out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normpath_matches_posixpath() {
        for (i, o) in [
            ("", "."),
            ("/", "/"),
            ("//", "//"),
            ("///", "/"),
            ("//a/b", "//a/b"),
            ("/a/./b/../c/", "/a/c"),
            ("a/../..", ".."),
            ("/../x", "/x"),
            ("a//b", "a/b"),
        ] {
            assert_eq!(normpath_str(i), o, "{i}");
        }
    }

    #[test]
    fn splitlines_and_strip() {
        assert_eq!(splitlines("a\r\nb\rc\n\nd"), vec!["a", "b", "c", "", "d"]);
        assert_eq!(splitlines("a\n"), vec!["a"]);
        assert_eq!(strip("\u{85} x \t"), "x");
        assert_eq!(split_ws("  a  b\tc "), vec!["a", "b", "c"]);
    }

    #[test]
    fn numbers_parse_like_python() {
        assert_eq!(parse_int(" 1_000 "), Some(1000));
        assert_eq!(parse_int("1__0"), None);
        assert_eq!(parse_int("+5"), Some(5));
        assert_eq!(parse_float("71.50"), Some(71.5));
        assert_eq!(parse_float("[N/A]"), None);
        assert!(parse_float("nan").is_some_and(f64::is_nan));
        assert_eq!(parse_float(".5"), Some(0.5));
        assert_eq!(parse_float("5."), Some(5.0));
        assert_eq!(parse_float("1e3"), Some(1000.0));
        assert_eq!(parse_float("."), None);
        assert_eq!(round(66.666_666, 1), 66.7);
        assert_eq!(round(0.25, 1), 0.2);
        assert_eq!(round(2.5, 0), 2.0);
    }

    #[test]
    fn shlex_split_posix() {
        assert_eq!(
            shlex_split("vllm serve m --api-key 'a b' \"c\\\"d\" e\\ f").unwrap_or_default(),
            vec!["vllm", "serve", "m", "--api-key", "a b", "c\"d", "e f"]
        );
        assert_eq!(shlex_split("'' x").unwrap_or_default(), vec!["", "x"]);
        assert!(shlex_split("'open").is_err());
        assert!(shlex_split("x\\").is_err());
    }

    #[test]
    fn fnmatch_cases() {
        assert!(fnmatchcase("lab-sglang-1", "lab-*"));
        assert!(!fnmatchcase("other", "lab-*"));
        assert!(fnmatchcase("a[b", "a[b"));
        assert!(fnmatchcase("ab", "a[!c]"));
        assert!(fnmatchcase("a-", "a[a-]"));
        assert!(fnmatchcase("x\ny", "x?y"));
        assert!(!fnmatchcase("Lab-1", "lab-*"));
    }

    #[test]
    fn dumps_like_python() {
        let v = serde_json::json!({"a": [1, 2.0, "é\n"], "b": {}, "c": []});
        assert_eq!(
            dumps(&v),
            "{\"a\": [1, 2.0, \"\\u00e9\\n\"], \"b\": {}, \"c\": []}"
        );
        assert_eq!(
            dumps_indent(&serde_json::json!({"a": [1], "b": {}}), 2),
            "{\n  \"a\": [\n    1\n  ],\n  \"b\": {}\n}"
        );
    }

    #[test]
    fn splitext_like_python() {
        assert_eq!(splitext_ext("a.ftw"), ".ftw");
        assert_eq!(splitext_ext(".bashrc"), "");
        assert_eq!(splitext_ext("..x.y"), ".y");
        assert_eq!(splitext_ext("noext"), "");
    }
}
