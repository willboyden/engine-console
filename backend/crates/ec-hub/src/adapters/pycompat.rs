//! Python semantics the engine adapters depend on, reproduced exactly, shared by every
//! adapter (first written as `vllm_parse/py.rs`; PORTING.md §A5 and §A7).
//!
//! The Python adapters lean on CPython behaviour that Rust's look-alikes do not share, and the port must give the same
//! answer for every input the API can carry:
//! * `re`: `\d` is Unicode `Nd` of Python 3.13 (Unicode 15.1), `\s` is `str.isspace()`, `\w` is letters, numbers and
//!   `_`; `re.IGNORECASE` also matches `İ` and `ı` for `i`. [`compile`] rewrites a pattern into an equivalent Rust regex.
//! * `float(str)` / `int(str)` accept Unicode digits, surrounding Unicode whitespace and `_` between digits.
//! * `str.strip()`, truthiness, `str(x)`, `int`/`float` of JSON values, the numeric tower (`2 * 2.0`, `%` signs,
//!   int/float comparisons) and the `TypeError`s Python raises on `None`.
//! * `sum()` of floats (Neumaier compensated summation since 3.12), `round(x, n)`, `format(x, ".1f")`.
//! * `json.loads` (NaN/Infinity, big integers, lone surrogates, its error messages, its nesting limit) and
//!   `json.dumps(..., separators=(",", ":"))` (ensure_ascii).
//! * `sorted()` of float keys that may hold NaN (CPython's run detection + binary insertion sort).
//!
//! Exceptions (PORTING.md §A5): where Python raises an exception the adapter does not catch, [`raise`] panics with a
//! [`PyExc`] payload. An HTTP caller gets the same generic 500 as Python's catch-all handler (app.rs `catch_unwind`);
//! every non-HTTP caller uses the adapter's `try_*` methods, whose default runs the plain method under [`catch`] and
//! returns the exception as `Err`. New adapter code may return `Err(exc(..))` from its `try_*` methods directly and
//! implement the plain method as `or_raise(self.try_x(..))`. Parsers never raise (see their modules).

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde_json::{Number, Value};

// ── raising ─────────────────────────────────────────────────────────────────────────────────────────────
/// A Python exception an adapter did not catch: its type (`TypeError`, `ValueError`, `ZeroDivisionError`,
/// `RecursionError`, `re.error`, ...) and message. The message is for the log only; an HTTP client sees the generic 500.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PyExc {
    pub kind: String,
    pub msg: String,
}

impl fmt::Display for PyExc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind, self.msg)
    }
}

impl std::error::Error for PyExc {}

/// What a fallible adapter method returns (`try_*`, PORTING.md §A5).
pub type PyResult<T> = Result<T, PyExc>;

/// The exception `kind(msg)`, as a value.
pub fn exc(kind: &str, msg: impl Into<String>) -> PyExc {
    PyExc {
        kind: kind.to_string(),
        msg: msg.into(),
    }
}

/// Python raising an exception nobody in the adapter catches. Panics with a [`PyExc`] payload, which [`catch`] turns
/// back into an `Err` and app.rs into the generic 500.
pub fn raise(kind: &str, msg: &str) -> ! {
    raise_exc(exc(kind, msg))
}

/// [`raise`] for an exception already held as a value.
#[allow(clippy::panic)] // the one place an adapter panics on purpose (PORTING.md §A5)
pub fn raise_exc(e: PyExc) -> ! {
    std::panic::panic_any(e)
}

/// The value, or raise the exception (the plain form of a method whose `try_*` form returns `Err`).
pub fn or_raise<T>(r: PyResult<T>) -> T {
    r.unwrap_or_else(|e| raise_exc(e))
}

/// Run `f`; an exception it raises ([`raise`]) comes back as `Err`. Any other panic is a bug and is resumed unchanged.
/// `f` must be synchronous (adapter methods are) and must not leave shared state half-changed when it raises (adapter
/// methods only read their arguments).
pub fn catch<T>(f: impl FnOnce() -> T) -> PyResult<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(v) => Ok(v),
        Err(payload) => match payload.downcast::<PyExc>() {
            Ok(e) => Err(*e),
            Err(other) => std::panic::resume_unwind(other),
        },
    }
}

/// Install (once per process) a panic hook that keeps a raised [`PyExc`] off stderr (a debug log line instead: it is
/// caught by [`catch`] or answered as a 500 by app.rs, which logs it) and hands every other panic to the previous hook.
pub fn install_quiet_panic_hook() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if let Some(e) = info.payload().downcast_ref::<PyExc>() {
                tracing::debug!(target: "engine_console.adapters", "raised {e}");
            } else {
                previous(info);
            }
        }));
    });
}

// ── character classes ───────────────────────────────────────────────────────────────────────────────────
/// Unicode `Nd` (decimal digits) of Python 3.13 (`unicodedata.unidata_version` 15.1.0): every run starts at digit 0.
pub const ND_RANGES: &[(u32, u32)] = &[
    (0x30, 0x39),
    (0x660, 0x669),
    (0x6f0, 0x6f9),
    (0x7c0, 0x7c9),
    (0x966, 0x96f),
    (0x9e6, 0x9ef),
    (0xa66, 0xa6f),
    (0xae6, 0xaef),
    (0xb66, 0xb6f),
    (0xbe6, 0xbef),
    (0xc66, 0xc6f),
    (0xce6, 0xcef),
    (0xd66, 0xd6f),
    (0xde6, 0xdef),
    (0xe50, 0xe59),
    (0xed0, 0xed9),
    (0xf20, 0xf29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x17e0, 0x17e9),
    (0x1810, 0x1819),
    (0x1946, 0x194f),
    (0x19d0, 0x19d9),
    (0x1a80, 0x1a89),
    (0x1a90, 0x1a99),
    (0x1b50, 0x1b59),
    (0x1bb0, 0x1bb9),
    (0x1c40, 0x1c49),
    (0x1c50, 0x1c59),
    (0xa620, 0xa629),
    (0xa8d0, 0xa8d9),
    (0xa900, 0xa909),
    (0xa9d0, 0xa9d9),
    (0xa9f0, 0xa9f9),
    (0xaa50, 0xaa59),
    (0xabf0, 0xabf9),
    (0xff10, 0xff19),
    (0x104a0, 0x104a9),
    (0x10d30, 0x10d39),
    (0x11066, 0x1106f),
    (0x110f0, 0x110f9),
    (0x11136, 0x1113f),
    (0x111d0, 0x111d9),
    (0x112f0, 0x112f9),
    (0x11450, 0x11459),
    (0x114d0, 0x114d9),
    (0x11650, 0x11659),
    (0x116c0, 0x116c9),
    (0x11730, 0x11739),
    (0x118e0, 0x118e9),
    (0x11950, 0x11959),
    (0x11c50, 0x11c59),
    (0x11d50, 0x11d59),
    (0x11da0, 0x11da9),
    (0x11f50, 0x11f59),
    (0x16a60, 0x16a69),
    (0x16ac0, 0x16ac9),
    (0x16b50, 0x16b59),
    (0x1d7ce, 0x1d7ff),
    (0x1e140, 0x1e149),
    (0x1e2f0, 0x1e2f9),
    (0x1e4f0, 0x1e4f9),
    (0x1e950, 0x1e959),
    (0x1fbf0, 0x1fbf9),
];

/// Characters for which Python 3.13's `str.isdigit()` is true but `str.isdecimal()` is not (superscripts, circled
/// digits...): `int()` refuses them.
pub const DIGIT_NOT_DECIMAL: &[(u32, u32)] = &[
    (0xb2, 0xb3),
    (0xb9, 0xb9),
    (0x1369, 0x1371),
    (0x19da, 0x19da),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247c),
    (0x2488, 0x2490),
    (0x24ea, 0x24ea),
    (0x24f5, 0x24fd),
    (0x24ff, 0x24ff),
    (0x2776, 0x277e),
    (0x2780, 0x2788),
    (0x278a, 0x2792),
    (0x10a40, 0x10a43),
    (0x10e60, 0x10e68),
    (0x11052, 0x1105a),
    (0x1f100, 0x1f10a),
];

/// `str.isspace()` (also what `re`'s `\s` matches and what `str.strip()` removes).
pub const SPACES: &[char] = &[
    '\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\u{85}',
    '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
    '\u{205f}', '\u{3000}',
];

pub fn is_space(c: char) -> bool {
    SPACES.contains(&c)
}

fn in_ranges(c: char, ranges: &[(u32, u32)]) -> Option<(u32, u32)> {
    let u = c as u32;
    ranges.iter().copied().find(|&(a, b)| a <= u && u <= b)
}

/// The decimal value of a Unicode `Nd` character (`unicodedata.decimal`).
pub fn decimal(c: char) -> Option<u32> {
    in_ranges(c, ND_RANGES).map(|(a, _)| (c as u32 - a) % 10)
}

/// `str.isdigit()` for one character.
pub fn is_digit_char(c: char) -> bool {
    decimal(c).is_some() || in_ranges(c, DIGIT_NOT_DECIMAL).is_some()
}

/// `str.isdigit()`.
pub fn isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_digit_char)
}

/// `str.isdecimal()`.
pub fn isdecimal(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| decimal(c).is_some())
}

/// `str.strip()` (no argument).
pub fn strip(s: &str) -> &str {
    s.trim_matches(is_space)
}

/// `s[-n:]` in characters.
pub fn tail_chars(s: &str, n: usize) -> &str {
    let count = s.chars().count();
    if count <= n {
        return s;
    }
    let skip = count - n;
    s.char_indices().nth(skip).map_or(s, |(i, _)| &s[i..])
}

fn ranges_body(ranges: &[(u32, u32)]) -> String {
    let mut out = String::new();
    for &(a, b) in ranges {
        if a == b {
            out.push_str(&format!("\\x{{{a:x}}}"));
        } else {
            out.push_str(&format!("\\x{{{a:x}}}-\\x{{{b:x}}}"));
        }
    }
    out
}

fn space_body() -> String {
    let mut out = String::new();
    for c in SPACES {
        out.push_str(&format!("\\x{{{:x}}}", *c as u32));
    }
    out
}

/// Translate a Python `re` pattern into a Rust regex with the same matches. Supports what the adapters use: literals,
/// groups, alternation, quantifiers (greedy and lazy), classes, `^`/`$`, `\d`, `\s`, `\S`, `\w`, `\.` style escapes and
/// `\xNN`. `ignorecase` is `re.IGNORECASE` (Python also folds `İ`/`ı` to `i`). `mode` anchors like `re.match`
/// (`Anchor::Start`) or `re.fullmatch` (`Anchor::Full`). Patterns are constants of this crate, so a translation error is
/// a programming error caught by the tests.
pub fn compile(pat: &str, ignorecase: bool, mode: Anchor) -> Regex {
    let nd = ranges_body(ND_RANGES);
    let sp = space_body();
    let mut out = String::new();
    if ignorecase {
        out.push_str("(?i)");
    }
    match mode {
        Anchor::Search => out.push_str("(?:"),
        Anchor::Start | Anchor::Full => out.push_str("^(?:"),
    }
    let chars: Vec<char> = pat.chars().collect();
    let mut i = 0;
    let mut in_class = false;
    while i < chars.len() {
        let c = chars[i];
        if c == '\\' && i + 1 < chars.len() {
            let n = chars[i + 1];
            i += 2;
            match (n, in_class) {
                ('d', false) => out.push_str(&format!("[{nd}]")),
                ('d', true) => out.push_str(&nd),
                ('s', false) => out.push_str(&format!("[{sp}]")),
                ('s', true) => out.push_str(&sp),
                ('S', false) => out.push_str(&format!("[^{sp}]")),
                ('w', false) => out.push_str(r"[\p{L}\p{N}_]"),
                ('D' | 'W' | 'b' | 'B' | 'A' | 'Z' | 'S' | 'w', _) => {
                    raise("re.error", &format!("unsupported escape \\{n} in {pat}"))
                }
                _ => {
                    out.push('\\');
                    out.push(n);
                }
            }
            continue;
        }
        if in_class {
            if c == ']' {
                in_class = false;
            } else if c.is_ascii_alphabetic() && ignorecase {
                raise(
                    "re.error",
                    &format!("letter class under IGNORECASE in {pat}"),
                );
            }
            out.push(c);
        } else if c == '[' {
            in_class = true;
            out.push(c);
            // a leading ']' or '^]' is a literal in both dialects
            if chars.get(i + 1) == Some(&'^') {
                out.push('^');
                i += 1;
            }
        } else if c == '('
            && chars.get(i + 1) == Some(&'?')
            && matches!(chars.get(i + 2), Some('=' | '!' | '<'))
        {
            raise("re.error", &format!("look-around is not supported: {pat}"));
        } else if ignorecase && (c == 'i' || c == 'I') {
            out.push_str("[iI\\x{130}\\x{131}]");
        } else {
            out.push(c);
        }
        i += 1;
    }
    out.push(')');
    if mode == Anchor::Full {
        out.push_str("\\z");
    }
    Regex::new(&out).unwrap_or_else(|e| raise("re.error", &format!("{pat}: {e}")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    /// `re.search`
    Search,
    /// `re.match`
    Start,
    /// `re.fullmatch`
    Full,
}

// ── numbers from text ───────────────────────────────────────────────────────────────────────────────────
/// `_PyUnicode_TransformDecimalAndSpaceToASCII`: Unicode digits to ASCII, Unicode spaces to ' '; any other non-ASCII
/// character makes the string invalid.
fn to_ascii_number_text(s: &str) -> Option<String> {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if (c as u32) < 127 {
            out.push(c); // ASCII is kept as it is: `\x1c` stays and is not stripped later
        } else if is_space(c) {
            out.push(' ');
        } else {
            out.push(char::from_digit(decimal(c)?, 10)?);
        }
    }
    Some(out)
}

/// Python's rule for `_` in a numeric literal given to `float()`/`int()`: only between two digits.
fn remove_underscores(s: &str) -> Option<String> {
    if !s.contains('_') {
        return Some(s.to_string());
    }
    let b = s.as_bytes();
    for (i, &c) in b.iter().enumerate() {
        if c == b'_' {
            let prev = i.checked_sub(1).map(|j| b[j]);
            let next = b.get(i + 1).copied();
            if !prev.is_some_and(|p| p.is_ascii_digit())
                || !next.is_some_and(|n| n.is_ascii_digit())
            {
                return None;
            }
        }
    }
    Some(s.replace('_', ""))
}

static FLOAT_TEXT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^[+-]?(?:(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:e[+-]?[0-9]+)?|inf|infinity|nan)$")
        .unwrap_or_else(|_| unreachable!())
});

/// `float(s)` of a Python str; None where Python raises ValueError.
pub fn float_str(s: &str) -> Option<f64> {
    let ascii = to_ascii_number_text(s)?;
    let t = ascii.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r'));
    let t = remove_underscores(t)?;
    if !FLOAT_TEXT.is_match(&t) {
        return None;
    }
    t.parse::<f64>().ok()
}

/// A Python int as text: optional '-', digits without leading zeros ("0" for zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigInt(pub String);

impl BigInt {
    fn normalized(neg: bool, digits: &str) -> Self {
        let d = digits.trim_start_matches('0');
        if d.is_empty() {
            Self("0".into())
        } else if neg {
            Self(format!("-{d}"))
        } else {
            Self(d.to_string())
        }
    }

    pub fn is_negative(&self) -> bool {
        self.0.starts_with('-')
    }

    pub fn is_zero(&self) -> bool {
        self.0 == "0"
    }

    /// i128 when it fits.
    pub fn small(&self) -> Option<i128> {
        self.0.parse().ok()
    }

    /// The JSON number Python would serialise. Beyond 64 bits the value becomes a float (serde_json without
    /// arbitrary precision cannot carry a bigger integer, and reads one back as a float too); None beyond the float
    /// range.
    pub fn to_value(&self) -> Option<Value> {
        if let Ok(i) = self.0.parse::<i64>() {
            return Some(Value::from(i));
        }
        if let Ok(u) = self.0.parse::<u64>() {
            return Some(Value::from(u));
        }
        self.0
            .parse::<f64>()
            .ok()
            .and_then(Number::from_f64)
            .map(Value::Number)
    }

    /// `int(float)`: the float's exact integer part. None for NaN (ValueError) or infinity (OverflowError).
    pub fn from_f64(x: f64) -> Option<Self> {
        if !x.is_finite() {
            return None;
        }
        let t = x.trunc();
        let s = format!("{t:.0}");
        let neg = s.starts_with('-');
        Some(Self::normalized(neg, s.trim_start_matches('-')))
    }
}

impl fmt::Display for BigInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// `int(s)` of a Python str (base 10); None where Python raises ValueError.
pub fn int_str(s: &str) -> Option<BigInt> {
    let ascii = to_ascii_number_text(s)?;
    let t = ascii.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\u{b}' | '\u{c}' | '\r'));
    let (neg, body) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() || body.starts_with('_') {
        return None;
    }
    let digits = remove_underscores(body)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(BigInt::normalized(neg, &digits))
}

// ── values (what `json.loads` gave the Python adapter) ──────────────────────────────────────────────────
/// `bool(v)`.
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// An optional value's truthiness (`bool(params.get(k))`).
pub fn truthy_opt(v: Option<&Value>) -> bool {
    v.is_some_and(truthy)
}

/// `str(v)`.
pub fn str_of(v: &Value) -> String {
    ec_proto::pyfmt::str_value(v)
}

/// Python's type name of a JSON value (for TypeError messages).
pub fn type_name(v: Option<&Value>) -> &'static str {
    match v {
        None | Some(Value::Null) => "NoneType",
        Some(Value::Bool(_)) => "bool",
        Some(Value::Number(n)) => {
            if n.is_f64() {
                "float"
            } else {
                "int"
            }
        }
        Some(Value::String(_)) => "str",
        Some(Value::Array(_)) => "list",
        Some(Value::Object(_)) => "dict",
    }
}

/// `value is None` for `params.get(key)`.
pub fn is_none(v: Option<&Value>) -> bool {
    matches!(v, None | Some(Value::Null))
}

/// `params.get(key, default)`: the default only when the key is absent (an explicit null stays None).
pub fn get_or<'a>(
    params: &'a serde_json::Map<String, Value>,
    key: &str,
    default: &'a Value,
) -> &'a Value {
    params.get(key).unwrap_or(default)
}

// ── the numeric tower ───────────────────────────────────────────────────────────────────────────────────
/// A Python int or float. Ints beyond i128 only come from products of huge JSON integers; they keep their digits.
#[derive(Debug, Clone, PartialEq)]
pub enum Num {
    I(i128),
    F(f64),
    Big(BigInt),
}

impl Num {
    /// The number a JSON value is in Python (bool is an int). None for null, str, list, dict.
    pub fn of(v: &Value) -> Option<Self> {
        match v {
            Value::Bool(b) => Some(Self::I(i128::from(*b))),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Some(Self::I(i128::from(i)))
                } else if let Some(u) = n.as_u64() {
                    Some(Self::I(i128::from(u)))
                } else {
                    n.as_f64().map(Self::F)
                }
            }
            _ => None,
        }
    }

    /// `Num::of`, raising TypeError for anything that is not a number (`op` names the operation for the log).
    pub fn need(v: Option<&Value>, op: &str) -> Self {
        v.and_then(Self::of).unwrap_or_else(|| {
            raise(
                "TypeError",
                &format!("unsupported operand type(s) for {op}: '{}'", type_name(v)),
            )
        })
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            Self::I(i) => *i as f64,
            Self::F(f) => *f,
            Self::Big(b) => b.0.parse().unwrap_or(f64::NAN),
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F(_))
    }

    /// `a * b`.
    pub fn mul(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::F(_), _) | (_, Self::F(_)) => Self::F(self.as_f64() * other.as_f64()),
            (Self::I(a), Self::I(b)) => a.checked_mul(*b).map_or_else(
                || Self::Big(big_mul(&a.to_string(), &b.to_string())),
                Self::I,
            ),
            _ => Self::Big(big_mul(&self.to_string(), &other.to_string())),
        }
    }

    /// `a % b` (sign of the divisor, ZeroDivisionError on zero).
    pub fn rem(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::I(a), Self::I(b)) => {
                if *b == 0 {
                    raise("ZeroDivisionError", "integer modulo by zero");
                }
                Self::I(((a % b) + b) % b)
            }
            (Self::F(_), _) | (_, Self::F(_)) => {
                let (x, w) = (self.as_f64(), other.as_f64());
                if w == 0.0 {
                    raise("ZeroDivisionError", "float modulo");
                }
                let mut m = x % w;
                if m != 0.0 {
                    if (w < 0.0) != (m < 0.0) {
                        m += w;
                    }
                } else {
                    m = 0.0_f64.copysign(w);
                }
                Self::F(m)
            }
            _ => raise(
                "OverflowError",
                "modulo of an integer beyond 128 bits is not reproduced",
            ),
        }
    }

    /// Exact comparison like Python's (an int and a float compare by value, never by rounding the int). None when
    /// unordered (NaN).
    pub fn compare(&self, other: &Self) -> Option<std::cmp::Ordering> {
        use std::cmp::Ordering;
        match (self, other) {
            (Self::I(a), Self::I(b)) => Some(a.cmp(b)),
            (Self::F(a), Self::F(b)) => a.partial_cmp(b),
            (Self::I(a), Self::F(b)) => cmp_int_float(*a, *b),
            (Self::F(a), Self::I(b)) => cmp_int_float(*b, *a).map(Ordering::reverse),
            (Self::Big(a), _) => Some(if a.is_negative() {
                Ordering::Less
            } else {
                Ordering::Greater
            }),
            (_, Self::Big(b)) => Some(if b.is_negative() {
                Ordering::Greater
            } else {
                Ordering::Less
            }),
        }
    }

    pub fn gt(&self, other: &Self) -> bool {
        self.compare(other) == Some(std::cmp::Ordering::Greater)
    }

    pub fn lt(&self, other: &Self) -> bool {
        self.compare(other) == Some(std::cmp::Ordering::Less)
    }

    pub fn same(&self, other: &Self) -> bool {
        self.compare(other) == Some(std::cmp::Ordering::Equal)
    }

    pub fn truthy(&self) -> bool {
        match self {
            Self::I(i) => *i != 0,
            Self::F(f) => *f != 0.0,
            Self::Big(b) => !b.is_zero(),
        }
    }
}

fn cmp_int_float(a: i128, b: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    if b.is_nan() {
        return None;
    }
    if b.is_infinite() {
        return Some(if b > 0.0 {
            Ordering::Less
        } else {
            Ordering::Greater
        });
    }
    let fl = b.floor();
    // every finite float beyond i128 is far from any i128 we hold
    if fl >= 1.7e38 {
        return Some(Ordering::Less);
    }
    if fl <= -1.7e38 {
        return Some(Ordering::Greater);
    }
    #[allow(clippy::cast_possible_truncation)]
    let fi = fl as i128;
    match a.cmp(&fi) {
        Ordering::Equal => Some(if b > fl {
            Ordering::Less
        } else {
            Ordering::Equal
        }),
        o => Some(o),
    }
}

impl fmt::Display for Num {
    /// `str(x)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I(i) => write!(f, "{i}"),
            Self::F(x) => f.write_str(&crate::pyjson::float_repr(*x)),
            Self::Big(b) => write!(f, "{b}"),
        }
    }
}

/// Decimal multiplication of two (possibly signed) integers given as text.
fn big_mul(a: &str, b: &str) -> BigInt {
    let neg = a.starts_with('-') != b.starts_with('-');
    let da: Vec<u32> = a
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|d| u32::from(d - b'0'))
        .rev()
        .collect();
    let db: Vec<u32> = b
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|d| u32::from(d - b'0'))
        .rev()
        .collect();
    let mut out = vec![0u64; da.len() + db.len() + 1];
    for (i, x) in da.iter().enumerate() {
        for (j, y) in db.iter().enumerate() {
            out[i + j] += u64::from(*x) * u64::from(*y);
        }
    }
    let mut carry = 0u64;
    for d in &mut out {
        let v = *d + carry;
        *d = v % 10;
        carry = v / 10;
    }
    let digits: String = out
        .iter()
        .rev()
        .map(|d| char::from(b'0' + u8::try_from(*d).unwrap_or(0)))
        .collect();
    BigInt::normalized(neg, &digits)
}

/// `float(x)` of a JSON value. Err(TypeError/ValueError message) where Python raises.
pub fn float_of(v: Option<&Value>) -> Result<f64, String> {
    match v {
        Some(Value::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Some(Value::Number(_)) => Ok(Num::need(v, "float()").as_f64()),
        Some(Value::String(s)) => float_str(s)
            .ok_or_else(|| format!("ValueError: could not convert string to float: {s:?}")),
        _ => Err(format!(
            "TypeError: float() argument must be a string or a real number, not '{}'",
            type_name(v)
        )),
    }
}

/// `int(x)` of a JSON value. Err where Python raises.
pub fn int_of(v: Option<&Value>) -> Result<Num, String> {
    match v {
        Some(Value::Bool(b)) => Ok(Num::I(i128::from(*b))),
        Some(Value::Number(n)) if n.is_f64() => {
            let f = n.as_f64().unwrap_or(f64::NAN);
            let b = BigInt::from_f64(f).ok_or_else(|| {
                "ValueError/OverflowError: int() of a non-finite float".to_string()
            })?;
            Ok(b.small().map_or(Num::Big(b), Num::I))
        }
        Some(Value::Number(_)) => Ok(Num::need(v, "int()")),
        Some(Value::String(s)) => int_str(s)
            .map(|b| b.small().map_or(Num::Big(b), Num::I))
            .ok_or_else(|| format!("ValueError: invalid literal for int() with base 10: {s:?}")),
        _ => Err(format!(
            "TypeError: int() argument must be a string, a bytes-like object or a real number, not '{}'",
            type_name(v)
        )),
    }
}

/// `float(x)`, raising like Python.
pub fn float_or_raise(v: Option<&Value>) -> f64 {
    float_of(v).unwrap_or_else(|m| raise("TypeError", &m))
}

/// `int(x)`, raising like Python.
pub fn int_or_raise(v: Option<&Value>) -> Num {
    int_of(v).unwrap_or_else(|m| raise("TypeError", &m))
}

/// `x or default` for a param: the value when truthy, else the default number.
pub fn or_num(v: Option<&Value>, default: Num) -> Num {
    match v {
        Some(x) if truthy(x) => Num::of(x).unwrap_or_else(|| {
            raise(
                "TypeError",
                &format!("a '{}' value where a number was expected", type_name(v)),
            )
        }),
        _ => default,
    }
}

/// The Python value `x or default` evaluates to (any type).
pub fn or_value<'a>(v: Option<&'a Value>, default: &'a Value) -> &'a Value {
    match v {
        Some(x) if truthy(x) => x,
        _ => default,
    }
}

/// `a == b` for JSON values the Python way (1 == 1.0 == True; containers compare element-wise).
pub fn py_eq(a: &Value, b: &Value) -> bool {
    match (Num::of(a), Num::of(b)) {
        (Some(x), Some(y)) => x.same(&y),
        (Some(_), None) | (None, Some(_)) => false,
        (None, None) => match (a, b) {
            (Value::Array(x), Value::Array(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
            }
            (Value::Object(x), Value::Object(y)) => {
                x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
            }
            _ => a == b,
        },
    }
}

// ── floats ──────────────────────────────────────────────────────────────────────────────────────────────
/// `sum(values)` of Python floats (CPython 3.12+: start 0, Neumaier compensated summation). `values` must be
/// non-empty (the callers never sum an empty list: Python would return the int 0).
pub fn sum(values: &[f64]) -> f64 {
    let mut it = values.iter();
    let Some(first) = it.next() else {
        return 0.0;
    };
    let mut f = 0.0 + *first;
    let mut c = 0.0_f64;
    for &x in it {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// `round(x, ndigits)` for a float (correctly rounded, ties to even on the exact binary value).
pub fn round(x: f64, ndigits: usize) -> f64 {
    if !x.is_finite() || x == 0.0 {
        return x;
    }
    format!("{x:.ndigits$}").parse().unwrap_or(x)
}

/// `format(x, f".{prec}f")`.
pub fn fixed(x: f64, prec: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{x:.prec$}")
}

/// `format(x, "g")`.
pub fn fmt_g(x: f64) -> String {
    ec_proto::pyfmt::fmt_g(x)
}

/// `repr(x)` of a float.
pub fn repr(x: f64) -> String {
    crate::pyjson::float_repr(x)
}

/// `sorted(items)` of `(float key, payload)` pairs with unique keys, as CPython orders them even when a key is NaN
/// (every comparison with NaN is false): run detection, then binary insertion sort (exactly CPython's algorithm for
/// lists shorter than 64; longer lists with NaN keys may differ, without NaN any correct sort gives the same order).
pub fn sorted_by_float_key<T: Clone>(items: &[(f64, T)]) -> Vec<(f64, T)> {
    let mut a: Vec<(f64, T)> = items.to_vec();
    if !a.iter().any(|(k, _)| k.is_nan()) {
        a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
        return a;
    }
    let n = a.len();
    if n < 2 {
        return a;
    }
    let lt = |x: f64, y: f64| x < y;
    // count_run (CPython 3.13)
    let mut k = 1;
    while k < n && !lt(a[k].0, a[k - 1].0) {
        k += 1;
    }
    let run = if k == n {
        n
    } else {
        let mut skip_desc = false;
        if k > 1 {
            if lt(a[0].0, a[k - 1].0) {
                skip_desc = true;
            } else {
                a[..k].reverse();
            }
        }
        if skip_desc {
            k
        } else {
            k += 1;
            let mut neq = 0usize;
            while k < n {
                if lt(a[k].0, a[k - 1].0) {
                    if neq > 0 {
                        let len = neq + 1;
                        a[k - len..k].reverse();
                        neq = 0;
                    }
                } else if lt(a[k - 1].0, a[k].0) {
                    break;
                } else {
                    neq += 1;
                }
                k += 1;
            }
            if neq > 0 {
                let len = neq + 1;
                a[k - len..k].reverse();
            }
            a[..k].reverse();
            // after reversing, a naturally increasing suffix may extend the run
            while k < n && !lt(a[k].0, a[k - 1].0) {
                k += 1;
            }
            k
        }
    };
    // binarysort from `run`
    for ok in run.max(1)..n {
        let pivot = a[ok].clone();
        let (mut l, mut r) = (0, ok);
        while l < r {
            let m = (l + r) >> 1;
            if lt(pivot.0, a[m].0) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        let mut m = ok;
        while m > l {
            a[m] = a[m - 1].clone();
            m -= 1;
        }
        a[l] = pivot;
    }
    a
}

/// A dict keyed by Python floats, in insertion order: 0.0 and -0.0 are one key (the first spelling is kept), every NaN
/// is its own key.
#[derive(Debug, Default, Clone)]
pub struct FloatDict {
    items: Vec<(f64, f64)>,
    index: std::collections::HashMap<u64, usize>,
}

impl FloatDict {
    /// `d[key] = d.get(key, 0.0) + v` (`defaultdict(float)` `+=`).
    pub fn add(&mut self, key: f64, v: f64) {
        if key.is_nan() {
            self.items.push((key, 0.0 + v));
            return;
        }
        let bits = if key == 0.0 { 0 } else { key.to_bits() };
        match self.index.get(&bits) {
            Some(&i) => self.items[i].1 += v,
            None => {
                self.index.insert(bits, self.items.len());
                self.items.push((key, 0.0 + v));
            }
        }
    }

    pub fn items(&self) -> &[(f64, f64)] {
        &self.items
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

// ── json ────────────────────────────────────────────────────────────────────────────────────────────────
/// The nesting depth at which Python 3.13's `json.loads` stops with RecursionError when called at the top of a stack
/// (measured: 9998 levels load, 9999 do not). Deeper in a server's stack Python gives up slightly earlier.
pub const JSON_MAX_DEPTH: usize = 9998;

/// A document as `json.loads` builds it, kept in an arena (no recursion when it is dropped or dumped).
#[derive(Debug, Clone)]
pub struct JsonDoc {
    nodes: Vec<Node>,
}

#[derive(Debug, Clone)]
enum Node {
    Null,
    Bool(bool),
    /// normalised integer text
    Int(String),
    Float(f64),
    /// code points (a lone surrogate from `\udXXX` stays one)
    Str(Vec<u32>),
    Arr(Vec<usize>),
    /// keys in first-insertion order, the last value wins (dict semantics)
    Obj(Vec<(Vec<u32>, usize)>),
}

/// `json.loads` failed: `e.msg` of the JSONDecodeError.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonError(pub &'static str);

enum Frame {
    Arr { node: usize },
    Obj { node: usize, key: Vec<u32> },
}

impl JsonDoc {
    /// `json.loads(s)`. Err(msg) for a JSONDecodeError; a nesting deeper than [`JSON_MAX_DEPTH`] raises
    /// RecursionError (never a JSONDecodeError in Python either).
    pub fn loads(s: &str) -> Result<Self, JsonError> {
        let chars: Vec<char> = s.chars().collect();
        if chars.first() == Some(&'\u{feff}') {
            return Err(JsonError("Unexpected UTF-8 BOM (decode using utf-8-sig)"));
        }
        let mut p = Parser {
            s: &chars,
            nodes: Vec::new(),
        };
        let start = p.ws(0);
        let (root, end) = p.value(start)?;
        let end = p.ws(end);
        if end != chars.len() {
            return Err(JsonError("Extra data"));
        }
        debug_assert_eq!(root, 0);
        Ok(Self { nodes: p.nodes })
    }

    /// The root is a dict or a list.
    pub fn is_container(&self) -> bool {
        matches!(self.nodes.first(), Some(Node::Arr(_) | Node::Obj(_)))
    }

    /// The root is a dict.
    pub fn is_dict(&self) -> bool {
        matches!(self.nodes.first(), Some(Node::Obj(_)))
    }

    /// `key in doc` for a dict root.
    pub fn has_key(&self, key: &str) -> bool {
        let k: Vec<u32> = key.chars().map(|c| c as u32).collect();
        matches!(self.nodes.first(), Some(Node::Obj(items)) if items.iter().any(|(x, _)| *x == k))
    }

    /// `json.dumps(doc, separators=(",", ":"))` (ensure_ascii, NaN/Infinity spelled as Python does).
    pub fn dumps_compact(&self) -> String {
        enum Step {
            Node(usize),
            Text(&'static str),
            Key(usize, usize),
        }
        let mut out = String::new();
        let mut stack = vec![Step::Node(0)];
        while let Some(step) = stack.pop() {
            match step {
                Step::Text(t) => out.push_str(t),
                Step::Key(obj, i) => {
                    if let Node::Obj(items) = &self.nodes[obj] {
                        write_code_points(&mut out, &items[i].0);
                        out.push(':');
                    }
                }
                Step::Node(i) => match &self.nodes[i] {
                    Node::Null => out.push_str("null"),
                    Node::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
                    Node::Int(t) => out.push_str(t),
                    Node::Float(f) => out.push_str(&json_float(*f)),
                    Node::Str(cps) => write_code_points(&mut out, cps),
                    Node::Arr(items) => {
                        out.push('[');
                        stack.push(Step::Text("]"));
                        for (j, child) in items.iter().enumerate().rev() {
                            stack.push(Step::Node(*child));
                            if j > 0 {
                                stack.push(Step::Text(","));
                            }
                        }
                    }
                    Node::Obj(items) => {
                        out.push('{');
                        stack.push(Step::Text("}"));
                        for (j, (_, child)) in items.iter().enumerate().rev() {
                            stack.push(Step::Node(*child));
                            stack.push(Step::Key(i, j));
                            if j > 0 {
                                stack.push(Step::Text(","));
                            }
                        }
                    }
                },
            }
        }
        out
    }
}

/// `json.dumps` of a float.
fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        }
    } else {
        repr(f)
    }
}

/// A JSON string with ensure_ascii escaping, from code points that may hold lone surrogates.
fn write_code_points(out: &mut String, cps: &[u32]) {
    out.push('"');
    for &u in cps {
        match u {
            0x22 => out.push_str("\\\""),
            0x5c => out.push_str("\\\\"),
            0x0a => out.push_str("\\n"),
            0x0d => out.push_str("\\r"),
            0x09 => out.push_str("\\t"),
            0x08 => out.push_str("\\b"),
            0x0c => out.push_str("\\f"),
            0x20..=0x7e => out.push(char::from_u32(u).unwrap_or('?')),
            0x10000.. => {
                let v = u - 0x10000;
                out.push_str(&format!(
                    "\\u{:04x}\\u{:04x}",
                    0xd800 + (v >> 10),
                    0xdc00 + (v & 0x3ff)
                ));
            }
            _ => out.push_str(&format!("\\u{u:04x}")),
        }
    }
    out.push('"');
}

struct Parser<'a> {
    s: &'a [char],
    nodes: Vec<Node>,
}

const EXPECTING_VALUE: JsonError = JsonError("Expecting value");

impl Parser<'_> {
    fn at(&self, i: usize) -> Option<char> {
        self.s.get(i).copied()
    }

    fn ws(&self, mut i: usize) -> usize {
        while matches!(self.at(i), Some(' ' | '\t' | '\n' | '\r')) {
            i += 1;
        }
        i
    }

    fn push(&mut self, n: Node) -> usize {
        self.nodes.push(n);
        self.nodes.len() - 1
    }

    fn attach(&mut self, frame: &mut Frame, child: usize) {
        match frame {
            Frame::Arr { node } => {
                if let Node::Arr(items) = &mut self.nodes[*node] {
                    items.push(child);
                }
            }
            Frame::Obj { node, key } => {
                if let Node::Obj(items) = &mut self.nodes[*node] {
                    let k = std::mem::take(key);
                    if let Some(slot) = items.iter_mut().find(|(x, _)| *x == k) {
                        slot.1 = child;
                    } else {
                        items.push((k, child));
                    }
                }
            }
        }
    }

    /// `scan_once` iteratively: returns (node, index after it).
    fn value(&mut self, start: usize) -> Result<(usize, usize), JsonError> {
        let mut stack: Vec<Frame> = Vec::new();
        let mut idx = start;
        loop {
            // ---- read one term at idx (or open a container)
            let (node, mut next) = match self.at(idx) {
                None => return Err(EXPECTING_VALUE),
                Some('{') => {
                    if stack.len() + 1 > JSON_MAX_DEPTH {
                        raise(
                            "RecursionError",
                            "maximum recursion depth exceeded while decoding a JSON object",
                        );
                    }
                    let node = self.push(Node::Obj(Vec::new()));
                    let i = self.ws(idx + 1);
                    if self.at(i) == Some('}') {
                        (node, i + 1)
                    } else {
                        let (key, after) = self.key(i)?;
                        stack.push(Frame::Obj { node, key });
                        idx = after;
                        continue;
                    }
                }
                Some('[') => {
                    if stack.len() + 1 > JSON_MAX_DEPTH {
                        raise(
                            "RecursionError",
                            "maximum recursion depth exceeded while decoding a JSON array",
                        );
                    }
                    let node = self.push(Node::Arr(Vec::new()));
                    let i = self.ws(idx + 1);
                    if self.at(i) == Some(']') {
                        (node, i + 1)
                    } else {
                        stack.push(Frame::Arr { node });
                        idx = i;
                        continue;
                    }
                }
                Some(_) => self.scalar(idx)?,
            };
            // ---- attach to the enclosing containers, closing as many as end here
            let mut child = node;
            loop {
                let Some(mut frame) = stack.pop() else {
                    return Ok((child, next));
                };
                self.attach(&mut frame, child);
                let i = self.ws(next);
                let (close, kind_obj) = match frame {
                    Frame::Arr { .. } => (']', false),
                    Frame::Obj { .. } => ('}', true),
                };
                if self.at(i) == Some(close) {
                    child = match frame {
                        Frame::Arr { node } | Frame::Obj { node, .. } => node,
                    };
                    next = i + 1;
                    continue;
                }
                if self.at(i) != Some(',') {
                    return Err(JsonError("Expecting ',' delimiter"));
                }
                let j = self.ws(i + 1);
                if self.at(j) == Some(close) {
                    return Err(JsonError(if kind_obj {
                        "Illegal trailing comma before end of object"
                    } else {
                        "Illegal trailing comma before end of array"
                    }));
                }
                if let Frame::Obj { node, .. } = frame {
                    let (key, after) = self.key(j)?;
                    stack.push(Frame::Obj { node, key });
                    idx = after;
                } else {
                    stack.push(frame);
                    idx = j;
                }
                break;
            }
        }
    }

    /// `"key"` then `:`; returns the key and the index of the value.
    fn key(&mut self, i: usize) -> Result<(Vec<u32>, usize), JsonError> {
        if self.at(i) != Some('"') {
            return Err(JsonError(
                "Expecting property name enclosed in double quotes",
            ));
        }
        let (key, after) = self.string(i + 1)?;
        let j = self.ws(after);
        if self.at(j) != Some(':') {
            return Err(JsonError("Expecting ':' delimiter"));
        }
        Ok((key, self.ws(j + 1)))
    }

    fn word(&self, i: usize, w: &str) -> bool {
        let n = w.chars().count();
        // CPython checks `idx + len - 1 < length` and each character
        i + n <= self.s.len() && self.s[i..i + n].iter().copied().eq(w.chars())
    }

    fn scalar(&mut self, idx: usize) -> Result<(usize, usize), JsonError> {
        let c = self.at(idx).ok_or(EXPECTING_VALUE)?;
        match c {
            '"' => {
                let (s, after) = self.string(idx + 1)?;
                return Ok((self.push(Node::Str(s)), after));
            }
            'n' if self.word(idx, "null") => return Ok((self.push(Node::Null), idx + 4)),
            't' if self.word(idx, "true") => return Ok((self.push(Node::Bool(true)), idx + 4)),
            'f' if self.word(idx, "false") => return Ok((self.push(Node::Bool(false)), idx + 5)),
            'N' if self.word(idx, "NaN") => return Ok((self.push(Node::Float(f64::NAN)), idx + 3)),
            'I' if self.word(idx, "Infinity") => {
                return Ok((self.push(Node::Float(f64::INFINITY)), idx + 8));
            }
            '-' if self.word(idx, "-Infinity") => {
                return Ok((self.push(Node::Float(f64::NEG_INFINITY)), idx + 9));
            }
            _ => {}
        }
        self.number(idx)
    }

    fn digit(&self, i: usize) -> bool {
        self.at(i).is_some_and(|c| c.is_ascii_digit())
    }

    fn number(&mut self, start: usize) -> Result<(usize, usize), JsonError> {
        let end = self.s.len();
        let mut idx = start;
        if self.at(idx) == Some('-') {
            idx += 1;
            if idx >= end {
                return Err(EXPECTING_VALUE);
            }
        }
        match self.at(idx) {
            Some('1'..='9') => {
                idx += 1;
                while self.digit(idx) {
                    idx += 1;
                }
            }
            Some('0') => idx += 1,
            _ => return Err(EXPECTING_VALUE),
        }
        let mut is_float = false;
        if idx + 1 < end && self.at(idx) == Some('.') && self.digit(idx + 1) {
            is_float = true;
            idx += 2;
            while self.digit(idx) {
                idx += 1;
            }
        }
        if idx + 1 < end && matches!(self.at(idx), Some('e' | 'E')) {
            let e_start = idx;
            idx += 1;
            if idx + 1 < end && matches!(self.at(idx), Some('-' | '+')) {
                idx += 1;
            }
            while self.digit(idx) {
                idx += 1;
            }
            if self.digit(idx - 1) {
                is_float = true;
            } else {
                idx = e_start;
            }
        }
        let text: String = self.s[start..idx].iter().collect();
        let node = if is_float {
            Node::Float(text.parse().unwrap_or(f64::NAN))
        } else {
            let neg = text.starts_with('-');
            Node::Int(BigInt::normalized(neg, text.trim_start_matches('-')).0)
        };
        Ok((self.push(node), idx))
    }

    /// `scanstring` (strict): returns the code points and the index after the closing quote.
    fn string(&self, begin: usize) -> Result<(Vec<u32>, usize), JsonError> {
        const UNTERMINATED: JsonError = JsonError("Unterminated string starting at");
        const BAD_U: JsonError = JsonError("Invalid \\uXXXX escape");
        let len = self.s.len();
        let mut out: Vec<u32> = Vec::new();
        let mut end = begin;
        loop {
            let mut next = end;
            let mut c = None;
            while next < len {
                let d = self.s[next];
                if d == '"' || d == '\\' {
                    c = Some(d);
                    break;
                }
                if (d as u32) <= 0x1f {
                    return Err(JsonError("Invalid control character at"));
                }
                out.push(d as u32);
                next += 1;
            }
            match c {
                Some('"') => return Ok((out, next + 1)),
                Some('\\') => {}
                _ => return Err(UNTERMINATED),
            }
            next += 1;
            if next == len {
                return Err(UNTERMINATED);
            }
            let e = self.s[next];
            if e != 'u' {
                end = next + 1;
                let v = match e {
                    '"' => 0x22,
                    '\\' => 0x5c,
                    '/' => 0x2f,
                    'b' => 0x08,
                    'f' => 0x0c,
                    'n' => 0x0a,
                    'r' => 0x0d,
                    't' => 0x09,
                    _ => return Err(JsonError("Invalid \\escape")),
                };
                out.push(v);
                continue;
            }
            next += 1;
            end = next + 4;
            if end >= len {
                return Err(BAD_U);
            }
            let hex = |from: usize| -> Option<u32> {
                let mut v = 0u32;
                for k in from..from + 4 {
                    v = (v << 4) | self.s.get(k)?.to_digit(16)?;
                }
                Some(v)
            };
            let mut cp = hex(next).ok_or(BAD_U)?;
            next = end;
            if (0xd800..=0xdbff).contains(&cp)
                && end + 6 < len
                && self.s[next] == '\\'
                && self.s[next + 1] == 'u'
            {
                let c2 = hex(next + 2).ok_or(BAD_U)?;
                if (0xdc00..=0xdfff).contains(&c2) {
                    cp = 0x10000 + (((cp - 0xd800) << 10) | (c2 - 0xdc00));
                    end += 6;
                }
            }
            out.push(cp);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_and_int_text_follow_python() {
        assert_eq!(float_str(" 1_000.5 "), Some(1000.5));
        assert_eq!(float_str("١٢"), Some(12.0));
        assert_eq!(float_str("-iNfInItY"), Some(f64::NEG_INFINITY));
        assert!(float_str("nan").is_some_and(f64::is_nan));
        for bad in [
            "", ".", "1.2.3", "1__0", "_1", "1_", "0x10", "e5", "1e", "abc", "1_.5",
        ] {
            assert_eq!(float_str(bad), None, "{bad}");
        }
        assert_eq!(float_str("1."), Some(1.0));
        assert_eq!(float_str(".5"), Some(0.5));
        assert_eq!(int_str(" 007 "), Some(BigInt("7".into())));
        assert_eq!(int_str("-0"), Some(BigInt("0".into())));
        assert_eq!(int_str("1.0"), None);
        assert_eq!(int_str("٣"), Some(BigInt("3".into())));
        assert_eq!(int_str("²"), None);
    }

    #[test]
    fn regex_translation_matches_python_classes() {
        let r = compile(r"(\d+)%", false, Anchor::Search);
        assert_eq!(&r.captures("x ٤٢% y").unwrap()[1], "٤٢");
        let s = compile(r"a\s+b", false, Anchor::Search);
        assert!(s.is_match("a\u{1c}b"));
        let i = compile(r"kimi", true, Anchor::Search);
        assert!(i.is_match("KIMI") && i.is_match("kımı") && i.is_match("kİmİ"));
        assert!(compile(r"\d+\.\d+", false, Anchor::Full).is_match("12.0"));
        assert!(!compile(r"\d+\.\d+", false, Anchor::Full).is_match("12.0x"));
    }

    #[test]
    fn neumaier_sum_matches_python() {
        assert_eq!(sum(&[0.1; 10]), 1.0);
        assert_eq!(sum(&[1e100, 1.0, -1e100]), 1.0);
        assert_eq!(sum(&[-0.0]).to_bits(), 0.0f64.to_bits());
        assert_eq!(sum(&[0.1, 0.2, 0.3]), 0.6);
    }

    #[test]
    fn numbers_compare_and_format_like_python() {
        assert!(Num::I(2).same(&Num::F(2.0)));
        assert_eq!(Num::I(36).rem(&Num::I(8)), Num::I(4));
        assert_eq!(Num::I(-7).rem(&Num::I(2)), Num::I(1));
        assert_eq!(Num::F(-7.0).rem(&Num::F(2.0)), Num::F(1.0));
        assert_eq!(Num::I(2).mul(&Num::F(1.0)).to_string(), "2.0");
        let big = Num::I(10_000_000_000_000_000_000).mul(&Num::I(10_000_000_000_000_000_000));
        assert_eq!(
            big.mul(&Num::I(10)).to_string(),
            format!("1{}", "0".repeat(39))
        );
        assert!(Num::I(9_007_199_254_740_993).gt(&Num::F(9_007_199_254_740_992.0)));
        assert_eq!(round(1.03125, 4), 1.0312);
        assert_eq!(fixed(0.125, 2), "0.12");
        assert!(
            py_eq(&json!(1), &json!(true))
                && py_eq(&json!(2.0), &json!(2))
                && !py_eq(&json!(null), &json!(1))
        );
    }

    #[test]
    fn json_loads_messages_and_dumps() {
        let cases = [
            ("", "Expecting value"),
            ("[", "Expecting value"),
            ("{", "Expecting property name enclosed in double quotes"),
            ("[1,]", "Illegal trailing comma before end of array"),
            ("{\"a\":1,}", "Illegal trailing comma before end of object"),
            ("{\"a\" 1}", "Expecting ':' delimiter"),
            ("[1 2]", "Expecting ',' delimiter"),
            ("1.", "Extra data"),
            ("\"\\x\"", "Invalid \\escape"),
            ("\"\\u12\"", "Invalid \\uXXXX escape"),
            ("\"a", "Unterminated string starting at"),
            ("\"\u{1}\"", "Invalid control character at"),
            ("\u{feff}1", "Unexpected UTF-8 BOM (decode using utf-8-sig)"),
            ("{\"a\":", "Expecting value"),
        ];
        for (text, msg) in cases {
            assert_eq!(JsonDoc::loads(text).err(), Some(JsonError(msg)), "{text:?}");
        }
        let d =
            JsonDoc::loads("{\"a\": 1, \"b\": [NaN, -0, 4.0, 1e400], \"a\": \"\\ud800\\u00e9\"}")
                .unwrap();
        assert_eq!(
            d.dumps_compact(),
            "{\"a\":\"\\ud800\\u00e9\",\"b\":[NaN,0,4.0,Infinity]}"
        );
        assert!(d.is_dict() && d.has_key("b") && !d.has_key("c"));
        let deep = format!(
            "{}{}",
            "[".repeat(JSON_MAX_DEPTH),
            "]".repeat(JSON_MAX_DEPTH)
        );
        assert!(JsonDoc::loads(&deep).is_ok());
    }

    #[test]
    fn cpython_sort_with_nan_keys() {
        let nan = f64::NAN;
        let got = sorted_by_float_key(&[(3.0, 'a'), (nan, 'b'), (1.0, 'c'), (2.0, 'd')]);
        let order: String = got.iter().map(|x| x.1).collect();
        // CPython 3.13 (checked): no key is "smaller" than its predecessor, so the list is one run and stays as it is
        assert_eq!(order, "abcd");
        let got = sorted_by_float_key(&[
            (5.0, 'a'),
            (0.1, 'b'),
            (f64::INFINITY, 'c'),
            (nan, 'd'),
            (2.0, 'e'),
        ]);
        let order: String = got.iter().map(|x| x.1).collect();
        // a descending pair [5, 0.1] reversed, then extended by the non-decreasing (NaN-blind) suffix
        assert_eq!(order, "bacde");
    }
}
