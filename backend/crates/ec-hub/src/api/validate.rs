//! Request validation with FastAPI/pydantic v2's exact 422 output (loc / msg / type).
//!
//! A small schema language mirrors the pydantic models of domain/models.py: [`T`] for types (lax-mode coercions of
//! JSON input), [`Model`] for objects. `validate` returns the coerced value (e.g. "3" -> 3 for an int field) or the
//! error list in pydantic's order (fields in declaration order; a model validator only runs when the fields pass).
//! The coerced value is then deserialised into the Rust struct. backend/parity/gen/validation_vectors.py records
//! pydantic's answers for edge inputs; tests/test_validation_vectors.rs replays them (PORTING.md §7).

use std::sync::LazyLock;

use crate::sync::Recover as _;
use regex::Regex;
use serde_json::{Map, Number, Value};

/// One pydantic error, without `input`/`ctx` (the problem handler drops them: they can echo secrets).
#[derive(Debug, Clone, PartialEq)]
pub struct Issue {
    pub loc: Vec<Value>,
    pub msg: String,
    pub typ: String,
}

impl Issue {
    pub fn new(loc: &[Value], msg: impl Into<String>, typ: &str) -> Self {
        Self {
            loc: loc.to_vec(),
            msg: msg.into(),
            typ: typ.to_string(),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("loc".into(), Value::Array(self.loc.clone()));
        m.insert("msg".into(), Value::String(self.msg.clone()));
        m.insert("type".into(), Value::String(self.typ.clone()));
        Value::Object(m)
    }
}

pub type AfterFn = fn(&str) -> Result<(), &'static str>;
pub type ModelAfterFn = fn(&Map<String, Value>) -> Result<(), String>;

/// A field type.
#[derive(Debug, Clone)]
pub enum T {
    /// `str` with optional constraints (min/max length in characters, a regex that must match somewhere, as
    /// pydantic's `pattern` does with `re.search`; anchor it with ^...$ when the whole string must match)
    Str {
        min: Option<usize>,
        max: Option<usize>,
        pattern: Option<&'static str>,
    },
    /// `int` with optional ge / le
    Int {
        ge: Option<i64>,
        le: Option<i64>,
    },
    Float,
    Bool,
    /// `Any`
    Any,
    /// `dict[str, Any]`
    AnyDict,
    /// `dict[str, T]`
    Dict(Box<T>),
    List(Box<T>),
    /// `T | None`
    Opt(Box<T>),
    Literal(&'static [&'static str]),
    Model(&'static Model),
    /// `Annotated[str, AfterValidator(f)]`: the inner type first, then `f` ("Value error, <msg>")
    After(Box<T>, AfterFn),
    /// `A | B | ...` (smart mode): each member tried; all failing -> every member's errors, loc + member name
    Union(Vec<(&'static str, T)>),
}

impl T {
    pub const fn str() -> Self {
        Self::Str {
            min: None,
            max: None,
            pattern: None,
        }
    }
    pub const fn int() -> Self {
        Self::Int { ge: None, le: None }
    }
    pub fn opt(t: T) -> Self {
        Self::Opt(Box::new(t))
    }
    pub fn list(t: T) -> Self {
        Self::List(Box::new(t))
    }
    pub fn dict(t: T) -> Self {
        Self::Dict(Box::new(t))
    }
}

#[derive(Debug, Clone)]
pub struct Field {
    pub name: &'static str,
    pub t: T,
    /// None: required. Some(v): the default when the key is absent.
    pub default: Option<Value>,
}

impl Field {
    pub fn req(name: &'static str, t: T) -> Self {
        Self {
            name,
            t,
            default: None,
        }
    }
    pub fn opt(name: &'static str, t: T, default: Value) -> Self {
        Self {
            name,
            t,
            default: Some(default),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    /// the class name (pydantic's `model_type` message names it)
    pub name: &'static str,
    pub fields: Vec<Field>,
    pub after: Option<ModelAfterFn>,
}

static PATTERNS: LazyLock<std::sync::Mutex<std::collections::HashMap<&'static str, Regex>>> =
    LazyLock::new(Default::default);

fn pattern_matches(pat: &'static str, s: &str) -> bool {
    let mut cache = PATTERNS.guard();
    let re = cache.entry(pat).or_insert_with(|| {
        Regex::new(pat).unwrap_or_else(|_| Regex::new("$^").unwrap_or_else(|_| unreachable!()))
    });
    // pydantic-core checks `pattern` with the Rust regex crate (search semantics): `$` never matches before a final
    // newline, unlike Python's `re`. Same engine here, same answers.
    re.is_match(s)
}

/// `'a', 'b' or 'c'`
pub fn literal_list(choices: &[&str]) -> String {
    let quoted: Vec<String> = choices.iter().map(|c| format!("'{c}'")).collect();
    match quoted.len() {
        0 => String::new(),
        1 => quoted[0].clone(),
        n => format!("{} or {}", quoted[..n - 1].join(", "), quoted[n - 1]),
    }
}

fn plural(n: usize, word: &str) -> String {
    format!("{n} {word}{}", if n == 1 { "" } else { "s" })
}

/// Validate `v` against `t` at `loc`; push issues; return the coerced value (Null when it failed).
pub fn check(v: &Value, t: &T, loc: &[Value], errs: &mut Vec<Issue>) -> Value {
    match t {
        T::Any => v.clone(),
        T::Opt(inner) => {
            if v.is_null() {
                Value::Null
            } else {
                check(v, inner, loc, errs)
            }
        }
        T::Str { min, max, pattern } => match v {
            Value::String(s) => {
                let n = s.chars().count();
                if let Some(m) = min
                    && n < *m
                {
                    errs.push(Issue::new(
                        loc,
                        format!("String should have at least {}", plural(*m, "character")),
                        "string_too_short",
                    ));
                    return Value::Null;
                }
                if let Some(m) = max
                    && n > *m
                {
                    errs.push(Issue::new(
                        loc,
                        format!("String should have at most {}", plural(*m, "character")),
                        "string_too_long",
                    ));
                    return Value::Null;
                }
                if let Some(p) = pattern
                    && !pattern_matches(p, s)
                {
                    errs.push(Issue::new(
                        loc,
                        format!("String should match pattern '{p}'"),
                        "string_pattern_mismatch",
                    ));
                    return Value::Null;
                }
                v.clone()
            }
            _ => {
                errs.push(Issue::new(
                    loc,
                    "Input should be a valid string",
                    "string_type",
                ));
                Value::Null
            }
        },
        T::Int { ge, le } => {
            let Some(i) = coerce_int(v, loc, errs) else {
                return Value::Null;
            };
            if let Some(g) = ge
                && i < *g
            {
                errs.push(Issue::new(
                    loc,
                    format!("Input should be greater than or equal to {g}"),
                    "greater_than_equal",
                ));
                return Value::Null;
            }
            if let Some(l) = le
                && i > *l
            {
                errs.push(Issue::new(
                    loc,
                    format!("Input should be less than or equal to {l}"),
                    "less_than_equal",
                ));
                return Value::Null;
            }
            Value::from(i)
        }
        T::Float => coerce_float(v, loc, errs).map_or(Value::Null, |f| {
            Number::from_f64(f).map_or(Value::Null, Value::Number)
        }),
        T::Bool => coerce_bool(v, loc, errs).map_or(Value::Null, Value::Bool),
        T::Literal(choices) => match v {
            Value::String(s) if choices.contains(&s.as_str()) => v.clone(),
            _ => {
                errs.push(Issue::new(
                    loc,
                    format!("Input should be {}", literal_list(choices)),
                    "literal_error",
                ));
                Value::Null
            }
        },
        T::AnyDict => match v {
            Value::Object(_) => v.clone(),
            _ => {
                errs.push(Issue::new(
                    loc,
                    "Input should be a valid dictionary",
                    "dict_type",
                ));
                Value::Null
            }
        },
        T::Dict(inner) => match v {
            Value::Object(o) => {
                let mut out = Map::new();
                for (k, x) in o {
                    let mut l = loc.to_vec();
                    l.push(Value::String(k.clone()));
                    out.insert(k.clone(), check(x, inner, &l, errs));
                }
                Value::Object(out)
            }
            _ => {
                errs.push(Issue::new(
                    loc,
                    "Input should be a valid dictionary",
                    "dict_type",
                ));
                Value::Null
            }
        },
        T::List(inner) => match v {
            Value::Array(a) => Value::Array(
                a.iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let mut l = loc.to_vec();
                        l.push(Value::from(i));
                        check(x, inner, &l, errs)
                    })
                    .collect(),
            ),
            _ => {
                errs.push(Issue::new(loc, "Input should be a valid list", "list_type"));
                Value::Null
            }
        },
        T::Model(m) => match v {
            Value::Object(o) => check_model(o, m, loc, errs),
            _ => {
                errs.push(Issue::new(
                    loc,
                    format!(
                        "Input should be a valid dictionary or instance of {}",
                        m.name
                    ),
                    "model_type",
                ));
                Value::Null
            }
        },
        T::After(inner, f) => {
            let before = errs.len();
            let out = check(v, inner, loc, errs);
            if errs.len() > before {
                return Value::Null;
            }
            match out.as_str().map(f) {
                Some(Err(msg)) => {
                    errs.push(Issue::new(
                        loc,
                        format!("Value error, {msg}"),
                        "value_error",
                    ));
                    Value::Null
                }
                _ => out,
            }
        }
        T::Union(members) => {
            let mut all = Vec::new();
            for (name, mt) in members {
                let mut e = Vec::new();
                let mut l = loc.to_vec();
                l.push(Value::String((*name).into()));
                let out = check(v, mt, &l, &mut e);
                if e.is_empty() {
                    return out;
                }
                all.extend(e);
            }
            errs.extend(all);
            Value::Null
        }
    }
}

/// An object against a model: every field in order (missing required -> "Field required"), extras ignored, then
/// the model validator when all fields passed.
pub fn check_model(
    o: &Map<String, Value>,
    m: &Model,
    loc: &[Value],
    errs: &mut Vec<Issue>,
) -> Value {
    let before = errs.len();
    let mut out = Map::new();
    for f in &m.fields {
        let mut l = loc.to_vec();
        l.push(Value::String(f.name.into()));
        match o.get(f.name) {
            Some(v) => {
                out.insert(f.name.into(), check(v, &f.t, &l, errs));
            }
            None => match &f.default {
                Some(d) => {
                    out.insert(f.name.into(), d.clone());
                }
                None => errs.push(Issue::new(&l, "Field required", "missing")),
            },
        }
    }
    if errs.len() == before
        && let Some(after) = m.after
        && let Err(msg) = after(&out)
    {
        errs.push(Issue::new(
            loc,
            format!("Value error, {msg}"),
            "value_error",
        ));
    }
    Value::Object(out)
}

fn int_parsing(loc: &[Value], errs: &mut Vec<Issue>) {
    errs.push(Issue::new(
        loc,
        "Input should be a valid integer, unable to parse string as an integer",
        "int_parsing",
    ));
}

/// pydantic lax int (python mode): ints; bools; integral floats (too large: int_parsing_size); strings holding an
/// integer or an integer with a zero fraction ("3", " 7 ", "1_000", "-1_000.0"), whitespace trimmed.
pub fn coerce_int(v: &Value, loc: &[Value], errs: &mut Vec<Issue>) -> Option<i64> {
    match v {
        Value::Bool(b) => Some(i64::from(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return Some(i);
            }
            let f = n.as_f64().unwrap_or(f64::NAN);
            if f.is_finite() && f.fract() == 0.0 {
                if f.abs() < 9.2e18 {
                    #[allow(clippy::cast_possible_truncation)]
                    return Some(f as i64);
                }
                errs.push(Issue::new(
                    loc,
                    "Unable to parse input string as an integer, exceeded maximum size",
                    "int_parsing_size",
                ));
            } else {
                errs.push(Issue::new(
                    loc,
                    "Input should be a valid integer, got a number with a fractional part",
                    "int_from_float",
                ));
            }
            None
        }
        Value::String(s) => {
            let t = s.trim();
            let (int_part, frac) = t.split_once('.').map_or((t, None), |(a, b)| (a, Some(b)));
            let frac_ok = frac.is_none_or(|f| !f.is_empty() && f.chars().all(|c| c == '0'));
            match parse_int_str(int_part) {
                Some(i) if frac_ok => Some(i),
                _ => {
                    int_parsing(loc, errs);
                    None
                }
            }
        }
        _ => {
            errs.push(Issue::new(
                loc,
                "Input should be a valid integer",
                "int_type",
            ));
            None
        }
    }
}

/// An integer string: optional sign, ASCII digits, single underscores between digits.
pub fn parse_int_str(t: &str) -> Option<i64> {
    let (neg, digits) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if !underscored_digits(digits) {
        return None;
    }
    let n: i64 = digits.replace('_', "").parse().ok()?;
    Some(if neg { -n } else { n })
}

fn underscored_digits(d: &str) -> bool {
    !d.is_empty()
        && d.chars().all(|c| c.is_ascii_digit() || c == '_')
        && !d.starts_with('_')
        && !d.ends_with('_')
        && !d.contains("__")
}

/// pydantic lax float: numbers; bools; numeric strings (whitespace trimmed, underscores between digits, inf/nan).
pub fn coerce_float(v: &Value, loc: &[Value], errs: &mut Vec<Issue>) -> Option<f64> {
    match v {
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Value::Number(n) => n.as_f64(),
        Value::String(s) => {
            let t = s.trim();
            let lower = t.to_ascii_lowercase();
            let parsed = match lower.as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
                "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
                "nan" | "+nan" | "-nan" => Some(f64::NAN),
                _ => {
                    // underscores only between digits (as Python's float() and pydantic accept them)
                    let b = t.as_bytes();
                    let ok_us = b.iter().enumerate().all(|(i, c)| {
                        *c != b'_'
                            || (i > 0
                                && i + 1 < b.len()
                                && b[i - 1].is_ascii_digit()
                                && b[i + 1].is_ascii_digit())
                    });
                    let ascii = t
                        .chars()
                        .all(|c| c.is_ascii_digit() || "+-.eE_".contains(c));
                    if ok_us && ascii {
                        t.replace('_', "").parse::<f64>().ok()
                    } else {
                        None
                    }
                }
            };
            if parsed.is_none() {
                errs.push(Issue::new(
                    loc,
                    "Input should be a valid number, unable to parse string as a number",
                    "float_parsing",
                ));
            }
            parsed
        }
        _ => {
            errs.push(Issue::new(
                loc,
                "Input should be a valid number",
                "float_type",
            ));
            None
        }
    }
}

/// pydantic lax bool: bools; the ints 0/1 (other ints: bool_parsing); floats 0.0/1.0 (other integral floats:
/// bool_parsing, the rest bool_type); the strings 0/1/true/false/yes/no/on/off/t/f/y/n in any case, untrimmed.
pub fn coerce_bool(v: &Value, loc: &[Value], errs: &mut Vec<Issue>) -> Option<bool> {
    let parsing = |errs: &mut Vec<Issue>| {
        errs.push(Issue::new(
            loc,
            "Input should be a valid boolean, unable to interpret input",
            "bool_parsing",
        ));
    };
    let typ = |errs: &mut Vec<Issue>| {
        errs.push(Issue::new(
            loc,
            "Input should be a valid boolean",
            "bool_type",
        ));
    };
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                return match i {
                    0 => Some(false),
                    1 => Some(true),
                    _ => {
                        parsing(errs);
                        None
                    }
                };
            }
            match n.as_f64() {
                Some(0.0) => Some(false),
                Some(1.0) => Some(true),
                Some(f) if f.fract() == 0.0 && f.abs() < 9.2e18 && !n.is_u64() => {
                    parsing(errs);
                    None
                }
                _ => {
                    typ(errs);
                    None
                }
            }
        }
        Value::String(s) => match parse_bool_str(s) {
            Some(b) => Some(b),
            None => {
                parsing(errs);
                None
            }
        },
        _ => {
            typ(errs);
            None
        }
    }
}

pub fn parse_bool_str(s: &str) -> Option<bool> {
    match s.to_ascii_lowercase().as_str() {
        "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
        "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
        _ => None,
    }
}

// ── query and path parameters (strings in, FastAPI's loc ["query"|"path", name]) ───────────────────────────
/// A request's query parameters: last value wins per name (Starlette's ImmutableMultiDict.get).
#[derive(Debug, Clone, Default)]
pub struct Query(pub Vec<(String, String)>);

impl Query {
    /// `parse_qsl(qs, keep_blank_values=True)`: '+' is a space, %XX decoded (invalid UTF-8 replaced).
    pub fn parse(qs: &str) -> Self {
        let mut out = Vec::new();
        for part in qs.split(['&', ';']) {
            if part.is_empty() {
                continue;
            }
            let (k, v) = part.split_once('=').unwrap_or((part, ""));
            out.push((qs_decode(k), qs_decode(v)));
        }
        Self(out)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

fn qs_decode(s: &str) -> String {
    let plus = s.replace('+', " ");
    percent_encoding::percent_decode_str(&plus)
        .decode_utf8_lossy()
        .into_owned()
}

/// Collects the 422 issues of one request's parameters and body.
#[derive(Debug, Default)]
pub struct Params {
    pub errs: Vec<Issue>,
}

impl Params {
    fn loc(src: &str, name: &str) -> Vec<Value> {
        vec![Value::String(src.into()), Value::String(name.into())]
    }

    /// `name: str = default` (or required when default is None)
    pub fn str_q(&mut self, q: &Query, name: &str, default: Option<&str>) -> Option<String> {
        match q.get(name) {
            Some(v) => Some(v.to_string()),
            None => {
                if default.is_none() {
                    self.errs.push(Issue::new(
                        &Self::loc("query", name),
                        "Field required",
                        "missing",
                    ));
                }
                default.map(str::to_string)
            }
        }
    }

    /// `name: str | None = None` with optional constraints
    pub fn opt_str_q(&mut self, q: &Query, name: &str, t: &T) -> Option<String> {
        let v = q.get(name)?;
        let out = check(
            &Value::String(v.into()),
            t,
            &Self::loc("query", name),
            &mut self.errs,
        );
        out.as_str().map(str::to_string)
    }

    pub fn int_q(&mut self, q: &Query, name: &str, default: i64) -> i64 {
        match q.get(name) {
            None => default,
            Some(v) => coerce_int(
                &Value::String(v.into()),
                &Self::loc("query", name),
                &mut self.errs,
            )
            .unwrap_or(default),
        }
    }

    pub fn opt_float_q(&mut self, q: &Query, name: &str) -> Option<f64> {
        let v = q.get(name)?;
        coerce_float(
            &Value::String(v.into()),
            &Self::loc("query", name),
            &mut self.errs,
        )
    }

    pub fn bool_q(&mut self, q: &Query, name: &str, default: bool) -> bool {
        match q.get(name) {
            None => default,
            Some(v) => coerce_bool(
                &Value::String(v.into()),
                &Self::loc("query", name),
                &mut self.errs,
            )
            .unwrap_or(default),
        }
    }

    pub fn opt_bool_q(&mut self, q: &Query, name: &str) -> Option<bool> {
        let v = q.get(name)?;
        coerce_bool(
            &Value::String(v.into()),
            &Self::loc("query", name),
            &mut self.errs,
        )
    }

    /// A path parameter checked against `t` (a pattern or an AfterValidator).
    pub fn path(&mut self, name: &str, value: &str, t: &T) -> String {
        check(
            &Value::String(value.into()),
            t,
            &Self::loc("path", name),
            &mut self.errs,
        );
        value.to_string()
    }

    /// The JSON body against `t` (loc starts with "body"); `None` = no body.
    pub fn body(&mut self, body: &Body, t: &T, required: bool) -> Value {
        let root = vec![Value::String("body".into())];
        match body {
            Body::Missing => {
                if required {
                    self.errs
                        .push(Issue::new(&root, "Field required", "missing"));
                }
                Value::Null
            }
            Body::Invalid(pos) => {
                self.errs.push(Issue::new(
                    &[Value::String("body".into()), Value::from(*pos)],
                    "JSON decode error",
                    "json_invalid",
                ));
                Value::Null
            }
            Body::NotJson => {
                // FastAPI hands the raw bytes to the model: never a dict
                let msg = match t {
                    T::Model(_) | T::Opt(_) => (
                        "Input should be a valid dictionary or object to extract fields from",
                        "model_attributes_type",
                    ),
                    _ => ("Input should be a valid dictionary", "dict_type"),
                };
                self.errs.push(Issue::new(&root, msg.0, msg.1));
                Value::Null
            }
            Body::Json(v) => match t {
                // the top-level body model: a non-object is model_attributes_type (FastAPI validates from attributes)
                T::Model(m) => match v {
                    Value::Object(o) => check_model(o, m, &root, &mut self.errs),
                    _ => {
                        self.errs.push(Issue::new(
                            &root,
                            "Input should be a valid dictionary or object to extract fields from",
                            "model_attributes_type",
                        ));
                        Value::Null
                    }
                },
                T::Opt(inner) if v.is_null() => {
                    let _ = inner;
                    Value::Null
                }
                T::Opt(inner) => match (inner.as_ref(), v) {
                    (T::Model(m), Value::Object(o)) => check_model(o, m, &root, &mut self.errs),
                    (T::Model(_), _) => {
                        self.errs.push(Issue::new(
                            &root,
                            "Input should be a valid dictionary or object to extract fields from",
                            "model_attributes_type",
                        ));
                        Value::Null
                    }
                    (other, _) => check(v, other, &root, &mut self.errs),
                },
                other => check(v, other, &root, &mut self.errs),
            },
        }
    }

    pub fn ok(&self) -> bool {
        self.errs.is_empty()
    }
}

/// The request body as FastAPI sees it.
#[derive(Debug, Clone, PartialEq)]
pub enum Body {
    /// no body bytes at all
    Missing,
    /// a JSON content-type with bytes that do not parse; the char position of the error
    Invalid(usize),
    /// bytes with a content-type that is not JSON
    NotJson,
    Json(Value),
}

impl Body {
    /// FastAPI's reading of a body: JSON when the content-type is application/json or */*+json (or absent? no:
    /// an absent content-type is not JSON in FastAPI 0.141's strict mode).
    pub fn from_bytes(content_type: Option<&str>, bytes: &[u8]) -> Self {
        if bytes.is_empty() {
            return Self::Missing;
        }
        let ct = content_type
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        let is_json =
            ct == "application/json" || (ct.starts_with("application/") && ct.ends_with("+json"));
        if !is_json {
            return Self::NotJson;
        }
        let text = String::from_utf8_lossy(bytes);
        match serde_json::from_str::<Value>(&text) {
            Ok(v) => Self::Json(v),
            Err(e) => Self::Invalid(json_error_pos(&text, e.line(), e.column())),
        }
    }
}

/// Python's `JSONDecodeError.pos` approximated from serde_json's 1-based line/column: the char offset of the
/// column, minus one (serde reports the column after the offending char). PORTING.md D-JSONPOS.
fn json_error_pos(text: &str, line: usize, column: usize) -> usize {
    let mut pos = 0usize;
    for (i, l) in text.split('\n').enumerate() {
        if i + 1 == line {
            let col_bytes = column.min(l.len());
            let chars = l.get(..col_bytes).map_or(col_bytes, |p| p.chars().count());
            return (pos + chars).saturating_sub(1);
        }
        pos += l.chars().count() + 1;
    }
    pos.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn errs_of(v: &Value, t: &T) -> Vec<(Vec<Value>, String, String)> {
        let mut e = Vec::new();
        check(v, t, &[json!("body"), json!("f")], &mut e);
        e.into_iter().map(|i| (i.loc, i.msg, i.typ)).collect()
    }

    #[test]
    fn test_lax_coercions() {
        let mut e = Vec::new();
        assert_eq!(check(&json!("3"), &T::int(), &[], &mut e), json!(3));
        assert_eq!(check(&json!(2.0), &T::int(), &[], &mut e), json!(2));
        assert_eq!(check(&json!(true), &T::int(), &[], &mut e), json!(1));
        assert_eq!(check(&json!("yes"), &T::Bool, &[], &mut e), json!(true));
        assert!(e.is_empty());
        assert_eq!(errs_of(&json!(2.5), &T::int())[0].2, "int_from_float");
        assert_eq!(errs_of(&json!("abc"), &T::int())[0].2, "int_parsing");
        assert_eq!(errs_of(&json!(null), &T::int())[0].2, "int_type");
        assert_eq!(
            errs_of(&json!(5), &T::str())[0].1,
            "Input should be a valid string"
        );
    }

    #[test]
    fn test_constraints_and_literals() {
        let t = T::Str {
            min: Some(1),
            max: Some(120),
            pattern: None,
        };
        assert_eq!(
            errs_of(&json!(""), &t)[0].1,
            "String should have at least 1 character"
        );
        assert_eq!(
            errs_of(&json!("x".repeat(121)), &t)[0].1,
            "String should have at most 120 characters"
        );
        let lit = T::Literal(&["hf", "gguf", "ollama", "plain", "other"]);
        assert_eq!(
            errs_of(&json!("zz"), &lit)[0].1,
            "Input should be 'hf', 'gguf', 'ollama', 'plain' or 'other'"
        );
        assert_eq!(literal_list(&["admin", "viewer"]), "'admin' or 'viewer'");
        let p = T::Str {
            min: None,
            max: None,
            pattern: Some(r"^[a-z0-9][a-z0-9-]{0,31}$"),
        };
        assert_eq!(errs_of(&json!("BAD"), &p)[0].2, "string_pattern_mismatch");
        assert!(errs_of(&json!("ok-1"), &p).is_empty());
    }

    #[test]
    fn test_union_reports_every_member() {
        let u = T::Union(vec![("str", T::str()), ("dict[str,any]", T::AnyDict)]);
        let e = errs_of(&json!(5), &u);
        assert_eq!(e.len(), 2);
        assert_eq!(e[0].0, vec![json!("body"), json!("f"), json!("str")]);
        assert_eq!(e[1].2, "dict_type");
    }

    #[test]
    fn test_body_reading_like_fastapi() {
        assert_eq!(
            Body::from_bytes(Some("application/json"), b""),
            Body::Missing
        );
        assert_eq!(Body::from_bytes(Some("text/plain"), b"{}"), Body::NotJson);
        assert_eq!(Body::from_bytes(None, b"{}"), Body::NotJson);
        assert_eq!(
            Body::from_bytes(Some("application/json; charset=utf-8"), b"{\"a\":1}"),
            Body::Json(json!({"a": 1}))
        );
        assert_eq!(
            Body::from_bytes(Some("application/json"), b"{bad"),
            Body::Invalid(1)
        );
        assert_eq!(
            Body::from_bytes(Some("application/json"), b"{\"name\":\"x\"} trailing"),
            Body::Invalid(13)
        );
    }

    #[test]
    fn test_query_last_value_wins_and_decoding() {
        let q = Query::parse("limit=1&limit=2&q=a+b%21&e=");
        assert_eq!(q.get("limit"), Some("2"));
        assert_eq!(q.get("q"), Some("a b!"));
        assert_eq!(q.get("e"), Some(""));
        let mut p = Params::default();
        assert_eq!(p.int_q(&q, "e", 7), 7);
        assert_eq!(p.errs[0].typ, "int_parsing");
    }
}
