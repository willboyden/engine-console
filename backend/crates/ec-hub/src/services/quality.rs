//! Ported from the lab's `model-quality-harness` (Hermes sandbox, `harness/judge.py` + `harness/run.py`).
//!
//! A deterministic tool-calling benchmark: for each embedded scenario the runner sends the scenario's
//! messages and tool schemas to a ready instance through its host's node, executes the model's tool
//! calls against the scenario's own `tool_impl` (constant results or rules — no network, no side
//! effects), loops up to the scenario's `max_rounds`, and scores the run rule-by-rule (no LLM judge).
//! The aggregate score is the share of passed criteria across the whole run (e.g. 51/52 = 98.1),
//! the same number the source harness reports.
//!
//! Provenance of the scenario files: `quality-scenarios/PROMOTIONS.md`. Scoring quirks the port
//! keeps (they were bugs once, then fixed and covered by tests there): position resolves WITHIN
//! calls of the same tool name, and numeric answers also match the thousands-separated form
//! ("1301.04" matches "1,301.04") without a spurious ".0" on integers.

use serde_json::{Map, Value, json};

use crate::services::nodehttp::{HttpOpts, JSON_HEADERS, NodeHttp};

/// The embedded scenarios, sorted by file name like the source harness's `load_scenarios`.
pub fn scenarios() -> Vec<(String, Value)> {
    let files: &[(&str, &str)] = &[
        (
            "bank_transfer_limit_split",
            include_str!("../../quality-scenarios/bank_transfer_limit_split.json"),
        ),
        (
            "flights_multiconstraint",
            include_str!("../../quality-scenarios/flights_multiconstraint.json"),
        ),
        (
            "fx_conversion_retry",
            include_str!("../../quality-scenarios/fx_conversion_retry.json"),
        ),
        (
            "hotel_cancel_refund",
            include_str!("../../quality-scenarios/hotel_cancel_refund.json"),
        ),
        (
            "inventory_then_reserve",
            include_str!("../../quality-scenarios/inventory_then_reserve.json"),
        ),
        (
            "pizza_distractor",
            include_str!("../../quality-scenarios/pizza_distractor.json"),
        ),
        (
            "playlist_merge_dedupe",
            include_str!("../../quality-scenarios/playlist_merge_dedupe.json"),
        ),
        (
            "report_two_step",
            include_str!("../../quality-scenarios/report_two_step.json"),
        ),
        (
            "table_booking",
            include_str!("../../quality-scenarios/table_booking.json"),
        ),
        (
            "unit_extraction",
            include_str!("../../quality-scenarios/unit_extraction.json"),
        ),
        (
            "warehouse_bin_relocation",
            include_str!("../../quality-scenarios/warehouse_bin_relocation.json"),
        ),
        (
            "weather_tokyo",
            include_str!("../../quality-scenarios/weather_tokyo.json"),
        ),
    ];
    files
        .iter()
        .filter_map(|(id, text)| {
            let v: Value = serde_json::from_str(text).ok()?;
            // the embedded file name is the id of record; a mismatched "id" field would only confuse
            let id = v
                .get("id")
                .and_then(Value::as_str)
                .filter(|s| *s == *id)
                .map(String::from)?;
            Some((id, v))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Normalization (judge.py norm_scalar / norm_params)
// ---------------------------------------------------------------------------

/// Python-equality for two JSON values: numbers compare as floats (400 == 400.0),
/// strings case-insensitively only where the judge normalizes them itself.
fn py_json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => x.as_f64() == y.as_f64(),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(u, v)| py_json_eq(u, v))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, u)| y.get(k).is_some_and(|v| py_json_eq(u, v)))
        }
        _ => a == b,
    }
}

/// Strip + casefold a param key or string value (`str.strip().casefold()`).
fn fold(s: &str) -> String {
    s.trim().to_lowercase()
}

/// judge.py `norm_scalar` for a JSON value: strings are stripped + casefolded, other scalars
/// compare through `py_json_eq`.
fn norm_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::String(fold(s)),
        other => other.clone(),
    }
}

// ---------------------------------------------------------------------------
// Tool implementation simulation (judge.py _eval_rule / simulate_tool)
// ---------------------------------------------------------------------------

fn eval_rule(cond: &Value, params: &Map<String, Value>) -> bool {
    let op = cond.get("op").and_then(Value::as_str).unwrap_or("");
    let value = cond.get("value");
    let param = cond.get("param").and_then(Value::as_str);
    match (param, op) {
        (Some(p), "eq" | "ne" | "lt" | "le" | "gt" | "ge" | "contains") => {
            let Some(pv) = params.get(p) else {
                return false;
            };
            match op {
                "eq" | "ne" => {
                    let Some(v) = value else { return false };
                    // norm_scalar casefolds strings; numbers compare as floats
                    let eq = match (pv.as_str(), v.as_str()) {
                        (Some(a), Some(b)) => fold(a) == fold(b),
                        _ => py_json_eq(pv, v),
                    };
                    if op == "eq" { eq } else { !eq }
                }
                "lt" | "le" | "gt" | "ge" => {
                    let (Some(a), Some(b)) = (pv.as_f64(), value.and_then(Value::as_f64)) else {
                        return false;
                    };
                    match op {
                        "lt" => a < b,
                        "le" => a <= b,
                        "gt" => a > b,
                        _ => a >= b,
                    }
                }
                _ => {
                    // "contains": str(value).casefold() in str(pv).casefold()
                    let as_lower = |v: &Value| {
                        v.as_str()
                            .map_or_else(|| v.to_string(), String::from)
                            .to_lowercase()
                    };
                    let pv = as_lower(pv);
                    let want = value.map(as_lower).unwrap_or_default();
                    pv.contains(&want)
                }
            }
        }
        // No param key: apply to the params dict as a whole (unused in the current suite).
        (None, "eq") => value.is_some_and(|v| {
            v.as_object().is_some_and(|want| {
                want.len() == params.len()
                    && want
                        .iter()
                        .all(|(k, u)| params.get(k).is_some_and(|got| py_json_eq(got, u)))
            })
        }),
        _ => false,
    }
}

/// Run the scenario's `tool_impl` for one call; returns the JSON content of the tool-role message.
/// `raw_args` is the parsed arguments object, or `None` when the arguments were not a JSON object.
pub fn simulate_tool(name: &str, raw_args: Option<&Value>, impls: &Value) -> Value {
    if raw_args.is_none() {
        return json!({"error": "tool call arguments were not valid JSON; tool not executed"});
    }
    let Some(Value::Object(_)) = raw_args else {
        return json!({"error": "tool call arguments must be a JSON object"});
    };
    let spec = impls.get(name);
    let Some(spec) = spec else {
        return json!({"error": format!("unknown tool '{name}'")});
    };
    match spec.get("kind").and_then(Value::as_str) {
        Some("constant") => spec.get("value").cloned().unwrap_or(Value::Null),
        Some("rule") => {
            let params = raw_args
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            let ok = spec
                .get("match")
                .and_then(Value::as_array)
                .is_some_and(|conds| conds.iter().all(|c| eval_rule(c, &params)));
            let which = if ok { "success" } else { "failure" };
            spec.get(which).cloned().unwrap_or(Value::Null)
        }
        _ => json!({"error": "unsupported tool_impl kind"}),
    }
}

// ---------------------------------------------------------------------------
// Schema validation (judge.py schema_info / validate_against_schema)
// ---------------------------------------------------------------------------

/// Missing/extra/mistyped params against the tool's declared schema (the subset of JSON Schema
/// the source harness checks: required, per-key type, enum for strings).
pub fn validate_against_schema(args: &Value, tool: &Value) -> Vec<String> {
    let mut problems = Vec::new();
    let Value::Object(args) = args else {
        return vec!["arguments are not a JSON object".into()];
    };
    let schema = tool.get("function").and_then(|f| f.get("parameters"));
    let required: Vec<String> = schema
        .and_then(|s| s.get("required"))
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(fold).collect())
        .unwrap_or_default();
    let props = schema
        .and_then(|s| s.get("properties"))
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let fold_key = |k: &String| fold(k);
    let props_by_fold: Map<String, Value> = props
        .iter()
        .map(|(k, v)| (fold_key(k), v.clone()))
        .collect();
    for r in &required {
        if !args.keys().any(|k| fold(k) == *r) {
            problems.push(format!("missing required param '{r}'"));
        }
    }
    for (k, v) in args {
        let Some(prop) = props_by_fold.get(&fold(k)) else {
            problems.push(format!("extra param '{k}' not in schema"));
            continue;
        };
        let ty = prop.get("type").and_then(Value::as_str).unwrap_or("");
        let is_bool = v.is_boolean();
        let is_int = v.is_i64() || v.is_u64();
        let is_num = v.is_number();
        let bad = match ty {
            "integer" => is_bool || !is_int,
            "number" => is_bool || !is_num,
            "string" => !v.is_string(),
            "boolean" => !is_bool,
            _ => false,
        };
        if bad {
            problems.push(format!(
                "param '{k}' should be {ty}, got {}",
                if is_bool {
                    "bool"
                } else if is_num {
                    "number"
                } else if v.is_string() {
                    "str"
                } else {
                    "other"
                }
            ));
        }
        if let (Some(enum_), Value::String(s)) = (prop.get("enum").and_then(Value::as_array), v)
            && !enum_.iter().any(|e| e == v)
        {
            problems.push(format!("param '{k}' value '{s}' not in enum {enum_:?}"));
        }
    }
    problems
}

// ---------------------------------------------------------------------------
// Criteria / scoring (judge.py check_criteria / score / expected_match)
// ---------------------------------------------------------------------------

fn calls_of_name<'a>(calls: &'a [CallRecord], name: &str) -> Vec<&'a CallRecord> {
    calls.iter().filter(|c| c.name == name).collect()
}

/// `position`: int (0-based) or "first" — resolved WITHIN calls of the same tool name.
fn nth_call<'a>(
    calls: &'a [CallRecord],
    name: &str,
    position: Option<&Value>,
) -> Option<&'a CallRecord> {
    let mine = calls_of_name(calls, name);
    if mine.is_empty() {
        return None;
    }
    let idx = match position {
        None | Some(Value::Null) => 0,
        Some(Value::String(s)) if s == "first" => 0,
        Some(Value::String(s)) => s.parse::<usize>().ok()?,
        Some(Value::Number(n)) => n
            .as_i64()
            .filter(|i| *i >= 0)
            .and_then(|i| usize::try_from(i).ok())?,
        Some(Value::Bool(b)) => usize::from(*b),
        _ => return None,
    };
    mine.get(idx).copied()
}

/// String variants for matching a numeric value inside an answer: as-is, as an integer, as a
/// float, and with thousands separators — "1301.04" also matches "1,301.04"; an integer
/// expectation never gains a spurious ".0" (the fixed source-harness bug).
#[allow(clippy::cast_possible_truncation)] // the integer branch is guarded by |f| < 1e15
fn num_variants(v: &str) -> Vec<String> {
    let mut out = vec![v.to_string()];
    let Ok(f) = v.parse::<f64>() else { return out };
    if !f.is_finite() {
        return out;
    }
    let sign = if f < 0.0 { "-" } else { "" };
    if f == f.trunc() && f.abs() < 1e15 {
        let digits = (f.abs() as i64).to_string();
        out.push(format!("{sign}{digits}"));
        out.push(format!("{sign}{}", group_thousands(&digits)));
    } else {
        // Rust's shortest float repr matches Python's %g for the magnitudes in the suite
        out.push(format!("{f}"));
        let s = format!("{f}");
        let (sign, rest) = if let Some(r) = s.strip_prefix('-') {
            ("-", r)
        } else {
            ("", s.as_str())
        };
        if let Some((int_part, frac)) = rest.split_once('.')
            && !int_part.is_empty()
            && int_part.chars().all(|c| c.is_ascii_digit())
        {
            out.push(format!("{sign}{}.{frac}", group_thousands(int_part)));
        }
    }
    let mut seen = std::collections::HashSet::new();
    out.retain(|x| !x.is_empty() && seen.insert(x.clone()));
    out
}

/// `f"{int(i):,}"` — Python's grouping (thousands separators).
fn group_thousands(digits: &str) -> String {
    let chars: Vec<char> = digits.chars().collect();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*c);
    }
    out
}

fn answer_hits(answer_lower: &str, value: &str) -> bool {
    let v = fold(value);
    if answer_lower.contains(&v) {
        return true;
    }
    num_variants(value)
        .iter()
        .any(|x| answer_lower.contains(&x.to_lowercase()))
}

/// One executed tool call, as the source harness's `calls` entries.
#[derive(Debug, Clone)]
pub struct CallRecord {
    pub name: String,
    pub args: Option<Value>,
    pub valid: bool,
    pub problems: Vec<String>,
    /// index of this call among calls with the same name (0-based)
    pub seq_index: usize,
}

pub fn check_criteria(
    calls: &[CallRecord],
    final_answer: Option<&str>,
    criteria: &[Value],
) -> Vec<(bool, String)> {
    let mut results = Vec::new();
    for c in criteria {
        let kind = c.get("type").and_then(Value::as_str).unwrap_or("");
        let mut passed = false;
        let mut detail = String::new();
        match kind {
            "call_made" => {
                let name = c.get("name").and_then(Value::as_str).unwrap_or("");
                let n = calls_of_name(calls, name).len();
                passed = n > 0;
                detail = format!("{n} call(s)");
            }
            "params_exact" => {
                let name = c.get("name").and_then(Value::as_str).unwrap_or("");
                match nth_call(calls, name, c.get("position")) {
                    None => detail = format!("no call to {name}"),
                    Some(call) => {
                        let want = c
                            .get("params")
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default();
                        let got = call
                            .args
                            .as_ref()
                            .and_then(Value::as_object)
                            .cloned()
                            .unwrap_or_default();
                        let mut diffs = Vec::new();
                        for (k, v) in &want {
                            match got.get(&fold(k)).map(norm_value) {
                                None => diffs.push(format!("missing '{k}'")),
                                Some(g) if !py_json_eq(&g, &norm_value(v)) => {
                                    diffs.push(format!("'{k}': got {g} want {v}"));
                                }
                                _ => {}
                            }
                        }
                        passed = diffs.is_empty();
                        detail = if diffs.is_empty() {
                            "exact match".into()
                        } else {
                            diffs.join("; ")
                        };
                    }
                }
            }
            "params_exact_any" => {
                let name = c.get("name").and_then(Value::as_str).unwrap_or("");
                let mine = calls_of_name(calls, name);
                let want = c
                    .get("params")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                for (i, call) in mine.iter().enumerate() {
                    let got = call
                        .args
                        .as_ref()
                        .and_then(Value::as_object)
                        .cloned()
                        .unwrap_or_default();
                    let diffs: Vec<String> = want
                        .iter()
                        .filter(|(k, v)| {
                            got.get(&fold(k))
                                .map(norm_value)
                                .is_none_or(|g| !py_json_eq(&g, &norm_value(v)))
                        })
                        .map(|(k, v)| format!("'{k}': got {:?} want {v}", got.get(&fold(k))))
                        .collect();
                    if diffs.is_empty() {
                        passed = true;
                        detail = format!("call #{i} matched");
                        break;
                    }
                }
                if !passed {
                    detail = format!("no call matched (checked {})", mine.len());
                }
            }
            "param_contains" => {
                let name = c.get("name").and_then(Value::as_str).unwrap_or("");
                match nth_call(calls, name, c.get("position")) {
                    None => detail = format!("no call to {name}"),
                    Some(call) => {
                        let param = c.get("param").and_then(Value::as_str).unwrap_or("");
                        let pv = call
                            .args
                            .as_ref()
                            .and_then(|a| a.get(param))
                            .map(Value::to_string)
                            .unwrap_or_default();
                        let value = c.get("value").and_then(Value::as_str).unwrap_or("");
                        passed = fold(&pv).contains(&fold(value));
                        detail = format!("got {pv:?}");
                    }
                }
            }
            "param_absent" => {
                let name = c.get("name").and_then(Value::as_str).unwrap_or("");
                match nth_call(calls, name, c.get("position")) {
                    None => detail = format!("no call to {name}"),
                    Some(call) => {
                        let param = c.get("param").and_then(Value::as_str).unwrap_or("");
                        passed = call.args.as_ref().is_none_or(|a| !a.get(param).is_some());
                        detail = if passed {
                            "absent".into()
                        } else {
                            "unexpectedly present".into()
                        };
                    }
                }
            }
            "answer_contains" => {
                let ans = fold(final_answer.unwrap_or(""));
                let missing: Vec<&Value> = c
                    .get("values")
                    .and_then(Value::as_array)
                    .map(|vs| {
                        vs.iter()
                            .filter(|v| {
                                // a JSON string's text, not its quoted repr
                                let s = v.as_str().map_or_else(|| v.to_string(), String::from);
                                !answer_hits(&ans, &s)
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                passed = missing.is_empty();
                detail = if missing.is_empty() {
                    "all found".into()
                } else {
                    format!("missing {missing:?}")
                };
            }
            _ => {}
        }
        results.push((passed, detail));
    }
    results
}

/// Does this call match its expected params (judge.py expected_match)? Only calls whose name has an
/// expectation are judged; the nth same-name call maps to the nth expectation (clamped to the last).
fn expected_match(call: &CallRecord, expected: &[Value]) -> bool {
    let mine: Vec<&Value> = expected
        .iter()
        .filter(|e| e.get("name").and_then(Value::as_str) == Some(&call.name))
        .collect();
    if mine.is_empty() {
        return false;
    }
    let exp = mine[call.seq_index.min(mine.len() - 1)];
    let want = exp
        .get("params")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    if want.is_empty() {
        return true; // expected call with free-form params
    }
    let got = call
        .args
        .as_ref()
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    want.iter().all(|(k, v)| {
        got.get(&fold(k))
            .map(norm_value)
            .is_some_and(|g| py_json_eq(&g, &norm_value(v)))
    })
}

/// The score block of one (scenario, rep) run — the fields the UI shows.
pub fn score_run(
    rounds_used: usize,
    max_rounds: usize,
    http_error: Option<&str>,
    calls: &[CallRecord],
    final_answer: Option<&str>,
    criteria: &[Value],
    expected: &[Value],
) -> Map<String, Value> {
    let results = check_criteria(calls, final_answer, criteria);
    let n_crit = results.len();
    let n_pass = results.iter().filter(|(p, _)| *p).count();
    let all_pass = n_pass == n_crit && n_crit > 0;

    let attempted = calls.len();
    let valid = calls.iter().filter(|c| c.valid).count();
    let expected_names: std::collections::HashSet<&str> = expected
        .iter()
        .filter_map(|e| e.get("name").and_then(Value::as_str))
        .collect();
    let checked: Vec<&CallRecord> = calls
        .iter()
        .filter(|c| c.valid && expected_names.contains(c.name.as_str()))
        .collect();
    let param_ok = checked
        .iter()
        .filter(|c| expected_match(c, expected))
        .count();

    let opt_bool = |b: bool| Value::Bool(b);
    let mut s = Map::new();
    s.insert("criteria_passed".into(), json!(n_pass));
    s.insert("criteria_total".into(), json!(n_crit));
    s.insert("all_criteria_passed".into(), json!(all_pass));
    s.insert(
        "completed".into(),
        json!(all_pass && rounds_used <= max_rounds && http_error.is_none()),
    );
    s.insert("rounds_used".into(), json!(rounds_used));
    s.insert("budget_ok".into(), json!(rounds_used <= max_rounds));
    s.insert("tool_calls_attempted".into(), json!(attempted));
    s.insert("tool_calls_valid".into(), json!(valid));
    s.insert(
        "valid_tool_call".into(),
        if attempted > 0 {
            opt_bool(valid == attempted)
        } else {
            Value::Null
        },
    );
    s.insert(
        "param_accuracy".into(),
        if !checked.is_empty() {
            opt_bool(param_ok == checked.len())
        } else {
            Value::Null
        },
    );
    s
}

// ---------------------------------------------------------------------------
// The runner: one scenario conversation, then the whole suite
// ---------------------------------------------------------------------------

pub struct QualityDeps<'a> {
    pub http: &'a NodeHttp,
    pub host: &'a str,
    pub port: i64,
    pub model: &'a str,
    pub max_tokens: i64,
    pub on_scenario: &'a (dyn Fn(usize, &str, usize, bool) + Send + Sync),
}

/// One request round: POST /v1/chat/completions (non-streaming, tools attached).
/// Returns the assistant `message` object, or an error string.
async fn chat_round(
    http: &NodeHttp,
    host: &str,
    port: i64,
    model: &str,
    messages: &[Value],
    tools: &Value,
    max_tokens: i64,
) -> Result<Value, String> {
    let body = crate::pyjson::dumps(&json!({
        "model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
        "max_tokens": max_tokens, "temperature": 0, "stream": false,
    }));
    let headers: Vec<(String, String)> = JSON_HEADERS
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let resp = http
        .request(
            host,
            port,
            "POST",
            "/v1/chat/completions",
            HttpOpts {
                body: Some(body.into_bytes()),
                headers,
                timeout_s: Some(300.0),
                max_bytes: None,
            },
        )
        .await
        .map_err(|e| format!("node request failed: {}", e.code()))?;
    if resp.status != 200 {
        return Err(format!("HTTP {}", resp.status));
    }
    let payload: Value = serde_json::from_slice(&resp.body)
        .map_err(|_| "the engine's reply was not JSON".to_string())?;
    let choice = payload
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned()
        .unwrap_or(Value::Null);
    Ok(choice.get("message").cloned().unwrap_or_else(|| json!({})))
}

/// Run one (scenario, rep). Returns the results entry for this scenario.
/// Mirrors run.py `run_one`: parse + validate every tool call, simulate each,
/// feed results back, stop at the first round with no tool call.
pub async fn run_one(deps: &QualityDeps<'_>, sc: &Value, rep: usize) -> Map<String, Value> {
    let id = sc
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let title = sc
        .get("title")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let max_rounds = usize::try_from(
        sc.get("max_rounds")
            .and_then(Value::as_i64)
            .unwrap_or(3)
            .max(1),
    )
    .unwrap_or(3);
    let tools = sc.get("tools").cloned().unwrap_or_else(|| json!([]));
    let impls = sc.get("tool_impl").cloned().unwrap_or_else(|| json!({}));
    let tool_names: std::collections::HashSet<String> = tools
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|t| {
                    t.pointer("/function/name")
                        .and_then(Value::as_str)
                        .map(String::from)
                })
                .collect()
        })
        .unwrap_or_default();

    let mut messages = vec![
        json!({"role": "system", "content": sc.get("system")}),
        json!({"role": "user", "content": sc.get("user")}),
    ];
    let mut calls: Vec<CallRecord> = Vec::new();
    let mut per_name_count: Map<String, Value> = Map::new();
    let mut final_answer: Option<String> = None;
    let mut error: Option<String> = None;
    let mut rounds_used = 0usize;

    for rnd in 1..=max_rounds {
        rounds_used = rnd;
        let msg = match chat_round(
            deps.http,
            deps.host,
            deps.port,
            deps.model,
            &messages,
            &tools,
            deps.max_tokens,
        )
        .await
        {
            Ok(m) => m,
            Err(e) => {
                error = Some(e);
                break;
            }
        };
        let content = msg.get("content").cloned().unwrap_or(Value::Null);
        let tool_calls = msg.get("tool_calls").cloned().unwrap_or_else(|| json!([]));
        // append the assistant turn (with its tool_calls) to the conversation
        let mut asst = json!({"role": "assistant", "content": content});
        if tool_calls.as_array().is_some_and(|a| !a.is_empty()) {
            asst["tool_calls"] = tool_calls.clone();
        }
        messages.push(asst);

        let Some(tcs) = tool_calls.as_array().filter(|a| !a.is_empty()) else {
            // no tool call this round: non-empty content is the final answer
            if content.as_str().is_some_and(|s| !s.trim().is_empty()) {
                final_answer = content.as_str().map(String::from);
            }
            break;
        };

        for tc in tcs {
            let name = tc
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let raw = tc
                .pointer("/function/arguments")
                .cloned()
                .unwrap_or(Value::Null);
            let mut problems: Vec<String> = Vec::new();
            // Python: `json.loads(raw_args) if raw_args else {}` — absent or empty arguments are {}
            let raw_str = raw.as_str().unwrap_or("");
            let parsed: Option<Value> = if raw.is_null() || raw_str.is_empty() {
                Some(json!({}))
            } else {
                serde_json::from_str(raw_str).ok()
            };
            if parsed.is_none() {
                problems.push("arguments are not valid JSON".into());
            }
            if !tool_names.contains(&name) {
                problems.push(format!("unknown tool '{name}'"));
            }
            if let (Some(p), Some(tool)) = (
                &parsed,
                tools.as_array().and_then(|a| {
                    a.iter().find(|t| {
                        t.pointer("/function/name").and_then(Value::as_str) == Some(name.as_str())
                    })
                }),
            ) {
                problems.extend(validate_against_schema(p, tool));
            }
            let valid = problems.is_empty();
            let seq = per_name_count
                .get(&name)
                .and_then(Value::as_i64)
                .unwrap_or(0);
            per_name_count.insert(name.clone(), json!(seq + 1));
            calls.push(CallRecord {
                name: name.clone(),
                args: parsed.clone(),
                valid,
                problems: problems.clone(),
                seq_index: usize::try_from(seq.max(0)).unwrap_or(0),
            });

            // simulate the tool and feed the result back
            let result = simulate_tool(&name, parsed.as_ref(), &impls);
            let call_id = tc
                .get("id")
                .and_then(Value::as_str)
                .map(String::from)
                .unwrap_or_else(|| format!("call_{}", calls.len()));
            messages.push(json!({"role": "tool", "tool_call_id": call_id, "content": crate::pyjson::dumps(&result)}));
        }
    }

    let criteria = sc
        .get("success_criteria")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let expected = sc
        .get("expected_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let scores = score_run(
        rounds_used,
        max_rounds,
        error.as_deref(),
        &calls,
        final_answer.as_deref(),
        &criteria,
        &expected,
    );
    let crit_results = check_criteria(&calls, final_answer.as_deref(), &criteria);
    let criteria_json: Vec<Value> = criteria
        .iter()
        .zip(&crit_results)
        .map(|(c, (passed, detail))| {
            json!({"kind": c.get("type").and_then(Value::as_str).unwrap_or(""), "passed": passed, "detail": detail})
        })
        .collect();
    let passed = scores
        .get("completed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut entry = Map::new();
    entry.insert("id".into(), json!(id));
    entry.insert("title".into(), json!(title));
    entry.insert("rep".into(), json!(rep));
    entry.insert("passed".into(), json!(passed));
    entry.insert("max_rounds".into(), json!(max_rounds));
    entry.insert(
        "error".into(),
        error.map(Value::String).unwrap_or(Value::Null),
    );
    entry.insert(
        "final_answer".into(),
        final_answer.map(Value::String).unwrap_or(Value::Null),
    );
    entry.extend(scores);
    entry.insert("criteria".into(), json!(criteria_json));
    entry
}

/// The whole quality run: every embedded scenario (or the `only` subset) × repeats, emitting
/// progress after each. Returns the `results` object for the bench_runs row.
pub async fn run_suite(
    deps: &QualityDeps<'_>,
    repeats: usize,
    scenario_count: usize,
    only: Option<&[String]>,
) -> Map<String, Value> {
    let scenarios = scenarios();
    let list: Vec<&Value> = scenarios
        .iter()
        .filter(|(id, _)| only.is_none_or(|o| o.iter().any(|x| x == id)))
        .map(|(_, sc)| sc)
        .collect();
    let mut entries: Vec<Value> = Vec::new();
    let mut crit_pass = 0usize;
    let mut crit_total = 0usize;
    let mut completed = 0usize;
    let mut errors = 0usize;
    for sc in &list {
        for rep in 1..=repeats {
            let entry = run_one(deps, sc, rep).await;
            let passed = entry
                .get("passed")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let add = |k: &str| {
                usize::try_from(entry.get(k).and_then(Value::as_i64).unwrap_or(0).max(0))
                    .unwrap_or(0)
            };
            crit_pass += add("criteria_passed");
            crit_total += add("criteria_total");
            completed += usize::from(passed);
            errors += usize::from(entry.get("error").is_some());
            let id = entry.get("id").and_then(Value::as_str).unwrap_or("");
            (deps.on_scenario)(entries.len(), id, rep, passed);
            entries.push(Value::Object(entry));
        }
    }
    let mut out = Map::new();
    out.insert("mode".into(), json!("quality"));
    out.insert("scenario_count".into(), json!(scenario_count));
    out.insert("repeats".into(), json!(repeats));
    out.insert("runs".into(), json!(list.len() * repeats));
    out.insert(
        "score".into(),
        if crit_total > 0 {
            json!(((crit_pass as f64 / crit_total as f64) * 1000.0).round() / 10.0)
        } else {
            Value::Null
        },
    );
    out.insert("criteria_passed".into(), json!(crit_pass));
    out.insert("criteria_total".into(), json!(crit_total));
    out.insert("completed_runs".into(), json!(completed));
    out.insert("errors".into(), json!(errors));
    out.insert("scenarios".into(), json!(entries));
    out
}

// ---------------------------------------------------------------------------
// Tests: the judge quirks the source harness fixed, ported as regression tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: Value, seq: usize) -> CallRecord {
        CallRecord {
            name: name.into(),
            args: Some(args),
            valid: true,
            problems: vec![],
            seq_index: seq,
        }
    }

    #[test]
    fn test_norm_numbers_compare_across_int_and_float() {
        let a = call("t", json!({"n": 400}), 0);
        let criteria = [json!({"type": "params_exact", "name": "t", "params": {"n": 400.0}})];
        let r = check_criteria(&[a], None, &criteria);
        assert!(r[0].0, "{r:?}");
    }

    #[test]
    fn test_position_resolves_within_calls_of_the_same_name() {
        // the generator once wrote position 1 meaning "second call overall"; the judge
        // resolves it within same-name calls — the regression the source suite hit
        let calls = [
            call("lookup", json!({"q": "x"}), 0),
            call("other", json!({"q": "y"}), 0),
            call("lookup", json!({"q": "z"}), 1),
        ];
        let criteria = [
            json!({"type": "params_exact", "name": "lookup", "position": 1, "params": {"q": "z"}}),
        ];
        let r = check_criteria(&calls, None, &criteria);
        assert!(r[0].0, "{r:?}");
        // "first" is the first same-name call, not the first call overall
        let criteria = [
            json!({"type": "params_exact", "name": "lookup", "position": "first", "params": {"q": "x"}}),
        ];
        let r = check_criteria(&calls, None, &criteria);
        assert!(r[0].0, "{r:?}");
    }

    #[test]
    fn test_numeric_answers_match_thousands_separators_without_a_spurious_dot_zero() {
        // "3,250" must match an expectation of 3250 (the source harness once emitted "3,250.0")
        for answer in ["It costs 3,250.", "3250", "3,250.0", "3250.0"] {
            assert!(answer_hits(&fold(answer), "3250"), "{answer}");
        }
        assert!(answer_hits(&fold("it costs 1,301.04"), "1301.04"));
        assert!(!answer_hits(&fold("nothing here"), "3250"));
    }

    #[test]
    fn test_params_exact_ignores_extra_optional_params() {
        let calls = [call("t", json!({"a": 1, "extra": "ok"}), 0)];
        let criteria = [json!({"type": "params_exact", "name": "t", "params": {"a": 1}})];
        let r = check_criteria(&calls, None, &criteria);
        assert!(r[0].0, "{r:?}");
    }

    #[test]
    fn test_param_absent_fails_when_present() {
        let calls = [call("t", json!({"a": 1, "promo_code": "X"}), 0)];
        let criteria = [
            json!({"type": "param_absent", "name": "t", "position": "first", "param": "promo_code"}),
        ];
        let r = check_criteria(&calls, None, &criteria);
        assert!(!r[0].0, "{r:?}");
        assert_eq!(r[0].1, "unexpectedly present");
    }

    #[test]
    fn test_simulate_tool_kinds() {
        let impls = json!({
            "const": {"kind": "constant", "value": {"temp_c": 22}},
            "rule": {"kind": "rule", "match": [{"op": "ge", "param": "amount", "value": 100}],
                     "success": {"status": "ok"}, "failure": {"error": "limit"}},
        });
        assert_eq!(
            simulate_tool("const", Some(&json!({})), &impls),
            json!({"temp_c": 22})
        );
        assert_eq!(
            simulate_tool("rule", Some(&json!({"amount": 150})), &impls),
            json!({"status": "ok"})
        );
        assert_eq!(
            simulate_tool("rule", Some(&json!({"amount": 5})), &impls),
            json!({"error": "limit"})
        );
        assert_eq!(
            simulate_tool("nope", Some(&json!({})), &impls),
            json!({"error": "unknown tool 'nope'"})
        );
        assert_eq!(
            simulate_tool("const", None, &impls),
            json!({"error": "tool call arguments were not valid JSON; tool not executed"})
        );
        assert_eq!(
            simulate_tool("const", Some(&json!([1])), &impls),
            json!({"error": "tool call arguments must be a JSON object"})
        );
    }

    #[test]
    fn test_rule_ops() {
        let impls = json!({"t": {"kind": "rule", "match": [{"op": "eq", "param": "s", "value": "TOKYO"},
                                                           {"op": "lt", "param": "n", "value": 10},
                                                           {"op": "contains", "param": "note", "value": "urgent"}],
                                  "success": {"ok": true}, "failure": {"ok": false}}});
        assert_eq!(
            simulate_tool(
                "t",
                Some(&json!({"s": "tokyo", "n": 9, "note": "VERY urgent please"})),
                &impls
            ),
            json!({"ok": true})
        );
        assert_eq!(
            simulate_tool(
                "t",
                Some(&json!({"s": "osaka", "n": 9, "note": "urgent"})),
                &impls
            ),
            json!({"ok": false})
        );
    }

    #[test]
    fn test_schema_validation() {
        let tool = json!({"function": {"name": "t", "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "n": {"type": "integer"},
                           "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"], "additionalProperties": false}}});
        assert!(validate_against_schema(&json!({"city": "tokyo", "n": 2}), &tool).is_empty());
        let p = validate_against_schema(&json!({"n": 2.5, "units": "kelvin"}), &tool);
        assert!(
            p.iter()
                .any(|x| x.contains("missing required param 'city'")),
            "{p:?}"
        );
        assert!(p.iter().any(|x| x.contains("should be integer")), "{p:?}");
        assert!(p.iter().any(|x| x.contains("not in enum")), "{p:?}");
        assert!(
            validate_against_schema(&json!({"city": "tokyo", "extra": 1}), &tool)
                .iter()
                .any(|x| x.contains("extra param"))
        );
    }

    #[test]
    fn test_scenarios_embed_and_are_well_formed() {
        let scs = scenarios();
        assert_eq!(scs.len(), 12);
        for (id, sc) in &scs {
            assert!(
                !sc.get("system")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .is_empty(),
                "{id}"
            );
            assert!(
                !sc.get("user")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .is_empty(),
                "{id}"
            );
            assert!(
                sc.get("tools")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty()),
                "{id}"
            );
            assert!(
                sc.get("max_rounds")
                    .and_then(Value::as_i64)
                    .is_some_and(|r| (1..=8).contains(&r)),
                "{id}"
            );
        }
    }

    #[test]
    fn test_group_thousands_matches_python_format() {
        assert_eq!(group_thousands("3250"), "3,250");
        assert_eq!(group_thousands("130104"), "130,104");
        assert_eq!(group_thousands("22"), "22");
    }
}
