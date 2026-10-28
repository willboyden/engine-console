//! Python formatting of JSON-decoded values, so messages that embed `str(x)`, `repr(x)` or `f"{x:g}"` read the same
//! from a Rust peer as from the Python one (e.g. the node's `unknown_op` text, `f"... {str(op)[:64]!r}"`).

use serde_json::Value;

/// `repr(s)` of a Python str.
pub fn repr_str(s: &str) -> String {
    crate::sanitize::py_repr(s)
}

/// `repr(f)` of a Python float: shortest round-trip digits, fixed notation for 1e-4 <= |f| < 1e16, else
/// scientific with a signed two-digit exponent (`1e+16`, `1.5e-05`), and always a `.0` on integral fixed values.
pub fn repr_float(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if f == 0.0 {
        return if f.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Rust's `{:e}` prints the shortest digits that round-trip, as `d.ddde[-]x`.
    let sci = format!("{f:e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };
    if (-4..16).contains(&exp) {
        // fixed notation
        let n = i32::try_from(digits.len()).unwrap_or(1);
        let point = exp + 1; // digits before the decimal point
        let s = if point <= 0 {
            format!(
                "0.{}{}",
                "0".repeat(usize::try_from(-point).unwrap_or(0)),
                digits
            )
        } else if point >= n {
            format!(
                "{digits}{}.0",
                "0".repeat(usize::try_from(point - n).unwrap_or(0))
            )
        } else {
            let p = usize::try_from(point).unwrap_or(0);
            format!("{}.{}", &digits[..p], &digits[p..])
        };
        format!("{sign}{s}")
    } else {
        let m = if digits.len() > 1 {
            format!("{}.{}", &digits[..1], &digits[1..])
        } else {
            digits
        };
        let es = if exp < 0 { "-" } else { "+" };
        format!("{sign}{m}e{es}{:02}", exp.abs())
    }
}

/// `repr(v)` of the Python object `json.loads` would build for `v`.
pub fn repr_value(v: &Value) -> String {
    match v {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(n) => {
            if n.is_f64() {
                repr_float(n.as_f64().unwrap_or(0.0))
            } else {
                n.to_string()
            }
        }
        Value::String(s) => repr_str(s),
        Value::Array(a) => {
            let parts: Vec<String> = a.iter().map(repr_value).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::Object(o) => {
            let parts: Vec<String> = o
                .iter()
                .map(|(k, x)| format!("{}: {}", repr_str(k), repr_value(x)))
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// `str(v)` of the Python object `json.loads` would build for `v` (a str is itself; everything else is its repr).
pub fn str_value(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => repr_value(v),
    }
}

/// `s[:n]` (characters).
pub fn take_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// `f"{x:g}"`: 6 significant digits, trailing zeros removed, scientific below 1e-4 or from 1e6.
pub fn fmt_g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    if !x.is_finite() {
        return repr_float(x);
    }
    let sci = format!("{x:.5e}");
    let (mant, exp) = sci.split_once('e').unwrap_or((sci.as_str(), "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    if (-4..6).contains(&exp) {
        let decimals = usize::try_from((5 - exp).max(0)).unwrap_or(0);
        let s = format!("{x:.decimals$}");
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s
        }
    } else {
        let m = mant.trim_end_matches('0').trim_end_matches('.');
        let es = if exp < 0 { "-" } else { "+" };
        format!("{m}e{es}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (f, want) in [
            (1.0, "1.0"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (123_456_789.0, "123456789.0"),
            (1_761_580_000.123_456_7, "1761580000.1234567"),
            (-2.5, "-2.5"),
            (1e22, "1e+22"),
            (9_999_999_999_999_998.0, "9999999999999998.0"),
        ] {
            assert_eq!(repr_float(f), want, "{f}");
        }
    }

    #[test]
    fn value_repr_matches_python() {
        assert_eq!(
            repr_value(&json!([1, "a", null, true, 1.5])),
            "[1, 'a', None, True, 1.5]"
        );
        assert_eq!(repr_value(&json!({"k": "it's"})), "{'k': \"it's\"}");
        assert_eq!(str_value(&json!("plain")), "plain");
    }

    #[test]
    fn g_format_matches_python() {
        for (f, want) in [
            (30.0, "30"),
            (0.6, "0.6"),
            (300.0, "300"),
            (1e-5, "1e-05"),
            (1234567.0, "1.23457e+06"),
        ] {
            assert_eq!(fmt_g(f), want, "{f}");
        }
    }
}
