//! Python's text formats, where the bytes matter (DB columns, the audit log, SSE frames, ETags, Prometheus text).
//!
//! * [`float_repr`]: `repr(float)` (shortest round-trip digits; exponent form below 1e-4 and from 1e16 up).
//! * [`dumps`]: `json.dumps(v)` with the default separators `", "` / `": "` and `ensure_ascii=True` (what Python
//!   writes into SQLite columns and the audit table), and [`dumps_compact`]: `separators=(",", ":")`, as used for SSE
//!   frames (`ensure_ascii` stays True there too).

use serde_json::Value;

/// `repr(x)` of a Python float.
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // shortest round-trip digits and decimal exponent, from Rust's own `{:e}` formatting (also shortest round-trip)
    let sci = format!("{x:e}"); // e.g. "-1.2345e-5", "1e16"
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let neg = mant.starts_with('-');
    let digits: String = mant.chars().filter(char::is_ascii_digit).collect();
    let sign = if neg { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        // d.ddd e±XX (at least two exponent digits)
        let (first, rest) = digits.split_at(1);
        let m = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        let es = if exp < 0 { "-" } else { "+" };
        return format!("{sign}{m}e{es}{:02}", exp.abs());
    }
    let n = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let body = if exp >= 0 {
        let int_len = exp + 1;
        if n <= int_len {
            let zeros = usize::try_from(int_len - n).unwrap_or(0);
            format!("{digits}{}.0", "0".repeat(zeros))
        } else {
            let k = usize::try_from(int_len).unwrap_or(0);
            format!("{}.{}", &digits[..k], &digits[k..])
        }
    } else {
        let zeros = usize::try_from(-exp - 1).unwrap_or(0);
        format!("0.{}{digits}", "0".repeat(zeros))
    };
    format!("{sign}{body}")
}

/// prometheus_client's `floatToGoString`.
pub fn go_float(d: f64) -> String {
    if d.is_infinite() {
        return if d > 0.0 {
            "+Inf".into()
        } else {
            "-Inf".into()
        };
    }
    if d.is_nan() {
        return "NaN".into();
    }
    let s = float_repr(d);
    if let Some(dot) = s.find('.')
        && d > 0.0
        && dot > 6
    {
        let mantissa = format!("{}.{}{}", &s[..1], &s[1..dot], &s[dot + 1..]);
        let mantissa = mantissa.trim_end_matches(['0', '.']);
        return format!("{mantissa}e+0{}", dot - 1);
    }
    s
}

/// `json.dumps(v)`: separators `", "` and `": "`, ensure_ascii.
pub fn dumps(v: &Value) -> String {
    let mut out = String::new();
    write(&mut out, v, ", ", ": ");
    out
}

/// `json.dumps(v, separators=(",", ":"))`, ensure_ascii.
pub fn dumps_compact(v: &Value) -> String {
    let mut out = String::new();
    write(&mut out, v, ",", ":");
    out
}

fn write(out: &mut String, v: &Value, item: &str, key: &str) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_f64() {
                out.push_str(&float_repr(n.as_f64().unwrap_or(0.0)));
            } else {
                out.push_str(&n.to_string());
            }
        }
        Value::String(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write(out, x, item, key);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write_str(out, k);
                out.push_str(key);
                write(out, x, item, key);
            }
            out.push('}');
        }
    }
}

/// A JSON string with Python's ensure_ascii escaping.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn float_repr_matches_python() {
        for (x, want) in [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123_456_789.0, "123456789.0"),
            (1_793_139_538.740_461_3, "1793139538.7404613"),
            (2.5e-7, "2.5e-07"),
            (-0.5, "-0.5"),
            (1.5e300, "1.5e+300"),
            (9_999_999_999_999_998.0, "9999999999999998.0"),
        ] {
            assert_eq!(float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn go_float_matches_prometheus_client() {
        assert_eq!(go_float(1.0), "1.0");
        assert_eq!(go_float(0.0123), "0.0123");
        assert_eq!(go_float(1_793_139_538.740_461_3), "1.7931395387404613e+09");
        assert_eq!(go_float(f64::INFINITY), "+Inf");
        assert_eq!(go_float(1_000_000.0), "1e+06");
    }

    #[test]
    fn dumps_matches_python() {
        let v = json!({"a": [1, 2.0, "\u{e9}\n"], "b": null, "c": {"d": true}});
        assert_eq!(
            dumps(&v),
            r#"{"a": [1, 2.0, "\u00e9\n"], "b": null, "c": {"d": true}}"#
        );
        assert_eq!(
            dumps_compact(&v),
            r#"{"a":[1,2.0,"\u00e9\n"],"b":null,"c":{"d":true}}"#
        );
        assert_eq!(dumps(&json!("\u{1f600}")), r#""\ud83d\ude00""#);
    }
}
