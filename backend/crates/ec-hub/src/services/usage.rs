//! Per-model / per-day / per-hour usage analytics from usage_events (port of services/usage.py).

use std::sync::Arc;

use crate::domain::errors::{AppResult, bad_request};
use crate::domain::models::UsageRow;
use crate::services::common::now;
use crate::services::store::Store;

/// group_by -> SQL expression (fixed table: never user input in SQL). Sorted names for the error message.
const GROUPS: &[(&str, &str)] = &[
    ("day", "strftime('%Y-%m-%d', ts, 'unixepoch')"),
    ("hour", "strftime('%Y-%m-%dT%H:00Z', ts, 'unixepoch')"),
    ("model", "model"),
];

#[derive(Debug)]
pub struct UsageService {
    db: Arc<Store>,
}

impl UsageService {
    pub fn new(db: Arc<Store>) -> Self {
        Self { db }
    }

    pub fn record(
        &self,
        model: &str,
        instance_id: Option<&str>,
        prompt_tokens: i64,
        completion_tokens: i64,
        latency_ms: f64,
        source: &str,
    ) -> AppResult<()> {
        self.db.execute(
            "INSERT INTO usage_events(ts,model,instance_id,prompt_tokens,completion_tokens,latency_ms,source) \
             VALUES(?,?,?,?,?,?,?)",
            &[&now(), &model, &instance_id, &prompt_tokens, &completion_tokens, &latency_ms, &source],
        )?;
        Ok(())
    }

    pub fn query(
        &self,
        group_by: &str,
        since: Option<f64>,
        until: Option<f64>,
    ) -> AppResult<Vec<UsageRow>> {
        let Some((_, expr)) = GROUPS.iter().find(|(g, _)| *g == group_by) else {
            let names: Vec<String> = GROUPS.iter().map(|(g, _)| format!("'{g}'")).collect();
            return Err(
                bad_request(format!("group_by must be one of [{}]", names.join(", ")))
                    .code("invalid_group_by")
                    .into(),
            );
        };
        // `since or 0.0` / `until or now() + 1`: a 0.0 bound counts as unset, as in Python
        let since = since.filter(|s| *s != 0.0).unwrap_or(0.0);
        let until = until.filter(|u| *u != 0.0).unwrap_or_else(|| now() + 1.0);
        let rows = self.db.all(
            &format!(
                "SELECT {expr} AS k, COUNT(*) AS n, SUM(prompt_tokens) AS p, SUM(completion_tokens) AS c, \
                 AVG(latency_ms) AS l FROM usage_events WHERE ts >= ? AND ts <= ? GROUP BY k ORDER BY k"
            ),
            &[&since, &until],
        )?;
        Ok(rows
            .iter()
            .map(|r| UsageRow {
                key: r.str_or("k", ""),
                requests: r.i64("n").unwrap_or(0),
                prompt_tokens: r.i64("p").unwrap_or(0),
                completion_tokens: r.i64("c").unwrap_or(0),
                avg_latency_ms: py_round2(r.f64("l").unwrap_or(0.0)),
            })
            .collect())
    }

    /// CSV with a header; keys starting with = + - @ are prefixed with ' (spreadsheet formula injection).
    pub fn export_csv(&self, group_by: &str) -> AppResult<String> {
        let mut out =
            String::from("key,requests,prompt_tokens,completion_tokens,avg_latency_ms\r\n");
        for r in self.query(group_by, None, None)? {
            let key = if r.key.starts_with(['=', '+', '-', '@']) {
                format!("'{}", r.key)
            } else {
                r.key.clone()
            };
            out.push_str(&format!(
                "{},{},{},{},{}\r\n",
                csv_field(&key),
                r.requests,
                r.prompt_tokens,
                r.completion_tokens,
                crate::pyjson::float_repr(r.avg_latency_ms)
            ));
        }
        Ok(out)
    }
}

/// Python's `round(x, 2)` (correctly rounded on the decimal value, ties to even).
pub fn py_round2(x: f64) -> f64 {
    format!("{x:.2}").parse().unwrap_or(x)
}

/// csv.writer's minimal quoting: quote when the field holds the delimiter, a quote, \r or \n; double the quotes.
pub fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_query_csv_and_bad_group() {
        let s = UsageService::new(Arc::new(Store::memory().unwrap()));
        s.record("=evil", Some("i1"), 10, 5, 100.0, "chat").unwrap();
        s.record("=evil", None, 1, 1, 50.0, "arena").unwrap();
        let rows = s.query("model", None, None).unwrap();
        assert_eq!(
            (
                rows[0].requests,
                rows[0].prompt_tokens,
                rows[0].avg_latency_ms
            ),
            (2, 11, 75.0)
        );
        let csv = s.export_csv("model").unwrap();
        assert_eq!(
            csv,
            "key,requests,prompt_tokens,completion_tokens,avg_latency_ms\r\n'=evil,2,11,6,75.0\r\n"
        );
        let e = s.query("week", None, None).unwrap_err();
        assert_eq!(
            e.problem().unwrap().detail,
            "group_by must be one of ['day', 'hour', 'model']"
        );
    }
}
