//! Append-only audit trail (port of services/audit.py; the DB rejects UPDATE/DELETE, migration 0001).

use std::sync::Arc;

use serde_json::{Map, Value};

use crate::domain::errors::AppResult;
use crate::domain::models::Page;
use crate::pyjson;
use crate::services::common::{now, redact};
use crate::services::store::{Store, paginate};

#[derive(Debug)]
pub struct AuditService {
    db: Arc<Store>,
}

impl AuditService {
    pub fn new(db: Arc<Store>) -> Self {
        Self { db }
    }

    /// One row; `params` is redacted and stored as Python's `json.dumps` text.
    pub fn record(
        &self,
        actor: &str,
        role: &str,
        method: &str,
        path: &str,
        status: u16,
        params: &Value,
    ) -> AppResult<()> {
        self.db.execute(
            "INSERT INTO audit(ts,actor,role,method,path,status,params) VALUES(?,?,?,?,?,?,?)",
            &[
                &now(),
                &actor,
                &role,
                &method,
                &path,
                &i64::from(status),
                &pyjson::dumps(&redact(params)),
            ],
        )?;
        Ok(())
    }

    pub fn list_page(&self, limit: i64, cursor: Option<&str>) -> AppResult<Page<Value>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = self.db.all(
            "SELECT * FROM audit ORDER BY id DESC LIMIT ? OFFSET ?",
            &[&(limit + 1), &off],
        )?;
        let n = usize::try_from(limit).unwrap_or(0);
        let items = rows
            .iter()
            .take(n)
            .map(|r| {
                let mut m = Map::new();
                for k in ["id", "ts", "actor", "role", "method", "path", "status"] {
                    m.insert(k.into(), r.json(k));
                }
                m.insert("params".into(), r.json_text("params"));
                Value::Object(m)
            })
            .collect();
        let next = (rows.len() > n).then(|| (off + limit).to_string());
        Ok(Page::new(items, next))
    }
}
