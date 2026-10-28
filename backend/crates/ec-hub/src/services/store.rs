//! SQLite (WAL) store with ordered SQL migrations (port of services/store.py).
//!
//! One connection behind a mutex: single node, short queries. The migrations are the Python package's own .sql
//! files, embedded at build time, applied in filename order and recorded in `schema_migrations` under the same
//! names, so a database written by the Python console opens as it is and vice versa (PORTING.md §6).
//!
//! The store's methods are synchronous and are called from async handlers. Every one of them runs through
//! [`off_worker`]: on a worker of the multi-threaded runtime the worker first hands its other tasks to another thread
//! (`block_in_place`), so a slow statement, or a caller waiting for the connection, never stalls unrelated requests.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};

use rusqlite::types::Value as SqlValue;
use rusqlite::{Connection, ToSql};
use serde_json::{Map, Value};

use crate::domain::errors::{AppError, AppResult, internal};

macro_rules! migration {
    ($name:literal) => {
        (
            $name,
            include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/migrations/", $name)),
        )
    };
}

/// (file name, SQL), in filename order. Never edit a shipped file: add 000N_*.sql in migrations/ (and here).
pub const MIGRATIONS: &[(&str, &str)] = &[
    migration!("0001_init.sql"),
    migration!("0002_container_port.sql"),
    migration!("0003_fleet.sql"),
    migration!("0004_library_and_transfers.sql"),
    migration!("0005_rollup_retention.sql"),
];

/// One result row: column name -> value, in column order (`sqlite3.Row`).
#[derive(Debug, Clone, PartialEq)]
pub struct SqlRow(pub Vec<(String, SqlValue)>);

impl SqlRow {
    pub fn get(&self, col: &str) -> Option<&SqlValue> {
        self.0.iter().find(|(c, _)| c == col).map(|(_, v)| v)
    }
    pub fn text(&self, col: &str) -> Option<String> {
        match self.get(col)? {
            SqlValue::Text(s) => Some(s.clone()),
            SqlValue::Integer(i) => Some(i.to_string()),
            SqlValue::Real(f) => Some(crate::pyjson::float_repr(*f)),
            _ => None,
        }
    }
    pub fn str_or(&self, col: &str, dflt: &str) -> String {
        self.text(col).unwrap_or_else(|| dflt.to_string())
    }
    pub fn i64(&self, col: &str) -> Option<i64> {
        match self.get(col)? {
            SqlValue::Integer(i) => Some(*i),
            #[allow(clippy::cast_possible_truncation)]
            SqlValue::Real(f) => Some(*f as i64),
            SqlValue::Text(s) => s.parse().ok(),
            _ => None,
        }
    }
    /// A REAL column as Python reads it: an INTEGER stored there comes back as an int in Python (sqlite3 returns
    /// the storage class); callers that build float fields convert, as pydantic does.
    pub fn f64(&self, col: &str) -> Option<f64> {
        match self.get(col)? {
            SqlValue::Real(f) => Some(*f),
            #[allow(clippy::cast_precision_loss)]
            SqlValue::Integer(i) => Some(*i as f64),
            SqlValue::Text(s) => s.parse().ok(),
            _ => None,
        }
    }
    pub fn bool(&self, col: &str) -> bool {
        self.i64(col).is_some_and(|i| i != 0)
    }
    /// The value as JSON (Python's sqlite3 -> json.dumps of the row value): INTEGER stays an int, REAL a float.
    pub fn json(&self, col: &str) -> Value {
        match self.get(col) {
            Some(SqlValue::Integer(i)) => Value::from(*i),
            Some(SqlValue::Real(f)) => {
                serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number)
            }
            Some(SqlValue::Text(s)) => Value::String(s.clone()),
            Some(SqlValue::Blob(b)) => Value::String(String::from_utf8_lossy(b).into_owned()),
            _ => Value::Null,
        }
    }
    /// A TEXT column holding JSON (`json.loads(row[col])`); null when NULL or unparsable.
    pub fn json_text(&self, col: &str) -> Value {
        self.text(col)
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(Value::Null)
    }
    pub fn to_map(&self) -> Map<String, Value> {
        self.0
            .iter()
            .map(|(c, _)| (c.clone(), self.json(c)))
            .collect()
    }
}

pub struct Store {
    conn: Mutex<Connection>,
}

impl std::fmt::Debug for Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Store")
    }
}

/// Run `f`, which may block (SQLite, or the connection's lock), without stalling the async runtime. On a worker of a
/// multi-threaded tokio runtime it runs under `block_in_place`: the worker's queued tasks and its driver move to
/// another thread first, so other requests go on while `f` waits. Anywhere else it simply runs: on a current-thread
/// runtime (where `block_in_place` panics; tests use one, and there is no other worker to free), on a blocking-pool
/// thread, in the runtime's own `block_on` thread, or outside any runtime.
pub fn off_worker<T>(f: impl FnOnce() -> T) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(h) if h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(f)
        }
        _ => f(),
    }
}

fn db_err(e: &rusqlite::Error) -> AppError {
    tracing::error!(target: "engine_console.store", "database error: {}", e);
    AppError::Problem(internal("unexpected server error"))
}

impl Store {
    /// Open (creating the parent 0700 and the file 0600), set WAL / foreign keys / synchronous=NORMAL, migrate.
    pub fn open(path: &Path) -> Result<Self, String> {
        let memory = path.as_os_str() == ":memory:";
        if !memory && let Some(parent) = path.parent() {
            create_dir_0700(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        if !memory {
            // SQLite opens these with a blocking open(2): a FIFO planted at any of them would hang start-up, and a
            // directory or device is never a database. Refuse them first (Python: sqlite3 would hang or fail).
            for p in [
                path.to_path_buf(),
                PathBuf::from(format!("{}-wal", path.display())),
                PathBuf::from(format!("{}-shm", path.display())),
            ] {
                crate::fsutil::require_regular_if_present(&p)
                    .map_err(|e| format!("{}: {e}", p.display()))?;
            }
        }
        let conn = Connection::open(path).map_err(|e| e.to_string())?;
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| e.to_string())?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| e.to_string())?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| e.to_string())?;
        if !memory {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| e.to_string())?;
        }
        let s = Self {
            conn: Mutex::new(conn),
        };
        s.migrate()?;
        Ok(s)
    }

    /// The connection. A panic while it was held poisons the lock; Python has no such state, so the guard is taken
    /// back, a transaction the panic left open is rolled back, and the store goes on serving.
    fn conn(&self) -> MutexGuard<'_, Connection> {
        match self.conn.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                let g = poisoned.into_inner();
                if !g.is_autocommit() {
                    let _ = g.execute_batch("ROLLBACK");
                }
                self.conn.clear_poison();
                g
            }
        }
    }

    pub fn memory() -> Result<Self, String> {
        Self::open(Path::new(":memory:"))
    }

    /// Apply the migrations not yet recorded; returns the names applied.
    pub fn migrate(&self) -> Result<Vec<String>, String> {
        off_worker(|| self.migrate_now())
    }

    fn migrate_now(&self) -> Result<Vec<String>, String> {
        let conn = self.conn();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)",
            [],
        )
        .map_err(|e| e.to_string())?;
        let done: Vec<String> = {
            let mut st = conn
                .prepare("SELECT name FROM schema_migrations")
                .map_err(|e| e.to_string())?;
            st.query_map([], |r| r.get::<_, String>(0))
                .map_err(|e| e.to_string())?
                .filter_map(Result::ok)
                .collect()
        };
        let mut applied = Vec::new();
        for (name, script) in MIGRATIONS {
            if done.iter().any(|d| d == name) {
                continue;
            }
            conn.execute_batch("BEGIN").map_err(|e| e.to_string())?;
            let res = (|| -> rusqlite::Result<()> {
                for stmt in split_sql(script) {
                    conn.execute_batch(&stmt)?;
                }
                conn.execute(
                    "INSERT INTO schema_migrations VALUES (?, strftime('%s','now'))",
                    [name],
                )?;
                Ok(())
            })();
            match res {
                Ok(()) => conn.execute_batch("COMMIT").map_err(|e| e.to_string())?,
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    return Err(format!("migration {name} failed: {e}"));
                }
            }
            applied.push((*name).to_string());
        }
        Ok(applied)
    }

    /// Run a statement; returns the number of rows changed (`cursor.rowcount`).
    pub fn execute(&self, sql: &str, params: &[&dyn ToSql]) -> AppResult<usize> {
        off_worker(|| {
            let conn = self.conn();
            conn.execute(sql, params).map_err(|e| db_err(&e))
        })
    }

    /// The same statement for every row of parameters (one transaction, like executemany in autocommit mode).
    pub fn executemany(&self, sql: &str, rows: &[Vec<SqlValue>]) -> AppResult<()> {
        off_worker(|| {
            let conn = self.conn();
            let mut st = conn.prepare_cached(sql).map_err(|e| db_err(&e))?;
            for r in rows {
                st.execute(rusqlite::params_from_iter(r.iter()))
                    .map_err(|e| db_err(&e))?;
            }
            Ok(())
        })
    }

    pub fn one(&self, sql: &str, params: &[&dyn ToSql]) -> AppResult<Option<SqlRow>> {
        Ok(self.all(sql, params)?.into_iter().next())
    }

    pub fn all(&self, sql: &str, params: &[&dyn ToSql]) -> AppResult<Vec<SqlRow>> {
        off_worker(|| self.all_now(sql, params))
    }

    fn all_now(&self, sql: &str, params: &[&dyn ToSql]) -> AppResult<Vec<SqlRow>> {
        let conn = self.conn();
        let mut st = conn.prepare_cached(sql).map_err(|e| db_err(&e))?;
        let cols: Vec<String> = st.column_names().iter().map(|c| (*c).to_string()).collect();
        let rows = st
            .query_map(params, |r| {
                let mut out = Vec::with_capacity(cols.len());
                for (i, c) in cols.iter().enumerate() {
                    out.push((c.clone(), r.get::<_, SqlValue>(i)?));
                }
                Ok(SqlRow(out))
            })
            .map_err(|e| db_err(&e))?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| db_err(&e))
    }

    /// Run `f` with the connection inside one transaction (for multi-statement updates).
    pub fn transaction<T>(
        &self,
        f: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> AppResult<T> {
        off_worker(|| {
            let conn = self.conn();
            conn.execute_batch("BEGIN").map_err(|e| db_err(&e))?;
            match f(&conn) {
                Ok(v) => {
                    conn.execute_batch("COMMIT").map_err(|e| db_err(&e))?;
                    Ok(v)
                }
                Err(e) => {
                    let _ = conn.execute_batch("ROLLBACK");
                    Err(db_err(&e))
                }
            }
        })
    }
}

/// `mkdir(parents=True, exist_ok=True, mode=0o700)`: new directories 0700 (the umask applies, as in Python).
pub fn create_dir_0700(p: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(p)
}

/// `_split_sql`: split on ';' at statement ends, keeping CREATE TRIGGER ... BEGIN ... END; bodies intact; comment
/// and blank lines dropped.
pub fn split_sql(script: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf: Vec<&str> = Vec::new();
    let mut in_trigger = false;
    for line in script.lines() {
        let s = line.trim();
        if s.is_empty() || s.starts_with("--") {
            continue;
        }
        buf.push(line);
        if s.to_ascii_uppercase().starts_with("CREATE TRIGGER") {
            in_trigger = true;
        }
        if in_trigger {
            if s.to_ascii_uppercase()
                .trim_end_matches(';')
                .ends_with("END")
            {
                out.push(buf.join("\n"));
                buf.clear();
                in_trigger = false;
            }
        } else if s.ends_with(';') {
            out.push(buf.join("\n"));
            buf.clear();
        }
    }
    if !buf.is_empty() {
        out.push(buf.join("\n"));
    }
    out
}

/// Opaque cursor == integer offset. Returns (limit, offset); callers fetch limit+1 to detect a next page.
pub fn paginate(limit: i64, cursor: Option<&str>) -> (i64, i64) {
    let offset = cursor
        .filter(|c| !c.is_empty())
        .map_or(0, |c| py_int(c).map_or(0, |v| v.max(0)));
    (limit.clamp(1, 500), offset)
}

/// Python `int(text)`: optional sign, digits (underscores between digits allowed), surrounding whitespace.
pub fn py_int(text: &str) -> Option<i64> {
    let t = text.trim();
    let (neg, digits) = match t.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
    {
        return None;
    }
    if !digits.chars().all(|c| c.is_ascii_digit() || c == '_') {
        return None;
    }
    let v: i64 = digits.replace('_', "").parse().ok()?;
    Some(if neg { -v } else { v })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Port of the store rows of tests/core/test_units.py (names mirror the Python tests).
    #[test]
    fn test_migrations_idempotent_and_audit_append_only() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("x.db");
        let s = Store::open(&p).unwrap();
        assert!(s.migrate().unwrap().is_empty());
        let names: Vec<String> = s
            .all("SELECT name FROM schema_migrations ORDER BY name", &[])
            .unwrap()
            .iter()
            .filter_map(|r| r.text("name"))
            .collect();
        assert_eq!(names[0], "0001_init.sql");
        assert!(names.iter().any(|n| n == "0003_fleet.sql"));
        assert_eq!(names.len(), MIGRATIONS.len());
        let mode = std::fs::metadata(&p)
            .map(|m| std::os::unix::fs::PermissionsExt::mode(&m.permissions()))
            .unwrap();
        assert_eq!(mode & 0o777, 0o600);
        s.execute(
            "INSERT INTO audit(ts,actor,role,method,path,status,params) VALUES(1,'a','admin','POST','/x',200,'{}')",
            &[],
        )
        .unwrap();
        assert!(s.execute("DELETE FROM audit", &[]).is_err());
        assert!(s.execute("UPDATE audit SET actor='b'", &[]).is_err());
        // the refusal is the trigger's own (the Store maps every database error to a generic 500 and logs it)
        let raw = Connection::open(&p).unwrap();
        for sql in ["DELETE FROM audit", "UPDATE audit SET actor='b'"] {
            let e = raw.execute(sql, []).unwrap_err().to_string();
            assert!(e.contains("append-only"), "{sql}: {e}");
        }
    }

    /// A panic inside a transaction poisons the connection's lock and leaves BEGIN open; the next caller gets the
    /// connection back with that transaction rolled back (hardening finding 3), and a FIFO at the database path is
    /// refused at once instead of hanging SQLite's open (finding 4).
    #[test]
    fn test_the_store_survives_a_panic_inside_a_transaction() {
        let dir = tempfile::tempdir().unwrap();
        let s = std::sync::Arc::new(Store::open(&dir.path().join("x.db")).unwrap());
        s.execute("CREATE TABLE t (v INTEGER)", &[]).unwrap();
        let s2 = std::sync::Arc::clone(&s);
        let _ = std::thread::spawn(move || {
            let _: AppResult<()> = s2.transaction(|c| {
                c.execute("INSERT INTO t VALUES (1)", [])?;
                #[allow(clippy::panic)]
                {
                    panic!("a bug mid-transaction");
                }
            });
        })
        .join();
        assert!(s.conn.is_poisoned());
        s.execute("INSERT INTO t VALUES (2)", &[]).unwrap();
        let vals: Vec<i64> = s
            .all("SELECT v FROM t ORDER BY v", &[])
            .unwrap()
            .iter()
            .filter_map(|r| r.i64("v"))
            .collect();
        assert_eq!(vals, [2], "the half-done transaction was rolled back");
        assert!(!s.conn.is_poisoned());
        let fifo = dir.path().join("fifo.db");
        nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
        let t = std::time::Instant::now();
        let e = Store::open(&fifo).err().unwrap_or_default();
        assert!(e.contains("not a regular file"), "{e}");
        let wal = dir.path().join("y.db-wal");
        nix::unistd::mkfifo(&wal, nix::sys::stat::Mode::from_bits_truncate(0o600)).unwrap();
        assert!(Store::open(&dir.path().join("y.db")).is_err());
        assert!(t.elapsed() < std::time::Duration::from_secs(2));
    }

    /// Audit F7: a store call that waits (for SQLite, or for the connection another caller holds) must not stall the
    /// runtime. With both workers of a two-worker runtime inside the store (one holding the connection, one waiting
    /// for it), a task that never touches the store still runs at once. Before the fix both workers were blocked on
    /// the std lock and that task waited for the holder (here: until the test released it).
    #[test]
    fn test_a_slow_store_call_does_not_stall_unrelated_tasks() {
        use std::sync::mpsc::channel;
        use std::time::Duration;
        let s = std::sync::Arc::new(Store::memory().unwrap());
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        let (held_tx, held_rx) = channel::<()>();
        let (release_tx, release_rx) = channel::<()>();
        let (waiting_tx, waiting_rx) = channel::<()>();
        let (ran_tx, ran_rx) = channel::<()>();
        // a slow statement holds the connection on one worker ...
        let s1 = std::sync::Arc::clone(&s);
        rt.spawn(async move {
            let _ = s1.transaction(|_c| {
                let _ = held_tx.send(());
                let _ = release_rx.recv_timeout(Duration::from_secs(30));
                Ok(())
            });
        });
        held_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        // ... a second caller waits for it on the other (no await point between the signal and the store call) ...
        let s2 = std::sync::Arc::clone(&s);
        rt.spawn(async move {
            let _ = waiting_tx.send(());
            let _ = s2.one("SELECT 1", &[]);
        });
        waiting_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        // ... and a task that never touches the store still runs
        rt.spawn(async move {
            let _ = ran_tx.send(());
        });
        let ran = ran_rx.recv_timeout(Duration::from_secs(3)).is_ok();
        let _ = release_tx.send(());
        rt.shutdown_timeout(Duration::from_secs(10));
        assert!(
            ran,
            "a task that does not use the store waited for a slow store call"
        );
    }

    /// The hand-off is only taken on a multi-threaded runtime's worker: on a current-thread runtime (where
    /// `block_in_place` would panic), on a blocking-pool thread and outside any runtime the call simply runs.
    #[test]
    fn test_store_calls_work_on_every_kind_of_thread() {
        let s = std::sync::Arc::new(Store::memory().unwrap());
        s.execute("CREATE TABLE t (v INTEGER)", &[]).unwrap();
        let ins = |s: &Store| s.execute("INSERT INTO t VALUES (1)", &[]).unwrap();
        ins(&s); // no runtime
        let ct = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        ct.block_on(async { ins(&s) });
        let s2 = std::sync::Arc::clone(&s);
        ct.block_on(async move { tokio::task::spawn_blocking(move || ins(&s2)).await })
            .unwrap();
        let mt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .build()
            .unwrap();
        mt.block_on(async { ins(&s) }); // the runtime's own block_on thread
        let s3 = std::sync::Arc::clone(&s);
        mt.block_on(async move { tokio::spawn(async move { ins(&s3) }).await })
            .unwrap(); // a worker
        let s4 = std::sync::Arc::clone(&s);
        mt.block_on(async move { tokio::task::spawn_blocking(move || ins(&s4)).await })
            .unwrap();
        let n = s.one("SELECT COUNT(*) AS n FROM t", &[]).unwrap().unwrap();
        assert_eq!(n.i64("n"), Some(6));
    }

    #[test]
    fn test_paginate_helper() {
        assert_eq!(paginate(10, None), (10, 0));
        assert_eq!(paginate(10, Some("20")), (10, 20));
        assert_eq!(paginate(10, Some("junk")), (10, 0));
        assert_eq!(paginate(9999, Some("-5")), (500, 0));
        assert_eq!(paginate(0, Some("7")), (1, 7));
    }

    const PRE_FLEET_ROW: &str = "INSERT INTO instances(id,name,engine,repo_id,params,gpu_ids,gpu_uuids,container_name,image,state,created_at) ";

    #[test]
    fn test_the_fleet_migration_adds_host_model_and_image_columns_with_local_as_the_default() {
        let dir = tempfile::tempdir().unwrap();
        let s = Store::open(&dir.path().join("x.db")).unwrap();
        let cols = s.all("PRAGMA table_info(instances)", &[]).unwrap();
        let col = |name: &str| {
            cols.iter()
                .find(|r| r.text("name").as_deref() == Some(name))
                .unwrap_or_else(|| panic!("no column {name}"))
        };
        for name in ["host_id", "model", "image_req", "container_port"] {
            col(name);
        }
        assert_eq!(col("host_id").i64("notnull"), Some(1));
        assert_eq!(
            col("host_id").text("dflt_value").as_deref(),
            Some("'local'")
        );
        assert_eq!(col("model").i64("notnull"), Some(0));
        // a pre-fleet row (no host_id given) lands on the console's own machine and reads back as a Hugging Face model
        s.execute(
            &format!("{PRE_FLEET_ROW}VALUES('i','n','fake','Qwen/Qwen3-32B','{{}}','[0]','[]','c','img','ready',1)"),
            &[],
        )
        .unwrap();
        let row = s
            .one("SELECT host_id, model FROM instances WHERE id='i'", &[])
            .unwrap()
            .unwrap();
        assert_eq!(row.text("host_id").as_deref(), Some("local"));
        assert_eq!(row.get("model"), Some(&SqlValue::Null));
    }

    #[test]
    fn test_an_upgrade_from_the_v1_schema_keeps_existing_instances() {
        // Apply only the v1 migrations, add a row, then let the Store apply the rest: nothing is lost, the row gains
        // host 'local'.
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("old.db");
        {
            let db = Connection::open(&p).unwrap();
            db.execute(
                "CREATE TABLE schema_migrations (name TEXT PRIMARY KEY, applied_at REAL)",
                [],
            )
            .unwrap();
            for (name, script) in MIGRATIONS {
                if ["0001_init.sql", "0002_container_port.sql"].contains(name) {
                    db.execute_batch(script).unwrap();
                    db.execute("INSERT INTO schema_migrations VALUES (?, 1)", [name])
                        .unwrap();
                }
            }
            db.execute(
                &format!("{PRE_FLEET_ROW}VALUES('old','n','vllm','a/b','{{}}','[0]','[]','ec-x','img:1','ready',1)"),
                [],
            )
            .unwrap();
        }
        let s = Store::open(&p).unwrap();
        let row = s
            .one(
                "SELECT host_id, model, repo_id FROM instances WHERE id='old'",
                &[],
            )
            .unwrap()
            .unwrap();
        assert_eq!(row.text("host_id").as_deref(), Some("local"));
        assert_eq!(row.get("model"), Some(&SqlValue::Null));
        assert_eq!(row.text("repo_id").as_deref(), Some("a/b"));
    }

    #[test]
    fn test_split_sql_keeps_trigger_bodies() {
        let parts = split_sql(
            "-- c\nCREATE TABLE a (x);\nCREATE TRIGGER t BEFORE UPDATE ON a BEGIN SELECT 1; END;\n",
        );
        assert_eq!(parts.len(), 2);
        assert!(parts[1].contains("SELECT 1; END;"));
    }
}
