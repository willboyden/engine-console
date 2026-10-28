//! Python: backend/src/engine_console/services/profiles.py
//! (tests/core/test_api_smoke.py profile parts, tests/data/test_routers.py).
//!
//! Launch profiles in SQLite: CRUD, diff, YAML export/import (PyYAML `safe_dump`/`safe_load` compatible text; the
//! export bytes are a contract: users keep these files).
//!
//! YAML is the hub's one PyYAML module (`crate::pyyaml`, PORTING.md §A9): the export is
//! `pyyaml::safe_dump(.., sort_keys=false)`, the import `pyyaml::safe_load` (PyYAML's reader, scanner, parser,
//! composer, resolver and SafeConstructor ported function by function, copied from the Nemo Console port) with the
//! Python objects it builds (`pyyaml::pyobj`). The import then does what `ProfileIn(**{k: doc[k] ...})` and `create`
//! do with those objects.

use std::sync::Arc;

use rusqlite::ErrorCode;
use serde_json::{Map, Value, json};

use crate::adapters::AdapterRegistry;
use crate::api::schemas::PROFILE_IN;
use crate::api::validate::{Body, Params, T};
use crate::domain::errors::{
    AppError, AppResult, Obj, bad_request, conflict, internal, not_found, unprocessable,
};
use crate::domain::models::{Page, Profile, ProfileDiff, ProfileIn};
use crate::services::common::{new_id, now};
use crate::services::secrets_store::{MARKER, split_secrets};
use crate::services::store::{SqlRow, Store, paginate};

use crate::pyyaml::pyobj::{Doc, Obj as PyObj, Py, PyInt};

/// The fields `import_yaml` takes from the document, in Python's order (never `preset`).
const IMPORT_KEYS: [&str; 5] = ["name", "engine", "repo_id", "description", "params"];

/// The most values one imported document may expand to through aliases (Python shares aliased objects, but
/// `json.dumps` expands them without bound: a MemoryError or a hang there, a 500 here).
const MAX_IMPORT_VALUES: usize = 1_000_000;

pub struct ProfileService {
    db: Arc<Store>,
    adapters: AdapterRegistry,
}

impl std::fmt::Debug for ProfileService {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ProfileService")
    }
}

fn server_error() -> AppError {
    AppError::Problem(internal("unexpected server error"))
}

/// Python raising an exception nothing in the profile code catches: the request ends in FastAPI's catch-all 500
/// (app.rs answers a raise the same way, PORTING.md §A5). The profile service only serves HTTP routes.
fn uncaught(kind: &str, msg: &str) -> ! {
    crate::adapters::pycompat::raise(kind, msg)
}

/// `sqlite3.IntegrityError`: a constraint failed (UNIQUE name, NOT NULL, ...).
fn integrity(e: &rusqlite::Error) -> bool {
    matches!(e, rusqlite::Error::SqliteFailure(f, _) if f.code == ErrorCode::ConstraintViolation)
}

/// Python `a == b` for JSON values: 1 == 1.0 == True, containers element-wise, dict key order ignored.
#[must_use]
pub fn py_eq(a: &Value, b: &Value) -> bool {
    fn num(v: &Value) -> Option<f64> {
        match v {
            Value::Bool(b) => Some(f64::from(u8::from(*b))),
            Value::Number(n) => n.as_f64(),
            _ => None,
        }
    }
    fn int(v: &Value) -> Option<i128> {
        match v {
            Value::Bool(b) => Some(i128::from(*b)),
            Value::Number(n) => n
                .as_i64()
                .map(i128::from)
                .or_else(|| n.as_u64().map(i128::from)),
            _ => None,
        }
    }
    match (a, b) {
        (Value::Bool(_) | Value::Number(_), Value::Bool(_) | Value::Number(_)) => {
            match (int(a), int(b)) {
                // two ints compare exactly (a float would round big ints)
                (Some(x), Some(y)) => x == y,
                _ => num(a) == num(b),
            }
        }
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| py_eq(v, w)))
        }
        _ => a == b,
    }
}

/// Turns the objects `yaml.safe_load` built into JSON for the request validator and the adapters' `validate`.
/// JSON-native objects map one to one. Inside `params`, anything else becomes a placeholder object and sets
/// `foreign` (Python's `json.dumps(params)` would raise on it, see `create_checked`); in a str field it becomes a
/// value the field rejects, as pydantic rejects the object.
struct Projector<'a> {
    doc: &'a Doc,
    budget: usize,
    foreign: bool,
    /// Parameters whose own value has no JSON type (bytes, a date, a set, a tuple, NaN, a big int), with Python's
    /// `str(value)`: see [`Imported::stand_ins`].
    top: Vec<(String, String)>,
}

/// What an import hands `create`: the validated document, and how to stand in for Python objects JSON lacks.
#[derive(Debug, Clone)]
pub struct Imported {
    pub fields: Value,
    /// Some value in `params` cannot be stored by `json.dumps` (Python raises after the engine's validation).
    pub foreign: bool,
    /// `(key, str(value))` of each parameter whose own value has no JSON type. The engine's checks get a stand-in of
    /// the same verdict: for an `enum` parameter the text (its message quotes `str(value)` and no choice is such a
    /// text), for a `json` one a number (neither a dict, a list nor a str), otherwise the placeholder object (not a
    /// bool, number, str or list).
    pub stand_ins: Vec<(String, String)>,
}

impl<'a> Projector<'a> {
    fn new(doc: &'a Doc) -> Self {
        Projector {
            doc,
            budget: MAX_IMPORT_VALUES,
            foreign: false,
            top: Vec::new(),
        }
    }

    /// A value no str or dict field accepts.
    fn rejected() -> Value {
        Value::Array(Vec::new())
    }

    fn placeholder(&mut self) -> Value {
        self.foreign = true;
        json!({ "<python object>": true })
    }

    /// A top-level ProfileIn field as pydantic's lax (python-mode) validation sees it: `bytes` validate as str
    /// (UTF-8 decoded); other objects JSON cannot hold are rejected by every field.
    fn field(&mut self, key: &str, v: &Py) -> Value {
        if key == "params" {
            return self.params(v);
        }
        match v {
            Py::Bytes(b) => std::str::from_utf8(b)
                .map_or_else(|_| Self::rejected(), |s| Value::String(s.to_owned())),
            Py::None => Value::Null,
            Py::Bool(b) => Value::Bool(*b),
            Py::Str(s) => Value::String(s.clone()),
            Py::Int(PyInt::Small(i)) => {
                i64::try_from(*i).map_or_else(|_| Self::rejected(), Value::from)
            }
            Py::Float(f) => {
                serde_json::Number::from_f64(*f).map_or_else(Self::rejected, Value::Number)
            }
            _ => Self::rejected(),
        }
    }

    /// `params: dict[str, Any]`: keys must be str (or UTF-8 bytes, which pydantic decodes); values are anything.
    fn params(&mut self, v: &Py) -> Value {
        let (Py::Dict(id), Some(items)) = (v, self.doc.dict(v)) else {
            return Self::rejected();
        };
        let mut out = Map::new();
        for (k, x) in items {
            let key = match k {
                Py::Str(s) => s.clone(),
                Py::Bytes(b) => match std::str::from_utf8(b) {
                    Ok(s) => s.to_owned(),
                    Err(_) => return Self::rejected(),
                },
                // ValidationError: "Input should be a valid string" for the key
                _ => return Self::rejected(),
            };
            let mut active = vec![*id];
            let value = self.value(x, &mut active, true);
            if no_json_type(x)
                && let Ok(text) = self.doc.str_of(x)
            {
                self.top.push((key.clone(), text));
            }
            out.insert(key, value);
        }
        Value::Object(out)
    }

    /// One value as `json.dumps` would write it (tuples as lists, int/float/bool/None dict keys as their JSON text).
    /// `top`: a parameter's own value, where the adapters' type checks see a tuple as a tuple (never a list).
    fn value(&mut self, v: &Py, active: &mut Vec<usize>, top: bool) -> Value {
        if self.budget == 0 {
            uncaught("MemoryError", "the document expands past the value budget");
        }
        self.budget -= 1;
        match v {
            Py::None => Value::Null,
            Py::Bool(b) => Value::Bool(*b),
            Py::Int(PyInt::Small(i)) => {
                if let Ok(x) = i64::try_from(*i) {
                    Value::from(x)
                } else if let Ok(x) = u64::try_from(*i) {
                    Value::from(x)
                } else {
                    self.placeholder()
                }
            }
            Py::Float(f) => match serde_json::Number::from_f64(*f) {
                Some(n) => Value::Number(n),
                None => self.placeholder(),
            },
            Py::Str(s) => Value::String(s.clone()),
            Py::Int(PyInt::Big(_)) | Py::Bytes(_) | Py::Date(_) | Py::DateTime(_) | Py::Set(_) => {
                self.placeholder()
            }
            Py::Tuple(items) if !top => {
                let mut out = Vec::with_capacity(items.len());
                for x in items {
                    out.push(self.value(x, active, false));
                }
                Value::Array(out)
            }
            Py::Tuple(_) => self.placeholder(),
            Py::List(id) | Py::Dict(id) => {
                if active.contains(id) || active.len() > crate::pyyaml::pyobj::MAX_DEPTH {
                    // json.dumps: "Circular reference detected" / RecursionError
                    return self.placeholder();
                }
                active.push(*id);
                let out = match self.doc.objs.get(*id) {
                    Some(PyObj::List(items)) => {
                        let mut out = Vec::with_capacity(items.len());
                        for x in items {
                            out.push(self.value(x, active, false));
                        }
                        Value::Array(out)
                    }
                    Some(PyObj::Dict(items)) => {
                        let mut out = Map::new();
                        for (k, x) in items {
                            let key = match k {
                                Py::Str(s) | Py::Int(PyInt::Big(s)) => s.clone(),
                                Py::Int(PyInt::Small(i)) => i.to_string(),
                                Py::Float(f) => json_float_key(*f),
                                Py::Bool(b) => if *b { "true" } else { "false" }.to_owned(),
                                Py::None => "null".to_owned(),
                                // TypeError: keys must be str, int, float, bool or None
                                _ => {
                                    self.foreign = true;
                                    continue;
                                }
                            };
                            let value = self.value(x, active, false);
                            out.insert(key, value);
                        }
                        Value::Object(out)
                    }
                    _ => self.placeholder(),
                };
                active.pop();
                out
            }
        }
    }
}

/// A value with no JSON type of its own (`json.dumps` writes a tuple as a list, but the engines' checks see a tuple).
fn no_json_type(v: &Py) -> bool {
    match v {
        Py::Int(PyInt::Small(i)) => i64::try_from(*i).is_err() && u64::try_from(*i).is_err(),
        Py::Float(f) => !f.is_finite(),
        Py::Int(PyInt::Big(_))
        | Py::Bytes(_)
        | Py::Date(_)
        | Py::DateTime(_)
        | Py::Set(_)
        | Py::Tuple(_) => true,
        Py::None | Py::Bool(_) | Py::Str(_) | Py::List(_) | Py::Dict(_) => false,
    }
}

/// `json.dumps` of a float dict key: `float.__repr__`, or `NaN` / `Infinity` / `-Infinity`.
fn json_float_key(f: f64) -> String {
    if f.is_nan() {
        "NaN".to_owned()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        crate::pyjson::float_repr(f)
    }
}

/// `yaml.safe_load(text)` and `{k: doc[k] for k in (...) if k in doc}`: the fields `ProfileIn` gets (as JSON, see
/// [`Projector`] and [`Imported`]). Errors: "invalid YAML: <class>" for a `yaml.YAMLError`, FastAPI's 500 for any
/// other exception, "profile YAML must be a mapping".
pub fn import_fields(text: &str) -> AppResult<Imported> {
    let doc = match crate::pyyaml::safe_load(text) {
        Ok(d) => d,
        Err(e) if e.is_yaml() => {
            return Err(bad_request(format!("invalid YAML: {}", e.class()))
                .code("invalid_yaml")
                .into());
        }
        // not a yaml.YAMLError (a constructor's ValueError, RecursionError, ...)
        Err(e) => uncaught(e.class(), "not a YAML error"),
    };
    let Some(items) = doc.dict(&doc.root) else {
        return Err(bad_request("profile YAML must be a mapping")
            .code("invalid_yaml")
            .into());
    };
    let mut proj = Projector::new(&doc);
    let mut fields = Map::new();
    for key in IMPORT_KEYS {
        if let Some(v) = Doc::get(items, key) {
            let value = proj.field(key, v);
            fields.insert(key.to_owned(), value);
        }
    }
    Ok(Imported {
        fields: Value::Object(fields),
        foreign: proj.foreign,
        stand_ins: proj.top,
    })
}

/// `ProfileIn(**fields)`: a ValidationError (a ValueError) is "invalid profile document".
fn profile_in(fields: &Value) -> AppResult<ProfileIn> {
    let invalid = || -> AppError {
        unprocessable("invalid profile document: ValidationError")
            .code("invalid_profile")
            .into()
    };
    let mut check = Params::default();
    let model = check.body(&Body::Json(fields.clone()), &T::Model(&PROFILE_IN), true);
    if !check.ok() {
        return Err(invalid());
    }
    serde_json::from_value(model).map_err(|_| invalid())
}

/// The export document for one profile, byte for byte as PyYAML writes it.
#[must_use]
pub fn export_text(p: &Profile) -> String {
    use crate::pyyaml::Node;
    let doc = Node::Map(vec![
        ("name".into(), Node::Str(p.name.clone())),
        ("engine".into(), Node::Str(p.engine.clone())),
        (
            "repo_id".into(),
            p.repo_id.clone().map_or(Node::Null, Node::Str),
        ),
        ("description".into(), Node::Str(p.description.clone())),
        (
            "params".into(),
            Node::from_json(&Value::Object(p.params.clone())),
        ),
    ]);
    crate::pyyaml::safe_dump(&doc, false)
}

impl ProfileService {
    pub fn new(db: Arc<Store>, adapters: AdapterRegistry) -> Self {
        Self { db, adapters }
    }

    fn to_model(r: &SqlRow) -> AppResult<Profile> {
        // pydantic refuses a non-dict `params`: FastAPI's 500
        let Ok(Value::Object(params)) = serde_json::from_str::<Value>(&r.str_or("params", ""))
        else {
            uncaught("ValidationError", "params is not a dict");
        };
        Ok(Profile {
            id: r.str_or("id", ""),
            name: r.str_or("name", ""),
            engine: r.str_or("engine", ""),
            repo_id: r.text("repo_id"),
            params,
            description: r.str_or("description", ""),
            created_at: r.f64("created_at").unwrap_or(0.0),
            updated_at: r.f64("updated_at").unwrap_or(0.0),
        })
    }

    /// `_resolve`: the preset, then the given params; validated by the engine; image keys refused; secret values
    /// dropped (the marker kept).
    fn resolve(&self, p: &ProfileIn) -> AppResult<Obj> {
        let adapter = self.adapters.get(&p.engine)?;
        let mut params = Obj::new();
        if let Some(preset) = p.preset.as_deref().filter(|s| !s.is_empty()) {
            let presets = adapter.presets();
            let Some((_, values)) = presets.iter().find(|(name, _)| name == preset) else {
                let mut available: Vec<String> = presets.iter().map(|(n, _)| n.clone()).collect();
                available.sort();
                return Err(
                    unprocessable(format!("unknown preset '{preset}' for {}", p.engine))
                        .code("unknown_preset")
                        .with("available", available)
                        .into(),
                );
            };
            for (k, v) in values {
                params.insert(k.clone(), v.clone());
            }
        }
        for (k, v) in &p.params {
            params.insert(k.clone(), v.clone());
        }
        let checked: Obj = params
            .iter()
            .filter(|(_, v)| v.as_str() != Some(MARKER))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let errs = adapter.validate(&checked);
        if !errs.is_empty() {
            return Err(unprocessable("invalid engine parameters")
                .with("errors", errs)
                .into());
        }
        if adapter
            .param_catalog()
            .iter()
            .any(|s| s.flag == "@image" && params.contains_key(&s.key))
        {
            return Err(unprocessable(
                "per-instance image overrides are not allowed; use settings pins",
            )
            .code("image_not_allowed")
            .into());
        }
        Ok(split_secrets(&params).0)
    }

    pub fn get(&self, pid: &str) -> AppResult<Profile> {
        match self.db.one("SELECT * FROM profiles WHERE id=?", &[&pid])? {
            Some(r) => Self::to_model(&r),
            None => Err(not_found(format!("no such profile: {pid}")).into()),
        }
    }

    pub fn list_page(
        &self,
        limit: i64,
        cursor: Option<&str>,
        engine: Option<&str>,
    ) -> AppResult<Page<Profile>> {
        let (limit, off) = paginate(limit, cursor);
        let rows = match engine.filter(|e| !e.is_empty()) {
            Some(e) => self.db.all(
                "SELECT * FROM profiles WHERE engine=? ORDER BY name LIMIT ? OFFSET ?",
                &[&e, &(limit + 1), &off],
            )?,
            None => self.db.all(
                "SELECT * FROM profiles ORDER BY name LIMIT ? OFFSET ?",
                &[&(limit + 1), &off],
            )?,
        };
        let more = i64::try_from(rows.len()).unwrap_or(i64::MAX) > limit;
        let take = usize::try_from(limit).unwrap_or(0);
        let items = rows
            .iter()
            .take(take)
            .map(Self::to_model)
            .collect::<AppResult<Vec<_>>>()?;
        Ok(Page::new(items, more.then(|| (off + limit).to_string())))
    }

    /// Run one statement; an IntegrityError is the "already exists" conflict (Python catches only that class).
    fn write(&self, sql: &str, args: &[&dyn rusqlite::ToSql], name: &str) -> AppResult<()> {
        // the closure hands the statement's own result out, so a constraint failure stays distinguishable
        match self.db.transaction(|c| Ok(c.execute(sql, args)))? {
            Ok(_) => Ok(()),
            Err(e) if integrity(&e) => {
                Err(conflict(format!("a profile named '{name}' already exists"))
                    .code("profile_exists")
                    .into())
            }
            Err(e) => {
                tracing::error!(target: "engine_console.store", "database error: {}", e);
                Err(server_error())
            }
        }
    }

    pub fn create(&self, p: &ProfileIn) -> AppResult<Profile> {
        self.create_checked(p, false)
    }

    /// `create`. `foreign`: the imported params held a Python object JSON has no type for; once the engine's
    /// validation passes, Python's `json.dumps(params)` raises (TypeError / ValueError) before the INSERT.
    fn create_checked(&self, p: &ProfileIn, foreign: bool) -> AppResult<Profile> {
        let params = self.resolve(p)?;
        if foreign {
            uncaught("TypeError", "Object is not JSON serializable");
        }
        let (pid, ts) = (new_id("prof_"), now());
        let text = crate::pyjson::dumps(&Value::Object(params));
        self.write(
            "INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?)",
            &[
                &pid,
                &p.name,
                &p.engine,
                &p.repo_id,
                &text,
                &p.description,
                &ts,
                &ts,
            ],
            &p.name,
        )?;
        self.get(&pid)
    }

    pub fn update(&self, pid: &str, p: &ProfileIn) -> AppResult<Profile> {
        self.get(pid)?;
        let params = self.resolve(p)?;
        let text = crate::pyjson::dumps(&Value::Object(params));
        self.write(
            "UPDATE profiles SET name=?, engine=?, repo_id=?, params=?, description=?, updated_at=? WHERE id=?",
            &[
                &p.name,
                &p.engine,
                &p.repo_id,
                &text,
                &p.description,
                &now(),
                &pid,
            ],
            &p.name,
        )?;
        self.get(pid)
    }

    pub fn delete(&self, pid: &str) -> AppResult<()> {
        self.get(pid)?;
        self.db
            .execute("DELETE FROM profiles WHERE id=?", &[&pid])?;
        Ok(())
    }

    /// Python builds `changed` / `only_a` / `only_b` from set operations, whose order follows the str hash (it
    /// varies per process); here they follow profile `a`'s order (`only_b`: `b`'s).
    pub fn diff(&self, a: &str, b: &str) -> AppResult<ProfileDiff> {
        let (pa, pb) = (self.get(a)?.params, self.get(b)?.params);
        let mut changed = Obj::new();
        let mut only_a = Obj::new();
        let mut only_b = Obj::new();
        for (k, va) in &pa {
            match pb.get(k) {
                Some(vb) if !py_eq(va, vb) => {
                    changed.insert(k.clone(), json!({"a": va, "b": vb}));
                }
                Some(_) => {}
                None => {
                    only_a.insert(k.clone(), va.clone());
                }
            }
        }
        for (k, vb) in &pb {
            if !pa.contains_key(k) {
                only_b.insert(k.clone(), vb.clone());
            }
        }
        Ok(ProfileDiff {
            a: a.to_owned(),
            b: b.to_owned(),
            changed,
            only_a,
            only_b,
        })
    }

    /// `yaml.safe_dump({"name", "engine", "repo_id", "description", "params"}, sort_keys=False)`.
    pub fn export_yaml(&self, pid: &str) -> AppResult<String> {
        Ok(export_text(&self.get(pid)?))
    }

    /// The body text (YAML, or the "yaml" member of a JSON body; the route extracts it).
    pub fn import_yaml(&self, text: &str) -> AppResult<Profile> {
        let imported = import_fields(text)?;
        let mut p = profile_in(&imported.fields)?;
        if !imported.stand_ins.is_empty() {
            // the engine must exist before its catalog says which stand-in each parameter gets (_resolve's order)
            let catalog = self.adapters.get(&p.engine)?.param_catalog();
            for (key, text) in &imported.stand_ins {
                match catalog
                    .iter()
                    .find(|s| &s.key == key)
                    .map(|s| s.type_.as_str())
                {
                    Some("enum") => {
                        p.params.insert(key.clone(), Value::String(text.clone()));
                    }
                    Some("json") => {
                        p.params.insert(key.clone(), Value::from(0));
                    }
                    _ => {}
                }
            }
        }
        self.create_checked(&p, imported.foreign)
    }
}
