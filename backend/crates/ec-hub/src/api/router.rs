//! Starlette's routing, exactly: routes are tried in registration order against the percent-decoded path; the
//! first route whose path AND method match handles the request; a route whose path matches but not its method is a
//! partial match, used only when nothing matches fully (405); nothing at all is 404 (or a slash redirect).
//! `{name}` matches one segment, `{name:path}` the rest; `$` also matches before one trailing "\n", like Python's.

use futures::future::BoxFuture;
use regex::Regex;

use crate::api::http::{Ctx, HResult};

pub type Handler = fn(Ctx) -> BoxFuture<'static, HResult>;

/// The role a route requires: its FastAPI role dependencies, enforced by the request pipeline (app.rs) after
/// authentication and before the handler (`api::deps::check_access`). `tests/data/route_roles.json`, generated from
/// the live Python app by `backend/parity/gen/route_roles_python.py`, pins every route's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Access {
    /// No role dependency (the OpenAPI document, /metrics, the SPA). Whether a key is needed at all is still the
    /// middleware's rule (`security::guarded`).
    Open,
    /// `enforce_roles`, the dependency of every API router: a safe method is open to any authenticated caller; a
    /// mutating one needs an admin, or a viewer on the exact viewer list (`security::viewer_may`).
    Caller,
    /// `enforce_roles`, then `_: Admin` (`require_admin`): an admin only.
    Admin,
}

impl Access {
    /// The name the golden file uses ("open", "caller", "admin").
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Open => "open",
            Self::Caller => "caller",
            Self::Admin => "admin",
        }
    }
}

pub struct Route {
    pub method: &'static str,
    /// the route template ("/api/v1/instances/{iid}"): the metrics label and the parity route id
    pub path: &'static str,
    /// the role the pipeline requires before the handler runs
    pub access: Access,
    re: Regex,
    names: Vec<String>,
    pub handler: Handler,
}

pub enum Match<'a> {
    Full(&'a Route, Vec<(String, String)>),
    /// path matched, method did not: 405
    Partial(&'a Route),
    None,
}

#[derive(Default)]
pub struct Router {
    pub routes: Vec<Route>,
}

impl Router {
    /// Register a route with the role it requires (registration order is matching order).
    pub fn add(
        &mut self,
        method: &'static str,
        path: &'static str,
        access: Access,
        handler: Handler,
    ) {
        let (re, names) = compile(path);
        self.routes.push(Route {
            method,
            path,
            access,
            re,
            names,
            handler,
        });
    }

    /// Every registered route as (method, path template, role), in matching order (the table the tests pin).
    pub fn table(&self) -> Vec<(&'static str, &'static str, Access)> {
        self.routes
            .iter()
            .map(|r| (r.method, r.path, r.access))
            .collect()
    }

    pub fn find(&self, method: &str, path: &str) -> Match<'_> {
        let mut partial: Option<&Route> = None;
        for r in &self.routes {
            if let Some(caps) = r.re.captures(path) {
                if r.method == method {
                    let params = r
                        .names
                        .iter()
                        .map(|n| {
                            (
                                n.clone(),
                                caps.name(n).map_or("", |m| m.as_str()).to_string(),
                            )
                        })
                        .collect();
                    return Match::Full(r, params);
                }
                if partial.is_none() {
                    partial = Some(r);
                }
            }
        }
        partial.map_or(Match::None, Match::Partial)
    }

    /// Starlette's redirect_slashes: the other slash variant of `path` matches some route fully.
    pub fn slash_redirect(&self, method: &str, path: &str) -> Option<String> {
        if path == "/" {
            return None;
        }
        let other = if let Some(p) = path.strip_suffix('/') {
            p.to_string()
        } else {
            format!("{path}/")
        };
        matches!(self.find(method, &other), Match::Full(..)).then_some(other)
    }
}

fn compile(path: &str) -> (Regex, Vec<String>) {
    let mut re = String::from("^");
    let mut names = Vec::new();
    let mut rest = path;
    while let Some(i) = rest.find('{') {
        re.push_str(&regex::escape(&rest[..i]));
        let after = &rest[i + 1..];
        let j = after.find('}').unwrap_or(after.len());
        let spec = &after[..j];
        let (name, conv) = spec.split_once(':').unwrap_or((spec, "str"));
        let pat = if conv == "path" { ".*" } else { "[^/]+" };
        re.push_str(&format!("(?P<{name}>{pat})"));
        names.push(name.to_string());
        rest = after.get(j + 1..).unwrap_or("");
    }
    re.push_str(&regex::escape(rest));
    re.push_str("\n?$");
    (
        Regex::new(&re).unwrap_or_else(|_| Regex::new("$^").unwrap_or_else(|_| unreachable!())),
        names,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(_: Ctx) -> BoxFuture<'static, HResult> {
        Box::pin(async { crate::api::http::no_content() })
    }

    #[test]
    fn test_starlette_matching() {
        let mut r = Router::default();
        r.add("GET", "/api/v1/instances/{iid}", Access::Caller, h);
        r.add("POST", "/api/v1/instances/preflight", Access::Caller, h);
        r.add("GET", "/api/v1/hf/models/{repo_id:path}", Access::Admin, h);
        r.add("GET", "/{path:path}", Access::Open, h);
        assert_eq!(
            r.table()[2],
            ("GET", "/api/v1/hf/models/{repo_id:path}", Access::Admin)
        );
        assert!(
            matches!(r.find("GET", "/api/v1/instances/abc"), Match::Full(rt, p) if rt.path == "/api/v1/instances/{iid}" && p[0].1 == "abc")
        );
        // registration order: GET /instances/preflight is the {iid} route
        assert!(
            matches!(r.find("GET", "/api/v1/instances/preflight"), Match::Full(rt, _) if rt.path == "/api/v1/instances/{iid}")
        );
        assert!(
            matches!(r.find("GET", "/api/v1/hf/models/a/b"), Match::Full(_, p) if p[0].1 == "a/b")
        );
        assert!(matches!(r.find("DELETE", "/anything"), Match::Partial(_)));
        assert!(
            matches!(r.find("GET", "/api/v1/instances/a/b"), Match::Full(rt, _) if rt.path == "/{path:path}")
        );
    }
}
