//! Every route of the Python app, registered in its exact order (backend/parity/routes.python.json is the reference;
//! the route-coverage check fails on a missing one). The handlers are thin: validate exactly as FastAPI
//! does (path/query/body errors together), call one service method, serialise. A service that is not ported
//! would answer 501 not_ported (D10).
//!
//! Each route declares the role its Python dependencies require ([`Access`]: `enforce_roles` on every API router,
//! `_: Admin` on the admin-only ones); the request pipeline (app.rs) enforces it after authentication and before
//! the handler, so no handler checks roles itself. `tests/test_route_roles.rs` compares this table with the
//! Python app's (tests/data/route_roles.json, from backend/parity/gen/route_roles_python.py).

pub mod admin;
pub mod chat;
pub mod downloads;
pub mod hosts;
pub mod instances;
pub mod library;
pub mod observe;
pub mod profiles;
pub mod scripts;
pub mod system;

use std::path::Path;

use crate::api::router::{Access, Router};

pub const OPENAPI: &str = "/api/v1/openapi.json";

macro_rules! h {
    ($f:path) => {
        |ctx| Box::pin($f(ctx))
    };
}

/// The route table. `frontend` is the resolved frontend dir when it exists (else no SPA route, as in Python).
pub fn router(frontend: Option<&Path>) -> Router {
    let mut r = Router::default();
    // openapi: a plain starlette Route (GET and HEAD), registered by FastAPI before the routers
    r.add("GET", OPENAPI, Access::Open, h!(system::openapi));
    r.add("HEAD", OPENAPI, Access::Open, h!(system::openapi));
    // system
    r.add("GET", "/api/v1/health", Access::Caller, h!(system::health));
    r.add(
        "GET",
        "/api/v1/hardware",
        Access::Caller,
        h!(system::hardware),
    );
    r.add("GET", "/api/v1/learn", Access::Caller, h!(system::learn));
    r.add(
        "GET",
        "/api/v1/engines",
        Access::Caller,
        h!(system::engines),
    );
    r.add(
        "GET",
        "/api/v1/engines/{engine}/params",
        Access::Caller,
        h!(system::engine_params),
    );
    r.add(
        "GET",
        "/api/v1/hf/search",
        Access::Admin,
        h!(system::hf_search),
    );
    r.add(
        "GET",
        "/api/v1/hf/models/{repo_id:path}",
        Access::Admin,
        h!(system::hf_model),
    );
    r.add("POST", "/api/v1/fit", Access::Caller, h!(system::fit));
    // downloads
    r.add(
        "GET",
        "/api/v1/downloads",
        Access::Caller,
        h!(downloads::list_downloads),
    );
    r.add(
        "POST",
        "/api/v1/downloads",
        Access::Caller,
        h!(downloads::create_download),
    );
    r.add(
        "GET",
        "/api/v1/downloads/stream",
        Access::Caller,
        h!(downloads::stream_downloads),
    );
    r.add(
        "GET",
        "/api/v1/downloads/{did}",
        Access::Caller,
        h!(downloads::get_download),
    );
    r.add(
        "POST",
        "/api/v1/downloads/{did}/pause",
        Access::Caller,
        h!(downloads::pause),
    );
    r.add(
        "POST",
        "/api/v1/downloads/{did}/resume",
        Access::Caller,
        h!(downloads::resume),
    );
    r.add(
        "POST",
        "/api/v1/downloads/{did}/cancel",
        Access::Caller,
        h!(downloads::cancel),
    );
    r.add(
        "DELETE",
        "/api/v1/downloads/{did}",
        Access::Caller,
        h!(downloads::delete_download),
    );
    r.add(
        "GET",
        "/api/v1/models",
        Access::Caller,
        h!(downloads::local_models),
    );
    r.add(
        "DELETE",
        "/api/v1/models/{repo_id:path}",
        Access::Admin,
        h!(downloads::delete_model),
    );
    // library
    r.add(
        "GET",
        "/api/v1/library",
        Access::Caller,
        h!(library::list_library),
    );
    r.add(
        "POST",
        "/api/v1/library/rescan",
        Access::Admin,
        h!(library::rescan_library),
    );
    r.add(
        "DELETE",
        "/api/v1/library/copies/{copy_id}",
        Access::Admin,
        h!(library::delete_copy),
    );
    r.add(
        "POST",
        "/api/v1/library/copies/{copy_id}/copy",
        Access::Admin,
        h!(library::copy_copy),
    );
    // instances
    r.add(
        "GET",
        "/api/v1/instances",
        Access::Caller,
        h!(instances::list_instances),
    );
    r.add(
        "POST",
        "/api/v1/instances",
        Access::Caller,
        h!(instances::create_instance),
    );
    r.add(
        "POST",
        "/api/v1/instances/preflight",
        Access::Caller,
        h!(instances::preflight),
    );
    r.add(
        "POST",
        "/api/v1/instances/discover",
        Access::Admin,
        h!(instances::discover),
    );
    r.add(
        "GET",
        "/api/v1/instances/{iid}",
        Access::Caller,
        h!(instances::get_instance),
    );
    r.add(
        "PATCH",
        "/api/v1/instances/{iid}",
        Access::Caller,
        h!(instances::patch_instance),
    );
    r.add(
        "POST",
        "/api/v1/instances/{iid}/stop",
        Access::Caller,
        h!(instances::stop),
    );
    r.add(
        "POST",
        "/api/v1/instances/{iid}/start",
        Access::Caller,
        h!(instances::start),
    );
    r.add(
        "POST",
        "/api/v1/instances/{iid}/restart",
        Access::Caller,
        h!(instances::restart),
    );
    r.add(
        "DELETE",
        "/api/v1/instances/{iid}",
        Access::Caller,
        h!(instances::remove),
    );
    r.add(
        "GET",
        "/api/v1/instances/{iid}/logs",
        Access::Caller,
        h!(instances::logs),
    );
    r.add(
        "GET",
        "/api/v1/instances/{iid}/logs/stream",
        Access::Caller,
        h!(instances::logs_stream),
    );
    r.add(
        "GET",
        "/api/v1/instances/{iid}/command",
        Access::Caller,
        h!(instances::command),
    );
    // profiles
    r.add(
        "GET",
        "/api/v1/profiles",
        Access::Caller,
        h!(profiles::list_profiles),
    );
    r.add(
        "POST",
        "/api/v1/profiles",
        Access::Caller,
        h!(profiles::create_profile),
    );
    r.add(
        "POST",
        "/api/v1/profiles/import",
        Access::Caller,
        h!(profiles::import_profile),
    );
    r.add(
        "GET",
        "/api/v1/profiles/{a}/diff/{b}",
        Access::Caller,
        h!(profiles::diff),
    );
    r.add(
        "GET",
        "/api/v1/profiles/{pid}/export",
        Access::Caller,
        h!(profiles::export),
    );
    r.add(
        "PUT",
        "/api/v1/profiles/{pid}",
        Access::Caller,
        h!(profiles::update_profile),
    );
    r.add(
        "DELETE",
        "/api/v1/profiles/{pid}",
        Access::Caller,
        h!(profiles::delete_profile),
    );
    // observe
    r.add(
        "GET",
        "/api/v1/metrics/system",
        Access::Caller,
        h!(observe::system_metrics),
    );
    r.add(
        "GET",
        "/api/v1/metrics/instances/{iid}",
        Access::Caller,
        h!(observe::instance_metrics),
    );
    r.add(
        "GET",
        "/api/v1/metrics/stream",
        Access::Caller,
        h!(observe::metrics_stream),
    );
    r.add(
        "POST",
        "/api/v1/bench",
        Access::Caller,
        h!(observe::start_bench),
    );
    r.add(
        "GET",
        "/api/v1/bench",
        Access::Caller,
        h!(observe::list_bench),
    );
    r.add(
        "GET",
        "/api/v1/bench/{bid}",
        Access::Caller,
        h!(observe::get_bench),
    );
    r.add(
        "GET",
        "/api/v1/bench/{bid}/stream",
        Access::Caller,
        h!(observe::bench_stream),
    );
    r.add("GET", "/api/v1/usage", Access::Caller, h!(observe::usage));
    r.add(
        "GET",
        "/api/v1/usage/export.csv",
        Access::Caller,
        h!(observe::usage_csv),
    );
    r.add("GET", "/api/v1/audit", Access::Admin, h!(observe::audit));
    // chat
    r.add(
        "POST",
        "/api/v1/chat/completions",
        Access::Caller,
        h!(chat::chat_completions),
    );
    r.add(
        "GET",
        "/api/v1/conversations",
        Access::Caller,
        h!(chat::list_conversations),
    );
    r.add(
        "POST",
        "/api/v1/conversations",
        Access::Caller,
        h!(chat::create_conversation),
    );
    r.add(
        "GET",
        "/api/v1/conversations/{cid}",
        Access::Caller,
        h!(chat::get_conversation),
    );
    r.add(
        "DELETE",
        "/api/v1/conversations/{cid}",
        Access::Caller,
        h!(chat::delete_conversation),
    );
    r.add(
        "GET",
        "/api/v1/prompts",
        Access::Caller,
        h!(chat::list_prompts),
    );
    r.add(
        "POST",
        "/api/v1/prompts",
        Access::Caller,
        h!(chat::create_prompt),
    );
    r.add(
        "PUT",
        "/api/v1/prompts/{pid}",
        Access::Caller,
        h!(chat::update_prompt),
    );
    r.add(
        "DELETE",
        "/api/v1/prompts/{pid}",
        Access::Caller,
        h!(chat::delete_prompt),
    );
    r.add(
        "POST",
        "/api/v1/arena/matches",
        Access::Caller,
        h!(chat::create_match),
    );
    r.add(
        "POST",
        "/api/v1/arena/matches/{mid}/vote",
        Access::Caller,
        h!(chat::vote),
    );
    r.add(
        "GET",
        "/api/v1/arena/leaderboard",
        Access::Caller,
        h!(chat::leaderboard),
    );
    // admin
    r.add(
        "GET",
        "/api/v1/settings",
        Access::Caller,
        h!(admin::get_settings),
    );
    r.add(
        "PUT",
        "/api/v1/settings",
        Access::Admin,
        h!(admin::put_settings),
    );
    r.add("GET", "/api/v1/keys", Access::Admin, h!(admin::list_keys));
    r.add("POST", "/api/v1/keys", Access::Admin, h!(admin::create_key));
    r.add(
        "DELETE",
        "/api/v1/keys/{kid}",
        Access::Admin,
        h!(admin::revoke_key),
    );
    // hosts
    r.add(
        "GET",
        "/api/v1/hosts",
        Access::Caller,
        h!(hosts::list_hosts),
    );
    r.add(
        "GET",
        "/api/v1/hosts/{host_id}",
        Access::Caller,
        h!(hosts::get_host),
    );
    r.add(
        "POST",
        "/api/v1/hosts/{host_id}/reconnect",
        Access::Admin,
        h!(hosts::reconnect_host),
    );
    r.add(
        "GET",
        "/api/v1/fleet/stream",
        Access::Caller,
        h!(hosts::fleet_stream),
    );
    // scripts
    r.add(
        "GET",
        "/api/v1/scripts",
        Access::Caller,
        h!(scripts::list_scripts),
    );
    r.add(
        "POST",
        "/api/v1/scripts/{host_id}/{name}/{action}",
        Access::Admin,
        h!(scripts::run_script),
    );
    r.add(
        "GET",
        "/api/v1/scripts/jobs/{job_id}",
        Access::Caller,
        h!(scripts::get_job),
    );
    r.add(
        "GET",
        "/api/v1/scripts/jobs/{job_id}/stream",
        Access::Caller,
        h!(scripts::job_stream),
    );
    // the console's own /metrics
    r.add("GET", "/metrics", Access::Open, h!(observe::prometheus));
    // the frontend (only when its directory exists)
    if frontend.is_some() {
        r.add("GET", "/{path:path}", Access::Open, h!(system::spa));
    }
    r
}
