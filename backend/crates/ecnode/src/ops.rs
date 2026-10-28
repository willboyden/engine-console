//! Python: node/src/ecnode/ops.py.
//!
//! One handler per protocol op (docs/FLEET-PROTOCOL.md §4). A handler receives arguments that already went through
//! `ec_proto::clean_args` + `missing_args` (server.rs does that, and refuses every op outside `MONITOR_OPS` with
//! `monitor_only` unless the policy says "manage"). It returns the op's result object or an error; policy checks
//! happen here, after cleaning.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use ec_proto::{
    BAD_ARGS, DOCKER_ERROR, MAX_LOG_TEXT, NOT_FOUND, Obj, POLICY_DENIED, PORT_IN_USE, UNSUPPORTED,
};
use serde_json::{Value, json};

use crate::agent::Agent;
use crate::discovery::container_view;
use crate::docker::{
    Docker, GATEWAY_NAME, INSTANCE, LABEL, NGINX_HOST, ROLE_LABEL, check_owned, container_name,
    docker_error, ensure_network, gateway_argv, label, labels_of, plan_run, remove_files, run_argv,
    write_files,
};
use crate::errors::{Call, HandlerError, OpError};
use crate::fsops::{fs_delete, fs_put_small, fs_read, fs_stat, fs_symlink, fs_write};
use crate::models::{Scanner, model_config};
use crate::policy::valid_image;
use crate::util::proc::{Timer, kill_group};
use crate::util::{
    EnvMap, child_env, fit_tail, iter_lines, lock, parse_time, py, run_tail, scrub, spawn,
};

pub const PULL_TIMEOUT_S: u64 = 4 * 3600;
/// docker.logs text, as encoded in the res frame (MAX_JSON is 1 MiB)
pub const LOG_JSON_BUDGET: usize = 900 * 1024;

/// The Python module that implements each op.
pub fn module_of(op: &str) -> &'static str {
    match op {
        "http.request" => "httpproxy",
        o if o.starts_with("fs.") => "fsops",
        o if o.starts_with("models.") => "models",
        "scripts.run" => "scripts",
        _ => "docker",
    }
}

fn arg<'a>(c: &'a Call<'_>, k: &str) -> &'a Value {
    c.args.get(k).unwrap_or(&Value::Null)
}

fn o(v: Value) -> Obj {
    match v {
        Value::Object(m) => m,
        _ => Obj::new(),
    }
}

fn err(code: &str, msg: impl Into<String>) -> HandlerError {
    OpError::new(code, msg).into()
}

fn scanner(a: &Agent) -> Result<Arc<Scanner>, HandlerError> {
    a.scanner()
        .ok_or_else(|| err(UNSUPPORTED, "the model scanner is not running on this node"))
}

fn doc_id(doc: &Value) -> Result<String, HandlerError> {
    match doc.get("Id") {
        Some(v) => Ok(py::str_of(v)),
        None => Err(HandlerError::internal("KeyError")),
    }
}

// ── node / metrics ───────────────────────────────────────────────────────────────────────────────────────
fn node_info(a: &Agent) -> Obj {
    let mut hello = a.hello(true);
    hello.remove("type");
    hello
}

fn metrics_configure(a: &Agent, c: &Call<'_>) -> Obj {
    let interval = arg(c, "interval_s").as_f64();
    let mut families = Vec::new();
    if let Value::Object(m) = arg(c, "families") {
        for (k, v) in m {
            let prefixes = v
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            families.push((k.clone(), prefixes));
        }
    }
    a.configure_metrics(interval, families);
    Obj::new()
}

// ── docker: runs ─────────────────────────────────────────────────────────────────────────────────────────
fn docker_render(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    let plan = plan_run(
        arg(c, "spec"),
        &policy,
        &a.gpu_uuids(),
        &a.paths.state,
        a.secret_scan_max_entries,
    )?;
    Ok(o(json!({"argv": run_argv(&a.docker_for(&policy), &plan)})))
}

fn docker_run(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    let d = a.docker_for(&policy);
    let plan = plan_run(
        arg(c, "spec"),
        &policy,
        &a.gpu_uuids(),
        &a.paths.state,
        a.secret_scan_max_entries,
    )?;
    ensure_network(&d, &policy.engine_network)?;
    write_files(&plan, &a.paths.state)?;
    let secrets: EnvMap = plan.secret_env.iter().cloned().collect();
    let res = a.runner.run(
        &run_argv(&d, &plan),
        120.0,
        Some(&child_env(Some(&secrets), &a.env)),
    );
    if !res.ok() {
        return Err(docker_error("run", &res).into());
    }
    Ok(o(json!({"id": last_line_id(&res.out)})))
}

fn last_line_id(out: &str) -> String {
    let lines = py::splitlines(py::strip(out));
    lines
        .last()
        .map(|l| py::take(py::strip(l), 64).to_string())
        .unwrap_or_default()
}

fn owned(a: &Agent, c: &Call<'_>) -> Result<(Docker, Value), HandlerError> {
    let name = container_name(arg(c, "name"))?;
    let d = a.docker();
    let Some(doc) = d.inspect_one(&json!(name))? else {
        return Err(err(
            NOT_FOUND,
            format!(
                "there is no container named {} on this machine",
                py::repr(&name)
            ),
        ));
    };
    check_owned(
        &doc,
        &name,
        arg(c, "role").as_str(),
        arg(c, "instance").as_str(),
    )?;
    Ok((d, doc))
}

fn docker_stop(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let (d, doc) = owned(a, c)?;
    let t = arg(c, "timeout_s").as_i64().map_or(30, |t| t.clamp(0, 600));
    let id = doc_id(&doc)?;
    d.ok(&["stop", "-t", &t.to_string(), &id], (t + 30) as f64, None)?;
    Ok(Obj::new())
}

fn docker_start(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let (d, doc) = owned(a, c)?;
    d.ok(&["start", &doc_id(&doc)?], 60.0, None)?;
    Ok(Obj::new())
}

fn docker_rm(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let (d, doc) = owned(a, c)?;
    d.ok(&["rm", "-f", &doc_id(&doc)?], 60.0, None)?;
    let labels = labels_of(&doc);
    if label(&labels, ROLE_LABEL) == Some("engine") {
        remove_files(
            &a.paths.state,
            label(&labels, "engine-console.instance").unwrap_or(""),
        );
    }
    Ok(Obj::new())
}

fn docker_inspect(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let name = container_name(arg(c, "name"))?;
    let Some(doc) = a.docker().inspect_one(&json!(name))? else {
        return Err(err(
            NOT_FOUND,
            format!(
                "there is no container named {} on this machine",
                py::repr(&name)
            ),
        ));
    };
    let roots: Vec<_> = a
        .policy()
        .model_roots
        .iter()
        .map(|r| (r.clone(), std::path::PathBuf::from(a.disk.resolve(&r.path))))
        .collect();
    let Some(view) = container_view(&doc, &roots, &a.gpu_by_index(), None) else {
        return Err(err(
            NOT_FOUND,
            format!("docker returned nothing usable for {}", py::repr(&name)),
        ));
    };
    Ok(o(json!({"info": view.info})))
}

fn tail_arg(c: &Call<'_>) -> i64 {
    arg(c, "tail").as_i64().map_or(200, |t| t.clamp(1, 5000))
}

fn docker_logs(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let (d, doc) = owned(a, c)?;
    let argv = d.argv(&["logs", "--tail", &tail_arg(c).to_string(), &doc_id(&doc)?]);
    let (rc, data, truncated) = run_tail(&argv, 30.0, MAX_LOG_TEXT, &child_env(None, &a.env));
    let decoded = String::from_utf8_lossy(&data);
    if rc != 0 {
        return Err(err(
            DOCKER_ERROR,
            format!("docker logs failed: {}", py::tail(&scrub(&decoded), 300)),
        ));
    }
    let text = scrub(&decoded);
    let text = py::tail(&text, MAX_LOG_TEXT);
    let (text, trimmed) = fit_tail(text, LOG_JSON_BUDGET);
    Ok(o(json!({"text": text, "truncated": truncated || trimmed})))
}

/// Run a streaming docker command, one event per output line, until it ends or the request is cancelled.
fn stream_lines(
    a: &Agent,
    c: &Call<'_>,
    argv: &[String],
    limit: usize,
    deadline: Option<Duration>,
    mut on_line: impl FnMut(&str),
) -> Result<i32, HandlerError> {
    let what = argv
        .get(argv.len().saturating_sub(1))
        .cloned()
        .unwrap_or_default();
    let _ = what;
    let sp = spawn(argv, &child_env(None, &a.env), true, None);
    let sp = match sp {
        Ok(sp) => sp,
        Err(e) => {
            return Err(HandlerError::Op(OpError::new(
                DOCKER_ERROR,
                py::io_strerror(&e),
            )));
        }
    };
    let proc = Arc::clone(&sp.proc);
    {
        let p = Arc::clone(&proc);
        c.stream
            .on_cancel(Box::new(move || kill_group(&p, Duration::from_secs(2))));
    }
    let mut timer = deadline.map(|d| {
        let p = Arc::clone(&proc);
        Timer::start(d, move || kill_group(&p, Duration::from_secs(2)))
    });
    for line in iter_lines(sp.stdout, limit) {
        on_line(&line);
    }
    let rc = proc.wait();
    if let Some(t) = timer.as_mut() {
        t.cancel();
    }
    kill_group(&proc, Duration::from_secs(2));
    Ok(rc)
}

fn docker_logs_follow(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let (d, doc) = owned(a, c)?;
    let argv = d.argv(&[
        "logs",
        "-f",
        "--tail",
        &tail_arg(c).to_string(),
        &doc_id(&doc)?,
    ]);
    let mut count = 0i64;
    stream_lines(a, c, &argv, 8192, None, |line| {
        let mut ev = Obj::new();
        ev.insert("line".into(), json!(scrub(line)));
        c.stream.event(ev);
        count += 1;
    })
    .map_err(|e| match e {
        HandlerError::Op(op) => err(
            DOCKER_ERROR,
            format!("cannot run docker logs: {}", op.message),
        ),
        other => other,
    })?;
    if c.stream.cancelled() {
        return Err(HandlerError::Cancelled);
    }
    Ok(o(json!({"lines": count})))
}

// ── docker: images ───────────────────────────────────────────────────────────────────────────────────────
fn image_arg(c: &Call<'_>) -> Result<String, HandlerError> {
    let image = arg(c, "image");
    match image.as_str() {
        Some(i) if valid_image(i) => Ok(i.to_string()),
        _ => Err(err(
            BAD_ARGS,
            format!(
                "{} is not a valid image reference",
                py::repr_take(&py::str_of(image), 100)
            ),
        )),
    }
}

fn docker_image_present(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let image = image_arg(c)?;
    let res = a.docker().call(
        &["image", "inspect", "--format", "{{.Id}}", &image],
        20.0,
        None,
    );
    if res.ok() {
        let id = py::take(py::strip(&res.out), 80).to_string();
        return Ok(o(
            json!({"present": true, "id": if id.is_empty() { None } else { Some(id) }}),
        ));
    }
    let text = format!("{}{}", res.err, res.out).to_lowercase();
    if text.contains("no such image") || text.contains("not found") {
        return Ok(o(json!({"present": false, "id": null})));
    }
    Err(docker_error("image inspect", &res).into())
}

fn docker_images(a: &Agent) -> Result<Obj, HandlerError> {
    let d = a.docker();
    let listed = d.ok(&["image", "ls", "-q", "--no-trunc"], 30.0, None)?;
    let mut ids: Vec<String> = Vec::new();
    for i in py::split_ws(&listed.out) {
        if !ids.iter().any(|x| x == i) {
            ids.push(i.to_string());
        }
    }
    ids.truncate(512);
    if ids.is_empty() {
        return Ok(o(json!({"images": []})));
    }
    let mut args = vec!["image", "inspect"];
    args.extend(ids.iter().map(String::as_str));
    let res = d.call(&args, 60.0, None);
    let docs: Value = if py::strip(&res.out).is_empty() {
        json!([])
    } else {
        serde_json::from_str(&res.out).map_err(|_| {
            err(
                DOCKER_ERROR,
                "docker image inspect returned something that is not JSON",
            )
        })?
    };
    let mut out: Vec<Value> = Vec::new();
    for doc in docs.as_array().map(Vec::as_slice).unwrap_or_default() {
        let Value::Object(doc) = doc else { continue };
        let refs = [doc.get("RepoTags"), doc.get("RepoDigests")]
            .into_iter()
            .flatten()
            .find(|v| py::truthy(v))
            .cloned()
            .unwrap_or_else(|| json!(["<none>"]));
        for r in refs.as_array().map(Vec::as_slice).unwrap_or_default() {
            if out.len() >= 512 {
                break;
            }
            let size = match doc.get("Size") {
                Some(Value::Number(n)) if !n.is_f64() => Value::Number(n.clone()),
                _ => Value::Null,
            };
            let s_or = |k: &str| match doc.get(k) {
                Some(v) if py::truthy(v) => py::str_of(v),
                _ => String::new(),
            };
            out.push(json!({
                "ref": py::take(&py::str_of(r), 256), "id": py::take(&s_or("Id"), 80), "size": size,
                "created": parse_time(doc.get("Created").unwrap_or(&Value::Null)),
                "arch": py::take(&s_or("Architecture"), 16),
            }));
        }
    }
    Ok(o(json!({"images": out})))
}

fn docker_pull(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    let image = image_arg(c)?;
    if !policy.allow_pull {
        return Err(err(
            POLICY_DENIED,
            format!(
                "this node's policy does not allow pulling images (allow_pull in {})",
                policy.where_()
            ),
        ));
    }
    if !policy.image_allowed(&image) {
        return Err(err(
            POLICY_DENIED,
            format!(
                "image {} is not allowed by this node's policy (engine_images)",
                py::repr(&image)
            ),
        ));
    }
    let d = a.docker_for(&policy);
    let last = Mutex::new(String::new());
    let rc = stream_lines(
        a,
        c,
        &d.argv(&["pull", &image]),
        1024,
        Some(Duration::from_secs(PULL_TIMEOUT_S)),
        |line| {
            if !line.is_empty() {
                *lock(&last) = line.to_string();
            }
            let mut ev = Obj::new();
            ev.insert("line".into(), json!(scrub(line)));
            c.stream.event(ev);
        },
    )
    .map_err(|e| match e {
        HandlerError::Op(op) => err(
            DOCKER_ERROR,
            format!("cannot run docker pull: {}", op.message),
        ),
        other => other,
    })?;
    if c.stream.cancelled() {
        return Err(HandlerError::Cancelled);
    }
    if rc != 0 {
        let last = lock(&last).clone();
        return Err(err(
            DOCKER_ERROR,
            format!("docker pull failed: {}", py::take(&scrub(&last), 300)),
        ));
    }
    let res = d.call(
        &["image", "inspect", "--format", "{{.Id}}", &image],
        20.0,
        None,
    );
    Ok(o(
        json!({"id": if res.ok() { Some(py::take(py::strip(&res.out), 80).to_string()) } else { None }}),
    ))
}

// ── docker: engines the console did not create ──────────────────────────────────────────────────────────
fn docker_external(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    if policy.external_control != "stop_start" {
        return Err(err(
            POLICY_DENIED,
            format!(
                "this node's policy does not let the console stop or start engines it did not create \
                 (external_control in {})",
                policy.where_()
            ),
        ));
    }
    let action = arg(c, "action").as_str().unwrap_or("");
    if action != "stop" && action != "start" {
        return Err(err(BAD_ARGS, "action must be \"stop\" or \"start\""));
    }
    let name = container_name(arg(c, "name"))?;
    let d = a.docker_for(&policy);
    let Some(doc) = d.inspect_one(&json!(name))? else {
        return Err(err(
            NOT_FOUND,
            format!(
                "there is no container named {} on this machine",
                py::repr(&name)
            ),
        ));
    };
    if label(&labels_of(&doc), LABEL) == Some("1") {
        return Err(err(
            POLICY_DENIED,
            format!(
                "{} is one of the console's own containers: use docker.stop / docker.start",
                py::repr(&name)
            ),
        ));
    }
    let view = container_view(&doc, &[], &std::collections::HashMap::new(), None);
    let engine = view
        .as_ref()
        .and_then(|v| v.info.get("engine"))
        .is_some_and(py::truthy);
    if !engine {
        return Err(err(
            POLICY_DENIED,
            format!(
                "{} is not recognised as an inference engine",
                py::repr(&name)
            ),
        ));
    }
    let vname = view
        .as_ref()
        .and_then(|v| v.info.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    if let Some(owner) = policy.scripts.iter().find(|s| s.matches(&vname)) {
        return Err(err(
            POLICY_DENIED,
            format!(
                "{} is managed by the script {}; use its actions",
                py::repr(&name),
                py::repr(&owner.name)
            ),
        ));
    }
    let id = doc_id(&doc)?;
    if action == "stop" {
        d.ok(&["stop", "-t", "30", &id], 90.0, None)?;
    } else {
        d.ok(&["start", &id], 90.0, None)?;
    }
    Ok(Obj::new())
}

// ── network and gateway ──────────────────────────────────────────────────────────────────────────────────
fn net_ensure_internal(a: &Agent) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    ensure_network(&a.docker_for(&policy), &policy.engine_network)?;
    Ok(o(json!({"network": policy.engine_network})))
}

fn gateway_start(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let policy = a.policy();
    let Some(image) = policy.gateway_image.clone() else {
        return Err(err(
            POLICY_DENIED,
            format!(
                "this node's policy names no gateway_image ({})",
                policy.where_()
            ),
        ));
    };
    let name = arg(c, "name").as_str().unwrap_or("").to_string();
    let instance = arg(c, "instance").as_str().unwrap_or("").to_string();
    if !GATEWAY_NAME.is_match(&name) {
        return Err(err(
            BAD_ARGS,
            "gateway name must look like ec-<lowercase letters, digits, hyphens>",
        ));
    }
    if !INSTANCE.is_match(&instance) {
        return Err(err(
            BAD_ARGS,
            "instance must be 1-64 letters, digits, '_' or '-'",
        ));
    }
    let port = arg(c, "host_port").as_i64().unwrap_or(0);
    let (lo, hi) = policy.port_range;
    if !(i64::from(lo)..=i64::from(hi)).contains(&port) {
        return Err(err(
            POLICY_DENIED,
            format!("host port {port} is outside this node's port_range {lo}-{hi}"),
        ));
    }
    let engine_host = arg(c, "engine_host").as_str().unwrap_or("").to_string();
    let engine_port = arg(c, "engine_port").as_i64().unwrap_or(0);
    if !NGINX_HOST.is_match(&engine_host) {
        return Err(err(
            BAD_ARGS,
            "engine_host must be the engine container's name",
        ));
    }
    if !(1..=65535).contains(&engine_port) {
        return Err(err(BAD_ARGS, "engine_port must be 1-65535"));
    }
    let d = a.docker_for(&policy);
    let Some(doc) = d.inspect_one(&json!(engine_host))? else {
        return Err(err(
            NOT_FOUND,
            format!(
                "there is no container named {} to forward to",
                py::repr(&engine_host)
            ),
        ));
    };
    check_owned(&doc, &engine_host, None, None)?;
    if label(&labels_of(&doc), ROLE_LABEL) == Some("gateway") {
        return Err(err(
            BAD_ARGS,
            format!("{} is a gateway, not an engine", py::repr(&engine_host)),
        ));
    }
    if !(a.port_free)(u16::try_from(port).unwrap_or(0)) {
        return Err(err(
            PORT_IN_USE,
            format!("port {port} on 127.0.0.1 is already in use"),
        ));
    }
    ensure_network(&d, &policy.engine_network)?;
    let argv = gateway_argv(
        &d,
        &image,
        &name,
        &instance,
        port,
        &engine_host,
        engine_port,
    )?;
    let res = a.runner.run(&argv, 60.0, None);
    if !res.ok() {
        return Err(docker_error("create", &res).into());
    }
    let connected = d
        .ok(
            &["network", "connect", &policy.engine_network, &name],
            30.0,
            None,
        )
        .and_then(|_| d.ok(&["start", &name], 60.0, None));
    if let Err(e) = connected {
        let _ = d.call(&["rm", "-f", &name], 30.0, None);
        return Err(e.into());
    }
    Ok(o(json!({"id": last_line_id(&res.out)})))
}

// ── HTTP, files, models, scripts ─────────────────────────────────────────────────────────────────────────
fn models_scan(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    let roots: Vec<String> = arg(c, "roots")
        .as_array()
        .map(|r| {
            r.iter()
                .filter_map(Value::as_str)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let force = py::truthy(arg(c, "force"));
    let sc = scanner(a)?;
    let generation = sc.request(if roots.is_empty() { None } else { Some(roots) }, force);
    Ok(o(json!({"generation": generation})))
}

fn models_list(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    Ok(scanner(a)?.page(arg(c, "cursor").as_i64(), arg(c, "limit").as_i64()))
}

pub fn dispatch(a: &Agent, c: &Call<'_>) -> Result<Obj, HandlerError> {
    match c.op {
        "node.info" => Ok(node_info(a)),
        "metrics.configure" => Ok(metrics_configure(a, c)),
        "docker.render" => docker_render(a, c),
        "docker.run" => docker_run(a, c),
        "docker.stop" => docker_stop(a, c),
        "docker.start" => docker_start(a, c),
        "docker.rm" => docker_rm(a, c),
        "docker.inspect" => docker_inspect(a, c),
        "docker.logs" => docker_logs(a, c),
        "docker.logs_follow" => docker_logs_follow(a, c),
        "docker.image_present" => docker_image_present(a, c),
        "docker.images" => docker_images(a),
        "docker.pull" => docker_pull(a, c),
        "docker.external" => docker_external(a, c),
        "net.ensure_internal" => net_ensure_internal(a),
        "gateway.start" => gateway_start(a, c),
        "http.request" => {
            let monitor = a.policy().mode != "manage";
            crate::httpproxy::request(
                c.args,
                c.stream,
                &|refresh| a.allowed_ports(refresh),
                monitor,
            )
        }
        "fs.stat" => fs_stat(&a.policy(), c.args),
        "fs.write" => fs_write(&a.policy(), c.args, c.stream, &a.write_locks),
        "fs.read" => fs_read(&a.policy(), c.args, c.stream),
        "fs.symlink" => fs_symlink(&a.policy(), c.args),
        "fs.put_small" => fs_put_small(&a.policy(), c.args),
        "fs.delete" => fs_delete(&a.policy(), c.args, &|| a.running_mounts()),
        "models.scan" => models_scan(a, c),
        "models.list" => models_list(a, c),
        "models.config" => model_config(&a.policy(), arg(c, "root"), arg(c, "rel").as_str()),
        "scripts.run" => a.scripts.run(&a.policy(), c.args, c.stream, &a.env),
        op => Err(err(UNSUPPORTED, format!("not ported: {}", module_of(op)))),
    }
}
