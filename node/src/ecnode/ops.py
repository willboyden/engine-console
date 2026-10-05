"""One handler per protocol op (docs/FLEET-PROTOCOL.md §4).

A handler receives arguments that already went through proto.clean_args + proto.missing_args (server.py does
that, and refuses every op outside proto.MONITOR_OPS with `monitor_only` unless the policy says "manage"). It
returns the op's result dict or raises errors.OpError; policy checks happen here, after cleaning.
"""
from __future__ import annotations

import contextlib
import json
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import proto
from .agent import Agent
from .discovery import container_view
from .docker import (
    GATEWAY_NAME,
    INSTANCE,
    LABEL,
    NGINX_HOST,
    ROLE_LABEL,
    check_owned,
    child_env,
    container_name,
    docker_error,
    ensure_network,
    gateway_argv,
    labels_of,
    plan_run,
    remove_files,
    run_argv,
    write_files,
)
from .errors import Cancelled, OpError, Stream
from .fsops import fs_delete, fs_put_small, fs_read, fs_stat, fs_symlink, fs_write
from .httpproxy import request as http_request
from .models import model_config
from .policy import valid_image
from .util import fit_tail, iter_lines, kill_group, parse_time, run_tail, scrub, spawn

PULL_TIMEOUT_S = 4 * 3600
LOG_JSON_BUDGET = 900 * 1024        # docker.logs text, as encoded in the res frame (MAX_JSON is 1 MiB)


@dataclass
class Call:
    op: str
    args: dict[str, Any]
    stream: Stream


Handler = Callable[[Agent, Call], dict[str, Any]]


def _scanner(a: Agent) -> Any:
    if a.scanner is None:
        raise OpError(proto.UNSUPPORTED, "the model scanner is not running on this node")
    return a.scanner


# ── node / metrics ───────────────────────────────────────────────────────────────────────────────────────
def node_info(a: Agent, c: Call) -> dict[str, Any]:
    hello = a.hello(reload=True)
    hello.pop("type", None)
    return hello


def metrics_configure(a: Agent, c: Call) -> dict[str, Any]:
    a.configure_metrics(c.args.get("interval_s"), c.args.get("families") or {})
    return {}


# ── docker: runs ─────────────────────────────────────────────────────────────────────────────────────────
def docker_render(a: Agent, c: Call) -> dict[str, Any]:
    plan = plan_run(c.args["spec"], a.policy, a.gpu_uuids, a.paths.state)
    return {"argv": run_argv(a.docker(), plan)}


def docker_run(a: Agent, c: Call) -> dict[str, Any]:
    policy = a.policy
    d = a.docker(policy)
    plan = plan_run(c.args["spec"], policy, a.gpu_uuids, a.paths.state)
    ensure_network(d, policy.engine_network)
    write_files(plan, a.paths.state)
    res = a.runner(run_argv(d, plan), timeout=120, env=child_env(plan.secret_env))
    if not res.ok:
        raise docker_error("run", res)
    lines = res.out.strip().splitlines()
    return {"id": lines[-1].strip()[:64] if lines else ""}


def _owned(a: Agent, c: Call) -> tuple[Any, dict[str, Any]]:
    name = container_name(c.args["name"])
    d = a.docker()
    doc = d.inspect_one(name)
    if doc is None:
        raise OpError(proto.NOT_FOUND, f"there is no container named {name!r} on this machine")
    check_owned(doc, name, role=c.args.get("role"), instance=c.args.get("instance"))
    return d, doc


def docker_stop(a: Agent, c: Call) -> dict[str, Any]:
    d, doc = _owned(a, c)
    t = c.args.get("timeout_s")
    t = 30 if t is None else max(0, min(int(t), 600))
    d.ok("stop", "-t", str(t), str(doc["Id"]), timeout=t + 30)
    return {}


def docker_start(a: Agent, c: Call) -> dict[str, Any]:
    d, doc = _owned(a, c)
    d.ok("start", str(doc["Id"]), timeout=60)
    return {}


def docker_rm(a: Agent, c: Call) -> dict[str, Any]:
    d, doc = _owned(a, c)
    d.ok("rm", "-f", str(doc["Id"]), timeout=60)
    labels = labels_of(doc)
    if labels.get(ROLE_LABEL) == "engine":
        remove_files(a.paths.state, labels.get("engine-console.instance", ""))
    return {}


def docker_inspect(a: Agent, c: Call) -> dict[str, Any]:
    name = container_name(c.args["name"])
    doc = a.docker().inspect_one(name)
    if doc is None:
        raise OpError(proto.NOT_FOUND, f"there is no container named {name!r} on this machine")
    roots = [(r, a.disk.resolve(r.path)) for r in a.policy.model_roots]
    view = container_view(doc, roots=roots, gpu_by_index=a.gpu_by_index())
    if view is None:
        raise OpError(proto.NOT_FOUND, f"docker returned nothing usable for {name!r}")
    return {"info": view.info}


def _tail(c: Call) -> int:
    t = c.args.get("tail")
    return 200 if t is None else max(1, min(int(t), 5000))


def docker_logs(a: Agent, c: Call) -> dict[str, Any]:
    d, doc = _owned(a, c)
    rc, data, truncated = run_tail(d.argv("logs", "--tail", str(_tail(c)), str(doc["Id"])), timeout=30,
                                   max_bytes=proto.MAX_LOG_TEXT)
    if rc != 0:
        raise OpError(proto.DOCKER_ERROR, f"docker logs failed: {scrub(data.decode('utf-8', 'replace'))[-300:]}")
    text = scrub(data.decode("utf-8", "replace"))[-proto.MAX_LOG_TEXT:]
    text, trimmed = fit_tail(text, LOG_JSON_BUDGET)
    return {"text": text, "truncated": truncated or trimmed}


def docker_logs_follow(a: Agent, c: Call) -> dict[str, Any]:
    d, doc = _owned(a, c)
    try:
        proc = spawn(d.argv("logs", "-f", "--tail", str(_tail(c)), str(doc["Id"])))
    except OSError as e:
        raise OpError(proto.DOCKER_ERROR, f"cannot run docker logs: {e.strerror}") from None
    c.stream.on_cancel(lambda: kill_group(proc, 2.0))
    count = 0
    try:
        assert proc.stdout is not None
        for line in iter_lines(proc.stdout, 8192):
            c.stream.event({"line": scrub(line)})
            count += 1
    finally:
        kill_group(proc, 2.0)
    if c.stream.cancelled:
        raise Cancelled
    return {"lines": count}


# ── docker: images ───────────────────────────────────────────────────────────────────────────────────────
def _image(c: Call) -> str:
    image = c.args["image"]
    if not valid_image(image):
        raise OpError(proto.BAD_ARGS, f"{str(image)[:100]!r} is not a valid image reference")
    return str(image)


def docker_image_present(a: Agent, c: Call) -> dict[str, Any]:
    image = _image(c)
    res = a.docker().call("image", "inspect", "--format", "{{.Id}}", image, timeout=20)
    if res.ok:
        return {"present": True, "id": res.out.strip()[:80] or None}
    text = (res.err + res.out).lower()
    if "no such image" in text or "not found" in text:
        return {"present": False, "id": None}
    raise docker_error("image inspect", res)


def docker_images(a: Agent, c: Call) -> dict[str, Any]:
    d = a.docker()
    ids = list(dict.fromkeys(d.ok("image", "ls", "-q", "--no-trunc", timeout=30).out.split()))[:512]
    if not ids:
        return {"images": []}
    res = d.call("image", "inspect", *ids, timeout=60)
    try:
        docs = json.loads(res.out) if res.out.strip() else []
    except ValueError:
        raise OpError(proto.DOCKER_ERROR, "docker image inspect returned something that is not JSON") from None
    out: list[dict[str, Any]] = []
    for doc in docs if isinstance(docs, list) else []:
        if not isinstance(doc, dict):
            continue
        refs = doc.get("RepoTags") or doc.get("RepoDigests") or ["<none>"]
        for ref in refs if isinstance(refs, list) else []:
            if len(out) >= 512:
                break
            size = doc.get("Size")
            out.append({"ref": str(ref)[:256], "id": str(doc.get("Id") or "")[:80],
                        "size": size if isinstance(size, int) and not isinstance(size, bool) else None,
                        "created": parse_time(doc.get("Created")), "arch": str(doc.get("Architecture") or "")[:16]})
    return {"images": out}


def docker_pull(a: Agent, c: Call) -> dict[str, Any]:
    policy = a.policy
    image = _image(c)
    if not policy.allow_pull:
        raise OpError(proto.POLICY_DENIED, f"this node's policy does not allow pulling images (allow_pull in "
                                           f"{policy.where()})")
    if not policy.image_allowed(image):
        raise OpError(proto.POLICY_DENIED, f"image {image!r} is not allowed by this node's policy (engine_images)")
    d = a.docker(policy)
    try:
        proc = spawn(d.argv("pull", image))
    except OSError as e:
        raise OpError(proto.DOCKER_ERROR, f"cannot run docker pull: {e.strerror}") from None
    c.stream.on_cancel(lambda: kill_group(proc, 2.0))
    last = ""
    timer = threading.Timer(PULL_TIMEOUT_S, kill_group, args=(proc, 2.0))
    timer.daemon = True
    timer.start()
    try:
        assert proc.stdout is not None
        for line in iter_lines(proc.stdout, 1024):
            last = line or last
            c.stream.event({"line": scrub(line)})
        rc = proc.wait()
    finally:
        timer.cancel()
        kill_group(proc, 2.0)
    if c.stream.cancelled:
        raise Cancelled
    if rc != 0:
        raise OpError(proto.DOCKER_ERROR, f"docker pull failed: {scrub(last)[:300]}")
    res = d.call("image", "inspect", "--format", "{{.Id}}", image, timeout=20)
    return {"id": res.out.strip()[:80] if res.ok else None}


# ── docker: engines the console did not create ──────────────────────────────────────────────────────────
def docker_external(a: Agent, c: Call) -> dict[str, Any]:
    policy = a.policy
    if policy.external_control != "stop_start":
        raise OpError(proto.POLICY_DENIED, "this node's policy does not let the console stop or start engines it "
                                           f"did not create (external_control in {policy.where()})")
    action = c.args["action"]
    if action not in ("stop", "start"):
        raise OpError(proto.BAD_ARGS, 'action must be "stop" or "start"')
    name = container_name(c.args["name"])
    d = a.docker(policy)
    doc = d.inspect_one(name)
    if doc is None:
        raise OpError(proto.NOT_FOUND, f"there is no container named {name!r} on this machine")
    if labels_of(doc).get(LABEL) == "1":
        raise OpError(proto.POLICY_DENIED, f"{name!r} is one of the console's own containers: use docker.stop / "
                                           "docker.start")
    view = container_view(doc)
    if view is None or not view.info["engine"]:
        raise OpError(proto.POLICY_DENIED, f"{name!r} is not recognised as an inference engine")
    owner = next((s for s in policy.scripts if s.matches(view.info["name"])), None)
    if owner is not None:
        raise OpError(proto.POLICY_DENIED, f"{name!r} is managed by the script {owner.name!r}; use its actions")
    if action == "stop":
        d.ok("stop", "-t", "30", str(doc["Id"]), timeout=90)
    else:
        d.ok("start", str(doc["Id"]), timeout=90)
    return {}


# ── network and gateway ──────────────────────────────────────────────────────────────────────────────────
def net_ensure_internal(a: Agent, c: Call) -> dict[str, Any]:
    policy = a.policy
    ensure_network(a.docker(policy), policy.engine_network)
    return {"network": policy.engine_network}


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def gateway_start(a: Agent, c: Call) -> dict[str, Any]:
    policy = a.policy
    if not policy.gateway_image:
        raise OpError(proto.POLICY_DENIED, f"this node's policy names no gateway_image ({policy.where()})")
    name, instance = c.args["name"], c.args["instance"]
    if not GATEWAY_NAME.fullmatch(name):
        raise OpError(proto.BAD_ARGS, "gateway name must look like ec-<lowercase letters, digits, hyphens>")
    if not INSTANCE.fullmatch(instance):
        raise OpError(proto.BAD_ARGS, "instance must be 1-64 letters, digits, '_' or '-'")
    port = c.args["host_port"]
    lo, hi = policy.port_range
    if not lo <= port <= hi:
        raise OpError(proto.POLICY_DENIED, f"host port {port} is outside this node's port_range {lo}-{hi}")
    engine_host, engine_port = c.args["engine_host"], c.args["engine_port"]
    if not NGINX_HOST.fullmatch(engine_host):
        raise OpError(proto.BAD_ARGS, "engine_host must be the engine container's name")
    if not 1 <= engine_port <= 65535:
        raise OpError(proto.BAD_ARGS, "engine_port must be 1-65535")
    d = a.docker(policy)
    doc = d.inspect_one(engine_host)
    if doc is None:
        raise OpError(proto.NOT_FOUND, f"there is no container named {engine_host!r} to forward to")
    check_owned(doc, engine_host)
    if labels_of(doc).get(ROLE_LABEL) == "gateway":
        raise OpError(proto.BAD_ARGS, f"{engine_host!r} is a gateway, not an engine")
    if not port_free(port):
        raise OpError(proto.PORT_IN_USE, f"port {port} on 127.0.0.1 is already in use")
    ensure_network(d, policy.engine_network)
    res = a.runner(gateway_argv(d, image=policy.gateway_image, name=name, instance=instance, host_port=port,
                                engine_host=engine_host, engine_port=engine_port), timeout=60)
    if not res.ok:
        raise docker_error("create", res)
    try:
        d.ok("network", "connect", policy.engine_network, name, timeout=30)
        d.ok("start", name, timeout=60)
    except OpError:
        with contextlib.suppress(OpError):
            d.call("rm", "-f", name, timeout=30)
        raise
    lines = res.out.strip().splitlines()
    return {"id": lines[-1].strip()[:64] if lines else ""}


# ── HTTP, files, models, scripts ─────────────────────────────────────────────────────────────────────────
def http_req(a: Agent, c: Call) -> dict[str, Any]:
    return http_request(c.args, c.stream, a.allowed_ports, monitor=a.policy.mode != "manage")


def models_scan(a: Agent, c: Call) -> dict[str, Any]:
    roots = [r for r in c.args.get("roots") or [] if r]
    return {"generation": _scanner(a).request(roots or None, bool(c.args.get("force")))}


def models_list(a: Agent, c: Call) -> dict[str, Any]:
    page: dict[str, Any] = _scanner(a).page(c.args.get("cursor"), c.args.get("limit"))
    return page


HANDLERS: dict[str, Handler] = {
    "node.info": node_info,
    "metrics.configure": metrics_configure,
    "docker.render": docker_render,
    "docker.run": docker_run,
    "docker.stop": docker_stop,
    "docker.start": docker_start,
    "docker.rm": docker_rm,
    "docker.inspect": docker_inspect,
    "docker.logs": docker_logs,
    "docker.logs_follow": docker_logs_follow,
    "docker.image_present": docker_image_present,
    "docker.images": docker_images,
    "docker.pull": docker_pull,
    "docker.external": docker_external,
    "net.ensure_internal": net_ensure_internal,
    "gateway.start": gateway_start,
    "http.request": http_req,
    "fs.stat": lambda a, c: fs_stat(a.policy, c.args),
    "fs.write": lambda a, c: fs_write(a.policy, c.args, c.stream, a.write_locks),
    "fs.read": lambda a, c: fs_read(a.policy, c.args, c.stream),
    "fs.symlink": lambda a, c: fs_symlink(a.policy, c.args),
    "fs.put_small": lambda a, c: fs_put_small(a.policy, c.args),
    "fs.delete": lambda a, c: fs_delete(a.policy, c.args, a.running_mounts),
    "models.scan": models_scan,
    "models.list": models_list,
    "models.config": lambda a, c: model_config(a.policy, c.args["root"], c.args["rel"]),
    "scripts.run": lambda a, c: a.scripts.run(a.policy, c.args, c.stream),
}
