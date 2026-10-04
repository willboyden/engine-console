"""Hub-side docker helpers: names, labels and the "copy as command" snippets. Pure functions.

The hub never runs docker (contract v2, §10.1 decision 14): every container action is a node operation
(`docker.run`, `gateway.start`, `docker.stop`, ...). The node renders the exact `docker run` argv it would use
(`docker.render`), and this module wraps it, and builds the compose file and the bare engine command from the
same run spec. Secrets are never in a snippet: a secret variable appears as a name only."""
from __future__ import annotations

import os
import shlex
from typing import Any

import yaml

LABEL = "engine-console"
ROLE_LABEL = "engine-console.role"
INSTANCE_LABEL = "engine-console.instance"


def gateway_name(container_name: str) -> str:
    """The per-instance gateway sidecar (the only container with a published port) is `<engine container>-gw`."""
    return f"{container_name}-gw"


def clean_render(argv: list[str]) -> str:
    """The node's rendered argv as one copyable `docker run ...` line.

    Accepts either `["run", ...]` or `["docker", "--context", "x", "run", ...]` (the node's choice): the binary is
    normalised to `docker` and a `--context` pair is dropped, because the context belongs to that machine."""
    toks = [str(a) for a in argv]
    if toks and os.path.basename(toks[0]) == "docker":
        toks = toks[1:]
    while toks[:1] == ["--context"] and len(toks) >= 2:
        toks = toks[2:]
    return shlex.join(["docker", *toks])


def command_snippets(spec: dict[str, Any], rendered: list[str] | None, *, host_paths: dict[str, str] | None = None,
                     network: str | None = None) -> dict[str, str]:
    """`docker_run`, `compose_yaml` and `engine_cli` for a run spec (secret_env values must already be absent).

    `host_paths` maps a mount's (root id) to that root's absolute path on the host, so a compose volume can name the
    real directory; a mount whose root is unknown is left out of the compose file rather than guessed."""
    paths = host_paths or {}
    env: dict[str, str] = dict(spec.get("env") or {})
    env.update({k: "${" + k + "}" for k in (spec.get("secret_env") or {})})
    volumes: list[str] = []
    for m in spec.get("mounts") or []:
        root = paths.get(str(m.get("root")))
        if root is None:
            continue
        host = root.rstrip("/") + ("/" + str(m["rel"]) if m.get("rel") else "")
        volumes.append(f"{host}:{m['dest']}{':ro' if m.get('ro', True) else ''}")
    volumes += [f"{v['name']}:{v['dest']}" for v in spec.get("volumes") or []]
    net = network or "engine-console-engines"
    service: dict[str, Any] = {
        "image": spec.get("image"), "command": list(spec.get("argv") or []), "restart": "no",
        "ipc": "host" if spec.get("ipc_host") else "private", "shm_size": spec.get("shm_size") or "16g",
        "environment": env, "volumes": volumes, "labels": {LABEL: "1", **(spec.get("labels") or {})},
        "networks": [net], "security_opt": ["no-new-privileges:true"], "cap_drop": ["ALL"], "pids_limit": 4096,
        "deploy": {"resources": {"reservations": {"devices": [
            {"driver": "nvidia", "device_ids": list(spec.get("gpu_uuids") or []), "capabilities": ["gpu"]}]}}},
    }
    compose = {"services": {spec.get("name") or "engine": service}, "networks": {net: {"external": True}}}
    text = yaml.safe_dump(compose, sort_keys=False)
    files = sorted((spec.get("files") or {}).keys())
    if files:   # the node writes these next to the instance and mounts them read-only; compose cannot know their content
        text = f"# generated config files the console mounts: {', '.join(files)}\n" + text
    docker_run = clean_render(rendered) if rendered else ""
    return {"docker_run": docker_run, "compose_yaml": text, "engine_cli": shlex.join(list(spec.get("argv") or []))}
