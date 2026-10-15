"""Human-facing commands: `ecnode check` (what this machine would report) and `ecnode policy-template`.

Both are read-only. `check` never prints an environment value, a container's command line or a secret: only
counts, names, sizes and the policy summary. `policy-template` prints a starter MONITOR-mode policy; it writes
nothing (install.sh decides whether to save it).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from collections.abc import Callable, Mapping
from typing import IO, Any

from . import __version__, proto
from .agent import Agent
from .disks import mount_of
from .docker import Docker
from .gpus import Gpus
from .policy import DEFAULT_DISCOVERY, DEFAULT_IMAGES, DEFAULT_NETWORK, DEFAULT_PORTS, PolicyStore, parse
from .util import Runner, run

GIB = 1024**3
# Written next to engine_images in the template: tag patterns are a starting point, digest pins are the goal.
IMAGES_NOTE = ("repository:* allows any tag of that repository, and a tag names whatever image was last pulled under "
               "it. Prefer exact references pinned by digest (repository@sha256:<digest>) for the images you have "
               "reviewed, and pin the same references for this machine in the console (host_image_pins). The console "
               "refers to llama.cpp by digest already.")


def _gib(n: object) -> str:
    if not isinstance(n, (int, float)) or isinstance(n, bool):
        return "?"
    return f"{n / GIB:,.1f} GiB" if n < 1024 * GIB else f"{n / GIB / 1024:,.2f} TiB"


def check(policy_path: str | None, out: IO[str], *, agent: Agent | None = None) -> int:
    """Print what this machine would report to a hub. Exit 1 when a policy file exists but was refused."""
    agent = agent or Agent(PolicyStore(policy_path))
    hello = agent.hello()
    pol = hello["policy"]

    def p(line: str = "") -> None:
        print(line, file=out)

    host = hello["host"]
    p(f"ecnode {__version__} · protocol {proto.PROTO} · Python {host['python']} · node id {hello['node_id']}")
    cores = f"{host['cpu_cores']} cores / {host['cpu_threads']} threads" if host.get("cpu_threads") else ""
    p(f"host     {host['hostname']} ({host['arch']}) · {host.get('os') or 'unknown OS'} · kernel {host['kernel']}")
    p(f"         {host.get('cpu_model') or 'unknown CPU'} · {cores} · RAM {_gib(host.get('ram_total_bytes'))}")
    g = hello["gpu"]
    p(f"GPUs     {len(hello['gpus'])} via {g['source']} · driver {g.get('driver') or '-'} · CUDA {g.get('cuda') or '-'}"
      + (f" · {g['error']}" if g.get("error") else ""))
    for gpu in hello["gpus"]:
        mem = "unified memory" if gpu.get("unified_memory") else _gib(gpu.get("total_bytes"))
        p(f"  {gpu['index']}  {gpu['name']} · {mem} · CC {gpu.get('compute_capability') or '?'} · {gpu['uuid']}")
    d = hello["docker"]
    if d.get("available"):
        p(f"docker   {d.get('version')} · context {d.get('context') or 'default'} · rootless "
          f"{'yes' if d.get('rootless') else 'no'} · nvidia runtime {'yes' if d.get('nvidia_runtime') else 'no'}")
    else:
        p(f"docker   not available: {d.get('error') or 'unknown reason'}")
    p(f"policy   {pol.get('path') or '(none)'} · mode {str(pol['mode']).upper()}")
    if pol.get("error"):
        p(f"  ! {pol['error']}")
    p(f"  engine images: {', '.join(pol['engine_images']) or '(none: nothing can be launched)'}")
    p(f"  gateway image: {pol.get('gateway_image') or '(none)'} · network {pol['engine_network']} · ports "
      f"{pol['port_range'][0]}-{pol['port_range'][1]} · discovery ports {', '.join(map(str, pol['discovery_ports']))}")
    p(f"  pull {'yes' if pol['allow_pull'] else 'no'} · delete {'yes' if pol['allow_delete'] else 'no'} · ipc host "
      f"{'yes' if pol['allow_ipc_host'] else 'no'} · external control {pol['external_control']} · max shm "
      f"{pol['max_shm_gib']:g} GiB")
    disks = agent.disk.report(agent.policy.model_roots)["mounts"]
    p("roots" + ("" if pol["model_roots"] else "    (none)"))
    mounts = agent.disk.mounts()
    for r in pol["model_roots"]:
        real = agent.disk.resolve(r["path"])
        exists = os.path.isdir(real)
        m = mount_of(real, mounts)
        info = next((x for x in disks if m is not None and x["mount"] == m.mount), None)
        where = "missing on this machine" if not exists else (
            f"on {info['mount']} ({info['fstype']}"
            + (f" from {info['remote_host']}:{info['remote_path']}" if info.get("remote") else "")
            + (f", {info['model']}" if info.get("model") else "") + f") · {_gib(info.get('free'))} free of "
            f"{_gib(info.get('size'))}" if info else "mount unknown")
        p(f"  {r['id']:<10} {r['kind']:<8} {r['path']}  ({'writable' if r['writable'] else 'read-only'})  {where}")
    if pol["scripts"]:
        p("scripts  " + "; ".join(f"{s['name']} ({', '.join(a['name'] for a in s['actions'])})"
                                  for s in pol["scripts"]))
    inv = agent.refresh_inventory()
    containers = inv.msg["containers"]
    engines = [c for c in containers if c.get("engine") and c.get("role") != "gateway"]
    owned = [c for c in containers if c.get("managed")]
    eps = inv.msg["endpoints"]
    p(f"running  {len([c for c in containers if c.get('state') == 'running'])} containers ({len(engines)} engines, "
      f"{len(owned)} console-owned) · {len(eps)} endpoint(s)"
      + (f" on ports {', '.join(str(e['port']) for e in eps)}" if eps else "")
      + (f" · docker: {inv.error}" if inv.error else ""))
    return 1 if pol.get("error") and pol.get("path") else 0


# ── policy-template ──────────────────────────────────────────────────────────────────────────────────────
def candidate_roots(env: Mapping[str, str] | None = None, home: str | None = None,
                    isdir: Callable[[str], bool] = os.path.isdir) -> list[dict[str, Any]]:
    """Model directories that exist on this machine. HF caches are listed as their hub/ directory and Ollama
    homes as their models/ directory, so files next to them (tokens, keys, history) are outside every root."""
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    out: list[dict[str, Any]] = []

    def add(rid: str, path: str, kind: str, label: str) -> None:
        path = os.path.normpath(path)
        if any(r["path"] == path for r in out) or any(r["id"] == rid for r in out):
            return
        try:
            parse({"model_roots": [{"id": rid, "path": path, "kind": kind}]}, None, home=home)
        except ValueError:
            return
        out.append({"id": rid, "path": path, "kind": kind, "writable": False, "label": label})

    hf_home = env.get("HF_HOME") or os.path.join(home, ".cache", "huggingface")
    hub = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE") or os.path.join(hf_home, "hub")
    if isdir(hub):
        add("hf", hub, "hf-hub", "Hugging Face cache")
    for i, base in enumerate((env.get("OLLAMA_MODELS"), os.path.join(home, ".ollama", "models"),
                              "/usr/share/ollama/.ollama/models", "/var/lib/ollama/models")):
        if base and isdir(os.path.join(base, "manifests")):
            add("ollama" if i == 0 or not any(r["kind"] == "ollama" for r in out) else f"ollama{i}", base,
                "ollama", "Ollama models")
    for path in (os.path.join(home, "models"), "/models", "/data/models", "/data/hf", "/data/gguf", "/srv/models",
                 "/mnt/models", "/opt/models"):
        if isdir(path):
            rid = os.path.basename(path).lower().replace(".", "-")[:32] or "models"
            add(rid if not any(r["id"] == rid for r in out) else f"{rid}{len(out)}", path, "auto", path)
    return out


def _gateway_digest(d: Docker) -> str | None:
    """A local nginx image pinned by digest, if there is one (the operator still has to read and accept it)."""
    res = d.call("images", "--digests", "--format", "{{.Repository}}@{{.Digest}}", "nginx", timeout=10)
    if not res.ok:
        return None
    for line in res.out.splitlines():
        repo, _, digest = line.strip().partition("@")
        if repo == "nginx" and digest.startswith("sha256:") and len(digest) == 71:
            return f"nginx@{digest}"
    return None


def detect_context(runner: Runner = run) -> str | None:
    """The docker context the node should use. A `rootless` context wins over "default": DOCKER_HOST often points
    the interactive shell at the rootless daemon, but the hub's ssh session (a scrubbed environment) has no
    DOCKER_HOST, and "default" would then mean the rootful socket."""
    shown = runner(["docker", "context", "show"], timeout=5)
    current = shown.out.strip() if shown.ok else ""
    if current and current != "default":
        return current
    listed = runner(["docker", "context", "ls", "--format", "{{.Name}}"], timeout=5)
    names = {line.strip().rstrip(" *") for line in listed.out.splitlines()} if listed.ok else set()
    return "rootless" if "rootless" in names else None


def policy_template(out: IO[str], *, runner: Runner = run, gpus: Gpus | None = None,
                    env: Mapping[str, str] | None = None, home: str | None = None) -> int:
    context = detect_context(runner)
    static, _info = (gpus or Gpus()).static()
    template: dict[str, Any] = {
        "_comment": ("Starter policy from `ecnode policy-template` "
                     f"({_dt.date.today().isoformat()}, ecnode {__version__}). MONITOR mode: Engine Console can "
                     "watch this machine but change nothing. Check every root, then set \"mode\": \"manage\" and "
                     "\"writable\": true where downloads may land. Reference: docs/FLEET-PROTOCOL.md section 6. "
                     "Keys starting with _ are comments."),
        "_detected_gpus": [f"{g['index']}: {g['name']} ({g['uuid']})" for g in static],
        "version": 1,
        "mode": "monitor",
        "docker": {"binary": "docker", "context": context},
        "_engine_images": IMAGES_NOTE,
        "engine_images": list(DEFAULT_IMAGES),
        "gateway_image": _gateway_digest(Docker("docker", context, runner=runner)),
        "engine_network": DEFAULT_NETWORK,
        "port_range": list(DEFAULT_PORTS),
        "discovery_ports": list(DEFAULT_DISCOVERY),
        "model_roots": candidate_roots(env, home),
        "scripts": [],
        "allow_pull": False,
        "allow_delete": False,
        "allow_ipc_host": True,
        "external_control": "none",
        "max_shm_gib": 64,
    }
    parse(template, None, home=home)    # the template must load as written
    print(json.dumps(template, indent=2), file=out)
    return 0
