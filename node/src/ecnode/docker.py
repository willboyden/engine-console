"""Docker through the `docker` CLI only: argv lists, timeouts, `--context` from the policy, never a shell.

What this module builds is what the node runs (docker.run) and what it shows (docker.render): the same argv,
never a secret value in it. Every engine container gets `--cap-drop ALL`, `no-new-privileges`, `--pids-limit
4096`, `--restart no`, `--pull never` (pulls are a separate, policy-gated op) and the policy's internal engine
network. The node changes only containers labelled `engine-console=1` (docs/FLEET-PROTOCOL.md §4 and §5).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import proto
from .confine import open_root, resolve_existing, split_rel
from .errors import OpError
from .policy import Policy, valid_image
from .util import CmdResult, Runner, clean_text, is_secret_name, run, scrub

LABEL = "engine-console"
ROLE_LABEL = "engine-console.role"
INSTANCE_LABEL = "engine-console.instance"
ENGINE_LABEL = "engine-console.engine"
GATEWAY_LISTEN = 8080
# constant shell text: the only variable input (the nginx config) travels in an env var, never in this string
GATEWAY_SH = 'printf "%s\\n" "$NGINX_CONF" > /tmp/nginx.conf && exec nginx -c /tmp/nginx.conf'  # noqa: S108

CONTAINER_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
RUN_NAME = re.compile(r"ec-[a-z0-9][a-z0-9-]{0,58}")
GATEWAY_NAME = re.compile(r"ec-[a-z0-9][a-z0-9-]{0,60}")
INSTANCE = re.compile(r"[A-Za-z0-9_-]{1,64}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
VOLUME = re.compile(r"ec-[a-z0-9][a-z0-9_.-]{0,60}")
SHM = re.compile(r"([0-9]{1,4})([kmg])")
LABEL_KEY = re.compile(r"engine-console\.[a-z0-9][a-z0-9_.-]{0,62}")
NGINX_HOST = re.compile(r"[a-z0-9][a-z0-9_.-]{0,62}")
# A plain `env` name that looks secret would put its value on the docker command line: it belongs in secret_env.
# Same rule the hub uses to split them, so a hub-built spec never trips it. A secret_env name must match it too:
# the hub sends nothing else there, and the rule keeps loader, runtime and tracing names out of docker's environment.
ENV_LOOKS_SECRET = re.compile(r"(key|token|secret|password)", re.I)
# secret_env values are set in the environment of the docker CLI process itself (util.child_env builds the rest from
# scratch). Names that would change what that process (or its loader, its Go runtime or its built-in OpenTelemetry
# exporter) does are refused: nothing a hub sends may run code, redirect docker or send data elsewhere from the node.
CHILD_ENV_DENY = re.compile(
    r"^(?:LD_|DYLD_|DOCKER_|XDG_|LC_|BUILDKIT_|COMPOSE_|PYTHON|NOTARY_|SSL_|OPENSSL_|GCONV_|MALLOC_|GLIBC_|"
    r"GIT_|SSH_|KRB5|NIX_|OTEL_|GRPC_)|^(?:PATH|HOME|USER|LOGNAME|SHELL|TMPDIR|TMP|TEMP|PWD|OLDPWD|IFS|ENV|"
    r"BASH_ENV|CDPATH|LANG|LANGUAGE|TZ|TZDIR|TERM|HTTP_PROXY|HTTPS_PROXY|NO_PROXY|ALL_PROXY|FTP_PROXY|HOSTALIASES|"
    r"LOCALDOMAIN|RES_OPTIONS|NLSPATH|LOCPATH|SSLKEYLOGFILE|ZONEINFO|GODEBUG|GOGC|GOMAXPROCS|GOMEMLIMIT|"
    r"GOTRACEBACK|GORACE|GOFLAGS|GOROOT|GOPATH|GOENV|GOTOOLCHAIN|GOCOVERDIR)$", re.I)
MAX_FILES = 4
DEFAULT_SOCKET = "/var/run/docker.sock"


def effective_context(binary: str, configured: str | None, *, runner: Runner = run,
                      env: Mapping[str, str] | None = None, default_socket: str = DEFAULT_SOCKET) -> str | None:
    """The `--context` to pass. The policy's context when it names one. Otherwise the CLI's own choice, with one
    exception: a node started with a scrubbed environment (ssh forced command, hub subprocess) has no DOCKER_HOST,
    so on a rootless-only machine "default" means a rootful socket this user cannot use. Then an existing
    `rootless` context is used instead. The default socket is only checked for access, never connected to."""
    if configured:
        return configured
    env = os.environ if env is None else env
    if env.get("DOCKER_HOST") or env.get("DOCKER_CONTEXT"):
        return None                       # the operator pointed the CLI somewhere: follow it
    shown = runner([binary, "context", "show"], timeout=5)
    if not shown.ok or shown.out.strip() != "default":
        return None                       # no CLI, or a current context chosen in the CLI config
    if os.access(default_socket, os.W_OK):
        return None                       # a default daemon this user may use
    listed = runner([binary, "context", "ls", "--format", "{{.Name}}"], timeout=5)
    names = {line.strip().rstrip(" *") for line in listed.out.splitlines()} if listed.ok else set()
    return "rootless" if "rootless" in names else None


class Docker:
    def __init__(self, binary: str = "docker", context: str | None = None, *, runner: Runner = run) -> None:
        self.binary = binary
        self.context = context
        self._run = runner

    @classmethod
    def from_policy(cls, policy: Policy, runner: Runner = run) -> Docker:
        return cls(policy.docker_binary, policy.docker_context, runner=runner)

    def argv(self, *args: str) -> list[str]:
        ctx = ["--context", self.context] if self.context else []
        return [self.binary, *ctx, *args]

    def call(self, *args: str, timeout: float = 30.0, env: Mapping[str, str] | None = None) -> CmdResult:
        return self._run(self.argv(*args), timeout=timeout, env=env)

    def ok(self, *args: str, timeout: float = 30.0, env: Mapping[str, str] | None = None) -> CmdResult:
        res = self.call(*args, timeout=timeout, env=env)
        if not res.ok:
            raise docker_error(args[0] if args else "", res)
        return res

    # ── reads ────────────────────────────────────────────────────────────────────────────────────────────
    def inspect(self, refs: Sequence[str]) -> list[dict[str, Any]]:
        """Raw `docker container inspect` documents (untrusted; never keep Config.Env). Missing ones are skipped."""
        if not refs:
            return []
        res = self.call("container", "inspect", *refs, timeout=30)
        try:
            docs = json.loads(res.out) if res.out.strip() else []
        except ValueError:
            docs = []
        if not res.ok and not docs and not _missing(res):
            raise docker_error("inspect", res)
        return [d for d in docs if isinstance(d, dict)] if isinstance(docs, list) else []

    def inspect_one(self, ref: str) -> dict[str, Any] | None:
        if not CONTAINER_REF.fullmatch(ref or ""):
            raise OpError(proto.BAD_ARGS, f"{str(ref)[:130]!r} is not a container name")
        docs = self.inspect([ref])
        return docs[0] if docs else None

    def ps(self, *, all_: bool = False, label: str | None = None) -> list[dict[str, str]]:
        """[{id, name, state, status}] from `docker ps` (no command column: it can hold secrets)."""
        args = ["ps", "--no-trunc", "--format", "{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Status}}"]
        if all_:
            args.insert(1, "-a")
        if label:
            args[1:1] = ["--filter", f"label={label}"]
        res = self.ok(*args, timeout=20)
        out = []
        for line in res.out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 4 and parts[0]:
                out.append({"id": parts[0], "name": parts[1].split(",")[0], "state": parts[2], "status": parts[3]})
        return out

    def facts(self) -> dict[str, Any]:
        """hello.docker: {available, context, rootless, version, nvidia_runtime, error}."""
        res = self.call("info", "--format", "{{json .}}", timeout=8)
        out: dict[str, Any] = {"available": False, "context": self.context, "rootless": None, "version": None,
                               "nvidia_runtime": None, "error": None}
        if res.rc == 127:
            out["error"] = f"the docker CLI ({self.binary}) is not installed or not on PATH"
            return out
        info: Any = None
        try:
            info = json.loads(res.out) if res.out.strip() else None
        except ValueError:
            info = None
        if not isinstance(info, dict):
            out["error"] = _first_line(res) or "docker info gave no answer"
            return out
        errors = info.get("ServerErrors")
        if isinstance(errors, list) and errors:
            out["error"] = clean_text(scrub(str(errors[0])), 300)
            return out
        sec = info.get("SecurityOptions") or []
        runtimes = info.get("Runtimes") or {}
        out.update({"available": res.ok, "version": clean_text(info.get("ServerVersion") or "", 64) or None,
                    "rootless": any("rootless" in str(s) for s in sec) if isinstance(sec, list) else None,
                    "nvidia_runtime": "nvidia" in runtimes if isinstance(runtimes, dict) else None})
        if not res.ok:
            out["error"] = _first_line(res)
        if not self.context:
            ctx = self.call("context", "show", timeout=5)
            out["context"] = clean_text(ctx.out.strip(), 64) or None if ctx.ok else None
        return out


def _first_line(res: CmdResult) -> str:
    text = (res.err or res.out).strip().splitlines()
    return clean_text(scrub(text[0] if text else f"exit status {res.rc}"), 300)


def _missing(res: CmdResult) -> bool:
    t = (res.err + res.out).lower()
    return "no such container" in t or "no such object" in t or "not found" in t


def docker_error(verb: str, res: CmdResult) -> OpError:
    """A docker failure as a protocol error with a sentence a person can act on (never a secret)."""
    text = (res.err + "\n" + res.out).lower()
    line = _first_line(res)
    if res.rc == 124:
        return OpError(proto.TIMEOUT, f"docker {verb} did not finish in time; it may or may not have taken effect")
    if res.rc == 127:
        return OpError(proto.DOCKER_ERROR, "the docker CLI is not installed on this machine (or not on PATH)")
    if "cannot connect to the docker daemon" in text or "permission denied while trying to connect" in text:
        return OpError(proto.DOCKER_ERROR, f"docker {verb}: cannot reach the docker daemon ({line})")
    if "no such container" in text or "no such object" in text:
        return OpError(proto.NOT_FOUND, f"docker {verb}: no such container ({line})")
    if "no such image" in text or "unable to find image" in text or "manifest unknown" in text:
        return OpError(proto.NOT_FOUND, f"docker {verb}: the image is not on this machine; pull it first ({line})")
    if "port is already allocated" in text or "address already in use" in text:
        return OpError(proto.PORT_IN_USE, f"docker {verb}: the host port is already in use ({line})")
    if "is already in use by container" in text or "conflict" in text:
        return OpError(proto.CONFLICT, f"docker {verb}: a container with that name already exists ({line})")
    return OpError(proto.DOCKER_ERROR, f"docker {verb} failed: {line}")


def labels_of(doc: Mapping[str, Any]) -> dict[str, str]:
    cfg = doc.get("Config") or {}
    labels = cfg.get("Labels") if isinstance(cfg, dict) else None
    return {str(k): str(v) for k, v in labels.items()} if isinstance(labels, dict) else {}


def check_owned(doc: Mapping[str, Any], name: str, *, role: str | None = None, instance: str | None = None) -> None:
    """The console touches only its own containers, and with role/instance exactly that role and instance."""
    labels = labels_of(doc)
    if labels.get(LABEL) != "1":
        raise OpError(proto.NOT_OWNED, f"container {name!r} was not created by Engine Console; this node will not "
                                       "touch it")
    if role and labels.get(ROLE_LABEL) != role:
        raise OpError(proto.NOT_OWNED, f"container {name!r} is not a console {role}")
    if instance and labels.get(INSTANCE_LABEL) != instance:
        raise OpError(proto.NOT_OWNED, f"container {name!r} belongs to another console instance")


def container_name(ref: object) -> str:
    if not isinstance(ref, str) or not CONTAINER_REF.fullmatch(ref):
        raise OpError(proto.BAD_ARGS, f"{str(ref)[:130]!r} is not a valid container name")
    return ref


# ── run specs (docs/FLEET-PROTOCOL.md §5) ───────────────────────────────────────────────────────────────
@dataclass
class RunPlan:
    name: str
    instance: str
    image: str
    argv: list[str]
    container_port: int
    network: str
    alias: str
    gpu_uuids: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    secret_env: dict[str, str] = field(default_factory=dict)   # values: child environment only
    mounts: list[tuple[str, str, bool]] = field(default_factory=list)
    volumes: list[tuple[str, str]] = field(default_factory=list)
    files: list[tuple[str, str, str]] = field(default_factory=list)   # (container path, host path, content)
    shm_size: str = "16g"
    ipc_host: bool = False
    ulimits: list[tuple[str, str]] = field(default_factory=list)


# Container ulimits an engine may ask for. `stack`: TensorRT-LLM's documented docker command raises it. `nofile`: many
# open model shards. `memlock` is deliberately absent: unlimited locked memory lets one container pin host RAM, and it
# only matters for RDMA, which needs its own policy switch when multi-machine engines arrive.
ULIMITS = frozenset({"stack", "nofile"})
ULIMIT_VALUE = re.compile(r"(?:-1|[0-9]{1,15})(?::(?:-1|[0-9]{1,15}))?")


def _bad(msg: str) -> OpError:
    return OpError(proto.BAD_ARGS, msg)


def check_dest(dest: object, what: str) -> str:
    if not isinstance(dest, str) or not dest.startswith("/") or len(dest) > 512 or dest == "/":
        raise _bad(f"{what}: the container path must be absolute (and not '/')")
    if any(c in dest for c in ":,\\") or any(ord(c) < 32 or ord(c) == 127 for c in dest):
        raise _bad(f"{what}: the container path {dest[:200]!r} may not contain ':', ',', '\\' or control characters")
    if any(seg in (".", "..") for seg in dest.split("/")):
        raise _bad(f"{what}: the container path {dest[:200]!r} may not contain '.' or '..' segments")
    return dest.rstrip("/") or dest


def shm_gib(value: str) -> float:
    m = SHM.fullmatch(value)
    if m is None:
        raise _bad(f"shm_size {value[:20]!r} must look like 16g, 512m or 1024k")
    return int(m.group(1)) * {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0}[m.group(2)]


# How far below a mount source the secret scan looks, and how many entries it reads before refusing. HF model trees
# (per-model folders, their blobs, snapshots and refs) are skipped: they hold only weights, configs and links.
SECRET_SCAN_DEPTH = 4
SECRET_SCAN_MAX_ENTRIES = 20000
_MODEL_TREE = re.compile(r"(?:models|datasets|spaces)--.+|blobs|snapshots|refs")


def secret_near_top(src: str) -> str | None:
    """The first key- or token-looking name in a mount source, down to SECRET_SCAN_DEPTH levels (an HF_HOME's `token`,
    an Ollama home's private key, a `creds/hf/token` tucked deeper), relative to the source; None when there is none.
    HF model trees are not walked. A source too big to check within SECRET_SCAN_MAX_ENTRIES is refused rather than
    waved through: mount a narrower folder."""
    if is_secret_name(os.path.basename(src)):
        return os.path.basename(src)
    seen = 0
    level = [""]
    for depth in range(SECRET_SCAN_DEPTH):
        nxt: list[str] = []
        for rel in level:
            try:
                with os.scandir(os.path.join(src, rel) if rel else src) as it:
                    entries = [(e.name, e.is_dir(follow_symlinks=False)) for e in it]
            except NotADirectoryError:
                continue
            except OSError as e:
                if depth == 0:
                    raise OpError(proto.IO_ERROR,
                                  f"cannot list {src[:200]!r} to check it before mounting: {e.strerror}") from None
                continue
            seen += len(entries)
            if seen > SECRET_SCAN_MAX_ENTRIES:
                raise OpError(proto.POLICY_DENIED,
                              f"{src[:200]!r} holds more than {SECRET_SCAN_MAX_ENTRIES} entries near its top, too many "
                              "to check for keys before mounting; mount a narrower folder")
            for name, is_dir in entries:
                path = f"{rel}/{name}" if rel else name
                if is_secret_name(name):
                    return path
                if is_dir and not _MODEL_TREE.fullmatch(name):
                    nxt.append(path)
        level = nxt
    return None


def plan_run(spec: object, policy: Policy, known_gpus: set[str], state_dir: Path) -> RunPlan:
    """Validate a cleaned RUN_SPEC against §5 and this node's policy; OpError says which field and why."""
    if not isinstance(spec, dict):
        raise _bad("spec is missing")
    name = spec.get("name")
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name):
        raise _bad("spec.name must look like ec-<lowercase letters, digits, hyphens> (at most 61 characters)")
    instance = spec.get("instance")
    if not isinstance(instance, str) or not INSTANCE.fullmatch(instance):
        raise _bad("spec.instance must be 1-64 letters, digits, '_' or '-'")
    image = spec.get("image")
    if not isinstance(image, str) or not valid_image(image):
        raise _bad("spec.image is not a valid image reference")
    if not policy.image_allowed(image):
        raise OpError(proto.POLICY_DENIED, f"image {image!r} is not allowed by this node's policy (engine_images in "
                                           f"{policy.where()})")
    argv = spec.get("argv") or []
    if len(argv) > 512 or any(len(a) > 4096 or "\x00" in a for a in argv):
        raise _bad("spec.argv: at most 512 arguments of at most 4096 characters, no NUL")
    port = spec.get("container_port")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise _bad("spec.container_port must be 1-65535")
    alias = spec.get("network_alias") or name
    if not RUN_NAME.fullmatch(alias):
        raise _bad("spec.network_alias must look like ec-<lowercase letters, digits, hyphens>")
    gpus: list[str] = []
    for u in spec.get("gpu_uuids") or []:
        if u not in known_gpus:
            raise OpError(proto.BAD_ARGS, f"GPU {u[:96]!r} is not a GPU of this machine (see node.info)")
        if u not in gpus:
            gpus.append(u)

    labels: dict[str, str] = {LABEL: "1", ROLE_LABEL: "engine", INSTANCE_LABEL: instance}
    for k, v in (spec.get("labels") or {}).items():
        if not LABEL_KEY.fullmatch(k):
            raise _bad(f"label {k[:80]!r}: keys must start with 'engine-console.' (lowercase, digits, . _ -)")
        if any(ord(c) < 32 or ord(c) == 127 for c in v):
            raise _bad(f"label {k!r} has a control character in its value")
        if k in labels and labels[k] != v:
            raise _bad(f"label {k!r} must be {labels[k]!r} for this container")
        labels[k] = v

    env: dict[str, str] = {}
    for k, v in (spec.get("env") or {}).items():
        if not ENV_NAME.fullmatch(k):
            raise _bad(f"env name {k[:80]!r} must be letters, digits and '_' (not starting with a digit)")
        if ENV_LOOKS_SECRET.search(k):
            raise _bad(f"env {k!r} looks like a secret: send it in secret_env so its value never reaches a command "
                       "line")
        env[k] = v
    secret_env: dict[str, str] = {}
    for k, v in (spec.get("secret_env") or {}).items():
        if not ENV_NAME.fullmatch(k):
            raise _bad(f"secret_env name {k[:80]!r} must be letters, digits and '_' (not starting with a digit)")
        if CHILD_ENV_DENY.search(k):
            raise OpError(proto.POLICY_DENIED, f"secret_env name {k!r} would change how the docker CLI itself runs "
                                               "on this machine; use another name")
        if not ENV_LOOKS_SECRET.search(k):
            raise _bad(f"secret_env name {k!r} does not look like a secret (key, token, secret or password): send it "
                       "in env")
        if k in env:          # unreachable while the two name rules above hold; kept in case one is relaxed
            raise _bad(f"{k!r} is in both env and secret_env")
        secret_env[k] = v

    dests: set[str] = set()

    def claim(dest: str, what: str) -> str:
        if dest in dests:
            raise _bad(f"{what}: container path {dest!r} is used twice")
        dests.add(dest)
        return dest

    mounts: list[tuple[str, str, bool]] = []
    for i, m in enumerate(spec.get("mounts") or []):
        what = f"mounts[{i}]"
        h = open_root(policy, m.get("root"))
        rel = m.get("rel")
        if m.get("ro") is False:
            # A read-write mount lets the container plant links and swap directories inside the root (and, on a
            # rootful daemon, setuid files). Engines only read weights, so every mount is read-only.
            raise OpError(proto.POLICY_DENIED, f"{what}: model roots are only ever mounted read-only on this node")
        # mounts only: an empty rel is the root directory itself (still realpath-confined)
        src = h.real if rel in (None, "") else resolve_existing(h, split_rel(rel, what=f"{what}.rel"))
        if any(c in src for c in ":,"):
            raise _bad(f"{what}: host path {src[:200]!r} contains ':' or ',' and cannot be mounted")
        secret = secret_near_top(src)
        if secret is not None:
            raise OpError(proto.POLICY_DENIED, f"{what}: {secret[:200]!r} looks like a key or token file, and a "
                                               "container would see it; mount a folder without it (for Hugging Face, "
                                               "a root of kind hf-hub on the hub/ folder, not the whole HF_HOME)")
        dest = claim(check_dest(m.get("dest"), what), what)
        mounts.append((src, dest, True))
    volumes: list[tuple[str, str]] = []
    for i, v in enumerate(spec.get("volumes") or []):
        what = f"volumes[{i}]"
        vname = v.get("name")
        if not isinstance(vname, str) or not VOLUME.fullmatch(vname):
            raise _bad(f"{what}: named volumes must be called ec-<name> (lowercase, digits, . _ -)")
        volumes.append((vname, claim(check_dest(v.get("dest"), what), what)))
    files: list[tuple[str, str, str]] = []
    fdir = state_dir / "instances" / instance / "files"
    for i, (dest, content) in enumerate(list((spec.get("files") or {}).items())[:MAX_FILES]):
        what = f"files[{i}]"
        dest = claim(check_dest(dest, what), what)
        base = re.sub(r"[^A-Za-z0-9._-]", "_", dest.rsplit("/", 1)[-1])[:100].lstrip(".") or "file"
        files.append((dest, str(fdir / f"{i}-{base}"), content))
    if any(c in str(fdir) for c in ":,"):
        raise _bad("the node's state directory contains ':' or ',' and cannot hold mounted files")

    shm = spec.get("shm_size")
    if shm is None:
        shm = "16g" if policy.max_shm_gib >= 16 else f"{max(1, int(policy.max_shm_gib * 1024))}m"
    if shm_gib(shm) > policy.max_shm_gib:
        raise OpError(proto.POLICY_DENIED, f"shm_size {shm} is above this node's policy limit of "
                                           f"{policy.max_shm_gib:g} GiB (max_shm_gib)")
    ipc_host = spec.get("ipc_host") is True
    if ipc_host and not policy.allow_ipc_host:
        raise OpError(proto.POLICY_DENIED, "this node's policy does not allow --ipc host (allow_ipc_host); multi-GPU "
                                           "engines need it for NCCL")
    ulimits: list[tuple[str, str]] = []
    for k, v in (spec.get("ulimits") or {}).items():
        if k not in ULIMITS:
            raise _bad(f"ulimit {k[:40]!r} is not one an engine may set here (allowed: {', '.join(sorted(ULIMITS))})")
        if not ULIMIT_VALUE.fullmatch(v):
            raise _bad(f"ulimit {k} value {v[:40]!r} must be -1 (unlimited), a number, or soft:hard numbers")
        ulimits.append((k, v))
    return RunPlan(name=name, instance=instance, image=image, argv=list(argv), container_port=port,
                   network=policy.engine_network, alias=alias, gpu_uuids=gpus, labels=labels, env=env,
                   secret_env=secret_env, mounts=mounts, volumes=volumes, files=files, shm_size=shm,
                   ipc_host=ipc_host, ulimits=ulimits)


def gpus_arg(uuids: Sequence[str]) -> str:
    # docker parses --gpus as CSV: several devices need the value wrapped in literal double quotes
    return f'"device={",".join(uuids)}"' if len(uuids) > 1 else f"device={uuids[0]}"


def run_argv(d: Docker, p: RunPlan) -> list[str]:
    """The exact argv docker.run executes. secret_env appears as bare `-e NAME`: values never reach argv."""
    a = ["run", "-d", "--pull", "never", "--name", p.name, "--network", p.network, "--network-alias", p.alias,
         "--restart", "no", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit", "4096"]
    if p.gpu_uuids:
        a += ["--gpus", gpus_arg(p.gpu_uuids)]
    a += ["--shm-size", p.shm_size, "--ipc", "host" if p.ipc_host else "private"]
    for k, v in p.ulimits:
        a += ["--ulimit", f"{k}={v}"]
    for k, v in p.labels.items():
        a += ["--label", f"{k}={v}"]
    for src, dest, ro in p.mounts:
        a += ["-v", f"{src}:{dest}{':ro' if ro else ''}"]
    for vname, dest in p.volumes:
        a += ["-v", f"{vname}:{dest}"]
    for dest, host, _content in p.files:
        a += ["-v", f"{host}:{dest}:ro"]
    for k, v in p.env.items():
        a += ["-e", f"{k}={v}"]
    for k in p.secret_env:
        a += ["-e", k]
    return d.argv(*a, p.image, *p.argv)


def write_files(p: RunPlan, state_dir: Path) -> None:
    """Write the spec's generated files under <state>/instances/<instance>/files/ (directories 0700, files 0644,
    each replaced atomically); stale files of earlier runs of that instance are removed."""
    fdir = state_dir / "instances" / p.instance / "files"
    try:
        for d in (state_dir, state_dir / "instances", state_dir / "instances" / p.instance, fdir):
            d.mkdir(mode=0o700, exist_ok=True)
            if d.is_symlink() or not d.is_dir():
                raise OpError(proto.IO_ERROR, f"{d} is not a plain directory")
            os.chmod(d, 0o700)
        keep = {os.path.basename(host) for _d, host, _c in p.files}
        for old in os.listdir(fdir):
            if old not in keep:
                (fdir / old).unlink(missing_ok=True)
        for _dest, host, content in p.files:
            tmp = f"{host}.{secrets.token_hex(4)}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o644)
            with os.fdopen(fd, "wb") as f:
                f.write(content.encode("utf-8"))
            os.chmod(tmp, 0o644)
            os.replace(tmp, host)
    except OSError as e:
        raise OpError(proto.IO_ERROR, f"cannot write the engine's config files under {state_dir}: "
                                      f"{e.strerror}") from None


def remove_files(state_dir: Path, instance: str) -> None:
    if INSTANCE.fullmatch(instance or ""):
        shutil.rmtree(state_dir / "instances" / instance, ignore_errors=True)


# ── network and gateway ──────────────────────────────────────────────────────────────────────────────────
def ensure_network(d: Docker, name: str) -> None:
    """Create the engine network with --internal, or verify an existing one really is internal."""
    res = d.call("network", "inspect", "--format", "{{.Internal}}", name, timeout=15)
    if res.ok:
        if res.out.strip().lower() != "true":
            raise OpError(proto.CONFLICT, f"docker network {name!r} exists but is not --internal; refusing to attach "
                                          "engines to it (remove it or name another engine_network in the policy)")
        return
    if not _missing(res):
        raise docker_error("network inspect", res)
    d.ok("network", "create", "--internal", "--label", f"{LABEL}=1", name, timeout=30)


def gateway_config(engine_host: str, engine_port: int) -> str:
    """nginx.conf for a pure TCP forward. Only a checked name and an int reach the template."""
    if not NGINX_HOST.fullmatch(engine_host):
        raise _bad("engine_host is not a safe container name")
    if not 1 <= engine_port <= 65535:
        raise _bad("engine_port must be 1-65535")
    return (
        # daemon off: nginx forks by default, which ends the container's PID 1 (found by live test, not mocks).
        "daemon off;\nworker_processes 1;\npid /tmp/nginx.pid;\nerror_log /dev/stderr warn;\n"
        "events { worker_connections 1024; }\n"
        f"stream {{\n  server {{\n    listen {GATEWAY_LISTEN};\n    proxy_pass {engine_host}:{engine_port};\n"
        "    proxy_connect_timeout 5s;\n    proxy_timeout 1h;\n  }\n}\n")


def gateway_argv(d: Docker, *, image: str, name: str, instance: str, host_port: int, engine_host: str,
                 engine_port: int) -> list[str]:
    """`docker create` on the default bridge (so it can publish on loopback); the engine network is attached
    afterwards with `docker network connect`, since one create accepts only one --network."""
    conf = gateway_config(engine_host, engine_port)
    return d.argv("create", "--pull", "never", "--name", name, "--network", "bridge", "--restart", "no",
                  "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--pids-limit",
                  "256", "--user", "101:101",
                  "--tmpfs", "/var/cache/nginx:rw,mode=1777,size=4m", "--tmpfs", "/var/run:rw,mode=1777,size=1m",
                  "--tmpfs", "/tmp:rw,mode=1777,size=1m",  # noqa: S108 - container-private tmpfs
                  "-p", f"127.0.0.1:{host_port}:{GATEWAY_LISTEN}",
                  "--label", f"{LABEL}=1", "--label", f"{ROLE_LABEL}=gateway", "--label", f"{INSTANCE_LABEL}={instance}",
                  "-e", f"NGINX_CONF={conf}", "--entrypoint", "/bin/sh", image, "-c", GATEWAY_SH)
