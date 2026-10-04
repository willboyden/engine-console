"""The node's local policy file (docs/FLEET-PROTOCOL.md §6): the hard boundary for everything a hub asks.

Lookup: `--policy PATH`, else /etc/ecnode/policy.json if it exists, else ~/.config/ecnode/policy.json.
The file is REFUSED (and the node runs in monitor mode, saying why in `error`) when it is a symlink or not a
regular file, group/world-writable, owned by anyone but root or this user, under /etc and not owned by root, in a
directory others can write to, too large, not JSON, or not valid against the schema below (unknown keys
included, so a typo in a safety switch is never silently ignored; keys starting with "_" are comments). No file
at all also means monitor mode. A hub can read the summary (it is in hello) but never change the file.
"""
from __future__ import annotations

import dataclasses
import errno
import fnmatch
import json
import os
import re
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

ETC_POLICY = "/etc/ecnode/policy.json"
USER_POLICY = os.path.join(".config", "ecnode", "policy.json")
MAX_POLICY_BYTES = 256 * 1024
SYSTEM_DIRS: tuple[str, ...] = ("/etc/",)

MODES = ("monitor", "manage")
KINDS = ("auto", "hf-home", "hf-hub", "gguf", "plain", "ollama", "other")
EXTERNAL = ("none", "stop_start")
DEFAULT_IMAGES = ("vllm/vllm-openai:*", "lmsysorg/sglang:*", "nvcr.io/nvidia/tensorrt-llm/release:*",
                  "ghcr.io/ggml-org/llama.cpp:*", "ghcr.io/ggml-org/llama.cpp@sha256:*", "ollama/ollama:*")
DEFAULT_NETWORK = "engine-console-engines"
DEFAULT_PORTS = (18000, 18099)
DEFAULT_DISCOVERY = (8000, 8080, 11434, 30000)

# Image references (docker's grammar, slightly stricter): [domain[:port]/]path[:tag][@sha256:digest]
_COMP = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
_DOMAIN = r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)*(?::[0-9]{1,5})?/)"
REPO_RE = _DOMAIN + "?" + _COMP + r"(?:/" + _COMP + r")*"
TAG_RE = r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
DIGEST_RE = r"sha256:[a-f0-9]{64}"
_IMAGE_REF = re.compile(r"(?P<repo>" + REPO_RE + r")(?::(?P<tag>" + TAG_RE + r"))?(?:@(?P<digest>" + DIGEST_RE + r"))?")
_PAT_TAG = re.compile(r"(?P<repo>" + REPO_RE + r"):\*")
_PAT_DIGEST = re.compile(r"(?P<repo>" + REPO_RE + r")@sha256:\*")

_ROOT_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_SCRIPT_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_NETWORK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_CONTEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
_BINARY = re.compile(r"/[A-Za-z0-9_./+-]{1,255}")
# Directories that must never be a model root themselves (a root below them is fine, e.g. /var/lib/ollama).
_FORBIDDEN_ROOTS = frozenset({"/", "/etc", "/proc", "/sys", "/dev", "/boot", "/root", "/run", "/var", "/usr",
                              "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/home", "/tmp", "/opt",  # noqa: S108
                              "/srv", "/mnt", "/media", "/snap", "/var/run", "/var/lib", "/usr/local"})


class PolicyError(ValueError):
    """The policy file is not acceptable; the message says which field and why."""


@dataclass(frozen=True)
class Root:
    id: str
    path: str
    kind: str = "auto"
    writable: bool = False
    label: str = ""

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "path": self.path, "kind": self.kind, "writable": self.writable,
                "label": self.label or self.id}


@dataclass(frozen=True)
class ScriptAction:
    name: str
    label: str
    argv: tuple[str, ...]
    args: tuple[str, ...] = ()
    confirm: bool = False
    danger: bool = False

    def summary(self) -> dict[str, Any]:   # argv stays on the node
        return {"name": self.name, "label": self.label, "args": list(self.args), "confirm": self.confirm,
                "danger": self.danger}


@dataclass(frozen=True)
class Script:
    name: str
    label: str
    engine: str | None = None
    description: str = ""
    containers: tuple[str, ...] = ()
    status_action: str | None = None
    timeout_s: int = 900
    actions: tuple[ScriptAction, ...] = ()

    def action(self, name: str) -> ScriptAction | None:
        return next((a for a in self.actions if a.name == name), None)

    def matches(self, container_name: str) -> bool:
        return any(fnmatch.fnmatchcase(container_name, p) for p in self.containers)

    def summary(self) -> dict[str, Any]:
        return {"name": self.name, "label": self.label, "engine": self.engine, "description": self.description,
                "actions": [a.summary() for a in self.actions], "containers": list(self.containers),
                "status_action": self.status_action}


@dataclass(frozen=True)
class Policy:
    path: str | None = None
    mode: str = "monitor"
    error: str | None = None
    docker_binary: str = "docker"
    docker_context: str | None = None
    engine_images: tuple[str, ...] = ()
    gateway_image: str | None = None
    engine_network: str = DEFAULT_NETWORK
    port_range: tuple[int, int] = DEFAULT_PORTS
    discovery_ports: tuple[int, ...] = DEFAULT_DISCOVERY
    model_roots: tuple[Root, ...] = ()
    scripts: tuple[Script, ...] = ()
    allow_pull: bool = False
    allow_delete: bool = False
    allow_ipc_host: bool = False
    external_control: str = "none"
    max_shm_gib: float = 64.0
    stamp: tuple[int, ...] = field(default=(), compare=False)   # file identity, to notice edits

    @property
    def manage(self) -> bool:
        return self.mode == "manage"

    def root(self, root_id: object) -> Root | None:
        return next((r for r in self.model_roots if r.id == root_id), None)

    def script(self, name: object) -> Script | None:
        return next((s for s in self.scripts if s.name == name), None)

    def image_allowed(self, image: str) -> bool:
        return any(image_matches(p, image) for p in self.engine_images)

    def where(self) -> str:
        return self.path or ETC_POLICY

    def summary(self) -> dict[str, Any]:
        return {"path": self.path, "mode": self.mode, "error": self.error, "engine_images": list(self.engine_images),
                "gateway_image": self.gateway_image, "engine_network": self.engine_network,
                "port_range": list(self.port_range), "discovery_ports": list(self.discovery_ports),
                "model_roots": [r.summary() for r in self.model_roots],
                "scripts": [s.summary() for s in self.scripts], "allow_pull": self.allow_pull,
                "allow_delete": self.allow_delete, "allow_ipc_host": self.allow_ipc_host,
                "external_control": self.external_control, "max_shm_gib": self.max_shm_gib}


# ── image references ─────────────────────────────────────────────────────────────────────────────────────
def split_image(ref: str) -> tuple[str, str | None, str | None] | None:
    """(repository, tag, digest) of a valid image reference, else None."""
    if not isinstance(ref, str) or len(ref) > 256:
        return None
    m = _IMAGE_REF.fullmatch(ref)
    if m is None:
        return None
    return m.group("repo"), m.group("tag"), m.group("digest")


def valid_image(ref: object) -> bool:
    return isinstance(ref, str) and split_image(ref) is not None


def image_matches(pattern: str, image: str) -> bool:
    """Exact reference, `repo:*` (any tag of exactly that repository) or `repo@sha256:*` (any digest of it).
    The repository is compared literally, so `vllm/vllm-openai:*` allows neither `evil/vllm/vllm-openai:x` nor
    `vllm/vllm-openai-evil:x`."""
    parts = split_image(image)
    if parts is None:
        return False
    repo, tag, digest = parts
    m = _PAT_TAG.fullmatch(pattern)
    if m is not None:
        return repo == m.group("repo") and tag is not None and digest is None
    m = _PAT_DIGEST.fullmatch(pattern)
    if m is not None:
        return repo == m.group("repo") and digest is not None and tag is None
    return valid_image(pattern) and image == pattern


def valid_image_pattern(p: object) -> bool:
    return isinstance(p, str) and bool(_PAT_TAG.fullmatch(p) or _PAT_DIGEST.fullmatch(p) or valid_image(p))


# ── schema ───────────────────────────────────────────────────────────────────────────────────────────────
def _keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    for k in obj:
        if not isinstance(k, str) or (not k.startswith("_") and k not in allowed):
            raise PolicyError(f"{where}: unknown key {k!r} (allowed: {', '.join(sorted(allowed))})")


def _bool(obj: dict[str, Any], key: str, where: str, default: bool) -> bool:
    v = obj.get(key, default)
    if not isinstance(v, bool):
        raise PolicyError(f"{where}.{key} must be true or false")
    return v


def _str(obj: dict[str, Any], key: str, where: str, *, limit: int, default: str | None = None,
         pattern: re.Pattern[str] | None = None, required: bool = False) -> str | None:
    if key not in obj or obj[key] is None:
        if required:
            raise PolicyError(f"{where}.{key} is required")
        return default
    v = obj[key]
    if not isinstance(v, str) or not v or len(v) > limit or "\x00" in v:
        raise PolicyError(f"{where}.{key} must be a non-empty string of at most {limit} characters")
    if pattern is not None and not pattern.fullmatch(v):
        raise PolicyError(f"{where}.{key}: {v!r} is not a valid value")
    return v


def _int(v: object, where: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise PolicyError(f"{where} must be a whole number from {lo} to {hi}")
    return v


def _list(obj: dict[str, Any], key: str, where: str, cap: int) -> list[Any]:
    v = obj.get(key, [])
    if v is None:
        return []
    if not isinstance(v, list) or len(v) > cap:
        raise PolicyError(f"{where}.{key} must be a list of at most {cap} items")
    return v


def _root_path(p: object, where: str, home: str | None) -> str:
    if not isinstance(p, str) or not p.startswith("/") or len(p) > 512:
        raise PolicyError(f"{where}.path must be an absolute path of at most 512 characters")
    if any(c in p for c in ":,\x00\n\r\\") or any(seg in (".", "..") for seg in p.split("/")):
        raise PolicyError(f"{where}.path: {p!r} must not contain ':', ',', '\\', '.' or '..' segments")
    norm = os.path.normpath(p)
    if norm in _FORBIDDEN_ROOTS or (home and norm == os.path.normpath(home)):
        raise PolicyError(f"{where}.path: {norm!r} is too broad for a model root; name the directory that holds "
                          "the models (for example /data/hf)")
    return norm


def parse(data: object, path: str | None, *, home: str | None = None) -> Policy:
    """Validate a decoded policy document; PolicyError says exactly what is wrong."""
    if not isinstance(data, dict):
        raise PolicyError("the policy must be one JSON object")
    _keys(data, {"version", "mode", "docker", "engine_images", "gateway_image", "engine_network", "port_range",
                 "discovery_ports", "model_roots", "scripts", "allow_pull", "allow_delete", "allow_ipc_host",
                 "external_control", "max_shm_gib"}, "policy")
    if data.get("version", 1) != 1:
        raise PolicyError("policy.version must be 1 (this node understands policy version 1)")
    mode = data.get("mode", "monitor")
    if mode not in MODES:
        raise PolicyError('policy.mode must be "monitor" or "manage"')

    dk = data.get("docker") or {}
    if not isinstance(dk, dict):
        raise PolicyError("policy.docker must be an object like {\"binary\": \"docker\", \"context\": null}")
    _keys(dk, {"binary", "context"}, "policy.docker")
    binary = _str(dk, "binary", "policy.docker", limit=256, default="docker") or "docker"
    if binary != "docker" and (not _BINARY.fullmatch(binary) or ".." in binary.split("/")):
        raise PolicyError('policy.docker.binary must be "docker" or an absolute path to the docker CLI')
    context = _str(dk, "context", "policy.docker", limit=64, pattern=_CONTEXT)

    images = _list(data, "engine_images", "policy", 64)
    for i, p in enumerate(images):
        if not valid_image_pattern(p):
            raise PolicyError(f"policy.engine_images[{i}]: {p!r} must be an exact image reference, "
                              "'repository:*' or 'repository@sha256:*' with a literal repository")
    gateway = data.get("gateway_image")
    if gateway is not None and not valid_image(gateway):
        raise PolicyError("policy.gateway_image must be an exact image reference (best: nginx@sha256:<digest>)")
    network = _str(data, "engine_network", "policy", limit=64, default=DEFAULT_NETWORK, pattern=_NETWORK)

    pr = data.get("port_range", list(DEFAULT_PORTS))
    if not isinstance(pr, list) or len(pr) != 2:
        raise PolicyError("policy.port_range must be [first, last], e.g. [18000, 18099]")
    lo = _int(pr[0], "policy.port_range[0]", 1024, 65535)
    hi = _int(pr[1], "policy.port_range[1]", lo, min(65535, lo + 1000))
    dports = tuple(dict.fromkeys(_int(p, f"policy.discovery_ports[{i}]", 1, 65535)
                                 for i, p in enumerate(_list(data, "discovery_ports", "policy", 128))))
    if "discovery_ports" not in data:
        dports = DEFAULT_DISCOVERY

    roots: list[Root] = []
    for i, r in enumerate(_list(data, "model_roots", "policy", 32)):
        where = f"policy.model_roots[{i}]"
        if not isinstance(r, dict):
            raise PolicyError(f"{where} must be an object")
        _keys(r, {"id", "path", "kind", "writable", "label"}, where)
        rid = _str(r, "id", where, limit=32, pattern=_ROOT_ID, required=True) or ""
        if any(x.id == rid for x in roots):
            raise PolicyError(f"{where}.id: {rid!r} is used twice")
        kind = _str(r, "kind", where, limit=16, default="auto") or "auto"
        if kind not in KINDS:
            raise PolicyError(f"{where}.kind must be one of {', '.join(KINDS)}")
        roots.append(Root(id=rid, path=_root_path(r.get("path"), where, home), kind=kind,
                          writable=_bool(r, "writable", where, False),
                          label=_str(r, "label", where, limit=64, default=rid) or rid))

    scripts: list[Script] = []
    for i, s in enumerate(_list(data, "scripts", "policy", 32)):
        scripts.append(_script(s, f"policy.scripts[{i}]", scripts))

    ext = data.get("external_control", "none")
    if ext not in EXTERNAL:
        raise PolicyError('policy.external_control must be "none" or "stop_start"')
    shm = data.get("max_shm_gib", 64)
    if isinstance(shm, bool) or not isinstance(shm, (int, float)) or not 0 < shm <= 4096:
        raise PolicyError("policy.max_shm_gib must be a number above 0 and at most 4096")
    return Policy(path=path, mode=mode, error=None, docker_binary=binary, docker_context=context,
                  engine_images=tuple(images), gateway_image=gateway, engine_network=network or DEFAULT_NETWORK,
                  port_range=(lo, hi), discovery_ports=dports, model_roots=tuple(roots), scripts=tuple(scripts),
                  allow_pull=_bool(data, "allow_pull", "policy", False),
                  allow_delete=_bool(data, "allow_delete", "policy", False),
                  allow_ipc_host=_bool(data, "allow_ipc_host", "policy", False),
                  external_control=ext, max_shm_gib=float(shm))


def _script(s: object, where: str, taken: Sequence[Script]) -> Script:
    if not isinstance(s, dict):
        raise PolicyError(f"{where} must be an object")
    _keys(s, {"name", "label", "engine", "description", "containers", "status_action", "timeout_s", "actions"},
          where)
    name = _str(s, "name", where, limit=32, pattern=_SCRIPT_NAME, required=True) or ""
    if any(x.name == name for x in taken):
        raise PolicyError(f"{where}.name: {name!r} is used twice")
    containers = _list(s, "containers", where, 16)
    for j, c in enumerate(containers):
        if not isinstance(c, str) or not c or len(c) > 128 or "\x00" in c:
            raise PolicyError(f"{where}.containers[{j}] must be a container name pattern like 'myproj-sglang-*'")
    actions: list[ScriptAction] = []
    for j, a in enumerate(_list(s, "actions", where, 16)):
        aw = f"{where}.actions[{j}]"
        if not isinstance(a, dict):
            raise PolicyError(f"{aw} must be an object")
        _keys(a, {"name", "label", "argv", "args", "confirm", "danger"}, aw)
        an = _str(a, "name", aw, limit=32, pattern=_SCRIPT_NAME, required=True) or ""
        if any(x.name == an for x in actions):
            raise PolicyError(f"{aw}.name: {an!r} is used twice")
        argv = a.get("argv")
        if (not isinstance(argv, list) or not 1 <= len(argv) <= 64
                or not all(isinstance(x, str) and 0 < len(x) <= 4096 and "\x00" not in x for x in argv)):
            raise PolicyError(f"{aw}.argv must be a list of 1-64 non-empty strings (the program, then its arguments)")
        if not argv[0].startswith("/") or "\n" in argv[0]:
            raise PolicyError(f"{aw}.argv[0] must be an absolute path to the program (no shell, no PATH lookup)")
        args = _list(a, "args", aw, 32)
        if not all(isinstance(x, str) and 0 < len(x) <= 64 and "\x00" not in x for x in args):
            raise PolicyError(f"{aw}.args must be a list of allowed argument values (each 1-64 characters)")
        actions.append(ScriptAction(name=an, label=_str(a, "label", aw, limit=64, default=an) or an,
                                    argv=tuple(argv), args=tuple(args), confirm=_bool(a, "confirm", aw, False),
                                    danger=_bool(a, "danger", aw, False)))
    status = _str(s, "status_action", where, limit=32)
    if status is not None and not any(x.name == status for x in actions):
        raise PolicyError(f"{where}.status_action must name one of the script's actions")
    timeout = _int(s.get("timeout_s", 900), f"{where}.timeout_s", 1, 86400)
    return Script(name=name, label=_str(s, "label", where, limit=64, default=name) or name,
                  engine=_str(s, "engine", where, limit=32), description=_str(s, "description", where, limit=500,
                                                                              default="") or "",
                  containers=tuple(containers), status_action=status, timeout_s=timeout, actions=tuple(actions))


# ── loading ──────────────────────────────────────────────────────────────────────────────────────────────
def monitor_policy(path: str | None, error: str | None, stamp: tuple[int, ...] = ()) -> Policy:
    """What the node runs with when there is no usable policy: monitor mode, no roots, no scripts."""
    return Policy(path=path, mode="monitor", error=error, stamp=stamp)


def candidate(explicit: str | None, home: str | None = None, etc_path: str = ETC_POLICY) -> tuple[str | None, bool]:
    """(path to load, whether it was named explicitly)."""
    if explicit:
        return explicit, True
    if os.path.lexists(etc_path):
        return etc_path, False
    user = os.path.join(home or os.path.expanduser("~"), USER_POLICY)
    return (user, False) if os.path.lexists(user) else (None, False)


def _under(path: str, prefixes: Sequence[str]) -> bool:
    ap = os.path.abspath(path)
    return any(ap.startswith(p) for p in prefixes)


def load(explicit: str | None = None, *, home: str | None = None, etc_path: str = ETC_POLICY,
         system_dirs: Sequence[str] = SYSTEM_DIRS, uid: int | None = None) -> Policy:
    """Read and validate the policy; never raises. Any problem -> monitor mode with `error` set."""
    path, named = candidate(explicit, home, etc_path)
    if path is None:
        return monitor_policy(None, "no policy file, so this node runs in monitor mode (write "
                                    f"{etc_path}, for example from `ecnode policy-template`)")
    me = os.getuid() if uid is None else uid
    stamp = file_stamp(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as e:
        if e.errno == errno.ELOOP:
            why = "is a symlink; point --policy at the real file"
        elif e.errno == errno.ENOENT:
            why = "does not exist" if named else "disappeared while being read"
        else:
            why = f"cannot be read ({e.strerror})"
        return monitor_policy(path, f"policy file {path} {why}; running in monitor mode", stamp)
    try:
        st = os.fstat(fd)
        problem = _file_problem(path, st, me, system_dirs)
        if problem:
            return monitor_policy(path, f"policy file {path} {problem}; running in monitor mode", stamp)
        raw = b""
        while len(raw) <= MAX_POLICY_BYTES:
            chunk = os.read(fd, MAX_POLICY_BYTES + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
    except OSError as e:
        return monitor_policy(path, f"policy file {path} cannot be read ({e.strerror}); running in monitor mode",
                              stamp)
    finally:
        os.close(fd)
    if len(raw) > MAX_POLICY_BYTES:
        return monitor_policy(path, f"policy file {path} is larger than {MAX_POLICY_BYTES} bytes; running in "
                                    "monitor mode", stamp)
    try:
        data = json.loads(raw.decode("utf-8"), parse_constant=_no_constant)
    except (UnicodeDecodeError, ValueError) as e:
        return monitor_policy(path, f"policy file {path} is not valid JSON ({str(e)[:160]}); running in monitor "
                                    "mode", stamp)
    try:
        pol = parse(data, path, home=home or os.path.expanduser("~"))
    except PolicyError as e:
        return monitor_policy(path, f"policy file {path}: {e}; running in monitor mode", stamp)
    return dataclasses.replace(pol, stamp=stamp)


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")


def _file_problem(path: str, st: os.stat_result, me: int, system_dirs: Sequence[str]) -> str | None:
    if not stat.S_ISREG(st.st_mode):
        return "is not a regular file"
    if st.st_mode & 0o022:
        return f"is group- or world-writable (run: sudo chmod 644 {path})"
    system = _under(path, system_dirs)
    if system and st.st_uid != 0:
        return f"must be owned by root (run: sudo chown root:root {path})"
    if st.st_uid not in (0, me):
        return "is owned by another user"
    # The directory matters too: whoever can write to it can swap the file. Under /etc it must be root's alone;
    # elsewhere a group-writable directory is tolerated (user-private groups with umask 002 make ~/.config
    # directories 0775), a world-writable one or one owned by another user is not.
    parent = os.path.dirname(os.path.abspath(path))
    try:
        pst = os.stat(parent)
    except OSError:
        return "has a directory that cannot be checked"
    if pst.st_mode & (0o022 if system else 0o002):
        return f"is in a directory others can write to ({parent})"
    if (system and pst.st_uid != 0) or pst.st_uid not in (0, me):
        return f"is in a directory owned by another user ({parent})"
    return None


def file_stamp(path: str) -> tuple[int, ...]:
    """Identity of whatever is at `path` now (not following a symlink), to notice an edit or a replacement."""
    try:
        st = os.lstat(path)
    except OSError:
        return ()
    return (st.st_dev, st.st_ino, st.st_mtime_ns, st.st_size, st.st_mode, st.st_uid)


class PolicyStore:
    """The current policy. Re-checks the file at most every `recheck_s` and reloads it when it changed (or when
    a file appears where there was none), so an administrator tightening the policy need not restart the node."""

    def __init__(self, explicit: str | None = None, *, home: str | None = None, etc_path: str = ETC_POLICY,
                 system_dirs: Sequence[str] = SYSTEM_DIRS, recheck_s: float = 2.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._explicit, self._home, self._etc = explicit, home, etc_path
        self._system_dirs = system_dirs
        self._recheck_s = recheck_s
        self._clock = clock
        self._policy = self._load()
        self._checked = clock()

    def _load(self) -> Policy:
        return load(self._explicit, home=self._home, etc_path=self._etc, system_dirs=self._system_dirs)

    def _identity(self) -> tuple[str | None, tuple[int, ...]]:
        path, _ = candidate(self._explicit, self._home, self._etc)
        return path, (file_stamp(path) if path else ())

    def current(self) -> Policy:
        now = self._clock()
        if now - self._checked >= self._recheck_s:
            self._checked = now
            if self._identity() != (self._policy.path, self._policy.stamp):
                self._policy = self._load()
        return self._policy

    def reload(self) -> Policy:
        self._policy = self._load()
        self._checked = self._clock()
        return self._policy
