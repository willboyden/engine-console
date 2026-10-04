"""The node's `docker.run` spec (ecnode.proto.RUN_SPEC) for one engine launch, and the model mounts it needs.

Pure functions: no I/O. A spec built here is validated against the same rules the node applies
(docs/FLEET-PROTOCOL.md §5) BEFORE it is sent, so a mistake is a clear 422 here instead of the node silently
dropping a field (proto.clean() drops what it cannot keep) or refusing with a generic message.

Mounts name a policy model root and a path inside it (`root` + `rel`), never a host path: the node resolves them.
An empty `rel` means the root itself."""
from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import Any

from engine_console.adapters.base import MODEL_MOUNT, LaunchSpec
from engine_console.domain.errors import Unprocessable
from engine_console.domain.refs import ModelRef
from engine_console.domain.repo_id import check_repo_id

CONTAINER_NAME = re.compile(r"ec-[a-z0-9][a-z0-9-]{0,58}")
INSTANCE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
VOLUME_NAME = re.compile(r"ec-[a-z0-9][a-z0-9_.-]{0,60}")
SHM_SIZE = re.compile(r"([0-9]{1,4})([kmg])")
OLLAMA_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@\-]{0,127}")
SECRET_ENV = re.compile(r"(key|token|secret|password)", re.I)
SHARD_SUFFIX = re.compile(r"-0*1-of-0*[0-9]+$")
OLLAMA_MODELS_DEST = "/root/.ollama/models"
MAX_FILES, MAX_FILE_BYTES = 4, 128 * 1024
ULIMITS = frozenset({"stack", "nofile"})            # mirrors the node's allowlist (ecnode/docker.py)
ULIMIT_VALUE = re.compile(r"(?:-1|[0-9]{1,15})(?::(?:-1|[0-9]{1,15}))?")


# ---- the model reference ------------------------------------------------------------------------------------------
def rel_ok(rel: str) -> bool:
    """The node's path rule for `rel`: relative, no backslash or NUL, no empty / `.` / `..` segment."""
    if not rel or rel.startswith("/") or "\\" in rel or "\x00" in rel:
        return False
    return all(seg not in ("", ".", "..") for seg in rel.split("/"))


def check_ref(ref: ModelRef) -> ModelRef:
    """Validate a ModelRef's fields for its format (422 `invalid_model`). Identifiers are matched with fullmatch."""
    bad = ""
    if ref.format == "hf":
        try:
            check_repo_id(ref.repo_id or "")
        except ValueError as e:
            bad = str(e)
    elif ref.format in ("gguf", "plain"):
        if not ref.root_id or not re.fullmatch(r"[A-Za-z0-9_.\-]{1,32}", ref.root_id):
            bad = f"a {ref.format} model needs the id of the model root it lives in"
        elif ref.format == "gguf" and not (ref.rel and rel_ok(ref.rel) and ref.rel.lower().endswith(".gguf")):
            bad = "a gguf model needs `rel`: the path of the .gguf file (or its first shard) inside the root"
        elif ref.format == "plain" and ref.rel and not rel_ok(ref.rel):
            bad = "`rel` must be a relative path without '..' segments"
    elif ref.format == "ollama" and ref.name is not None and ref.name != "" and not OLLAMA_NAME.fullmatch(ref.name):
        bad = "an ollama model name looks like 'llama3:8b'"
    if bad:
        raise Unprocessable(bad, code="invalid_model")
    return ref


def display_of(ref: ModelRef) -> str:
    """The name an engine serves the model under and the UI shows: what chat sends as `model`."""
    if ref.format == "hf":
        return ref.repo_id or "model"
    if ref.format == "ollama":
        return ref.name or "ollama"
    base = posixpath.basename((ref.rel or "").rstrip("/")) or (ref.root_id or "model")
    if ref.format == "gguf":
        base = SHARD_SUFFIX.sub("", base[: -len(".gguf")] if base.lower().endswith(".gguf") else base)
    return base or "model"


# ---- mounts -------------------------------------------------------------------------------------------------------
def hf_layout(root_kind: str | None, copy_rel: str) -> str:
    """"home": the root is an HF_HOME (models under `<root>/hub/`); "hub": the root is the hub cache itself."""
    if root_kind == "hf-hub":
        return "hub"
    if root_kind == "hf-home":
        return "home"
    return "hub" if copy_rel.startswith("models--") else "home"


def hf_mounts(root_id: str, root_kind: str | None, copy_rel: str, hf_path: str, ro: bool) -> list[dict[str, Any]]:
    """The host's HF cache root, read-only, where the engine's HF_HOME expects it. A hub-layout root is mounted at
    `<HF_HOME>/hub` so `models--org--name` folders resolve the same way."""
    dest = hf_path if hf_layout(root_kind, copy_rel) == "home" else f"{hf_path.rstrip('/')}/hub"
    return [{"root": root_id, "rel": "", "dest": dest, "ro": ro}]


def model_mounts(ref: ModelRef, *, copy_rel: str | None, ollama_root: tuple[str, str] | None
                 ) -> tuple[list[dict[str, Any]], str | None]:
    """(mounts, model_path) for the non-HF formats (HF is mounted by `hf_mounts`).

    gguf: the directory holding the file, so every shard of a split set is visible; model_path is the file.
    plain: the model directory at MODEL_MOUNT. ollama: the models dir of an Ollama home, or the home's `models/`
    (`ollama_root` is (root id, root path))."""
    if ref.format == "gguf":
        rel = ref.rel or ""
        repo, sep, inner = rel.partition("/snapshots/")
        if sep and repo and inner:
            # A quant inside an HF cache repo: its snapshot file is a relative link into the repo's `blobs/`, so the
            # whole repo directory is mounted and the model path keeps the snapshot path below it.
            return ([{"root": ref.root_id, "rel": repo, "dest": MODEL_MOUNT, "ro": True}],
                    f"{MODEL_MOUNT}/snapshots/{inner}")
        return ([{"root": ref.root_id, "rel": posixpath.dirname(rel), "dest": MODEL_MOUNT, "ro": True}],
                f"{MODEL_MOUNT}/{posixpath.basename(rel)}")
    if ref.format == "plain":
        return [{"root": ref.root_id, "rel": ref.rel or "", "dest": MODEL_MOUNT, "ro": True}], MODEL_MOUNT
    if ref.format == "ollama" and ollama_root is not None:
        root_id, root_path = ollama_root
        first = (copy_rel or "").split("/", 1)[0]
        if copy_rel and first == "manifests":
            rel = ""
        elif copy_rel and first == "models":
            rel = "models"
        else:
            rel = "" if posixpath.basename(root_path.rstrip("/")) == "models" else "models"
        return [{"root": root_id, "rel": rel, "dest": OLLAMA_MODELS_DEST, "ro": True}], None
    return [], None


# ---- the spec -----------------------------------------------------------------------------------------------------
def split_env(launch_env: Mapping[str, str], *, offline: bool) -> tuple[dict[str, str], dict[str, str]]:
    """(plain env, secret env). A variable whose NAME looks secret travels in `secret_env` (docker only ever sees its
    name on the command line); everything else is plain. `HF_HUB_OFFLINE=1` unless an adapter set it."""
    env: dict[str, str] = {}
    secret: dict[str, str] = {}
    for k, v in launch_env.items():
        (secret if SECRET_ENV.search(k) else env)[k] = str(v)
    if offline:
        env.setdefault("HF_HUB_OFFLINE", "1")
    return env, secret


def engine_labels(*, instance: str, engine: str, display: str, ref: ModelRef, port: int, name: str,
                  gpu_ids: list[int], gpu_uuids: list[str]) -> dict[str, str]:
    """`engine-console.*` labels (the node adds `engine-console=1`). Values are capped at 256 characters by the node,
    so a value that would not fit is left out rather than truncated into something misleading."""
    labels = {"engine-console.role": "engine", "engine-console.instance": instance, "engine-console.engine": engine,
              "engine-console.model": display, "engine-console.port": str(port), "engine-console.name": name,
              "engine-console.format": ref.format, "engine-console.gpu_ids": ",".join(str(i) for i in gpu_ids)}
    for key, val in (("engine-console.gpus", ",".join(gpu_uuids)), ("engine-console.root", ref.root_id or ""),
                     ("engine-console.rel", ref.rel or "")):
        if val and len(val) <= 256:
            labels[key] = val
    return {k: v for k, v in labels.items() if v and len(v) <= 256}


def build_run_spec(*, name: str, instance: str, image: str, launch: LaunchSpec, gpu_uuids: list[str],
                   labels: dict[str, str], env: dict[str, str], secret_env: dict[str, str],
                   mounts: list[dict[str, Any]], volumes: list[tuple[str, str]], ipc_host: bool) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "name": name, "instance": instance, "image": image, "argv": list(launch.argv),
        "container_port": launch.container_port, "gpu_uuids": list(gpu_uuids), "network_alias": name,
        "labels": labels, "env": env, "secret_env": secret_env, "mounts": mounts,
        "volumes": [{"name": n, "dest": d} for n, d in volumes], "shm_size": launch.shm_size, "ipc_host": ipc_host}
    if launch.files:
        spec["files"] = dict(launch.files)
    if launch.ulimits:
        spec["ulimits"] = dict(launch.ulimits)
    return spec


def validate_run_spec(spec: Mapping[str, Any], *, max_shm_gib: float | None = None) -> list[str]:
    """The node's validation rules (docs/FLEET-PROTOCOL.md §5), evaluated hub-side. Empty list = acceptable."""
    errs: list[str] = []
    if not CONTAINER_NAME.fullmatch(str(spec.get("name", ""))):
        errs.append("container name must look like ec-<letters, digits, hyphens>")
    if not INSTANCE_ID.fullmatch(str(spec.get("instance", ""))):
        errs.append("instance id has characters the node does not accept")
    argv = spec.get("argv") or []
    if len(argv) > 512 or any(len(str(a)) > 4096 or "\x00" in str(a) for a in argv):
        errs.append("the engine command line is too long (over 512 arguments, or an argument over 4096 characters)")
    port = spec.get("container_port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        errs.append("container port must be 1-65535")
    if len(spec.get("gpu_uuids") or []) > 16:
        errs.append("at most 16 GPUs per engine")
    labels = spec.get("labels") or {}
    if len(labels) > 32 or any(not str(k).startswith("engine-console.") or len(str(v)) > 256 for k, v in labels.items()):
        errs.append("labels must start with 'engine-console.', be at most 32, and each value at most 256 characters")
    for field in ("env", "secret_env"):
        vals = spec.get(field) or {}
        if len(vals) > (128 if field == "env" else 16) or any(
                not ENV_NAME.fullmatch(str(k)) or len(str(v)) > 4096 for k, v in vals.items()):
            errs.append(f"{field}: too many variables, or a bad name or value (names: letters, digits, underscore)")
    if len(spec.get("mounts") or []) > 16:
        errs.append("at most 16 mounts")
    for m in spec.get("mounts") or []:
        rel = str(m.get("rel", ""))
        dest = str(m.get("dest", ""))
        if (rel and not rel_ok(rel)) or not dest.startswith("/") or ".." in dest.split("/"):
            errs.append(f"mount of root {m.get('root')!r}: bad path ({rel!r} -> {dest!r})")
    if len(spec.get("volumes") or []) > 8 or any(not VOLUME_NAME.fullmatch(str(v.get("name", ""))) or
                                                  not str(v.get("dest", "")).startswith("/") for v in spec.get("volumes") or []):
        errs.append("named volumes must be called ec-<name> and have an absolute destination")
    shm = SHM_SIZE.fullmatch(str(spec.get("shm_size", "")))
    if not shm:
        errs.append("shm_size must look like 16g")
    elif max_shm_gib is not None:
        gib = int(shm.group(1)) * {"k": 1 / 1024**2, "m": 1 / 1024, "g": 1.0}[shm.group(2)]
        if gib > max_shm_gib:
            errs.append(f"shm_size {spec.get('shm_size')} is above this machine's policy limit of {max_shm_gib:g} GiB")
    files = spec.get("files") or {}
    if len(files) > MAX_FILES or any(not str(p).startswith("/") or len(str(c).encode()) > MAX_FILE_BYTES
                                     for p, c in files.items()):
        errs.append("at most 4 generated files, absolute paths, 128 KiB each")
    for k, v in (spec.get("ulimits") or {}).items():
        if k not in ULIMITS or not ULIMIT_VALUE.fullmatch(str(v)):
            errs.append(f"ulimit {k}={v}: only {', '.join(sorted(ULIMITS))} with -1, a number or soft:hard numbers")
    return errs
