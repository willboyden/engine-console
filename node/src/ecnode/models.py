"""Model scanner (docs/FLEET-PROTOCOL.md §6.1): what models each policy root holds, in the background.

Root kinds: hf-home (<root>/hub/models--org--name), hf-hub (<root>/models--org--name), gguf (.gguf files, split
sets grouped), plain (directories with config.json + weights), ollama (manifests -> layers -> blobs), other (one
entry per top-level directory) and auto (detected per root). Bounded per root: at most 50 000 directory entries
and 120 s. Metadata files are read only when their realpath stays inside the root, through O_NOFOLLOW opens, with
size caps; names that look like keys or secrets are never opened (an Ollama home keeps its own ssh-style keys).
"""
from __future__ import annotations

import functools
import json
import os
import re
import stat
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from . import gguf, proto
from .confine import inside, open_inside, open_root, read_inside, split_rel
from .errors import OpError
from .policy import Policy, Root
from .util import Guarded, clean_text, is_secret_name

MAX_ENTRIES = 50_000
MAX_SECONDS = 120.0
HUNG_GRACE_S = 30.0
RESCAN_S = 600.0
CHECK_S = 30.0
FRESH_S = 10.0
CONFIG_MAX = 512 * 1024
MANIFEST_MAX = 256 * 1024
SAFETENSORS_HEADER_MAX = 16 * 1024 * 1024
SAFETENSORS_MAX_TENSORS = 200_000      # the largest real shards hold a few thousand tensors
SAFETENSORS_MAX_DIMS = 32
PARAMS_MAX = 2**53
MAX_SAFETENSORS_FILES = 64
PAGE_BYTES = 768 * 1024
WEIGHT_EXT = (".safetensors", ".bin", ".pt", ".pth")
SPLIT = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<n>\d{5})\.gguf$", re.I)
QUANT_IN_NAME = re.compile(
    r"(?:^|[-_.])((?:UD-)?(?:I?Q[1-8](?:_[0-9A-Z]{1,4}){0,3}|BF16|F16|F32|FP16|FP8|MXFP4(?:_MOE)?|TQ[12]_0))(?=$|[-_.])",
    re.I)
HF_QUANT_IN_NAME = re.compile(r"(?:^|[-_.])(NVFP4|MXFP4|FP8|AWQ|GPTQ|INT4|INT8|W4A16|W8A8|W4A4|BNB-4BIT|GGUF)"
                              r"(?=$|[-_.])", re.I)
META_KEYS = ("architectures", "model_type", "torch_dtype", "dtype", "hidden_size", "intermediate_size",
             "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings",
             "vocab_size", "rope_scaling", "rope_theta", "sliding_window", "num_experts", "num_local_experts",
             "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size", "kv_lora_rank", "q_lora_rank",
             "qk_rope_head_dim", "v_head_dim", "tie_word_embeddings", "layer_types")
QUANT_KEYS = ("quant_method", "bits", "group_size", "quant_algo", "format", "weight_block_size", "fmt",
              "load_in_4bit", "load_in_8bit", "kv_cache_quant_algo")


class BudgetExceeded(Exception):
    """The per-root scan budget ran out; the entries found so far are still used."""


class Budget:
    def __init__(self, entries: int = MAX_ENTRIES, seconds: float = MAX_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.left = entries
        self.limit = entries
        self.deadline = clock() + seconds
        self.seconds = seconds
        self.clock = clock

    def tick(self, n: int = 1) -> None:
        self.left -= n
        if self.left < 0:
            raise BudgetExceeded(f"stopped after {self.limit} directory entries (scan budget)")
        if self.clock() > self.deadline:
            raise BudgetExceeded(f"stopped after {self.seconds:g} s (scan budget)")


def _entry(root: str, rel: str, fmt: str, name: str, **kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {k: None for k in proto.MODEL_ENTRY}
    e.update(root=root, rel=rel, format=fmt, name=clean_text(name, 256), warnings=[])
    e.update(kw)
    return e


def _scandir(path: str, budget: Budget) -> list[os.DirEntry[str]]:
    try:
        with os.scandir(path) as it:
            out = []
            for d in it:
                budget.tick()
                out.append(d)
    except OSError:
        return []
    return sorted(out, key=lambda d: d.name)


def _lstat(path: str) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError:
        return None


def _json(data: bytes | None) -> Any:
    if data is None:
        return None
    try:
        return json.loads(data.decode("utf-8", "replace"))
    except (ValueError, RecursionError):
        return None


def _subset(cfg: dict[str, Any], depth: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {k: cfg[k] for k in META_KEYS if k in cfg}
    qc = cfg.get("quantization_config")
    if isinstance(qc, dict):
        out["quantization_config"] = {k: qc[k] for k in QUANT_KEYS if k in qc}
    if depth == 0:
        for sub in ("text_config", "llm_config", "language_config"):
            if isinstance(cfg.get(sub), dict):
                out[sub] = _subset(cfg[sub], 1)
    clean = proto.clean_json(out)
    return clean if isinstance(clean, dict) else {}


def _cfg_get(cfg: dict[str, Any], key: str) -> Any:
    if key in cfg:
        return cfg[key]
    for sub in ("text_config", "llm_config", "language_config"):
        inner = cfg.get(sub)
        if isinstance(inner, dict) and key in inner:
            return inner[key]
    return None


def quant_of_config(cfg: dict[str, Any] | None, name: str) -> str | None:
    """fp8 / nvfp4 / awq-4bit / gptq-4bit / bnb-4bit / w4a16 ... from quantization_config, else from the name,
    else the stored dtype (bf16, fp16, fp32)."""
    qc = cfg.get("quantization_config") if isinstance(cfg, dict) else None
    if not isinstance(qc, dict) and isinstance(cfg, dict):
        qc = _cfg_get(cfg, "quantization_config")
    if isinstance(qc, dict):
        method = str(qc.get("quant_method") or "").lower()
        bits = qc.get("bits")
        if method in ("fp8", "fbgemm_fp8"):
            return "fp8"
        if method in ("awq", "gptq") and isinstance(bits, int):
            return f"{method}-{bits}bit"
        if method == "bitsandbytes":
            return "bnb-4bit" if qc.get("load_in_4bit") else "bnb-8bit"
        if method == "modelopt":
            algo = str(qc.get("quant_algo") or "").lower()
            return algo or "modelopt"
        if method == "compressed-tensors":
            fmt = str(qc.get("format") or "").lower()
            if "nvfp4" in fmt:
                return "nvfp4"
            if "float" in fmt or "fp8" in fmt:
                return "fp8"
            groups = qc.get("config_groups")
            if isinstance(groups, dict):
                for g in groups.values():
                    w = g.get("weights") if isinstance(g, dict) else None
                    if isinstance(w, dict) and isinstance(w.get("num_bits"), int):
                        kind = "fp" if w.get("type") == "float" else "w"
                        return f"fp{w['num_bits']}" if kind == "fp" else f"w{w['num_bits']}a16"
            return "compressed-tensors"
        if method:
            return clean_text(method, 32)
    m = HF_QUANT_IN_NAME.search(name)
    if m:
        return m.group(1).lower()
    dtype = (_cfg_get(cfg, "torch_dtype") or _cfg_get(cfg, "dtype")) if isinstance(cfg, dict) else None
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(str(dtype))


def _arch_ctx(cfg: dict[str, Any] | None) -> tuple[str | None, int | None]:
    if not isinstance(cfg, dict):
        return None, None
    archs = cfg.get("architectures")
    arch = archs[0] if isinstance(archs, list) and archs and isinstance(archs[0], str) else cfg.get("model_type")
    ctx = _cfg_get(cfg, "max_position_embeddings")
    return (clean_text(arch, 64) if isinstance(arch, str) else None,
            ctx if isinstance(ctx, int) and not isinstance(ctx, bool) and 0 < ctx < 2**31 else None)


def _safetensors_params(root_real: str, files: list[str], index: Any) -> int | None:
    """Total parameters: the index's metadata.total_parameters, else the sum of tensor element counts read from
    each file's header (bounded), which counts packed 4-bit weights wrongly, so callers skip it for those."""
    if isinstance(index, dict):
        meta = index.get("metadata")
        tp = meta.get("total_parameters") if isinstance(meta, dict) else None
        if isinstance(tp, int) and not isinstance(tp, bool) and tp > 0:
            return tp
    if not files or len(files) > MAX_SAFETENSORS_FILES:
        return None
    total = 0
    for path in files:
        fd = open_inside(root_real, path)
        if fd is None:
            return None
        try:
            head = os.read(fd, 8)
            if len(head) != 8:
                return None
            n = int.from_bytes(head, "little")
            if n <= 0 or n > SAFETENSORS_HEADER_MAX:
                return None
            buf = b""
            while len(buf) < n:
                chunk = os.read(fd, n - len(buf))
                if not chunk:
                    return None
                buf += chunk
        except OSError:
            return None
        finally:
            os.close(fd)
        hdr = _json(buf)
        if not isinstance(hdr, dict) or len(hdr) > SAFETENSORS_MAX_TENSORS:
            return None
        for k, t in hdr.items():
            if k == "__metadata__" or not isinstance(t, dict) or not isinstance(t.get("shape"), list):
                continue
            # bounded: a hostile header with millions of dims would otherwise cost minutes of big-int CPU per file
            if len(t["shape"]) > SAFETENSORS_MAX_DIMS:
                return None
            count = 1
            for dim in t["shape"]:
                if not isinstance(dim, int) or dim < 0:
                    return None
                count *= dim
                if count > PARAMS_MAX:
                    return None
            total += count
            if total > PARAMS_MAX:
                return None
    return total if total > 0 else None


def gguf_name_quant(filename: str, header: dict[str, Any] | None) -> tuple[str, str | None]:
    """(display name without the split suffix and quant token, quant)."""
    stem = filename[:-5] if filename.lower().endswith(".gguf") else filename
    m = SPLIT.match(filename)
    if m:
        stem = m.group("base")
    quant = (header or {}).get("quant")
    qm = QUANT_IN_NAME.search(stem)
    if qm:
        quant = quant or qm.group(1).upper()
        if qm.group(1).upper() == str(quant).upper() or qm.group(1).upper().replace("UD-", "") == str(quant).upper():
            stem = (stem[: qm.start()] + stem[qm.end():]).strip("-_. ") or stem
    return stem, quant


# ── per-kind scanners ────────────────────────────────────────────────────────────────────────────────────
@dataclass
class Ctx:
    root: Root
    real: str
    budget: Budget


def _walk(ctx: Ctx, rel: str, depth: int, max_depth: int) -> Iterator[tuple[str, os.DirEntry[str]]]:
    """(rel of entry, entry) below ctx.real/rel, depth-first, no symlinked dirs, no hidden or secret names."""
    path = os.path.join(ctx.real, rel) if rel else ctx.real
    for d in _scandir(path, ctx.budget):
        if d.name.startswith(".") or is_secret_name(d.name):
            continue
        r = f"{rel}/{d.name}" if rel else d.name
        yield r, d
        if depth < max_depth and d.is_dir(follow_symlinks=False):
            yield from _walk(ctx, r, depth + 1, max_depth)


def _gguf_groups(files: list[tuple[str, int, float, float]]) -> list[list[tuple[str, int, float, float]]]:
    """Group (rel, size, mtime, atime) of .gguf files into split sets (same dir, base and count) and singles."""
    sets: dict[tuple[str, str, str], list[tuple[str, int, float, float]]] = {}
    singles: list[list[tuple[str, int, float, float]]] = []
    for f in files:
        d, _, fname = f[0].rpartition("/")
        m = SPLIT.match(fname)
        if m:
            sets.setdefault((d, m.group("base"), m.group("n")), []).append(f)
        else:
            singles.append([f])
    return singles + [sorted(v) for v in sets.values()]


def gguf_header(root_real: str, path: str) -> dict[str, Any] | None:
    """GGUF facts of a file whose realpath is inside the root (never anything else)."""
    fd = open_inside(root_real, path)
    return gguf.read_file(path, fd) if fd is not None else None


def _gguf_entry(ctx: Ctx, group: list[tuple[str, int, float, float]], *, name: str | None = None,
                revision: str | None = None, extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    first = group[0]
    fname = first[0].rsplit("/", 1)[-1]
    header = gguf_header(ctx.real, os.path.join(ctx.real, first[0]))
    warnings: list[str] = []
    m = SPLIT.match(fname)
    shards = None
    params = header.get("params") if header else None
    if m:
        shards = int(m.group("n"))
        if len(group) != shards:
            warnings.append(f"split set has {len(group)} of {shards} files")
        if int(m.group("idx")) != 1:
            warnings.append("the first file of the split set is missing")
        for f in group[1:]:          # every shard lists its own tensors: add them up for the parameter count
            h2 = gguf_header(ctx.real, os.path.join(ctx.real, f[0]))
            params = params + h2["params"] if params is not None and h2 and h2.get("params") is not None else None
    if header is None:
        warnings.append("could not read the GGUF header")
    elif not header.get("complete"):
        warnings.append("GGUF header only partly read")
    disp, quant = gguf_name_quant(fname, header)
    meta: dict[str, Any] = {k: header.get(k) for k in ("version", "name", "size_label", "block_count",
                                                     "embedding_length", "head_count", "head_count_kv",
                                                     "expert_count", "expert_used_count", "file_type")
                            if header and header.get(k) is not None}
    if "mmproj" in fname.lower() or (header and header.get("arch") == "clip"):
        meta["role"] = "mmproj"
    if m:
        meta["files"] = [f[0].rsplit("/", 1)[-1] for f in group][:64]
    meta.update(extra_meta or {})
    return _entry(ctx.root.id, first[0], "gguf", name or disp, revision=revision, size=sum(f[1] for f in group),
                  files=len(group), mtime=max(f[2] for f in group), atime=max(f[3] for f in group), quant=quant,
                  params=params, arch=header.get("arch") if header else None,
                  context=header.get("context") if header else None, shards=shards,
                  meta=proto.clean_json(meta), warnings=warnings)


def scan_gguf(ctx: Ctx, sink: list[dict[str, Any]], base: str = "", max_depth: int = 6) -> None:
    files: list[tuple[str, int, float, float]] = []
    try:
        for rel, d in _walk(ctx, base, 1, max_depth):
            if not d.name.lower().endswith(".gguf"):
                continue
            try:
                st = d.stat(follow_symlinks=True)
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            if d.is_symlink() and not inside(os.path.realpath(d.path), ctx.real):
                continue
            files.append((rel, st.st_size, st.st_mtime, st.st_atime))
    finally:      # the budget may have run out: still list what was found
        sink.extend(_gguf_entry(ctx, g) for g in _gguf_groups(files))


def _dir_size(ctx: Ctx, path: str, max_depth: int = 8) -> tuple[int, int, float, float, bool]:
    """(bytes, files, newest mtime, newest atime, complete) of a directory tree (no symlinks followed)."""
    total = count = 0
    mtime = atime = 0.0
    stack = [(path, 0)]
    try:
        while stack:
            p, depth = stack.pop()
            for d in _scandir(p, ctx.budget):
                if is_secret_name(d.name):
                    continue
                try:
                    st = d.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode) and depth < max_depth:
                    stack.append((d.path, depth + 1))
                elif stat.S_ISREG(st.st_mode):
                    total += st.st_size
                    count += 1
                    mtime, atime = max(mtime, st.st_mtime), max(atime, st.st_atime)
    except BudgetExceeded:
        return total, count, mtime, atime, False
    return total, count, mtime, atime, True


def _plain_entry(ctx: Ctx, rel: str, path: str) -> dict[str, Any]:
    cfg = _json(read_inside(ctx.real, os.path.join(path, "config.json"), CONFIG_MAX))
    cfg = cfg if isinstance(cfg, dict) else None
    size, count, mtime, atime, complete = _dir_size(ctx, path)
    name = rel.rsplit("/", 1)[-1] if rel else os.path.basename(ctx.real)
    quant = quant_of_config(cfg, name)
    arch, context = _arch_ctx(cfg)
    st_files = sorted(os.path.join(path, f) for f in _names(path) if f.endswith(".safetensors"))
    index = _json(read_inside(ctx.real, os.path.join(path, "model.safetensors.index.json"), 8 << 20))
    params = _safetensors_params(ctx.real, st_files, index) if _unpacked(quant) else None
    warnings = [] if complete else ["size is a lower bound: the scan budget ran out inside this directory"]
    names = _names(path)
    if not any(n.endswith(WEIGHT_EXT) for n in names):
        exts = sorted({os.path.splitext(n)[1] or n for n in _other_weights(path, names)})[:4]
        warnings.append(f"the weights are in a format the common engines do not load ({', '.join(exts)}); "
                        "listed so its disk space is visible")
    return _entry(ctx.root.id, rel, "plain", name, size=size, files=count, mtime=mtime or None, atime=atime or None,
                  quant=quant, params=params, arch=arch, context=context,
                  shards=len(st_files) or None, meta=_subset(cfg) if cfg else None, warnings=warnings)


def _unpacked(quant: str | None) -> bool:
    return quant in (None, "fp8", "bf16", "fp16", "fp32")


def _names(path: str) -> list[str]:
    try:
        return [n for n in os.listdir(path) if not is_secret_name(n)][:10000]
    except OSError:
        return []


BIG_FILE = 64 * 1024 * 1024
_NOT_WEIGHTS = (".json", ".md", ".txt", ".jinja", ".py", ".model", ".tiktoken", ".gitattributes")


def _other_weights(path: str, names: list[str]) -> list[str]:
    """Large regular files that are not a known weight format (a custom checkpoint format, e.g. `*.ftw`)."""
    out: list[str] = []
    for n in names[:2000]:
        if n.endswith(WEIGHT_EXT) or n.endswith(_NOT_WEIGHTS):
            continue
        try:
            st = os.lstat(os.path.join(path, n))
        except OSError:
            continue
        if stat.S_ISREG(st.st_mode) and st.st_size >= BIG_FILE:
            out.append(n)
    return out


def _is_model_dir(path: str) -> bool:
    """config.json plus weights. Weights in an unknown format still count (with a warning on the entry): the directory
    is a model taking disk space even when no engine here can load it."""
    names = _names(path)
    if "config.json" not in names:
        return False
    return any(n.endswith(WEIGHT_EXT) for n in names) or bool(_other_weights(path, names))


def scan_plain(ctx: Ctx, sink: list[dict[str, Any]], max_depth: int = 3) -> None:
    if _is_model_dir(ctx.real):
        sink.append(_plain_entry(ctx, "", ctx.real))
        return
    stack = [("", 0)]
    while stack:
        rel, depth = stack.pop()
        path = os.path.join(ctx.real, rel) if rel else ctx.real
        for d in _scandir(path, ctx.budget):
            if d.name.startswith(".") or is_secret_name(d.name) or not d.is_dir(follow_symlinks=False):
                continue
            r = f"{rel}/{d.name}" if rel else d.name
            if _is_model_dir(d.path):
                sink.append(_plain_entry(ctx, r, d.path))
            elif depth + 1 < max_depth:
                stack.append((r, depth + 1))


def scan_other(ctx: Ctx, sink: list[dict[str, Any]]) -> None:
    for d in _scandir(ctx.real, ctx.budget):
        if d.name.startswith(".") or is_secret_name(d.name) or not d.is_dir(follow_symlinks=False):
            continue
        size, count, mtime, atime, complete = _dir_size(ctx, d.path, max_depth=32)
        sink.append(_entry(ctx.root.id, d.name, "other", d.name, size=size, files=count, mtime=mtime or None,
                           atime=atime or None,
                           warnings=[] if complete else ["size is a lower bound: the scan budget ran out"]))
        if not complete:
            raise BudgetExceeded("the scan budget ran out")


# ── Hugging Face cache ───────────────────────────────────────────────────────────────────────────────────
_REV = re.compile(r"[0-9a-f]{40}")


def scan_hf(ctx: Ctx, sink: list[dict[str, Any]], base: str) -> None:
    path = os.path.join(ctx.real, base) if base else ctx.real
    for d in _scandir(path, ctx.budget):
        if d.name.startswith("models--") and d.is_dir(follow_symlinks=False):
            sink.extend(_hf_repo(ctx, d.path, f"{base}/{d.name}" if base else d.name, d.name))


def _hf_snapshot(ctx: Ctx, repo: str) -> tuple[str | None, list[str]]:
    """(revision to show, warnings): refs/main when its snapshot exists, else the newest snapshot."""
    warnings: list[str] = []
    ref = (read_inside(ctx.real, os.path.join(repo, "refs", "main"), 256) or b"").decode("ascii", "replace").strip()
    snaps = [d for d in _scandir(os.path.join(repo, "snapshots"), ctx.budget) if d.is_dir(follow_symlinks=False)]
    names = {d.name for d in snaps}
    if ref and _REV.fullmatch(ref) and ref in names:
        if len(snaps) > 1:
            warnings.append(f"{len(snaps)} revisions on disk")
        return ref, warnings
    if not snaps:
        return None, ["no snapshot: the download has not finished (or was removed)"]
    newest = max(snaps, key=lambda d: getattr(_lstat(d.path), "st_mtime", 0.0))
    warnings.append("refs/main is missing or points nowhere: showing the newest snapshot")
    return newest.name, warnings


def _hf_repo(ctx: Ctx, repo: str, rel: str, dirname: str) -> list[dict[str, Any]]:
    repo_id = dirname[len("models--"):].replace("--", "/")
    rev, warnings = _hf_snapshot(ctx, repo)
    blobs_dir = os.path.join(repo, "blobs")
    blobs: dict[str, os.stat_result] = {}
    incomplete = 0
    for d in _scandir(blobs_dir, ctx.budget):
        bst = _lstat(d.path)
        if bst is None or not stat.S_ISREG(bst.st_mode):
            continue
        if d.name.endswith(".incomplete"):
            incomplete += 1
        else:
            blobs[d.name] = bst
    if incomplete:
        warnings.append(f"{incomplete} file(s) still downloading")
    disk = sum(st.st_size for st in blobs.values())
    if rev is None:
        return [_entry(ctx.root.id, rel, "hf", repo_id, size=disk, files=0, warnings=warnings,
                       meta={"disk_bytes": disk})]
    snap = os.path.join(repo, "snapshots", rev)
    files: list[tuple[str, int, float, float]] = []          # (path inside the snapshot, size, mtime, atime)
    used: set[str] = set()
    size = 0
    stack = [("", 0)]
    while stack:
        sub, depth = stack.pop()
        for d in _scandir(os.path.join(snap, sub) if sub else snap, ctx.budget):
            r = f"{sub}/{d.name}" if sub else d.name
            if is_secret_name(d.name):
                continue
            if d.is_dir(follow_symlinks=False):
                if depth < 6:
                    stack.append((r, depth + 1))
                continue
            st: os.stat_result | None = None
            blob: str | None = None
            if d.is_symlink():
                try:
                    target = os.path.normpath(os.path.join(os.path.dirname(d.path), os.readlink(d.path)))
                except OSError:
                    continue
                if os.path.dirname(target) == blobs_dir and os.path.basename(target) in blobs:
                    blob = os.path.basename(target)
                    st = blobs[blob]
                elif inside(os.path.realpath(d.path), ctx.real):
                    st = _lstat(os.path.realpath(d.path))
            else:
                st = _lstat(d.path)
            if st is None or not stat.S_ISREG(st.st_mode):
                if len(warnings) < 8:
                    warnings.append(f"{r[:120]} points to a missing blob")
                continue
            if blob is None or blob not in used:
                size += st.st_size
                if blob is not None:
                    used.add(blob)
            files.append((r, st.st_size, st.st_mtime, st.st_atime))
    names = [f[0] for f in files]
    ggufs = [f for f in files if f[0].lower().endswith(".gguf")]
    weights = [n for n in names if n.endswith(WEIGHT_EXT)]
    if ggufs and not weights:
        return [_hf_gguf_entry(ctx, repo, rel, repo_id, rev, ggufs, size=size, files=len(files),
                               warnings=warnings)]
    cfg = _json(read_inside(ctx.real, os.path.join(snap, "config.json"), CONFIG_MAX))
    cfg = cfg if isinstance(cfg, dict) else None
    if cfg is None:
        warnings.append("no config.json in the snapshot")
    quant = quant_of_config(cfg, repo_id)
    arch, context = _arch_ctx(cfg)
    st_files = sorted(os.path.join(snap, n) for n in names if n.endswith(".safetensors") and "/" not in n)
    index = _json(read_inside(ctx.real, os.path.join(snap, "model.safetensors.index.json"), 8 << 20))
    params = _safetensors_params(ctx.real, st_files, index) if _unpacked(quant) else None
    meta: dict[str, Any] = _subset(cfg) if cfg else {}
    if disk != size:
        meta["disk_bytes"] = disk
    return [_entry(ctx.root.id, rel, "hf", repo_id, revision=rev, size=size, files=len(files),
                   mtime=max((f[2] for f in files), default=None), atime=max((f[3] for f in files), default=None),
                   quant=quant, params=params, arch=arch, context=context, shards=len(st_files) or None,
                   meta=proto.clean_json(meta) if meta else None, warnings=warnings[:8])]


def _hf_gguf_entry(ctx: Ctx, repo: str, rel: str, repo_id: str, rev: str,
                   ggufs: list[tuple[str, int, float, float]], *, size: int, files: int,
                   warnings: list[str]) -> dict[str, Any]:
    """One entry for a GGUF-only HF repo: rel is the repo directory like every HF entry (the hub mounts the
    cache); each quant file (or split set) is listed in meta.gguf_files with its path below the repo directory."""
    groups = _gguf_groups([(f"snapshots/{rev}/{f[0]}", f[1], f[2], f[3]) for f in ggufs])
    listed: list[dict[str, Any]] = []
    first: dict[str, Any] | None = None
    for g in groups[:32]:
        fname = g[0][0].rsplit("/", 1)[-1]
        header = gguf_header(ctx.real, os.path.join(repo, g[0][0]))
        first = first or header
        _name, quant = gguf_name_quant(fname, header)
        m = SPLIT.match(fname)
        listed.append({"rel": g[0][0], "quant": quant, "size": sum(f[1] for f in g), "files": len(g),
                       "shards": int(m.group("n")) if m else None,
                       "role": "mmproj" if "mmproj" in fname.lower() else None})
    quants = [x["quant"] for x in listed if x["role"] is None]
    if len(groups) > 32:
        warnings.append(f"{len(groups)} GGUF files or sets: only the first 32 are listed")
    return _entry(ctx.root.id, rel, "gguf", repo_id, revision=rev, size=size, files=files,
                  mtime=max(f[2] for f in ggufs), atime=max(f[3] for f in ggufs),
                  quant=quants[0] if len(quants) == 1 else None,
                  params=first.get("params") if first and len(quants) == 1 else None,
                  arch=first.get("arch") if first else None, context=first.get("context") if first else None,
                  meta=proto.clean_json({"hf_repo": repo_id, "gguf_files": listed}), warnings=warnings[:8])


# ── Ollama ───────────────────────────────────────────────────────────────────────────────────────────────
_DIGEST = re.compile(r"sha256[:-]([0-9a-f]{64})")
_PARAM_SIZE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMBT])\s*$", re.I)


def scan_ollama(ctx: Ctx, sink: list[dict[str, Any]], base: str) -> None:
    models = os.path.join(ctx.real, base) if base else ctx.real
    manifests = os.path.join(models, "manifests")
    stack: list[tuple[str, list[str]]] = [(manifests, [])]
    seen: list[tuple[str, list[str]]] = []
    try:
        while stack:
            path, parts = stack.pop()
            for d in _scandir(path, ctx.budget):
                if d.name.startswith(".") or is_secret_name(d.name):
                    continue
                if d.is_dir(follow_symlinks=False):
                    if len(parts) < 5:
                        stack.append((d.path, [*parts, d.name]))
                elif d.is_file(follow_symlinks=False) and len(parts) >= 2:
                    seen.append((d.path, [*parts, d.name]))   # <registry>/<namespace>/<model>/<tag>
    finally:
        for path, parts in sorted(seen):
            e = _ollama_entry(ctx, models, path, parts)
            if e is not None:
                sink.append(e)


def ollama_name(parts: list[str]) -> str:
    """name:tag from a manifest path below manifests/: library models short, others with their namespace."""
    if len(parts) >= 4:
        registry, namespace, model, tag = parts[0], "/".join(parts[1:-2]), parts[-2], parts[-1]
    else:
        registry, (namespace, model, tag) = "", parts[-3:]
    if registry in ("registry.ollama.ai", "") and namespace == "library":
        return f"{model}:{tag}"
    if registry in ("registry.ollama.ai", ""):
        return f"{namespace}/{model}:{tag}"
    return f"{registry}/{namespace}/{model}:{tag}"


def _params_from_label(label: object) -> int | None:
    m = _PARAM_SIZE.match(str(label)) if isinstance(label, str) else None
    if not m:
        return None
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[m.group(2).upper()]
    return int(float(m.group(1)) * mult)


def _ollama_entry(ctx: Ctx, models: str, path: str, parts: list[str]) -> dict[str, Any] | None:
    doc = _json(read_inside(ctx.real, path, MANIFEST_MAX))
    if not isinstance(doc, dict):
        return None
    raw_layers = doc.get("layers")
    layers: list[Any] = raw_layers if isinstance(raw_layers, list) else []
    blobs = os.path.join(models, "blobs")
    size = 0
    warnings: list[str] = []
    model_blob: str | None = None
    atime: float | None = None
    for layer in layers[:64]:
        if not isinstance(layer, dict):
            continue
        m = _DIGEST.fullmatch(str(layer.get("digest") or ""))
        if m is None:
            continue
        blob = os.path.join(blobs, f"sha256-{m.group(1)}")
        st = _lstat(blob)
        if st is not None and stat.S_ISREG(st.st_mode):
            size += st.st_size
        else:
            declared = layer.get("size")
            size += declared if isinstance(declared, int) and declared > 0 else 0
            warnings.append(f"layer blob sha256-{m.group(1)[:12]}... is missing")
        if str(layer.get("mediaType", "")).endswith(".model"):
            model_blob = blob
            atime = st.st_atime if st is not None else None
    cfg_digest = _DIGEST.fullmatch(str((doc.get("config") or {}).get("digest") or "")) \
        if isinstance(doc.get("config"), dict) else None
    facts = _json(read_inside(ctx.real, os.path.join(blobs, f"sha256-{cfg_digest.group(1)}"), 64 * 1024)) \
        if cfg_digest else None
    facts = facts if isinstance(facts, dict) else {}
    header = gguf_header(ctx.real, model_blob) if model_blob else None
    rel = os.path.relpath(path, ctx.real)
    st = _lstat(path)
    params = header.get("params") if header else None
    params = params or _params_from_label(facts.get("model_type"))
    meta = {k: facts[k] for k in ("model_format", "model_family", "model_families", "model_type", "file_type")
            if k in facts}
    if header:
        meta.update({k: header[k] for k in ("block_count", "embedding_length", "head_count", "head_count_kv",
                                            "expert_count", "expert_used_count") if header.get(k) is not None})
    rev = _DIGEST.fullmatch(os.path.basename(model_blob or ""))
    return _entry(ctx.root.id, rel, "ollama", ollama_name(parts), revision=rev.group(1)[:12] if rev else None,
                  size=size, files=len(layers), mtime=st.st_mtime if st else None, atime=atime,
                  quant=clean_text(facts.get("file_type"), 32) if isinstance(facts.get("file_type"), str)
                  else (header.get("quant") if header else None),
                  params=params, arch=clean_text(facts.get("model_family"), 64)
                  if isinstance(facts.get("model_family"), str) else (header.get("arch") if header else None),
                  context=header.get("context") if header else None, meta=proto.clean_json(meta) or None,
                  warnings=warnings[:8])


# ── a whole root ─────────────────────────────────────────────────────────────────────────────────────────
def detect(real: str, budget: Budget) -> list[tuple[str, str]]:
    """[(kind, base rel)] for an `auto` root."""
    if os.path.isdir(os.path.join(real, "hub")):
        return [("hf-home", "hub")]
    if any(d.name.startswith("models--") for d in _scandir(real, budget)):
        return [("hf-hub", "")]
    if os.path.isdir(os.path.join(real, "models", "manifests")):
        return [("ollama", "models")]
    if os.path.isdir(os.path.join(real, "manifests")):
        return [("ollama", "")]
    return [("plain", ""), ("gguf", "")]


def scan_root(root: Root, budget: Budget) -> tuple[list[dict[str, Any]], str | None]:
    """(every model entry of one root, why the scan stopped early or None)."""
    real = os.path.realpath(root.path)
    if not os.path.isdir(real):
        raise FileNotFoundError(root.path)
    ctx = Ctx(root, real, budget)
    kinds: list[tuple[str, str]]
    if root.kind == "auto":
        kinds = detect(real, budget)
    elif root.kind == "hf-home":
        kinds = [("hf-home", "hub")]
    elif root.kind == "ollama":
        kinds = [("ollama", "models" if os.path.isdir(os.path.join(real, "models", "manifests")) else "")]
    else:
        kinds = [(root.kind, "")]
    found: list[dict[str, Any]] = []
    try:
        for kind, base in kinds:
            if kind in ("hf-home", "hf-hub"):
                scan_hf(ctx, found, base)
            elif kind == "gguf":
                scan_gguf(ctx, found)
            elif kind == "plain":
                scan_plain(ctx, found)
            elif kind == "ollama":
                scan_ollama(ctx, found, base)
            else:
                scan_other(ctx, found)
    except BudgetExceeded as e:
        return _dedupe(found), str(e)
    return _dedupe(found), None


def _dedupe(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        out.setdefault(str(e["rel"]), e)
    return list(out.values())


def model_config(policy: Policy, root_id: str, rel: str) -> dict[str, Any]:
    """models.config: the bounded config.json of a model (an HF repo dir, a plain dir, or a config.json path);
    for a .gguf file, the facts from its header."""
    h = open_root(policy, root_id)
    parts = split_rel(rel)
    path = h.path(parts)
    last = parts[-1]
    if last.lower().endswith(".gguf"):
        header = gguf_header(h.real, path)
        if header is None:
            raise OpError(proto.NOT_FOUND, f"{rel[:300]!r} is not a readable GGUF file in model root {h.root.id!r}")
        return {"config": proto.clean_json({"gguf": header})}
    if last == "config.json":
        candidate = path
    elif last.startswith("models--"):
        rev, _w = _hf_snapshot(Ctx(h.root, h.real, Budget(5000, 10.0)), path)
        candidate = os.path.join(path, "snapshots", rev, "config.json") if rev else ""
    else:
        candidate = os.path.join(path, "config.json")
    data = read_inside(h.real, candidate, CONFIG_MAX) if candidate else None
    if data is None:
        raise OpError(proto.NOT_FOUND, f"there is no readable config.json for {rel[:300]!r} in model root "
                                       f"{h.root.id!r}")
    cfg = _json(data)
    if not isinstance(cfg, dict):
        raise OpError(proto.IO_ERROR, f"the config.json of {rel[:300]!r} is not a JSON object")
    return {"config": proto.clean_json(cfg)}


# ── the background scanner ───────────────────────────────────────────────────────────────────────────────
def _comparable(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Entries without access times: reading a model must not count as a library change."""
    return [tuple(sorted((k, json.dumps(v, sort_keys=True)) for k, v in e.items() if k != "atime")) for e in entries]


def _signature(root: Root, limit: int = 3000) -> tuple[Any, ...]:
    """Cheap change detector: mtimes of the directories that change when a model appears or disappears."""
    real = os.path.realpath(root.path)
    sig: list[tuple[str, int]] = []
    stack = [(real, 0)]
    while stack and len(sig) < limit:
        path, depth = stack.pop()
        st = _lstat(path)
        if st is None:
            continue
        sig.append((path, st.st_mtime_ns))
        if depth >= 3:
            continue
        try:
            with os.scandir(path) as it:
                for d in it:
                    if len(sig) + len(stack) >= limit:
                        break
                    if d.is_dir(follow_symlinks=False) and not d.name.startswith(".") and not is_secret_name(d.name):
                        stack.append((d.path, depth + 1))
        except OSError:
            continue
    return tuple(sorted(sig))


class Scanner:
    """Scans every policy root in the background and keeps the latest result as numbered generations.

    The first scan runs at start. After that a root is rescanned when its directory signature changes (checked
    every CHECK_S), every RESCAN_S anyway, and when the hub asks (models.scan). A scan that finds no difference
    publishes nothing, except a requested one: the hub waits for the generation models.scan promised."""

    def __init__(self, policy_fn: Callable[[], Policy], on_change: Callable[[int], None], *,
                 entries: int = MAX_ENTRIES, seconds: float = MAX_SECONDS, rescan_s: float = RESCAN_S,
                 check_s: float = CHECK_S, clock: Callable[[], float] = time.time,
                 mono: Callable[[], float] = time.monotonic) -> None:
        self._policy_fn = policy_fn
        self._on_change = on_change
        self._entries, self._seconds = entries, seconds
        self._rescan_s, self._check_s = rescan_s, check_s
        self._clock, self._mono = clock, mono
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._by_root: dict[str, list[dict[str, Any]]] = {}
        self._scans: dict[str, dict[str, Any]] = {}
        self._snapshot: list[dict[str, Any]] = []
        self._generation = 0
        self._assigned = 0
        self._pending = False
        self._pending_all = False
        self._pending_roots: set[str] = set()
        self._notify = False
        self._running = False
        self._done_at = -1e18
        self._next_full = 0.0
        self._sigs: dict[str, tuple[Any, ...]] = {}
        self._guard = Guarded()
        self._thread: threading.Thread | None = None

    # ── control ──────────────────────────────────────────────────────────────────────────────────────────
    def start(self) -> None:
        self.request(None, force=True)
        self._thread = threading.Thread(target=self._loop, name="ecnode-models", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()

    def request(self, roots: list[str] | None = None, force: bool = False) -> int:
        """Ask for a rescan; returns the generation that scan will publish (or the current one when the last
        scan finished less than FRESH_S ago and nothing was forced)."""
        with self._cv:
            idle = not self._running and not self._pending
            if not force and not roots and idle and self._generation and self._mono() - self._done_at < FRESH_S:
                return self._generation
            self._pending = True
            self._notify = True
            if roots:
                self._pending_roots.update(roots)
            else:
                self._pending_all = True
            self._cv.notify_all()
            return self._assigned + 1

    @property
    def generation(self) -> int:
        with self._cv:
            return self._generation

    # ── reading ──────────────────────────────────────────────────────────────────────────────────────────
    def page(self, cursor: int | None, limit: int | None) -> dict[str, Any]:
        """models.list: one page of the latest generation (cursor = offset; next_cursor None at the end)."""
        policy = self._policy_fn()
        with self._cv:
            snap, gen, scans = self._snapshot, self._generation, dict(self._scans)
        start = max(0, cursor or 0)
        n = max(1, min(limit or 200, 200))
        page: list[dict[str, Any]] = []
        used = 0
        for e in snap[start:start + n]:
            size = len(json.dumps(e, default=str))
            if page and used + size > PAGE_BYTES:
                break
            page.append(e)
            used += size
        end = start + len(page)
        roots = [scans.get(r.id) or {"root": r.id, "scanned_at": None, "complete": False, "entries": 0,
                                     "error": None, "duration_s": None} for r in policy.model_roots]
        return {"generation": gen, "entries": page, "next_cursor": end if end < len(snap) else None,
                "roots": roots}

    def count(self) -> tuple[int, int]:
        with self._cv:
            return len(self._snapshot), sum(int(e.get("size") or 0) for e in self._snapshot)

    # ── the loop ─────────────────────────────────────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._cv:
                if not self._pending:
                    self._cv.wait(self._check_s)
                if self._stop.is_set():
                    return
                pending = self._pending
            if not pending:
                policy = self._policy_fn()
                stale = [r.id for r in policy.model_roots if self._changed(r)]
                full = self._mono() >= self._next_full
                if not stale and not full:
                    continue
                with self._cv:
                    if not self._pending:
                        self._pending = True
                        if full:
                            self._pending_all = True
                        self._pending_roots.update(stale)
            self.run_once()

    def run_once(self) -> int | None:
        """Run the pending scan now (the loop calls this; tests may too). Returns the published generation."""
        with self._cv:
            if not self._pending:
                return None
            roots = None if self._pending_all else set(self._pending_roots)
            notify = self._notify
            self._pending = self._pending_all = self._notify = False
            self._pending_roots = set()
            self._assigned += 1
            gen = self._assigned
            self._running = True
        try:
            policy = self._policy_fn()
            results = {r.id: self._scan_one(r) for r in policy.model_roots if roots is None or r.id in roots}
        finally:
            with self._cv:
                self._running = False
        with self._cv:
            for rid, (entries, scan) in results.items():
                if entries is not None:
                    self._by_root[rid] = entries
                self._scans[rid] = scan
            ids = [r.id for r in policy.model_roots]
            self._by_root = {k: v for k, v in self._by_root.items() if k in ids}
            self._scans = {k: v for k, v in self._scans.items() if k in ids}
            new = sorted((e for rid in ids for e in self._by_root.get(rid, [])),
                         key=lambda e: (str(e["root"]), str(e["rel"])))
            changed = _comparable(new) != _comparable(self._snapshot)
            publish = changed or notify
            if publish:
                self._generation = gen
                self._snapshot = new
            self._done_at = self._mono()
            if roots is None:
                self._next_full = self._mono() + self._rescan_s
        for r in policy.model_roots:
            if roots is None or r.id in roots:
                self._sigs[r.id] = self._guard.call("sig:" + r.id, functools.partial(_signature, r), 10.0) or ()
        if publish:
            self._on_change(gen)
        return gen if publish else None

    def _changed(self, root: Root) -> bool:
        sig = self._guard.call("sig:" + root.id, lambda: _signature(root), 10.0)
        if sig is None:
            return False
        return root.id in self._sigs and sig != self._sigs[root.id]

    def _scan_one(self, root: Root) -> tuple[list[dict[str, Any]] | None, dict[str, Any]]:
        started, t0 = self._clock(), self._mono()

        def work() -> tuple[str, list[dict[str, Any]], str | None]:
            try:
                entries, err = scan_root(root, Budget(self._entries, self._seconds, self._mono))
            except FileNotFoundError:
                return "missing", [], None
            except OSError as e:
                return "error", [], f"cannot read the root: {e.strerror}"
            return "ok", entries, err

        res = self._guard.call("scan:" + root.id, work, self._seconds + HUNG_GRACE_S)
        took = round(self._mono() - t0, 3)
        prev = self._by_root.get(root.id, [])
        if res is None:
            return None, {"root": root.id, "scanned_at": started, "complete": False, "entries": len(prev),
                          "duration_s": took, "error": "the filesystem did not answer in time (a hung network "
                                                       "mount?); showing the previous scan"}
        status, entries, err = res
        if status == "missing":
            return [], {"root": root.id, "scanned_at": started, "complete": False, "entries": 0, "duration_s": took,
                        "error": f"{root.path} does not exist on this machine (not mounted?)"}
        if status == "error":
            return [], {"root": root.id, "scanned_at": started, "complete": False, "entries": 0, "duration_s": took,
                        "error": err}
        return entries, {"root": root.id, "scanned_at": started, "complete": err is None, "entries": len(entries),
                         "duration_s": took, "error": err}
