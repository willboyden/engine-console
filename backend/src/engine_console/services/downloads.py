"""Downloads and copies into any machine's model roots (contract v2, docs/ARCHITECTURE.md 10.6).

Bytes never touch the hub's own disk and the hub never opens a model directory: it streams them.

* **Download** (`kind="hf"`): Hugging Face -> hub (allowlisted egress, optional proxy, the HF token stays here) ->
  the target machine's node (`fs.write`, which hashes the whole file and only then renames it into place).
* **Copy** (`kind="copy"`): a library copy's machine (`fs.read`) -> hub -> the target's node (`fs.write`), with the same
  resume and the same verification.

Where the files land depends on the target root's kind (`layout_of`):

* `hf-home` / `hf-hub`: the Hugging Face cache layout, `blobs/<sha256>`, `snapshots/<commit>/<path>` (relative links)
  and `refs/<revision>`, so an engine resolves the model by repo id with HF_HUB_OFFLINE=1. The links and the ref are
  created only once every blob is in place, so a half-downloaded repository never looks like a model.
* `gguf`: the files themselves under `<org>--<name>/<path>`. (Not `models--org--name`: an `auto` root would take that
  for an HF cache.) A file is written under its final name, so an existing file of the same size proves nothing: it is
  replaced (atomically, by the node) unless this very row finished it earlier.
* `plain`: a model directory (config.json, weights, tokenizer files) under `<dir>/`; only a copy of a plain model
  directory goes there.
* `auto`: whatever its last scan showed (an HF cache, loose model files), or an empty root takes what is being written.

State machine, pause/resume/cancel, the queue and the SSE progress are those of v1. A cancel asks the node to delete
the file in flight (its `.incomplete` partial) when that node's policy allows deletes; otherwise the partial stays and
a later download of the same file resumes from it.
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import fnmatch
import hashlib
import json
import logging
import posixpath
import re
import time
from collections.abc import AsyncGenerator, Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

from ecnode import proto

from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError, Unprocessable
from engine_console.domain.fleet import CopyRequest, FleetPort, NodeClient, NodeError, NodeUnavailable
from engine_console.domain.models import Download, DownloadRequest, Page
from engine_console.domain.ports import ByteStream, HubClient, RepoFile
from engine_console.domain.repo_id import check_repo_id, require_repo_id
from engine_console.services.common import EventBus, new_id, now, scrub
from engine_console.services.gguf_info import shard_names
from engine_console.services.hf import HfService
from engine_console.services.library import CopySource, LibraryService, RootSpec, RootUsage
from engine_console.services.nodeerr import problem_from_node
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store, paginate

log = logging.getLogger(__name__)
TOPIC = "downloads"
DISK_MARGIN = 1.02
STALL_S = 120.0                    # a node must accept a chunk, or answer a small call, within this long
VERIFY_BPS = 20e6                  # after the last byte the node hashes the whole file: allow this slow a disk
STAT_BATCH = 256                   # paths per fs.stat (proto.ARGS)
MAX_PATTERNS = 64
Layout = Literal["hf-home", "hf-hub", "files"]
HF_LAYOUTS: tuple[Layout, ...] = ("hf-home", "hf-hub")
HF_KINDS = ("hf-home", "hf-hub", "gguf", "auto")      # root kinds an HF repository may be downloaded or copied into
# Names a model directory usually holds. The node protocol has no directory listing, so a plain directory is copied by
# probing these (plus the weight shards its index names) and then PROVING nothing else is there: the byte total and file
# count must equal what the node's scan reported for the directory.
PLAIN_NAMES: tuple[str, ...] = (
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
    "tokenizer.tiktoken", "special_tokens_map.json", "vocab.json", "vocab.txt", "merges.txt", "added_tokens.json",
    "spiece.model", "sentencepiece.bpe.model", "bpe.model", "chat_template.json", "chat_template.jinja",
    "preprocessor_config.json", "processor_config.json", "image_processor_config.json", "video_preprocessor_config.json",
    "feature_extractor_config.json", "quantization_config.json", "hf_quant_config.json", "adapter_config.json",
    "params.json", "tekken.json", "model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
    "pytorch_model.bin.index.json", "consolidated.safetensors", "README.md", "LICENSE", "LICENSE.txt", "NOTICE",
    "USE_POLICY.md", ".gitattributes")
PLAIN_INDEXES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
INDEX_MAX = 4 * 1024 * 1024
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# An HF cache directory as a node lists it: [hub/]models--org--name, optionally down to one snapshot file.
_HF_REL = re.compile(r"^(?P<repo>(?:hub/)?models--(?P<org>[^/]+?)--(?P<name>[^/]+))"
                     r"(?:/snapshots/(?P<rev>[^/]+)(?:/(?P<path>.+))?)?$")


def _safe_rel_path(path: str) -> bool:
    """Hub-supplied file names are untrusted: no absolute paths, '..', backslashes or NULs."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    return all(seg not in ("", ".", "..") for seg in path.split("/"))


def _match(path: str, patterns: list[str] | None) -> bool:
    return not patterns or any(path == p or fnmatch.fnmatchcase(path, p) for p in patterns)


# ---- layouts (pure, unit-tested) -------------------------------------------------------------------------------------
def hf_repo_rel(layout: str, repo_id: str) -> str:
    """The repo's cache directory inside a root: under `hub/` for an HF_HOME, directly in a hub cache."""
    folder = "models--" + repo_id.replace("/", "--")
    return f"hub/{folder}" if layout == "hf-home" else folder


def files_dir(repo_id: str) -> str:
    return repo_id.replace("/", "--")


def blob_name(f: RepoFile, commit: str) -> str:
    """LFS files are content addressed (their sha256). A small file has no hash from the hub, so its blob is named
    after the commit and path: a newer commit's config.json is a new blob, never the stale one."""
    return f.sha256 or "nolfs-" + hashlib.sha256(f"{commit}\0{f.path}".encode()).hexdigest()[:40]


def snapshot_link(repo_rel: str, commit: str, path: str, blob: str) -> tuple[str, str]:
    """(link path, relative target) of one snapshot file, both inside the root."""
    link = f"{repo_rel}/snapshots/{commit}/{path}"
    return link, posixpath.relpath(f"{repo_rel}/blobs/{blob}", posixpath.dirname(link))


def layout_of(spec: RootSpec, usage: RootUsage, empty_auto: Layout = "hf-home") -> Layout | None:
    """How a transfer writes into a root, or None when the root cannot hold one. An `auto` root is what its last scan
    found: an HF cache stays one, loose model files get files, and an empty root takes `empty_auto` (an HF cache for a
    repository, plain files for a loose model). A root that holds only other kinds of store (Ollama...) is refused
    rather than turned into something the node scans differently."""
    if spec.kind in HF_LAYOUTS:
        return "hf-home" if spec.kind == "hf-home" else "hf-hub"
    if spec.kind in ("gguf", "plain"):
        return "files"
    if spec.kind == "auto":
        if usage.hf_home:
            return "hf-home"
        if usage.hf_hub:
            return "hf-hub"
        if usage.loose and not usage.ollama and not usage.other:
            return "files"
        if usage.entries == 0:
            return empty_auto
    return None


@dataclass(frozen=True)
class HfSource:
    """A copy that is a Hugging Face cache directory on its machine."""
    repo_id: str
    repo_rel: str                       # the repo directory inside the source root
    revision: str | None                # a commit, when the entry says which
    only: tuple[str, ...] | None        # snapshot paths to copy (one GGUF quant), None: the whole repo


@dataclass(frozen=True)
class GgufSource:
    """A copy that is one .gguf file (or every shard of a split set) outside any HF cache."""
    rels: tuple[str, ...]


@dataclass(frozen=True)
class PlainSource:
    """A plain model directory (config.json + weights). Its files cannot be listed through the node protocol, so
    `files` and `size` (what the node's scan counted) are what a copy checks its own enumeration against."""
    rel: str                            # "" when the root itself is the model directory
    name: str
    files: int
    size: int


def classify_copy_source(src: CopySource) -> HfSource | GgufSource | PlainSource | None:
    """What a copy of this library copy is made of, from where it sits. None: not copyable with the node protocol as
    it is (it has no directory listing, so an Ollama store or `other` cannot be enumerated at all)."""
    rel = src.copy.rel
    m = _HF_REL.match(rel)
    if m is not None:
        repo_id = f"{m['org']}/{m['name']}"
        try:
            check_repo_id(repo_id)
        except ValueError:
            return None
        commit = next((c for c in (m["rev"], src.revision) if c and _COMMIT.fullmatch(c)), None)
        only: tuple[str, ...] | None = None
        if m["path"] is not None:
            if not m["path"].lower().endswith(".gguf"):
                return None
            only = tuple(shard_names(m["path"], src.shards))
        return HfSource(repo_id=repo_id, repo_rel=m["repo"], revision=commit, only=only)
    if src.format == "gguf" and rel.lower().endswith(".gguf") and _safe_rel_path(rel):
        return GgufSource(rels=tuple(shard_names(rel, src.shards)))
    if src.format == "plain" and (rel == "" or _safe_rel_path(rel)):
        counted = src.copy.files
        exact = counted is not None and counted > 0 and not any("lower bound" in w for w in src.warnings)
        # without exact totals nothing could prove a copy complete: classify it as not copyable
        return PlainSource(rel=rel, name=src.name, files=counted or 0, size=src.copy.size_bytes) if exact else None
    return None


# ---- plans ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Target:
    host_id: str
    label: str
    root_id: str
    root_path: str
    layout: Layout
    kind: str                           # the root's policy kind: "gguf" roots only make sense for .gguf files
    node: NodeClient
    can_delete: bool                    # the node's policy allows fs.delete: a cancel may remove the partial


@dataclass
class _Item:
    path: str                           # shown as the current file
    size: int
    sha256: str | None
    dest: str                           # where the bytes land, relative to the target root
    link: str | None = None             # HF layouts: the snapshot link and its relative target
    link_target: str | None = None
    from_hf: tuple[str, str, str] | None = None        # (repo id, commit, repo path)
    from_node: tuple[str, str, str] | None = None      # (host id, root id, rel) on the source machine


@dataclass
class _Plan:
    items: list[_Item]
    layout: Layout
    commit: str | None = None
    repo_rel: str | None = None         # HF layouts: the repo directory on the target
    refs_name: str = "main"


@dataclass(frozen=True)
class _Stat:
    exists: bool = False
    size: int = 0
    partial: int = 0


class _Progress:
    """File and byte progress of one running row: persisted about once a second, published to the SSE about four times
    a second, with speed and ETA from the bytes moved since this run started."""

    def __init__(self, svc: DownloadService, did: str, total: int, done: int) -> None:
        self.svc, self.did, self.total, self.done = svc, did, total, done
        self.base = done
        self.started = svc._clock()
        self.last_pub = self.last_db = 0.0
        self.speed = 0.0

    def advance(self, n: int) -> None:
        self.done += n
        t = self.svc._clock()
        if t - self.last_pub < 0.25:
            return
        self.speed = (self.done - self.base) / max(t - self.started, 1e-6)
        eta = (self.total - self.done) / self.speed if self.speed > 0 else None
        self.last_pub = t
        if t - self.last_db >= 1.0:
            self.svc._set(self.did, done_bytes=self.done)
            self.last_db = t
        self.svc._bus.publish(TOPIC, {**self.svc._row(self.did).model_dump(), "done_bytes": self.done,
                                      "speed_bps": self.speed, "eta_s": eta})


async def _hf_chunks(stream: ByteStream) -> AsyncGenerator[bytes]:
    async for chunk in stream.chunks():
        yield chunk


class DownloadService:
    def __init__(self, store: Store, hub: HubClient, hf: HfService, settings: SettingsService, bus: EventBus,
                 fleet: FleetPort, library: LibraryService, *, concurrency: int = 2, max_queue: int = 50,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._db = store
        self._hub = hub
        self._hf = hf
        self._settings = settings
        self._bus = bus
        self._fleet = fleet
        self._library = library
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._intent: dict[str, str] = {}
        self._max_queue = max_queue
        self._clock = clock

    # -- persistence ---------------------------------------------------------------------------
    def _row(self, did: str) -> Download:
        r = self._db.one("SELECT * FROM downloads WHERE id=?", (did,))
        if r is None:
            raise NotFound(f"no such download: {did}")
        return self._to_model(r)

    @staticmethod
    def _to_model(r: Any, speed: float = 0.0, eta: float | None = None) -> Download:
        pats = json.loads(r["allow_patterns"]) if r["allow_patterns"] else None
        return Download(id=r["id"], repo_id=r["repo_id"], revision=r["revision"], commit_sha=r["commit_sha"],
                        allow_patterns=pats, state=r["state"], total_bytes=r["total_bytes"], done_bytes=r["done_bytes"],
                        speed_bps=speed, eta_s=eta, files_total=r["files_total"], files_done=r["files_done"],
                        current_file=r["current_file"], error=r["error"], error_code=r["error_code"],
                        created_at=r["created_at"], updated_at=r["updated_at"], host_id=r["host_id"],
                        root_id=r["root_id"], kind=r["kind"], source_copy_id=r["source_copy_id"])

    def _set(self, did: str, **cols: Any) -> None:
        cols["updated_at"] = now()
        sets = ", ".join(f"{k}=?" for k in cols)   # column names are literals from this module
        self._db.execute(f"UPDATE downloads SET {sets} WHERE id=?", [*cols.values(), did])  # noqa: S608

    def _publish(self, did: str, speed: float = 0.0, eta: float | None = None) -> None:
        r = self._db.one("SELECT * FROM downloads WHERE id=?", (did,))
        if r is not None:
            self._bus.publish(TOPIC, self._to_model(r, speed, eta).model_dump())

    def get(self, did: str) -> Download:
        return self._row(did)

    def list_page(self, limit: int = 50, cursor: str | None = None) -> Page[Download]:
        limit, off = paginate(limit, cursor)
        rows = self._db.all("SELECT * FROM downloads ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit + 1, off))
        return Page[Download](items=[self._to_model(r) for r in rows[:limit]],
                              next_cursor=str(off + limit) if len(rows) > limit else None)

    def recover(self) -> None:
        """A crashed process leaves running/queued rows behind: park them as paused so they can be resumed."""
        self._db.execute("UPDATE downloads SET state='paused' WHERE state IN ('queued','running')")

    # -- target machine and root ---------------------------------------------------------------
    def _label(self, host_id: str) -> str:
        return self._library.host_label(host_id)

    def _resolve(self, host_id: str, root_id: str | None, *, kinds: Collection[str], accept: Collection[str],
                 default: Collection[str], what: str, exclude: Collection[tuple[str, str]] = (),
                 empty_auto: Layout = "hf-home") -> _Target:
        """The machine and root a transfer writes into, or the problem that says why not: 404 unknown machine, 503 not
        connected, 409 monitor-only policy / read-only root / nothing suitable, 422 no such root or the wrong kind.
        Only roots of a policy kind in `kinds` qualify. A root named by the caller may then be any of the `accept`
        layouts; with none named, the first writable root of a `default` layout is used (for a Hugging Face
        repository: the machine's first writable HF cache)."""
        node = self._fleet.node(host_id)               # NotFound(code="unknown_host")
        label = node.host.label
        state = node.link().state
        if state != "up":
            raise problem_from_node(NodeUnavailable(host_id, f"it is not connected ({state})"), label)
        facts = self._library.host_facts(host_id)
        if facts is None:
            raise problem_from_node(NodeUnavailable(host_id, "its node has not reported yet"), label)
        if facts.mode != "manage":
            raise ProblemError(f"{label}: this machine's node policy is monitor-only, so nothing can be written to it",
                               code="monitor_only", status=409, host_id=host_id)
        chosen: tuple[RootSpec, Layout] | None = None
        if root_id is not None:
            spec = facts.root(root_id)
            if spec is None:
                raise Unprocessable(f"{label} has no model root {root_id[:40]!r}", code="unknown_root")
            if (host_id, spec.id) in exclude:
                raise Conflict(f"{label}: model root {spec.id!r} is where the model already is (or storage it shares)",
                               code="copy_same_location")
            if not spec.writable:
                raise Conflict(f"{label}: model root {spec.id!r} is read-only in its node policy",
                               code="root_not_writable")
            layout = layout_of(spec, self._library.root_usage(host_id, spec.id), empty_auto)
            if spec.kind not in kinds or layout is None or layout not in accept:
                raise Unprocessable(f"{label}: model root {spec.id!r} (kind {spec.kind}) cannot hold {what}; use a root "
                                    f"of kind {', '.join(kinds)}", code="root_kind_unsupported")
            chosen = (spec, layout)
        else:
            for spec in facts.roots:
                layout = (layout_of(spec, self._library.root_usage(host_id, spec.id), empty_auto)
                          if spec.writable and spec.kind in kinds else None)
                if layout is not None and layout in default and (host_id, spec.id) not in exclude:
                    chosen = (spec, layout)
                    break
            if chosen is None:
                raise Conflict(f"{label} has no writable model root that can hold {what}", code="no_writable_root")
        spec, layout = chosen
        return _Target(host_id=host_id, label=label, root_id=spec.id, root_path=spec.path, layout=layout,
                       kind=spec.kind, node=node, can_delete=facts.allow_delete)

    def _resolve_copy_target(self, host_id: str, root_id: str | None, kind: HfSource | GgufSource | PlainSource,
                             src: CopySource) -> _Target:
        """A repository's cache copies into an HF cache (or, flattened, into a gguf root); a loose .gguf file only into
        a gguf root; a plain model directory only into a plain root. Never onto the storage the copy is already on, on
        any machine that shares it."""
        if isinstance(kind, HfSource):
            return self._resolve(host_id, root_id, kinds=HF_KINDS, accept=("hf-home", "hf-hub", "files"),
                                 default=HF_LAYOUTS, exclude=src.locations, what="that model")
        if isinstance(kind, GgufSource):
            return self._resolve(host_id, root_id, kinds=("gguf", "auto"), accept=("files",), default=("files",),
                                 exclude=src.locations, what="a loose .gguf file", empty_auto="files")
        return self._resolve(host_id, root_id, kinds=("plain", "auto"), accept=("files",), default=("files",),
                             exclude=src.locations, what="a plain model directory", empty_auto="files")

    # -- commands ------------------------------------------------------------------------------
    def _check_capacity(self) -> None:
        pending = self._db.one("SELECT COUNT(*) AS n FROM downloads WHERE state IN ('queued','running','paused')")
        if pending is not None and pending["n"] >= self._max_queue:
            raise Conflict("too many unfinished downloads; cancel or finish some first", code="download_queue_full")

    def _check_duplicate(self, kind: str, name: str, host_id: str, root_id: str, source: str | None = None) -> None:
        """Two rows writing the same files into the same root would fight over one `.incomplete` partial."""
        row = self._db.one(
            "SELECT 1 FROM downloads WHERE kind=? AND repo_id=? AND host_id=? AND root_id=? AND COALESCE(source_copy_id,'')=?"
            " AND state IN ('queued','running','paused')", (kind, name, host_id, root_id, source or ""))
        if row is not None:
            raise Conflict("this is already being downloaded to that location (queued, running or paused)",
                           code="download_exists")

    async def create(self, req: DownloadRequest) -> Download:
        require_repo_id(req.repo_id)
        pats = req.allow_patterns
        if pats and (len(pats) > MAX_PATTERNS or any(not p or len(p) > 256 for p in pats)):
            raise BadRequest(f"allow_patterns: at most {MAX_PATTERNS} patterns of 1-256 characters", code="invalid_params")
        self._check_capacity()
        target = self._resolve(req.host_id, req.root_id, kinds=HF_KINDS, accept=("hf-home", "hf-hub", "files"),
                               default=HF_LAYOUTS, what="a Hugging Face repository")
        self._check_duplicate("hf", req.repo_id, target.host_id, target.root_id)
        did, ts = new_id("dl_"), now()
        self._db.execute(
            "INSERT INTO downloads(id,repo_id,revision,allow_patterns,state,created_at,updated_at,host_id,root_id,kind)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (did, req.repo_id, req.revision, json.dumps(pats) if pats else None, "queued", ts, ts, target.host_id,
             target.root_id, "hf"))
        self._start(did)
        return self._row(did)

    async def copy(self, copy_id: str, req: CopyRequest) -> Download:
        """Copy one library copy onto another machine or drive (`POST /library/copies/{id}/copy`)."""
        src = self._library.source(copy_id)
        if src is None:
            raise NotFound("no such model copy", code="copy_not_found")
        kind = classify_copy_source(src)
        if kind is None:
            raise ProblemError(
                f"a {src.format} model of this kind cannot be copied yet: only Hugging Face cache folders, .gguf files "
                "and plain model directories whose scan counted every file can be (the node protocol has no directory "
                "listing to enumerate anything else)", code="copy_unsupported", status=501)
        self._check_capacity()
        src_node = self._fleet.node(src.copy.host_id)
        if src_node.link().state != "up":
            raise problem_from_node(NodeUnavailable(src.copy.host_id, f"it is not connected ({src_node.link().state})"),
                               src_node.host.label)
        target = self._resolve_copy_target(req.host_id, req.root_id, kind, src)
        self._check_duplicate("copy", src.name, target.host_id, target.root_id, copy_id)
        did, ts = new_id("dl_"), now()
        self._db.execute(
            "INSERT INTO downloads(id,repo_id,revision,allow_patterns,state,created_at,updated_at,host_id,root_id,kind,"
            "source_copy_id) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (did, src.name, src.revision, None, "queued", ts, ts, target.host_id, target.root_id, "copy", copy_id))
        self._start(did)
        return self._row(did)

    def _start(self, did: str) -> None:
        self._intent.pop(did, None)
        self._tasks[did] = asyncio.create_task(self._run(did), name=f"download-{did}")

    async def pause(self, did: str) -> Download:
        dl = self._row(did)
        if dl.state not in ("queued", "running"):
            raise Conflict(f"cannot pause a {dl.state} download")
        await self._stop_task(did, "paused")
        if self._row(did).state in ("queued", "running"):     # cancelled before its first step: nobody else will say so
            self._set(did, state="paused")
        return self._row(did)

    async def resume(self, did: str) -> Download:
        dl = self._row(did)
        if dl.state not in ("paused", "failed"):
            raise Conflict(f"cannot resume a {dl.state} download")
        self._set(did, state="queued", error=None, error_code=None)
        self._start(did)
        return self._row(did)

    async def cancel(self, did: str) -> Download:
        dl = self._row(did)
        if dl.state in ("completed", "cancelled"):
            raise Conflict(f"download already {dl.state}")
        live = self._is_live(did)
        await self._stop_task(did, "cancelled")
        self._set(did, state="cancelled")
        if not live:                            # a live task cleans up in its own cancel handler
            await self._cleanup_partial(did)
        self._publish(did)
        return self._row(did)

    async def delete(self, did: str) -> None:
        dl = self._row(did)
        live = self._is_live(did)
        await self._stop_task(did, "cancelled")
        if dl.state != "completed" and not live:
            await self._cleanup_partial(did)
        self._db.execute("DELETE FROM downloads WHERE id=?", (did,))

    def _is_live(self, did: str) -> bool:
        """A task that has not ended (a failed or finished row keeps its dead task in the table until it is stopped)."""
        t = self._tasks.get(did)
        return t is not None and not t.done()

    async def _stop_task(self, did: str, intent: str) -> None:
        task = self._tasks.pop(did, None)
        self._intent[did] = intent
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        elif intent == "paused":
            self._set(did, state="paused")

    async def wait(self, did: str) -> None:
        """Test/ops helper: block until the download task ends."""
        t = self._tasks.get(did)
        if t:
            await asyncio.gather(t, return_exceptions=True)

    async def shutdown(self) -> None:
        for did in list(self._tasks):
            await self._stop_task(did, "paused")

    # -- worker --------------------------------------------------------------------------------------------------
    def _failure(self, e: NodeError) -> tuple[str, str]:
        """(error_code, text) of a node refusal. A full disk reads the same whoever noticed it first."""
        code = "insufficient_disk" if e.code == proto.NO_SPACE else e.code
        return code, scrub(f"{self._label(e.host_id)}: {e.message}")[:500]

    async def _run(self, did: str) -> None:
        dl = self._row(did)
        try:
            async with self._sem:
                self._set(did, state="running")
                if dl.kind == "copy":
                    await self._copy(did, dl)
                else:
                    await self._download(did, dl)
        except asyncio.CancelledError:
            intent = self._intent.get(did, "paused")
            self._set(did, state=intent)
            if intent == "cancelled":
                await self._cleanup_partial(did)
            self._publish(did)
        except NodeError as e:
            code, text = self._failure(e)
            self._set(did, state="failed", error=text, error_code=code)
            self._publish(did)
        except ProblemError as e:
            self._set(did, state="failed", error=scrub(e.detail)[:500], error_code=e.code)
            self._publish(did)
        except Exception as e:  # noqa: BLE001 - a worker must never die silently
            log.exception("download %s crashed", did)
            self._set(did, state="failed", error=f"{type(e).__name__}: {scrub(str(e))}"[:500], error_code="download_failed")
            self._publish(did)

    async def _cleanup_partial(self, did: str) -> None:
        """Remove the file in flight's `.incomplete` partial, when the node's policy allows deletes. Best effort: if it
        cannot be done the partial stays, and downloading the same file again resumes from it."""
        r = self._db.one("SELECT host_id, root_id, current_dest FROM downloads WHERE id=?", (did,))
        if r is None or not r["current_dest"] or not r["root_id"]:
            return
        facts = self._library.host_facts(r["host_id"])
        if facts is None or not facts.allow_delete or facts.mode != "manage":
            return
        with contextlib.suppress(NodeError, NotFound, TimeoutError):
            node = self._fleet.node(r["host_id"])
            await asyncio.wait_for(node.call("fs.delete", {"root": r["root_id"], "rel": r["current_dest"] + ".incomplete"},
                                             timeout=STALL_S), STALL_S + 10)
            self._set(did, current_dest=None)

    # -- planning ------------------------------------------------------------------------------------------------
    @staticmethod
    def _checked_listing(files: list[RepoFile], sha: str | None) -> str:
        """The commit the listing is for, after the hub-supplied names, sizes and hashes have been vetted."""
        if not sha or not _COMMIT.fullmatch(sha):
            raise ProblemError("hub returned a missing or malformed commit sha", code="hf_bad_sha")
        for f in files:
            if not _safe_rel_path(f.path):
                raise ProblemError("hub listed an unsafe file path; refusing the whole download", code="hf_bad_path")
            if f.sha256 is not None and not _SHA256.fullmatch(f.sha256):
                raise ProblemError(f"hub returned a malformed sha256 for {f.path[:80]}", code="hf_bad_hash")
            if f.size < 0:
                raise ProblemError("hub returned a negative file size", code="hf_bad_size")
        return sha

    @staticmethod
    def _hf_items(target: _Target, repo_id: str, commit: str, files: list[RepoFile],
                  source: Callable[[RepoFile], tuple[str, tuple[str, str, str]] | None]) -> _Plan:
        """The plan for `files` of one repo at one commit. `source(f)` says where each file's bytes come from
        ("hf" or "node", and the coordinates), or None to leave a file out."""
        repo_rel = hf_repo_rel(target.layout, repo_id)
        items: list[_Item] = []
        for f in files:
            src = source(f)
            if src is None:
                continue
            item = _Item(path=f.path, size=f.size, sha256=f.sha256, dest="")
            if target.layout == "files":
                item.dest = f"{files_dir(repo_id)}/{f.path}"
            else:
                blob = blob_name(f, commit)
                item.dest = f"{repo_rel}/blobs/{blob}"
                item.link, item.link_target = snapshot_link(repo_rel, commit, f.path, blob)
            if src[0] == "hf":
                item.from_hf = src[1]
            else:
                item.from_node = src[1]
            items.append(item)
        return _Plan(items=items, layout=target.layout, commit=commit, repo_rel=repo_rel)

    async def _download(self, did: str, dl: Download) -> None:
        target = self._resolve(dl.host_id, dl.root_id, kinds=HF_KINDS, accept=("hf-home", "hf-hub", "files"),
                               default=HF_LAYOUTS, what="a Hugging Face repository")
        hub = await self._hf.list_files(dl.repo_id, dl.revision)
        commit = self._checked_listing(hub.files, hub.sha)
        files = [f for f in hub.files if _match(f.path, dl.allow_patterns)]
        if not files:
            raise BadRequest("no files match allow_patterns", code="no_files")
        if target.kind == "gguf" and not any(f.path.lower().endswith(".gguf") for f in files):
            raise BadRequest("a gguf model root holds .gguf files, and none of the selected files is one",
                             code="no_gguf_files")
        plan = self._hf_items(target, dl.repo_id, commit, files, lambda f: ("hf", (dl.repo_id, commit, f.path)))
        plan.refs_name = dl.revision or "main"
        await self._execute(did, target, plan)

    async def _copy(self, did: str, dl: Download) -> None:
        src = self._library.source(dl.source_copy_id or "")
        if src is None:
            raise NotFound("the model copy this row was copying is no longer in the library", code="copy_source_gone")
        kind = classify_copy_source(src)
        if kind is None:
            raise ProblemError("this model can no longer be copied", code="copy_unsupported", status=501)
        target = self._resolve_copy_target(dl.host_id, dl.root_id, kind, src)
        src_node = self._fleet.node(src.copy.host_id)
        if src_node.link().state != "up":
            raise problem_from_node(NodeUnavailable(src.copy.host_id, f"it is not connected ({src_node.link().state})"),
                               src_node.host.label)
        if isinstance(kind, HfSource):
            plan = await self._plan_hf_copy(target, src, kind, src_node)
        elif isinstance(kind, GgufSource):
            plan = await self._plan_gguf_copy(target, src, kind, src_node)
        else:
            plan = await self._plan_plain_copy(target, src, kind, src_node)
        await self._execute(did, target, plan)

    async def _plan_hf_copy(self, target: _Target, src: CopySource, kind: HfSource, src_node: NodeClient) -> _Plan:
        """The source machine has the large files under their sha256 names; the hub's listing of the repository gives
        every file's path, size and hash. Large files that the source holds are copied machine to machine; the few
        small files (which a cache may name after its own git object ids) are fetched from Hugging Face, so a copy
        needs the hub's egress for them. A file the source lacks is skipped: a partial copy stays partial."""
        hub = await self._hf.list_files(kind.repo_id, kind.revision)
        commit = self._checked_listing(hub.files, hub.sha)
        files = [f for f in hub.files if kind.only is None or f.path in kind.only]
        if not files:
            raise BadRequest("none of the model's files are in the repository listing", code="no_files")
        large = [f for f in files if f.sha256]
        stats = await self._stat(src_node, src.copy.host_id, src.copy.root_id,
                                 [f"{kind.repo_rel}/blobs/{f.sha256}" for f in large])
        held = {f.path for f in large
                if stats[f"{kind.repo_rel}/blobs/{f.sha256}"].exists
                and stats[f"{kind.repo_rel}/blobs/{f.sha256}"].size == f.size}
        if large and not held:
            raise ProblemError("none of the repository's large files are on the source copy at the revision the hub "
                               "lists (the copy is at another revision, or is incomplete)", code="copy_source_mismatch",
                               status=409)

        def where(f: RepoFile) -> tuple[str, tuple[str, str, str]] | None:
            if not f.sha256:
                return "hf", (kind.repo_id, commit, f.path)
            if f.path in held:
                return "node", (src.copy.host_id, src.copy.root_id, f"{kind.repo_rel}/blobs/{f.sha256}")
            return None

        return self._hf_items(target, kind.repo_id, commit, files, where)

    async def _plan_gguf_copy(self, target: _Target, src: CopySource, kind: GgufSource, src_node: NodeClient) -> _Plan:
        stats = await self._stat(src_node, src.copy.host_id, src.copy.root_id, list(kind.rels))
        items: list[_Item] = []
        for rel in kind.rels:
            st = stats[rel]
            if not st.exists:
                raise ProblemError(f"{rel.rsplit('/', 1)[-1][:120]} is missing on the source machine (a split set needs "
                                   "every shard)", code="copy_source_mismatch", status=409)
            items.append(_Item(path=posixpath.basename(rel), size=st.size, sha256=None, dest=rel,
                               from_node=(src.copy.host_id, src.copy.root_id, rel)))
        return _Plan(items=items, layout="files")

    async def _plan_plain_copy(self, target: _Target, src: CopySource, kind: PlainSource, src_node: NodeClient) -> _Plan:
        """A plain model directory cannot be listed, so its files are found by name: the usual ones, plus the weight shards
        its index names. The enumeration is then PROVED complete (not just plausible): the files found must add up to
        exactly the byte total and file count the node's scan reported for the directory. Anything else (a file with an
        unusual name, a subfolder, a scan that was only a lower bound, a directory that changed since the scan) is
        refused with the numbers, never copied half."""
        host, root = src.copy.host_id, src.copy.root_id
        base = f"{kind.rel}/" if kind.rel else ""
        found = await self._probe(src_node, host, root, base, list(PLAIN_NAMES))
        shards: list[str] = []
        for index in PLAIN_INDEXES:
            if index in found:
                shards += await self._weight_files(src_node, host, root, base + index, found[index].size)
        found.update(await self._probe(src_node, host, root, base, [n for n in dict.fromkeys(shards) if n not in found]))
        total = sum(st.size for st in found.values())
        if len(found) != kind.files or total != kind.size:
            raise ProblemError(
                f"found {len(found)} of the directory's {kind.files} files ({total} of {kind.size} bytes) by their usual "
                "names; the node protocol cannot list a directory, so a copy would be incomplete (a file with an unusual "
                "name, a subfolder, or a change since the last scan: rescan, or copy it another way)",
                code="copy_listing_incomplete", status=501)
        dirname = re.sub(r"[^A-Za-z0-9._-]", "_", posixpath.basename(kind.rel) or kind.name).lstrip(".")[:100] or "model"
        items = [_Item(path=name, size=st.size, sha256=None, dest=f"{dirname}/{name}",
                       from_node=(host, root, base + name)) for name, st in sorted(found.items())]
        return _Plan(items=items, layout="files")

    async def _probe(self, node: NodeClient, host_id: str, root_id: str, base: str, names: list[str]) -> dict[str, _Stat]:
        """The regular files among `names` (relative to `base`) that exist, with their sizes."""
        stats = await self._stat(node, host_id, root_id, [base + n for n in names])
        return {n: stats[base + n] for n in names if stats[base + n].exists}

    async def _weight_files(self, node: NodeClient, host_id: str, root_id: str, rel: str, size: int) -> list[str]:
        """The shard file names a weight index (model.safetensors.index.json) lists. Untrusted JSON: only plain
        relative names survive."""
        if size > INDEX_MAX:
            raise ProblemError("a weight index is larger than 4 MiB; its shards cannot be listed", code="copy_listing_incomplete",
                               status=501)
        buf = bytearray()
        async with contextlib.aclosing(self._node_chunks(node, root_id, rel, 0)) as chunks:
            async for chunk in chunks:
                buf += chunk
                if len(buf) > INDEX_MAX:
                    raise ProblemError("a weight index is larger than 4 MiB; its shards cannot be listed",
                                       code="copy_listing_incomplete", status=501)
        try:
            doc = json.loads(bytes(buf))
        except ValueError:
            return []
        weight_map = doc.get("weight_map") if isinstance(doc, dict) else None
        names = {v for v in weight_map.values() if isinstance(v, str)} if isinstance(weight_map, dict) else set()
        return sorted(n for n in names if _safe_rel_path(n) and len(n) <= 200)

    # -- node calls ------------------------------------------------------------------------------------------------
    async def _call(self, node: NodeClient, op: str, args: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
        return await asyncio.wait_for(node.call(op, args, timeout=timeout), timeout + 10.0)

    async def _stat(self, node: NodeClient, host_id: str, root_id: str, rels: list[str]) -> dict[str, _Stat]:
        """exists / size / partial of each path, `STAT_BATCH` at a time. A path the node does not mention does not
        exist."""
        out: dict[str, _Stat] = {}
        for i in range(0, len(rels), STAT_BATCH):
            batch = rels[i:i + STAT_BATCH]
            res = await self._call(node, "fs.stat", {"root": root_id, "paths": batch})
            by_rel = {it.get("rel"): it for it in (res.get("items") or []) if isinstance(it, dict)}
            for rel in batch:
                it = by_rel.get(rel) or {}
                size, partial = it.get("size"), it.get("partial")
                out[rel] = _Stat(exists=it.get("exists") is True and it.get("is_dir") is not True,
                                 size=size if isinstance(size, int) and size > 0 else 0,
                                 partial=partial if isinstance(partial, int) and partial > 0 else 0)
        return out

    # -- execution ----------------------------------------------------------------------------------------------------
    def _done_paths(self, did: str) -> set[str]:
        r = self._db.one("SELECT done_json FROM downloads WHERE id=?", (did,))
        try:
            raw = json.loads(r["done_json"]) if r is not None and r["done_json"] else []
        except ValueError:
            raw = []
        return {p for p in raw if isinstance(p, str)} if isinstance(raw, list) else set()

    async def _execute(self, did: str, target: _Target, plan: _Plan) -> None:
        items = plan.items
        total = sum(i.size for i in items)
        stats = await self._stat(target.node, target.host_id, target.root_id, [i.dest for i in items])
        finished = self._done_paths(did)
        complete: list[bool] = []
        already = partial_sum = 0
        for it in items:
            st = stats[it.dest]
            # A content-addressed blob that exists is that content. A file under its final name could be an older
            # file of the same size, so only one this row itself finished counts.
            ok = st.exists and st.size == it.size and (plan.layout != "files" or it.dest in finished)
            complete.append(ok)
            if ok:
                already += it.size
            elif 0 < st.partial <= it.size:
                partial_sum += st.partial
        need = total - already - partial_sum
        free = self._library.free_bytes(target.host_id, target.root_path)
        if free is not None and need * DISK_MARGIN > free:
            raise ProblemError(f"{target.label}: need {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB free at "
                               f"{target.root_path}", code="insufficient_disk", status=507)
        files_done = sum(complete)
        self._set(did, commit_sha=plan.commit, total_bytes=total, files_total=len(items),
                  done_bytes=already + partial_sum, files_done=files_done)
        self._publish(did)
        progress = _Progress(self, did, total, already + partial_sum)
        written: set[str] = set()
        for it, ok in zip(items, complete, strict=True):
            if ok:
                continue
            if it.dest in written:                # two files with the same content share one blob
                progress.done += it.size
                files_done += 1
                continue
            st = stats[it.dest]
            self._set(did, current_file=it.path, current_dest=it.dest)
            await self._transfer(target, it, st.partial if 0 < st.partial <= it.size else 0, progress)
            written.add(it.dest)
            finished.add(it.dest)
            self._set(did, done_json=json.dumps(sorted(finished)), current_dest=None)
            files_done += 1
            self._set(did, files_done=files_done, done_bytes=progress.done)
        if plan.layout != "files" and plan.repo_rel is not None and plan.commit is not None:
            self._set(did, current_file="snapshot links", current_dest=None)
            for it in items:
                await self._link(target, it)
            await self._call(target.node, "fs.put_small", {
                "root": target.root_id, "rel": f"{plan.repo_rel}/refs/{plan.refs_name}",
                "content_b64": base64.b64encode(plan.commit.encode()).decode()})
        self._set(did, state="completed", done_bytes=total, files_done=len(items), current_file=None, current_dest=None)
        self._publish(did)
        # the node finds the new model on its next scan; ask for it now so the library shows it at once
        with contextlib.suppress(Exception):
            await self._library.rescan(target.host_id)

    async def _link(self, target: _Target, it: _Item) -> None:
        if it.link is None or it.link_target is None:
            return
        try:
            await self._call(target.node, "fs.symlink", {"root": target.root_id, "link": it.link,
                                                         "target": it.link_target})
        except NodeError as e:
            # the link of an earlier run is still there: fine, as long as it is
            if e.code not in (proto.CONFLICT, proto.IO_ERROR) or not (
                    await self._stat(target.node, target.host_id, target.root_id, [it.link]))[it.link].exists:
                raise

    async def _transfer(self, target: _Target, it: _Item, partial: int, progress: _Progress) -> None:
        """One file: from its source into `fs.write`, resuming a partial the node holds. The node checks the size and
        the sha256 before it renames the partial into place."""
        start = partial
        hf_stream: ByteStream | None = None
        try:
            chunks: AsyncGenerator[bytes]
            if start >= it.size:
                chunks = _no_chunks()                 # the partial is whole: only the node's verification is left
            elif it.from_hf is not None:
                repo_id, commit, path = it.from_hf
                hf_stream = await self._hub.open_file(repo_id, commit, path, start)
                if hf_stream.offset != start:         # the server ignored the Range: this file starts over
                    progress.done -= start - hf_stream.offset
                    start = hf_stream.offset
                chunks = _hf_chunks(hf_stream)
            elif it.from_node is not None:
                src_host, src_root, src_rel = it.from_node
                chunks = self._node_chunks(self._fleet.node(src_host), src_root, src_rel, start)
            else:
                raise ProblemError("internal: a plan item has no source", code="internal_error")
            await self._write_remote(target, it, start, chunks, progress)
        finally:
            if hf_stream is not None:
                await hf_stream.aclose()

    async def _write_remote(self, target: _Target, it: _Item, offset: int, chunks: AsyncGenerator[bytes],
                            progress: _Progress) -> None:
        args: dict[str, Any] = {"root": target.root_id, "rel": it.dest, "size": it.size, "offset": offset}
        if it.sha256:
            args["sha256"] = it.sha256
        stream = target.node.open("fs.write", args)
        sent = offset
        try:
            async with contextlib.aclosing(chunks):
                async for chunk in chunks:
                    if sent + len(chunk) > it.size:   # never trust a source to stop at the declared size
                        raise ProblemError(f"{it.path[:80]}: the source sent more than the declared {it.size} bytes",
                                           code="size_mismatch", status=502)
                    for i in range(0, len(chunk), proto.MAX_DATA):
                        await asyncio.wait_for(stream.write(chunk[i:i + proto.MAX_DATA]), STALL_S)
                    sent += len(chunk)
                    progress.advance(len(chunk))
            if sent != it.size:
                # no end(): the node would discard a short file; cancelling keeps the partial for the next try
                raise ProblemError(f"{it.path[:80]}: the source ended after {sent} of {it.size} bytes",
                                   code="size_mismatch", status=502)
            await asyncio.wait_for(stream.end(), STALL_S)
            await asyncio.wait_for(stream.result(), STALL_S + it.size / VERIFY_BPS)
        except TimeoutError:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.cancel(), 5.0)
            raise NodeError(target.host_id, proto.TIMEOUT, "the machine stopped accepting data; resume to continue") from None
        except BaseException:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.cancel(), 5.0)
            raise

    @staticmethod
    async def _node_chunks(node: NodeClient, root_id: str, rel: str, offset: int) -> AsyncGenerator[bytes]:
        stream = node.open("fs.read", {"root": root_id, "rel": rel, "offset": offset})
        try:
            async for item in stream:
                if item.kind == "data" and item.data:
                    yield item.data
            await stream.result()                     # a failure on the source machine surfaces here
        except BaseException:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(stream.cancel(), 5.0)
            raise


async def _no_chunks() -> AsyncGenerator[bytes]:
    return
    yield b""   # pragma: no cover - makes this an async generator
