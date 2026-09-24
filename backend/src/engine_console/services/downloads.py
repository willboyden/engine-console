"""Resumable snapshot downloads into the HF hub cache layout (blobs/ + snapshots/<sha>/ symlinks + refs/),
so engines mounting the cache resolve the model by repo id with HF_HUB_OFFLINE=1."""
from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir
from huggingface_hub.errors import HFValidationError
from huggingface_hub.file_download import repo_folder_name

from engine_console.domain.errors import BadRequest, Conflict, NotFound, ProblemError
from engine_console.domain.models import Download, DownloadRequest, LocalModel, Page
from engine_console.domain.ports import HubClient, HubModel, RepoFile
from engine_console.domain.repo_id import require_repo_id
from engine_console.services.common import EventBus, new_id, now
from engine_console.services.hf import HfService
from engine_console.services.settings import SettingsService
from engine_console.services.store import Store, paginate

log = logging.getLogger(__name__)
DISK_MARGIN = 1.02
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _safe_rel_path(path: str) -> bool:
    """Hub-supplied file names are untrusted: no absolute paths, '..', backslashes or NULs."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return False
    return all(seg not in ("", ".", "..") for seg in path.split("/"))
TOPIC = "downloads"


def _blob_name(f: RepoFile) -> str:
    return f.sha256 or "nolfs-" + hashlib.sha256(f.path.encode()).hexdigest()[:40]


def _match(path: str, patterns: list[str] | None) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, p) for p in patterns)


def _existing_parent(p: Path) -> Path:
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class DownloadService:
    def __init__(self, store: Store, hub: HubClient, hf: HfService, settings: SettingsService, bus: EventBus,
                 *, concurrency: int = 2, in_use: Callable[[str], bool] | None = None, max_queue: int = 50,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._db = store
        self._hub = hub
        self._hf = hf
        self._settings = settings
        self._bus = bus
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._intent: dict[str, str] = {}
        self._in_use = in_use or (lambda _r: False)
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
                        created_at=r["created_at"], updated_at=r["updated_at"])

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
        self._db.execute("UPDATE downloads SET state='paused' WHERE state IN ('running','queued')")

    # -- commands ------------------------------------------------------------------------------
    async def create(self, req: DownloadRequest) -> Download:
        require_repo_id(req.repo_id)
        pending = self._db.one("SELECT COUNT(*) AS n FROM downloads WHERE state IN ('queued','running','paused')")
        if pending is not None and pending["n"] >= self._max_queue:
            raise Conflict("too many unfinished downloads; cancel or finish some first", code="download_queue_full")
        did, ts = new_id("dl_"), now()
        self._db.execute("INSERT INTO downloads(id,repo_id,revision,allow_patterns,state,created_at,updated_at)"
                         " VALUES(?,?,?,?,?,?,?)",
                         (did, req.repo_id, req.revision, json.dumps(req.allow_patterns) if req.allow_patterns else None,
                          "queued", ts, ts))
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
        await self._stop_task(did, "cancelled")
        self._set(did, state="cancelled")
        self._publish(did)
        return self._row(did)

    async def delete(self, did: str) -> None:
        self._row(did)
        await self._stop_task(did, "cancelled")
        self._db.execute("DELETE FROM downloads WHERE id=?", (did,))

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

    # -- worker --------------------------------------------------------------------------------
    async def _run(self, did: str) -> None:
        dl = self._row(did)
        try:
            async with self._sem:
                self._set(did, state="running")
                await self._download(did, dl)
        except asyncio.CancelledError:
            intent = self._intent.get(did, "paused")
            self._set(did, state=intent)
            if intent == "cancelled":
                self._cleanup_partial(dl)
            self._publish(did)
        except ProblemError as e:
            self._set(did, state="failed", error=e.detail, error_code=e.code)
            self._publish(did)
        except Exception as e:  # noqa: BLE001 - a worker must never die silently
            log.exception("download %s crashed", did)
            self._set(did, state="failed", error=f"{type(e).__name__}: {e}", error_code="download_failed")
            self._publish(did)

    def _repo_dir(self, repo_id: str) -> Path:
        require_repo_id(repo_id)
        try:
            return self._settings.hub_dir() / repo_folder_name(repo_id=repo_id, repo_type="model")
        except HFValidationError:
            raise BadRequest("repo_id must look like 'org/name'", code="invalid_repo_id") from None

    def _cleanup_partial(self, dl: Download) -> None:
        blobs = self._repo_dir(dl.repo_id) / "blobs"
        if blobs.is_dir():
            for p in blobs.glob("*.incomplete"):
                p.unlink(missing_ok=True)

    async def _download(self, did: str, dl: Download) -> None:
        hub: HubModel = await self._hf.list_files(dl.repo_id, dl.revision)
        if not hub.sha or not _COMMIT.fullmatch(hub.sha):
            raise ProblemError("hub returned a missing or malformed commit sha", code="hf_bad_sha")
        for f in hub.files:
            if not _safe_rel_path(f.path):
                raise ProblemError("hub listed an unsafe file path; refusing the whole download", code="hf_bad_path")
            if f.sha256 is not None and not _SHA256.fullmatch(f.sha256):
                raise ProblemError(f"hub returned a malformed sha256 for {f.path[:80]}", code="hf_bad_hash")
            if f.size < 0:
                raise ProblemError("hub returned a negative file size", code="hf_bad_size")
        files = [f for f in hub.files if _match(f.path, dl.allow_patterns)]
        if not files:
            raise BadRequest("no files match allow_patterns", code="no_files")
        repo = self._repo_dir(dl.repo_id)
        blobs, snap = repo / "blobs", repo / "snapshots" / hub.sha
        total = sum(f.size for f in files)
        already = sum(f.size for f in files if (blobs / _blob_name(f)).is_file())
        partial = sum((blobs / (_blob_name(f) + ".incomplete")).stat().st_size for f in files
                      if (blobs / (_blob_name(f) + ".incomplete")).is_file())
        need = total - already - partial
        free = shutil.disk_usage(_existing_parent(blobs)).free
        if need * DISK_MARGIN > free:
            raise ProblemError(f"need {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB free at {repo.parent}",
                               code="insufficient_disk", status=507)
        blobs.mkdir(parents=True, exist_ok=True)
        snap.mkdir(parents=True, exist_ok=True)
        self._set(did, commit_sha=hub.sha, total_bytes=total, files_total=len(files), done_bytes=already,
                  files_done=sum(1 for f in files if (blobs / _blob_name(f)).is_file()))
        done = already
        files_done = sum(1 for f in files if (blobs / _blob_name(f)).is_file())
        started, base = self._clock(), done
        last_pub = last_db = 0.0
        speed = 0.0
        for f in files:
            blob = blobs / _blob_name(f)
            if not blob.is_file():
                part = blob.with_name(blob.name + ".incomplete")
                start = part.stat().st_size if part.is_file() else 0
                self._set(did, current_file=f.path)
                if start > f.size:      # corrupt partial: start over
                    part.unlink()
                    start = 0
                if start < f.size or f.size == 0:
                    stream = await self._hub.open_file(dl.repo_id, hub.sha, f.path, start)
                    try:
                        if stream.offset != start:   # server ignored Range: restart this file cleanly
                            done -= start - stream.offset
                            start = stream.offset
                        with part.open("ab" if start else "wb") as fh:
                            got = start
                            async for chunk in stream.chunks():
                                got += len(chunk)
                                if got > f.size:   # never trust the server to stop at the declared size
                                    raise ProblemError(f"{f.path[:80]}: server sent more than the declared {f.size} bytes",
                                                       code="size_mismatch")
                                await asyncio.to_thread(fh.write, chunk)
                                done += len(chunk)
                                t = self._clock()
                                if t - last_pub >= 0.25:
                                    speed = (done - base) / max(t - started, 1e-6)
                                    eta = (total - done) / speed if speed > 0 else None
                                    last_pub = t
                                    if t - last_db >= 1.0:
                                        self._set(did, done_bytes=done)
                                        last_db = t
                                    self._bus.publish(TOPIC, {**self._row(did).model_dump(), "done_bytes": done,
                                                              "speed_bps": speed, "eta_s": eta})
                    finally:
                        await stream.aclose()
                if part.stat().st_size != f.size:
                    raise ProblemError(f"{f.path}: got {part.stat().st_size} bytes, expected {f.size}",
                                       code="size_mismatch")
                if f.sha256:
                    digest = await asyncio.to_thread(_sha256_file, part)
                    if digest != f.sha256:
                        part.unlink(missing_ok=True)
                        raise ProblemError(f"{f.path}: sha256 mismatch (partial file discarded, resume to refetch)",
                                           code="hash_mismatch")
                os.replace(part, blob)
            files_done += 1
            link = snap / f.path
            snap_real = snap.resolve()

            def inside(p: Path, root: Path = snap_real) -> bool:
                r = p.resolve()
                return r == root or root in r.parents

            if not inside(link.parent):   # before mkdir/unlink/symlink touch anything
                raise ProblemError("file path escapes the snapshot directory", code="hf_bad_path")
            link.parent.mkdir(parents=True, exist_ok=True)
            if not inside(link.parent):
                raise ProblemError("file path escapes the snapshot directory", code="hf_bad_path")
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.relpath(blob, link.parent))
            self._set(did, files_done=files_done, done_bytes=done)
        refs = repo / "refs"
        refs.mkdir(exist_ok=True)
        (refs / (dl.revision or "main")).write_text(hub.sha)
        self._set(did, state="completed", done_bytes=total, current_file=None)
        self._publish(did)

    # -- local library ---------------------------------------------------------------------------
    def list_local(self) -> list[LocalModel]:
        hub_dir = self._settings.hub_dir()
        if not hub_dir.is_dir():
            return []
        try:
            info = scan_cache_dir(hub_dir)
        except Exception:  # noqa: BLE001 - corrupt/partial cache must not break the UI
            log.warning("scan_cache_dir failed for %s", hub_dir)
            return []
        out = [LocalModel(repo_id=r.repo_id, size_bytes=r.size_on_disk, revisions=sorted(v.commit_hash for v in r.revisions),
                          last_used=r.last_accessed or None, path=str(r.repo_path))
               for r in info.repos if r.repo_type == "model"]
        return sorted(out, key=lambda m: m.repo_id)

    def local_snapshot(self, repo_id: str) -> tuple[Path, list[RepoFile]] | None:
        repo = self._repo_dir(repo_id)
        ref = repo / "refs" / "main"
        if not ref.is_file():
            snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
            if not snaps:
                return None
            snap = snaps[-1]
        else:
            snap = repo / "snapshots" / ref.read_text().strip()
        if not snap.is_dir():
            return None
        files = [RepoFile(path=str(p.relative_to(snap)), size=p.stat().st_size) for p in snap.rglob("*")
                 if p.is_file() or p.is_symlink()]
        return snap, files

    def is_cached(self, repo_id: str) -> bool:
        return self.local_snapshot(repo_id) is not None

    def delete_local(self, repo_id: str) -> None:
        repo = self._repo_dir(repo_id).resolve()
        hub = self._settings.hub_dir().resolve()
        if repo.parent != hub or not repo.name.startswith("models--"):
            raise BadRequest("refusing to delete outside the hub cache", code="invalid_repo_id")
        if not repo.is_dir():
            raise NotFound(f"model not in local cache: {repo_id}")
        if self._in_use(repo_id):
            raise Conflict("model is used by a running instance; stop it first", code="model_in_use")
        shutil.rmtree(repo)

    def local_hub_model(self, repo_id: str) -> HubModel | None:
        """Offline HubModel from the cached snapshot (config.json + on-disk sizes), for preflight and the library."""
        snap_files = self.local_snapshot(repo_id)
        if snap_files is None:
            return None
        snap, files = snap_files
        cfg: dict[str, Any] | None = None
        cfg_path = snap / "config.json"
        if cfg_path.is_file():
            try:
                parsed = json.loads(cfg_path.read_text())
                cfg = parsed if isinstance(parsed, dict) else None
            except (OSError, ValueError):
                cfg = None
        return HubModel(repo_id=repo_id, sha=snap.name, gated=False, license=None, pipeline_tag=None,
                        library_name=None, tags=[], files=files, config=cfg)
