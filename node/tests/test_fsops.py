"""fs.* ops: path rules, symlink escapes, secret names, resumable uploads with whole-file sha256, deletes."""
from __future__ import annotations

import base64
import hashlib
import os
import threading
from typing import Any

import pytest

from ecnode import fsops, proto
from ecnode.errors import Cancelled, OpError
from ecnode.fsops import WriteLocks, fs_delete, fs_put_small, fs_read, fs_stat, fs_symlink, fs_write
from ecnode.policy import parse


class FakeStream:
    def __init__(self, chunks: list[bytes] | None = None, cancel_after: int | None = None) -> None:
        self.chunks = list(chunks or [])
        self.cancel_after = cancel_after
        self.reads = 0
        self.events: list[dict[str, Any]] = []
        self.data: list[bytes] = []
        self._cancelled = False
        self.hooks: list[Any] = []

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def check(self) -> None:
        if self._cancelled:
            raise Cancelled

    def event(self, ev: dict[str, Any]) -> None:
        self.events.append(ev)

    def send_data(self, data: bytes) -> None:
        self.data.append(data)

    def read_upload(self) -> bytes:
        if self.cancel_after is not None and self.reads >= self.cancel_after:
            self._cancelled = True
            raise Cancelled
        self.reads += 1
        return self.chunks.pop(0) if self.chunks else b""

    def drain_upload(self) -> None:
        self.chunks.clear()

    def on_cancel(self, fn: Any) -> None:
        self.hooks.append(fn)


@pytest.fixture
def roots(tmp_path):
    base = tmp_path / "models"
    for d in ("w", "ro", "w/inner"):
        (base / d).mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("outside the root")
    data = {"mode": "manage", "allow_delete": True, "model_roots": [
        {"id": "w", "path": str(base / "w"), "kind": "gguf", "writable": True},
        {"id": "ro", "path": str(base / "ro"), "kind": "gguf", "writable": False},
        {"id": "inner", "path": str(base / "w" / "inner"), "kind": "gguf", "writable": True}]}
    return base, tmp_path / "outside", data


def pol(data, **over):
    return parse({**data, **over}, "/test/policy.json")


def code(fn, *a) -> str:
    with pytest.raises(OpError) as e:
        fn(*a)
    return e.value.code


@pytest.mark.parametrize("rel", ["../x", "/abs/path", "a/../b", "a//b", "a/./b", "", ".", "a\\b", "a\x00b", "a/",
                                 "x\ny"])
def test_path_rules_refuse_traversal_and_odd_paths(roots, rel):
    _base, _out, data = roots
    p = pol(data)
    assert code(fs_read, p, {"root": "w", "rel": rel, "offset": None, "length": None}, FakeStream()) == proto.BAD_ARGS
    assert code(fs_stat, p, {"root": "w", "paths": [rel]}) == proto.BAD_ARGS


def test_unknown_root_and_missing_root_dir(roots, tmp_path):
    _base, _out, data = roots
    assert code(fs_stat, pol(data), {"root": "nope", "paths": ["a"]}) == proto.NOT_FOUND
    gone = pol(data, model_roots=[{"id": "g", "path": str(tmp_path / "unmounted")}])
    assert code(fs_stat, gone, {"root": "g", "paths": ["a"]}) == proto.NOT_FOUND


def test_symlinked_directory_escaping_the_root_is_refused(roots):
    base, outside, data = roots
    (base / "w" / "escape").symlink_to(outside)
    p = pol(data)
    args = {"root": "w", "rel": "escape/secret.txt", "offset": None, "length": None}
    assert code(fs_read, p, args, FakeStream()) == proto.POLICY_DENIED
    assert code(fs_write, p, {"root": "w", "rel": "escape/new.bin", "size": 1, "sha256": None, "offset": None},
                FakeStream([b"x"]), WriteLocks()) == proto.POLICY_DENIED
    assert not (outside / "new.bin").exists() and not (outside / "new.bin.incomplete").exists()


def test_final_symlink_is_never_followed_by_fs_read(roots):
    base, outside, data = roots
    (base / "w" / "link.gguf").symlink_to(outside / "secret.txt")
    assert code(fs_read, pol(data), {"root": "w", "rel": "link.gguf", "offset": None, "length": None},
                FakeStream()) == proto.BAD_ARGS


@pytest.mark.parametrize("rel", ["token", "sub/id_ed25519", ".env", "creds/credentials.json", ".ssh/model.gguf",
                                 "server.pem"])
def test_secret_looking_names_are_never_read(roots, rel):
    base, _out, data = roots
    f = base / "w" / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("do not read")
    s = FakeStream()
    assert code(fs_read, pol(data), {"root": "w", "rel": rel, "offset": None, "length": None}, s) == \
        proto.POLICY_DENIED
    assert s.data == []


def test_symlink_inside_the_root_cannot_reach_a_secret_directory_inside_it(roots):
    base, _out, data = roots
    (base / "w" / ".ssh").mkdir()
    (base / "w" / ".ssh" / "config").write_text("Host *")
    (base / "w" / "innocent").symlink_to(".ssh")
    s = FakeStream()
    assert code(fs_read, pol(data), {"root": "w", "rel": "innocent/config", "offset": None, "length": None}, s) == \
        proto.POLICY_DENIED
    assert s.data == []
    (base / "w" / "cfg").symlink_to(".ssh/config")           # the scanner's reader refuses the same trick
    from ecnode.confine import read_inside
    assert read_inside(str(base / "w"), str(base / "w" / "cfg"), 1000) is None


def test_stat_reports_files_dirs_links_and_partials(roots):
    base, outside, data = roots
    w = base / "w"
    (w / "a.bin").write_bytes(b"12345")
    (w / "b.bin.incomplete").write_bytes(b"123")
    (w / "d").mkdir()
    (w / "blobs").mkdir()
    (w / "blobs" / "x").write_bytes(b"blob!")
    (w / "snap").mkdir()
    (w / "snap" / "f").symlink_to("../blobs/x")
    (w / "esc").symlink_to(outside / "secret.txt")
    items = {i["rel"]: i for i in fs_stat(pol(data), {"root": "w", "paths": ["a.bin", "b.bin", "d", "snap/f", "esc",
                                                                          "nope/x"]})["items"]}
    assert items["a.bin"] == {"rel": "a.bin", "exists": True, "is_dir": False, "size": 5, "partial": 0}
    assert items["b.bin"]["exists"] is False and items["b.bin"]["partial"] == 3
    assert items["d"]["is_dir"] is True
    assert items["snap/f"]["exists"] is True and items["snap/f"]["size"] == 5
    assert items["esc"]["exists"] is False                 # a link leaving the root does not count
    assert items["nope/x"]["exists"] is False


# ── fs.write ─────────────────────────────────────────────────────────────────────────────────────────────
def _write(p, rel, chunks, size, sha=None, offset=None, locks=None):
    return fs_write(p, {"root": "w", "rel": rel, "size": size, "sha256": sha, "offset": offset}, FakeStream(chunks),
                    locks or WriteLocks())


def test_write_creates_dirs_checks_the_whole_file_hash_and_renames(roots):
    base, _out, data = roots
    payload = os.urandom(300_000)
    sha = hashlib.sha256(payload).hexdigest()
    res = _write(pol(data), "hub/models--org--m/blobs/abc", [payload[:100_000], payload[100_000:]], len(payload), sha)
    assert res == {"size": len(payload), "sha256": sha}
    out = base / "w" / "hub" / "models--org--m" / "blobs" / "abc"
    assert out.read_bytes() == payload and not (out.parent / "abc.incomplete").exists()


def test_write_resumes_from_the_partial_and_hashes_the_whole_file(roots):
    base, _out, data = roots
    payload = os.urandom(50_000)
    (base / "w" / "f.bin.incomplete").write_bytes(payload[:20_000])
    sha = hashlib.sha256(payload).hexdigest()
    assert _write(pol(data), "f.bin", [payload[20_000:]], len(payload), sha, offset=20_000)["sha256"] == sha
    assert (base / "w" / "f.bin").read_bytes() == payload


def test_write_offset_must_match_the_partial(roots):
    base, _out, data = roots
    (base / "w" / "f.bin.incomplete").write_bytes(b"x" * 10)
    with pytest.raises(OpError) as e:
        _write(pol(data), "f.bin", [b"y"], 20, offset=5)
    assert e.value.code == proto.CONFLICT and "resume from 10" in e.value.message
    _write(pol(data), "f.bin", [b"z" * 20], 20, offset=0)      # 0 truncates and starts over
    assert (base / "w" / "f.bin").read_bytes() == b"z" * 20


@pytest.mark.parametrize("chunks,size,sha,expected", [
    ([b"abc"], 3, "0" * 64, proto.HASH_MISMATCH),
    ([b"ab"], 3, None, proto.SIZE_MISMATCH),
    ([b"abcd"], 3, None, proto.SIZE_MISMATCH),
])
def test_write_mismatch_deletes_the_partial(roots, chunks, size, sha, expected):
    base, _out, data = roots
    with pytest.raises(OpError) as e:
        _write(pol(data), "m.bin", chunks, size, sha)
    assert e.value.code == expected
    assert not (base / "w" / "m.bin").exists() and not (base / "w" / "m.bin.incomplete").exists()


def test_cancelled_write_keeps_the_partial_for_a_resume(roots):
    base, _out, data = roots
    s = FakeStream([b"a" * 1000, b"b" * 1000], cancel_after=1)
    with pytest.raises(Cancelled):
        fs_write(pol(data), {"root": "w", "rel": "c.bin", "size": 5000, "sha256": None, "offset": None}, s,
                 WriteLocks())
    assert (base / "w" / "c.bin.incomplete").read_bytes() == b"a" * 1000


def test_write_needs_a_writable_root_and_free_space(roots, monkeypatch):
    _base, _out, data = roots
    assert code(fs_write, pol(data), {"root": "ro", "rel": "x", "size": 1, "sha256": None, "offset": None},
                FakeStream([b"x"]), WriteLocks()) == proto.POLICY_DENIED
    def tiny(fd):
        return os.statvfs_result((4096, 4096, 1000, 100, 100, 10, 10, 10, 0, 255))   # 400 KiB free

    monkeypatch.setattr(fsops.os, "fstatvfs", tiny)
    assert code(fs_write, pol(data), {"root": "w", "rel": "big", "size": 10, "sha256": None, "offset": None},
                FakeStream([b"x" * 10]), WriteLocks()) == proto.NO_SPACE


def test_two_uploads_of_one_file_conflict(roots):
    _base, _out, data = roots
    locks = WriteLocks()
    gate, started = threading.Event(), threading.Event()

    class Slow(FakeStream):
        def read_upload(self) -> bytes:
            started.set()
            gate.wait(5)
            return super().read_upload()

    t = threading.Thread(target=fs_write, args=(pol(data), {"root": "w", "rel": "s.bin", "size": 1, "sha256": None,
                                                            "offset": None}, Slow([b"x"]), locks))
    t.start()
    started.wait(5)
    try:
        assert code(fs_write, pol(data), {"root": "w", "rel": "s.bin", "size": 1, "sha256": None, "offset": None},
                    FakeStream([b"y"]), locks) == proto.CONFLICT
    finally:
        gate.set()
        t.join(5)


# ── fs.read ──────────────────────────────────────────────────────────────────────────────────────────────
def test_read_streams_a_range(roots):
    base, _out, data = roots
    payload = os.urandom(fsops.CHUNK * 2 + 10)
    (base / "w" / "f.gguf").write_bytes(payload)
    s = FakeStream()
    assert fs_read(pol(data), {"root": "w", "rel": "f.gguf", "offset": 5, "length": None}, s) == \
        {"size": len(payload) - 5}
    assert b"".join(s.data) == payload[5:] and max(len(c) for c in s.data) <= fsops.CHUNK
    s2 = FakeStream()
    fs_read(pol(data), {"root": "w", "rel": "f.gguf", "offset": 0, "length": 7}, s2)
    assert b"".join(s2.data) == payload[:7]
    assert code(fs_read, pol(data), {"root": "w", "rel": "f.gguf", "offset": len(payload) + 1, "length": None},
                FakeStream()) == proto.BAD_ARGS


# ── fs.symlink / fs.put_small ────────────────────────────────────────────────────────────────────────────
def test_symlink_inside_the_root_and_atomic_replace(roots):
    base, _out, data = roots
    p = pol(data)
    (base / "w" / "repo" / "blobs").mkdir(parents=True)
    (base / "w" / "repo" / "blobs" / "h1").write_text("one")
    fs_symlink(p, {"root": "w", "link": "repo/snapshots/rev/config.json", "target": "../../blobs/h1"})
    link = base / "w" / "repo" / "snapshots" / "rev" / "config.json"
    assert link.is_symlink() and link.read_text() == "one"
    (base / "w" / "repo" / "blobs" / "h2").write_text("two")
    fs_symlink(p, {"root": "w", "link": "repo/snapshots/rev/config.json", "target": "../../blobs/h2"})
    assert link.read_text() == "two"


@pytest.mark.parametrize("target,expected", [
    ("../../../../../../etc/passwd", proto.POLICY_DENIED),
    ("/etc/passwd", proto.BAD_ARGS),
    ("a//b", proto.BAD_ARGS),
    ("a\\b", proto.BAD_ARGS),
])
def test_symlink_targets_must_stay_inside(roots, target, expected):
    _base, _out, data = roots
    assert code(fs_symlink, pol(data), {"root": "w", "link": "x/l", "target": target}) == expected


def test_symlink_never_replaces_a_directory_and_needs_a_writable_root(roots):
    base, _out, data = roots
    (base / "w" / "dir").mkdir()
    assert code(fs_symlink, pol(data), {"root": "w", "link": "dir", "target": "x"}) == proto.CONFLICT
    assert code(fs_symlink, pol(data), {"root": "ro", "link": "l", "target": "x"}) == proto.POLICY_DENIED


def test_put_small(roots):
    base, _out, data = roots
    fs_put_small(pol(data), {"root": "w", "rel": "hub/models--a--b/refs/main",
                             "content_b64": base64.b64encode(b"0" * 40).decode()})
    assert (base / "w" / "hub" / "models--a--b" / "refs" / "main").read_bytes() == b"0" * 40
    assert code(fs_put_small, pol(data), {"root": "w", "rel": "x", "content_b64": "%%%"}) == proto.BAD_ARGS
    big = base64.b64encode(b"x" * (fsops.PUT_SMALL_MAX + 1)).decode()
    assert code(fs_put_small, pol(data), {"root": "w", "rel": "x", "content_b64": big}) == proto.BAD_ARGS


# ── fs.delete ────────────────────────────────────────────────────────────────────────────────────────────
def test_delete_rules(roots):
    base, _out, data = roots
    m = base / "w" / "models--org--m"
    (m / "blobs").mkdir(parents=True)
    (m / "blobs" / "a").write_bytes(b"x" * 1000)
    (m / "blobs" / "b").write_bytes(b"y" * 24)
    p = pol(data)
    assert code(fs_delete, pol(data, allow_delete=False), {"root": "w", "rel": "models--org--m"},
                lambda: []) == proto.POLICY_DENIED
    assert code(fs_delete, p, {"root": "ro", "rel": "x"}, lambda: []) == proto.POLICY_DENIED
    assert code(fs_delete, p, {"root": "w", "rel": "nope"}, lambda: []) == proto.NOT_FOUND
    # a running container mounting the target (or something under it) blocks the delete
    assert code(fs_delete, p, {"root": "w", "rel": "models--org--m"},
                lambda: [str(m / "blobs")]) == proto.IN_USE
    # a container mounting the PARENT (the whole cache) does not
    assert fs_delete(p, {"root": "w", "rel": "models--org--m"}, lambda: [str(base / "w")]) == {"freed": 1024}
    assert not m.exists()


def test_delete_refuses_the_root_itself_and_other_roots(roots):
    base, _out, data = roots
    (base / "w" / "self").symlink_to(base / "w")
    p = pol(data)
    assert code(fs_delete, p, {"root": "w", "rel": "self/inner"}, lambda: []) == proto.POLICY_DENIED   # holds root
    os.unlink(base / "w" / "self")
    assert code(fs_delete, p, {"root": "w", "rel": "inner"}, lambda: []) == proto.POLICY_DENIED
    (base / "w" / "loop").symlink_to(".")
    # deleting a symlink removes the link only, never what it points at
    assert fs_delete(p, {"root": "w", "rel": "loop"}, lambda: []) == {"freed": 0}
    assert (base / "w" / "inner").is_dir()


def test_delete_when_docker_cannot_say_what_runs(roots):
    base, _out, data = roots
    (base / "w" / "f.gguf").write_bytes(b"1")

    def broken() -> list[str]:
        raise OpError(proto.DOCKER_ERROR, "cannot ask docker")

    assert code(fs_delete, pol(data), {"root": "w", "rel": "f.gguf"}, broken) == proto.DOCKER_ERROR
    assert (base / "w" / "f.gguf").exists()
