"""Golden vectors for the Rust node's path boundary (crates/ecnode src/confine.rs and src/util/py.rs), from the Python
reference (node/src/ecnode/confine.py and posixpath).

    "$(node backend/parity/lib/python-release.mjs)/node/.venv/bin/python" -I backend/crates/ecnode/tests/gen/wsd_path_vectors.py

(from the repo root; the Python reference is the last Python release, backend/parity/lib/python-release.mjs)

writes backend/crates/ecnode/tests/data/wsd_path_vectors.json, read by tests/test_confine.rs
(test_wsd_path_vectors_match_python). Standard library plus the reference `ecnode` package only; deterministic
(seeded); it touches nothing but its own temp dir.

Families: split_rel (segments, or the exact OpError code + message), inside, secret_inside, normpath, join, relpath,
splitext, and over a symlink tree the generator builds (described in the JSON, relative link targets only, so the
Rust test rebuilds it in its own temp dir): realpath, confine.open_dir (non-create: where the fd really points, or
the refusal) and confine.open_inside (whether a metadata read is allowed). Paths in the tree families are written
relative to the temp dir as "<TMP>/...".
"""
from __future__ import annotations

import json
import os
import posixpath
import random
import shutil
import sys
import tempfile
from pathlib import Path

from ecnode import confine
from ecnode.errors import OpError
from ecnode.policy import Policy, Root

OUT = Path(__file__).resolve().parents[1] / "data" / "wsd_path_vectors.json"
rng = random.Random(20261027)

SEGS = ["a", "b", "model.gguf", ".", "..", "", "x y", "été", "a\\b", "a\x00b", "c\nd", "e\x7f",
        "a" * 255, "a" * 256, "ü" * 127, "ü" * 128, "-", "...", ".hidden", "token"]


def split_rel_vectors() -> list[dict]:
    rels: list[object] = [None, "", ".", "..", "/", "/abs", "a/", "a//b", "a/./b", "a/../b", "../x", "a\\b", "a\x00b",
                          "x\ny", "\t", "a/b/c.gguf", "models--org--m/blobs/abc", "a" * 255, "a" * 256,
                          "ü" * 127 + "/x", "ü" * 128, "x" * 300 + "/..", "/" + "y" * 300, "..." + "/b"]
    for _ in range(260):
        n = rng.randint(1, 4)
        rel = "/".join(rng.choice(SEGS) for _ in range(n))
        if rng.random() < 0.15:
            rel = "/" + rel
        rels.append(rel)
    out = []
    for i, rel in enumerate(rels):
        what = ("path", "link", "mounts[0].rel")[i % 3]
        try:
            out.append({"rel": rel, "what": what, "ok": confine.split_rel(rel, what=what)})
        except OpError as e:
            out.append({"rel": rel, "what": what, "code": e.code, "message": e.message})
    return out


PATHS = ["/", "/a", "/a/", "/a/b", "/a/b/", "/ab", "/a/bc", "/a//b", "//a", "/a/b/c", "a", "a/b", "", "/a.b",
         "/models", "/models/hf", "/models/hf-old", "/models/hf/hub"]


def inside_vectors() -> list[dict]:
    return [{"child": c, "root": r, "inside": confine.inside(c, r)} for c in PATHS for r in PATHS]


def secret_inside_vectors() -> list[dict]:
    reals = ["/r", "/r/a", "/r/.ssh", "/r/.ssh/config", "/r/a/id_rsa", "/r/tokenizer.json", "/r/hf_token",
             "/r/a/b/.env.local", "/x/.ssh", "/x/model", "/r/models--a--b/blobs/x", "/r/credentials/x", "/rr/.aws/x"]
    return [{"real": real, "root": root, "secret": confine.secret_inside(real, root)}
            for real in reals for root in ("/r", "/r/", "/", "/x")]


def normpath_vectors() -> list[dict]:
    ins = ["", ".", "..", "/", "//", "///", "////a", "//a/b", "a//b", "a/./b", "a/../b", "/../a", "a/..", "a/../..",
           "../../a", "/a/b/../../..", "./", "a/", "/a/b/", "...", "a/.../b"]
    parts = ["a", "b", ".", "..", "", "c d"]
    for _ in range(140):
        p = "/".join(rng.choice(parts) for _ in range(rng.randint(1, 6)))
        p = "/" * rng.choice([0, 0, 1, 1, 2, 3]) + p
        ins.append(p)
    return [{"in": p, "out": posixpath.normpath(p)} for p in ins]


def join_vectors() -> list[dict]:
    left = ["", "/", "/a", "/a/", "a", "a/b", "//x"]
    right = ["", "b", "/b", "b/", "../c", "."]
    return [{"a": a, "b": b, "out": posixpath.join(a, b)} for a in left for b in right]


def relpath_vectors() -> list[dict]:
    paths = ["/", "/a", "/a/b", "/a/b/c", "/a/x", "/b", "/a/b/../c", "/a//b/"]
    return [{"path": p, "start": s, "out": posixpath.relpath(p, s)} for p in paths for s in paths]


def splitext_vectors() -> list[dict]:
    names = ["a.ftw", ".bashrc", "..x.y", "noext", "a.b.c", "a.", ".", "..", "...", "x/.y", "x.y/z", "x.y/.z",
             "model.safetensors", "dir.d/file", "a..b", ".a.b"]
    return [{"name": n, "ext": posixpath.splitext(n)[1]} for n in names]


# ── a symlink tree ──────────────────────────────────────────────────────────────────────────────────────
TREE = {
    "dirs": ["root", "root/d1", "root/d1/d2", "root/d3", "root/.ssh", "root/keys", "outside", "outside/o1"],
    "files": ["root/d1/f", "root/d3/g", "root/d1/d2/h", "root/.ssh/config", "root/keys/id_rsa", "root/d1/config.json",
              "outside/secret.txt", "outside/o1/config.json"],
    "links": [
        ["root/l1", "d1"], ["root/l2", "l1/d2"], ["root/l3", "../outside"], ["root/l4", "d1/../../outside/o1"],
        ["root/loop1", "loop2"], ["root/loop2", "loop1"], ["root/self", "."], ["root/dd", "d1/.."],
        ["root/broken", "nope"], ["root/d1/up", ".."], ["root/d1/up2", "../.."], ["root/sshlink", ".ssh"],
        ["root/d3/cfg", "../.ssh/config"], ["root/d3/f2", "../d1/f"], ["root/d3/esc", "../../outside/secret.txt"],
        ["root/d1/d2/back", "../../d3"], ["root/chain", "l2/../f"],
    ],
}
NAMES = ["d1", "d2", "d3", "f", "g", "h", "l1", "l2", "l3", "l4", "loop1", "self", "dd", "broken", "up", "up2",
         "sshlink", "config", "cfg", "f2", "esc", "back", "chain", "keys", "id_rsa", "..", ".", "", "nope"]


def build(tmp: str) -> None:
    for d in TREE["dirs"]:
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
    for f in TREE["files"]:
        with open(os.path.join(tmp, f), "w") as fh:
            fh.write("x")
    for link, target in TREE["links"]:
        os.symlink(target, os.path.join(tmp, link))


def rel_to_tmp(p: str | None, tmp: str) -> str | None:
    if p is None:
        return None
    if p == tmp or p.startswith(tmp + "/"):
        return "<TMP>" + p[len(tmp):]
    return p


def tree_vectors() -> dict:
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="wsd-paths-"))
    try:
        build(tmp)
        root = os.path.join(tmp, "root")
        queries = ["", "d1", "d1/f", "l1/f", "l2/h", "l3", "l3/secret.txt", "l4/config.json", "loop1", "loop1/x",
                   "self/self/d1/f", "dd/d3/g", "broken", "broken/x", "d1/../l1/./f", "l1/../d3", "d1/up/d3",
                   "d1/up2/outside", "chain", "d1/d2/back/g", "sshlink/config", "d3/cfg", "d3/esc", "/abs/x"]
        for _ in range(170):
            queries.append("/".join(rng.choice(NAMES) for _ in range(rng.randint(1, 5))))
        realpath = [{"in": q, "out": rel_to_tmp(os.path.realpath(os.path.join(root, q) if not q.startswith("/")
                                                                  else q), tmp)} for q in queries]
        policy = Policy(path="/x/policy.json", model_roots=(Root("t", root, writable=True),))
        h = confine.open_root(policy, "t")
        open_dir = []
        for q in queries:
            try:
                parts = confine.split_rel(q)
            except OpError:
                continue
            try:
                fd = confine.open_dir(h, parts)
            except OpError as e:
                open_dir.append({"parts": parts, "code": e.code, "message": e.message})
                continue
            try:
                open_dir.append({"parts": parts, "ok": rel_to_tmp(confine.fd_path(fd), tmp)})
            finally:
                os.close(fd)
        open_inside = []
        for q in queries:
            if q.startswith("/"):
                continue
            fd = confine.open_inside(root, os.path.join(root, q))
            if fd is not None:
                os.close(fd)
            data = confine.read_inside(root, os.path.join(root, q), 1)
            open_inside.append({"in": q, "open": fd is not None, "read1": None if data is None else data.decode()})
        return {"tree": TREE, "realpath": realpath, "open_dir": open_dir, "open_inside": open_inside}
    finally:
        shutil.rmtree(tmp)


def main() -> int:
    doc = {
        "_generated_by": "backend/crates/ecnode/tests/gen/wsd_path_vectors.py (Python " + sys.version.split()[0] + ")",
        "split_rel": split_rel_vectors(),
        "inside": inside_vectors(),
        "secret_inside": secret_inside_vectors(),
        "normpath": normpath_vectors(),
        "join": join_vectors(),
        "relpath": relpath_vectors(),
        "splitext": splitext_vectors(),
        **tree_vectors(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, ensure_ascii=True) + "\n")
    counts = {k: len(v) for k, v in doc.items() if isinstance(v, list)}
    print(f"wrote {OUT}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
