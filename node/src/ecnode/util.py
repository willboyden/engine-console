"""Shared helpers: subprocesses (argv lists, never a shell, always a timeout), text cleaning, secret masking.

Every child process gets stdin from /dev/null and captured output, so nothing a child does can read or write
the protocol stream on the node's own stdin/stdout.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import json
import os
import re
import shlex
import signal
import subprocess
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Protocol, TypeVar

REDACTED = "[redacted]"
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


# ── subprocesses ─────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CmdResult:
    rc: int          # 124: timed out, 126: could not start, 127: not found
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.rc == 0


class Runner(Protocol):
    def __call__(self, argv: Sequence[str], *, timeout: float, env: Mapping[str, str] | None = None) -> CmdResult:
        ...


def run(argv: Sequence[str], *, timeout: float, env: Mapping[str, str] | None = None) -> CmdResult:
    """Run argv with a timeout; never raises for a missing binary or a timeout (see CmdResult.rc)."""
    try:
        p = subprocess.run(list(argv), stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout,
                           env=dict(env) if env is not None else None, check=False)
    except FileNotFoundError:
        return CmdResult(127, "", f"{argv[0]}: command not found")
    except subprocess.TimeoutExpired:
        return CmdResult(124, "", f"{argv[0]} did not finish within {timeout:g} s")
    except OSError as e:
        return CmdResult(126, "", f"{argv[0]}: {e.strerror or e}")
    return CmdResult(p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace"))


def spawn(argv: Sequence[str], *, env: Mapping[str, str] | None = None, merge_stderr: bool = True,
          cwd: str | None = None) -> subprocess.Popen[bytes]:
    """A streaming child in its own session (so kill_group reaches everything it started)."""
    return subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                            env=dict(env) if env is not None else None, cwd=cwd, start_new_session=True)


def kill_group(proc: subprocess.Popen[bytes], grace_s: float = 5.0) -> None:
    """SIGTERM the child's process group, SIGKILL it if it is still there after grace_s."""
    if proc.poll() is not None:
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace_s)


def run_tail(argv: Sequence[str], *, timeout: float, max_bytes: int,
             env: Mapping[str, str] | None = None) -> tuple[int, bytes, bool]:
    """(rc, last max_bytes of stdout+stderr, truncated). Memory stays bounded however much the child prints."""
    try:
        proc = spawn(argv, env=env)
    except FileNotFoundError:
        return 127, f"{argv[0]}: command not found".encode(), False
    except OSError as e:
        return 126, f"{argv[0]}: {e.strerror or e}".encode(), False
    timer = threading.Timer(timeout, kill_group, args=(proc, 2.0))
    timer.daemon = True
    timer.start()
    buf = bytearray()
    truncated = False
    out = proc.stdout
    try:
        assert out is not None
        while True:
            chunk = out.read1(65536)  # type: ignore[attr-defined]  # a BufferedReader at runtime
            if not chunk:
                break
            buf += chunk
            if len(buf) > max_bytes:
                del buf[: len(buf) - max_bytes]
                truncated = True
        rc = proc.wait()
    finally:
        timer.cancel()
        kill_group(proc, 1.0)
    return rc, bytes(buf), truncated


def iter_lines(f: IO[bytes], limit: int = 8192) -> Iterator[str]:
    """Lines of a child's output as str (newline stripped, capped at `limit`); stops at EOF."""
    for raw in iter(lambda: f.readline(1024 * 1024), b""):
        yield raw.decode("utf-8", "replace").rstrip("\r\n")[:limit]


# ── text ─────────────────────────────────────────────────────────────────────────────────────────────────
def json_size(obj: object) -> int:
    """Bytes `obj` takes in a protocol frame (same encoder settings as proto.encode_json)."""
    return len(json.dumps(obj, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8", "replace"))


def fit_tail(text: str, budget: int) -> tuple[str, bool]:
    """The longest tail of `text` whose JSON encoding fits in `budget` bytes (escapes such as ANSI codes can make
    the encoding several times longer than the text). (tail, trimmed)."""
    if json_size(text) <= budget:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:                       # smallest cut that fits
        mid = (lo + hi) // 2
        if json_size(text[mid:]) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:], True


def clean_text(s: object, limit: int = 200) -> str:
    """One display line: control characters removed, capped."""
    return _CTRL.sub("", str(s))[:limit]


def as_int(v: object) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        try:
            return int(v.strip())
        except ValueError:
            return None
    return None


def as_float(v: object) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    try:
        f = float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


_TS = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(\.\d+)?\s*(Z|[+-]\d{2}:?\d{2})?")


def parse_time(s: object) -> float | None:
    """RFC 3339 timestamp as docker prints it (nanoseconds, 'Z') -> epoch seconds; None for zero/unparsable."""
    if not isinstance(s, str) or not s or s.startswith("0001-01-01"):
        return None
    m = _TS.match(s.strip())
    if not m:
        return None
    try:
        base = _dt.datetime.strptime(f"{m.group(1)}T{m.group(2)}", "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    tz = m.group(4)
    offset = 0
    if tz and tz != "Z":
        sign = -1 if tz[0] == "-" else 1
        hh, mm = int(tz[1:3]), int(tz[-2:])
        offset = sign * (hh * 3600 + mm * 60)
    frac = float("0" + m.group(3)) if m.group(3) else 0.0
    return base.replace(tzinfo=_dt.timezone.utc).timestamp() - offset + frac


# ── secrets ──────────────────────────────────────────────────────────────────────────────────────────────
# Words that make a flag, key or variable name secret. "token" only as a whole word, so --tokenizer,
# max_tokens and TOKENIZERS_PARALLELISM stay visible while --hf-token and HF_TOKEN are masked.
SENSITIVE_WORD = (r"(?:api[-_]?key|apikey|secret|passw(?:or)?d|passphrase|credentials?|access[-_]?key|auth[-_]?key|"
               r"private[-_]?key|(?<![a-z])token(?![a-z])|bearer|authorization)")
SECRET_NAME_RE = re.compile(SENSITIVE_WORD, re.I)
_SECRET_VALUES = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{16,}|ec_[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|xox[abprs]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|nvapi-[A-Za-z0-9_\-]{20,}|"
    r"eyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,})")
_AUTH_HEADER = re.compile(r"\b(Bearer|Basic|Token)\s+[A-Za-z0-9._~+/=\-]{8,}", re.I)
_URL_PASSWORD = re.compile(r"([a-z][a-z0-9+.\-]*://[^/\s:@]{0,128}:)[^@\s/]{1,256}@", re.I)
# key=value / "key": "value" / key: ['value'] where the key looks secret (bounded prefix: no quadratic backtracking)
_KEY_VALUE = re.compile(r"([\w.\-]{0,40}" + SENSITIVE_WORD + r"[\w.\-]{0,40}['\"]?\s{0,4}[:=]\s{0,4}\[?\s{0,4}['\"]?)"
                        r"([^'\"\s,\]}]{1,4096})", re.I)
# --api-key value / -token=value in free text
_FLAG_VALUE = re.compile(r"(--?[\w\-]{0,40}" + SENSITIVE_WORD + r"[\w\-]{0,40})(\s+|=)(['\"]?)([^\s'\"]{1,4096})", re.I)


def scrub(text: str) -> str:
    """Mask secret-looking values in free text (log lines, argv tokens, error messages)."""
    if not text:
        return text
    text = _SECRET_VALUES.sub(REDACTED, text)
    text = _AUTH_HEADER.sub(lambda m: f"{m.group(1)} {REDACTED}", text)
    text = _URL_PASSWORD.sub(lambda m: f"{m.group(1)}{REDACTED}@", text)
    text = _FLAG_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", text)
    return _KEY_VALUE.sub(lambda m: f"{m.group(1)}{REDACTED}", text)


def _secret_flag(head: str) -> bool:
    return bool(SECRET_NAME_RE.search(head))


def redact_argv(argv: Sequence[object], *, max_items: int = 400, max_len: int = 1000, _depth: int = 0) -> list[str]:
    """Mask secret flag values BEFORE anything else looks at an argv: `--api-key X`, `--token=X`, `NAME=X`, and
    the same inside one-string shell commands (`sh -c "vllm serve --api-key X"`)."""
    out: list[str] = []
    skip = False
    for raw in list(argv)[:max_items]:
        tok = _CTRL.sub(" ", str(raw))[:max_len]
        if skip:
            out.append(REDACTED)
            skip = False
            continue
        if _depth < 2 and any(c.isspace() for c in tok.strip()):
            try:
                parts = shlex.split(tok)
            except ValueError:
                parts = tok.split()
            out.append(" ".join(redact_argv(parts, max_items=max_items, max_len=max_len, _depth=_depth + 1)))
            continue
        head, eq, _ = tok.partition("=")
        if _secret_flag(head) and (head.startswith("-") or eq):
            if eq:
                out.append(f"{head}={REDACTED}")
            else:
                out.append(tok)
                skip = True
            continue
        out.append(scrub(tok))
    return out


# Names a node never reads, even inside a model root (FLEET-PROTOCOL.md §4 "Path rules").
_SECRET_FILE = re.compile(
    r"^(?:id_.*|.*\.pem|.*\.key|.*\.p12|.*\.pfx|.*\.ppk|\.env.*|tokens?(?:[^a-z].*)?|credentials?(?:[^a-z].*)?|"
    r"stored_tokens|\.netrc|\.git-credentials|\.pypirc|\.npmrc|authorized_keys.*|known_hosts.*)$", re.I | re.S)
# Secret words anywhere in a name (`hf_token`, `my-secret.txt`, `apikey`). A bare `token` substring is NOT one of
# them: every model ships tokenizer.json, special_tokens_map.json and added_tokens.json, so `token` counts only
# with a credential-style prefix.
_SECRET_WORD = re.compile(
    r"secret|credential|passw(?:or)?d|api[-_.]?key|private[-_.]?key|"
    r"(?:hf|api|access|auth|bearer|refresh|session|github|gh|hub|openai|ngc)[-_.]?tokens?(?![a-z])", re.I)
SECRET_DIRS = frozenset({".ssh", ".gnupg", ".aws", ".docker", ".kube", ".config", ".password-store", ".azure",
                         ".gcloud", ".local"})


def is_secret_name(name: str) -> bool:
    """True for a file or directory name that looks like a key, token or credential store."""
    return bool(_SECRET_FILE.match(name) or _SECRET_WORD.search(name)) or name.lower() in SECRET_DIRS


T = TypeVar("T")


class Guarded:
    """Blocking filesystem calls (statvfs, realpath on a network mount) with a timeout. A call that does not
    return is abandoned (its daemon thread stays blocked in the kernel) and its key is skipped until that call
    does return, so a dead NFS server costs one stuck thread, not one more per poll."""

    def __init__(self) -> None:
        self._busy: set[str] = set()
        self._lock = threading.Lock()

    def call(self, key: str, fn: Callable[[], T], timeout: float) -> T | None:
        with self._lock:
            if key in self._busy:
                return None
            self._busy.add(key)
        box: list[T] = []
        done = threading.Event()

        def target() -> None:
            try:
                box.append(fn())
            except Exception:       # the caller only needs "no answer"
                box.clear()
            finally:
                with self._lock:
                    self._busy.discard(key)
                done.set()

        threading.Thread(target=target, name="ecnode-guarded", daemon=True).start()
        if done.wait(timeout) and box:
            return box[0]
        return None
