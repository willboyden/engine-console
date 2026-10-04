"""Which machines the console manages and exactly how each one is reached (contract v2, docs/ARCHITECTURE.md §10.3).

hosts.json decides what this process runs `ssh` against, so it is read once at start-up, from disk only, and only
if it could not have been planted by someone else: a plain file (no symlink), owned by the running user, writable
by that user alone. Nothing a browser or a node sends can change it or reach an ssh argument.

Everything that ends up on a command line is validated here with `fullmatch` (never `match` with `$`, which would
accept a trailing newline) and is passed as its own argv element: no shell is ever involved.
"""
from __future__ import annotations

import errno
import getpass
import json
import math
import os
import re
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from engine_console.config import Settings
from engine_console.domain.fleet import HOST_ID_PATTERN, LOCAL, HostConfig

MAX_HOSTS = 32
MAX_FILE_BYTES = 64 * 1024
LABEL_MAX = 64
HOST_KEYS = frozenset({"id", "label", "ssh", "port", "ssh_config", "enabled"})
KEY_COMMENT = "engine-console-hub"
DEFAULT_REMOTE_COMMAND = "/usr/local/lib/ecnode/ecnode --stdio"

_HOST_ID = re.compile(HOST_ID_PATTERN)
# [user@]host. No leading '-' (ssh would read it as an option, e.g. -oProxyCommand=...), no '%' (ssh token
# expansion), no whitespace or shell metacharacters. An ssh_config alias has the same shape.
_SSH_DEST = re.compile(r"(?:[A-Za-z0-9._-]{1,32}@)?[A-Za-z0-9][A-Za-z0-9._-]{0,252}")
# What sshd accepts in `from="..."`: comma-separated address/host patterns (CIDR, '*', '?', '!'). No quote or space,
# so the value can never close the option string it is written into.
_FROM = re.compile(r"[0-9A-Za-z.:/*?!,-]{1,256}")
# An absolute path and plain arguments: this string is both the remote command and the `command="..."` of the
# authorized_keys line, so it must not be able to break out of either.
_REMOTE_CMD = re.compile(r"/[A-Za-z0-9_./+-]{1,200}(?: [A-Za-z0-9_./=:+-]{1,64}){0,8}")
_PUBKEY_TYPE = re.compile(
    r"ssh-ed25519|ssh-rsa|ecdsa-sha2-nistp(?:256|384|521)|sk-ssh-ed25519@openssh\.com|sk-ecdsa-sha2-nistp256@openssh\.com")
_PUBKEY_B64 = re.compile(r"[A-Za-z0-9+/]{16,4096}={0,3}")
_LOCAL_ENV_KEYS = ("HOME", "PATH", "USER", "LOGNAME", "LANG", "XDG_RUNTIME_DIR", "DOCKER_HOST", "DOCKER_CONTEXT",
                   "DOCKER_CONFIG")


class HostsError(ValueError):
    """hosts.json (or a key file, or an ssh argument) is not acceptable. The text says what to fix."""


class NoHubKey(HostsError):
    """The console's own ssh key does not exist yet (`engine-console hosts init-key` creates it)."""


# ── hosts.json ───────────────────────────────────────────────────────────────────────────────────────────
def default_hosts() -> list[HostConfig]:
    """No hosts.json: the console manages the machine it runs on and nothing else."""
    name = "".join(ch for ch in socket.gethostname() if ch.isprintable())[:LABEL_MAX]
    return [HostConfig(id=LOCAL, label=name or "This machine")]


def _reject_constant(name: str) -> Any:
    raise HostsError(f"{name} is not allowed: the file holds text, booleans and whole numbers only")


def _finite_float(text: str) -> float:
    v = float(text)               # "1e999" is not a NaN/Infinity constant but parses to inf
    if not math.isfinite(v):
        raise HostsError(f"{text[:20]} is not a finite number")
    return v


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise HostsError(f"the key {k!r} appears twice in one object")
        out[k] = v
    return out


def parse_hosts(doc: Any) -> list[HostConfig]:
    """Validate a decoded hosts.json document. Strict on purpose: unknown keys are errors, not ignored, so a typo
    such as "sshconfig" can never silently leave a safer default in place."""
    if not isinstance(doc, dict) or set(doc) != {"hosts"}:
        raise HostsError('the file must be {"hosts": [ ... ]} and nothing else')
    raw = doc["hosts"]
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_HOSTS:
        raise HostsError(f"hosts must be a list of 1-{MAX_HOSTS} machines")
    out: list[HostConfig] = []
    seen: set[str] = set()
    for i, h in enumerate(raw):
        cfg = _parse_host(h, i)
        if cfg.id in seen:
            raise HostsError(f"hosts[{i}]: id {cfg.id!r} is used twice")
        seen.add(cfg.id)
        out.append(cfg)
    return out


def _parse_host(h: Any, i: int) -> HostConfig:
    where = f"hosts[{i}]"
    if not isinstance(h, dict):
        raise HostsError(f"{where} must be an object")
    extra = set(h) - HOST_KEYS
    if extra:
        raise HostsError(f"{where}: unknown key(s) {', '.join(sorted(map(str, extra)))}")
    hid = h.get("id")
    if not isinstance(hid, str) or _HOST_ID.fullmatch(hid) is None:
        raise HostsError(f"{where}: id must be lowercase letters, digits and '-', 1-32 long, not starting with '-'")
    where = f"host {hid!r}"
    label = h.get("label", hid)      # a label is a convenience: the id is shown when none is given
    if not isinstance(label, str) or not 1 <= len(label) <= LABEL_MAX or not label.isprintable():
        raise HostsError(f"{where}: label must be 1-{LABEL_MAX} printable characters")
    dest, port = h.get("ssh"), h.get("port")
    ssh_config, enabled = h.get("ssh_config", False), h.get("enabled", True)
    if not isinstance(ssh_config, bool) or not isinstance(enabled, bool):
        raise HostsError(f"{where}: ssh_config and enabled must be true or false")
    if hid == LOCAL:
        if dest is not None or port is not None or ssh_config:
            raise HostsError(f'{where}: "local" is the machine the console runs on; it takes no ssh, port or ssh_config')
    else:
        if not isinstance(dest, str) or _SSH_DEST.fullmatch(dest) is None:
            raise HostsError(f"{where}: ssh must be user@address or an ssh_config alias (letters, digits, '.', '_', '-'; "
                             "not starting with '-')")
        if port is not None and (isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535):
            raise HostsError(f"{where}: port must be a whole number from 1 to 65535")
    try:
        return HostConfig(id=hid, label=label, ssh=dest, port=port, ssh_config=ssh_config, enabled=enabled)
    except ValidationError as e:     # unreachable unless HostConfig grows rules this loader does not know yet
        raise HostsError(f"{where}: {e.errors()[0]['msg']}") from None


def _read_capped(fd: int, limit: int) -> bytes:
    chunks: list[bytes] = []
    while limit > 0 and (chunk := os.read(fd, min(limit, 64 * 1024))):
        chunks.append(chunk)
        limit -= len(chunk)
    return b"".join(chunks)


def load_hosts(path: Path | str) -> list[HostConfig]:
    """The configured machines, or just this one when the file does not exist. Raises HostsError (message starts
    with the path) for anything else: the console must refuse to start rather than guess."""
    p = os.fspath(path)
    try:
        # O_NOFOLLOW: a symlink in the last position is refused, so the file cannot be redirected somewhere the
        # owner checks below would not cover. O_NONBLOCK: opening a FIFO must not hang start-up.
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except FileNotFoundError:
        return default_hosts()
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise HostsError(f"{p}: is a symlink; it decides what the console runs ssh against, so it must be a plain file") from None
        raise HostsError(f"{p}: cannot open it: {e.strerror}") from None
    try:
        # Every check is made on the descriptor we will read from, not on the path again, so the file cannot be
        # swapped between the check and the read.
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise HostsError(f"{p}: is not a regular file")
        if st.st_uid != os.geteuid():
            raise HostsError(f"{p}: must be owned by the user running the console: it decides what the console runs ssh against")
        if st.st_mode & 0o022:
            raise HostsError(f"{p}: must not be group- or world-writable (chmod 600)")
        data = _read_capped(fd, MAX_FILE_BYTES + 1)
    except OSError as e:
        raise HostsError(f"{p}: cannot read it: {e.strerror}") from None
    finally:
        os.close(fd)
    if len(data) > MAX_FILE_BYTES:
        raise HostsError(f"{p}: is over {MAX_FILE_BYTES // 1024} KiB")
    try:
        doc = json.loads(data.decode("utf-8"), parse_constant=_reject_constant, parse_float=_finite_float,
                         object_pairs_hook=_no_duplicate_keys)
        return parse_hosts(doc)
    except HostsError as e:
        raise HostsError(f"{p}: {e}") from None
    except (ValueError, RecursionError) as e:    # JSONDecodeError, bad UTF-8, absurdly long integers, deep nesting
        raise HostsError(f"{p}: not valid JSON: {str(e)[:120]}") from None


# ── how each machine is reached ──────────────────────────────────────────────────────────────────────────
def check_remote_command(cmd: str) -> str:
    if not isinstance(cmd, str) or _REMOTE_CMD.fullmatch(cmd) is None:
        raise HostsError("the remote node command must be an absolute path followed by plain arguments, "
                         f"for example {DEFAULT_REMOTE_COMMAND!r}")
    return cmd


def ssh_argv(host: HostConfig, key_file: Path | str, remote_cmd: str) -> list[str]:
    """The exact ssh command for one remote host (docs/ARCHITECTURE.md §10.3).

    Only the hub's own key is offered (`-F none` unless the host opts into the user's ssh config, IdentitiesOnly,
    no agent, no passwords; raises NoHubKey rather than build a command ssh would complete with other keys);
    known_hosts is read, never written (StrictHostKeyChecking=yes, UpdateHostKeys=no), so
    an unknown or changed host key is an error and not a silent trust. `--` ends option parsing, so the
    destination can never be taken for an option even if a bug let a '-' through the checks above it.

    ssh joins everything after the destination with spaces into one remote command line, so `remote_cmd` as a
    single element is the same on the wire as its words as separate elements."""
    if host.id == LOCAL or host.ssh is None:
        raise HostsError(f"host {host.id!r} is reached without ssh")
    if _SSH_DEST.fullmatch(host.ssh) is None:
        raise HostsError(f"host {host.id!r}: ssh must be user@address or an ssh_config alias")
    cmd = check_remote_command(remote_cmd)
    key = os.fspath(key_file)
    if not key or key.startswith("-") or "\x00" in key or "\n" in key:
        raise HostsError("the hub key path is not usable")
    if any(c in key for c in " \t\"'%"):
        # The key is passed as `-o IdentityFile=<path>`, which ssh parses like a config line: whitespace and quotes
        # would split or quote it and `%` would be expanded as a token.
        raise HostsError("the hub key path must not contain spaces, quotes or '%'")
    # Checked here, not left to ssh: when the `-i` file does not exist ssh only warns and then offers the user's
    # default identity files (the id_* files in the user's own ssh directory) instead, even with IdentitiesOnly=yes
    # (observed with `ssh -G`). That would log in to machines with the user's own unrestricted key instead of the
    # console's restricted one.
    if not os.path.isfile(key):
        raise NoHubKey("the console has no ssh key yet: run `engine-console hosts init-key --from <this machine's "
                       "address>` and add the line it prints to authorized_keys on each machine")
    argv = ["ssh", "-T", "-e", "none"]
    if not host.ssh_config:
        argv += ["-F", "none"]
    for opt in ("BatchMode=yes", "StrictHostKeyChecking=yes", "UpdateHostKeys=no", "VerifyHostKeyDNS=no",
                "IdentityAgent=none", "PKCS11Provider=none", "PasswordAuthentication=no",
                "KbdInteractiveAuthentication=no", "ControlMaster=no", "ControlPath=none", "ForwardAgent=no",
                "ForwardX11=no", "ClearAllForwardings=yes", "PermitLocalCommand=no", "ConnectTimeout=5",
                "ServerAliveInterval=5", "ServerAliveCountMax=3"):
        argv += ["-o", opt]
    # `-o IdentityFile=` rather than `-i`: ssh registers it without checking that the file exists, so if the key
    # vanishes between the check above and the spawn, the login fails instead of falling back to default identities.
    argv += ["-o", f"IdentityFile={key}", "-o", "IdentitiesOnly=yes"]
    if host.port:
        argv += ["-p", str(host.port)]
    return [*argv, "--", host.ssh, cmd]


def _username() -> str:
    try:
        return getpass.getuser()
    except (KeyError, OSError, ImportError):
        return ""


def ssh_env() -> dict[str, str]:
    """ssh gets a scrubbed environment: no SSH_AUTH_SOCK (so a terminal and a service authenticate identically,
    with the hub's own key), no proxies, and nothing else of the console's environment leaks into the child."""
    env = {"HOME": os.environ.get("HOME") or str(Path.home()),
           "PATH": os.environ.get("PATH") or "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8"}
    user = os.environ.get("USER") or _username()
    if user:
        env["USER"] = user
    return env


def local_env() -> dict[str, str]:
    """Environment of the local node process: the few variables a node needs to find its home and a rootless
    docker. The console's own environment (HF_TOKEN, engine API keys, ...) must never reach it: the node starts
    docker and scripts, and secrets travel to docker only as the explicit `secret_env` of one request."""
    env = {k: v for k in _LOCAL_ENV_KEYS if (v := os.environ.get(k))}
    env.setdefault("HOME", str(Path.home()))
    env.setdefault("PATH", "/usr/local/bin:/usr/bin:/bin")
    env.setdefault("LANG", "C.UTF-8")
    if "USER" not in env and (user := _username()):
        env["USER"] = user
    return env


def local_argv(cfg: Settings) -> list[str]:
    """argv of the local node. `node_command` replaces the whole thing (development and tests); only the default
    command takes `--policy`, because an override is the complete command line by definition."""
    if cfg.node_command:
        return list(cfg.node_command)
    argv = [sys.executable, "-I", "-m", "ecnode", "--stdio"]
    if cfg.node_policy:
        argv += ["--policy", os.fspath(cfg.node_policy)]
    return argv


# ── the hub's ssh key ────────────────────────────────────────────────────────────────────────────────────
def key_line(pubkey: str, from_: str | None = None, remote_cmd: str = DEFAULT_REMOTE_COMMAND) -> str:
    """The one line to add to the login user's authorized_keys on each managed machine. `restrict` removes every
    capability (shell, pty, forwarding, agent), `command` pins the key to the node, `from` pins it to the hub's
    address: a stolen key can start a node, from the hub, and nothing else. The public key's own comment is
    dropped, so nothing in it can add options or lines."""
    text = pubkey.strip()
    parts = text.split()
    if "\n" in text or "\r" in text or len(parts) < 2 or _PUBKEY_TYPE.fullmatch(parts[0]) is None \
            or _PUBKEY_B64.fullmatch(parts[1]) is None:
        raise HostsError("that is not a single OpenSSH public key line")
    cmd = check_remote_command(remote_cmd)
    opts = ["restrict"]
    if from_ is not None:
        if _FROM.fullmatch(from_) is None:
            raise HostsError("--from must be addresses or patterns like 10.0.0.5 or 10.0.0.0/24 (comma-separated)")
        opts.append(f'from="{from_}"')
    opts.append(f'command="{cmd}"')
    return f"{','.join(opts)} {parts[0]} {parts[1]} {KEY_COMMENT}"


def _run_keygen(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL, env=ssh_env())
    except FileNotFoundError:
        raise HostsError("ssh-keygen was not found: install the OpenSSH client") from None
    except subprocess.TimeoutExpired:
        raise HostsError("ssh-keygen did not finish in time") from None
    except subprocess.CalledProcessError as e:
        raise HostsError(f"ssh-keygen failed: {(e.stderr or '').strip()[:200] or f'exit {e.returncode}'}") from None
    except OSError as e:
        raise HostsError(f"could not run ssh-keygen: {e.strerror}") from None


def init_key(key_file: Path | str, from_: str | None, remote_cmd: str = DEFAULT_REMOTE_COMMAND, *,
             keygen: str = "ssh-keygen", timeout: float = 30.0) -> str:
    """Create the hub's ed25519 key (no passphrase: the console uses it unattended; what limits it is the
    authorized_keys line) unless one exists, and return that line. An existing key is NEVER replaced: its public
    half is read (or re-derived) and the same line is returned, so running this twice is harmless.

    Missing directories are created 0700 and the key 0600 (umask 077 while ssh-keygen runs). The key is never
    printed or returned; only its public half is."""
    key = os.fspath(key_file)
    pub = key + ".pub"
    key_line("ssh-ed25519 AAAAAAAAAAAAAAAAAAAAAA", from_, remote_cmd)   # validate the arguments before touching disk
    if os.path.lexists(key):
        if not os.path.exists(key):
            raise HostsError(f"{key} is a dangling symlink: remove it first")
    else:
        if os.path.lexists(pub):
            raise HostsError(f"{pub} exists without its private key: move it away so it is not overwritten")
        old = os.umask(0o077)
        try:
            os.makedirs(os.path.dirname(key) or ".", mode=0o700, exist_ok=True)
            _run_keygen([keygen, "-t", "ed25519", "-N", "", "-C", KEY_COMMENT, "-q", "-f", key], timeout)
        finally:
            os.umask(old)
        if os.stat(key).st_mode & 0o077:
            raise HostsError(f"{key} was created readable by others; delete it and check the umask")
    try:
        with open(pub, encoding="utf-8") as f:
            text = f.read(8192)
    except FileNotFoundError:            # the public half can always be derived again
        text = _run_keygen([keygen, "-y", "-f", key], timeout).stdout
    except OSError as e:
        raise HostsError(f"cannot read {pub}: {e.strerror}") from None
    return key_line(text, from_, remote_cmd)
