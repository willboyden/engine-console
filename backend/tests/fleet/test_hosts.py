"""hosts.json loading, the exact ssh command line, the scrubbed environments, and the hub's ssh key."""
from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from engine_console.domain.fleet import HostConfig
from engine_console.services import hosts as H
from engine_console.services.hosts import (
    HostsError,
    NoHubKey,
    init_key,
    key_line,
    load_hosts,
    local_argv,
    local_env,
    parse_hosts,
    ssh_argv,
    ssh_env,
)

REMOTE_CMD = "/usr/local/lib/ecnode/ecnode --stdio"
PUB = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl user@somewhere"


def write_hosts(path: Path, doc: Any, mode: int = 0o600) -> Path:
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    path.chmod(mode)
    return path


def host(**kw: Any) -> dict[str, Any]:
    return {"id": "gpu-box-2", "ssh": "ops@10.0.0.5", **kw}


# ── hosts.json: what is accepted ────────────────────────────────────────────────────────────────────────
def test_no_file_means_this_machine_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "my-workstation")
    assert load_hosts(tmp_path / "absent.json") == [HostConfig(id="local", label="my-workstation")]


def test_no_file_label_is_never_empty_or_unprintable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "gethostname", lambda: "")
    assert load_hosts(tmp_path / "absent.json")[0].label == "This machine"
    monkeypatch.setattr(socket, "gethostname", lambda: "ok\x07name\n" + "x" * 100)
    label = load_hosts(tmp_path / "absent.json")[0].label
    assert label.isprintable() and 1 <= len(label) <= 64


def test_a_full_file(tmp_path: Path) -> None:
    doc = {"hosts": [
        {"id": "local", "label": "Workstation"},
        {"id": "gpu-box-2", "label": "GPU box 2", "ssh": "ops@10.0.0.5", "port": 2222},
        {"id": "edge", "ssh": "edge-alias", "ssh_config": True, "enabled": False},
    ]}
    got = load_hosts(write_hosts(tmp_path / "hosts.json", doc))
    assert got == [
        HostConfig(id="local", label="Workstation"),
        HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5", port=2222),
        HostConfig(id="edge", label="edge", ssh="edge-alias", ssh_config=True, enabled=False),   # label defaults to the id
    ]


def test_leaving_local_out_is_allowed(tmp_path: Path) -> None:
    assert [h.id for h in load_hosts(write_hosts(tmp_path / "h.json", {"hosts": [host()]}))] == ["gpu-box-2"]


@pytest.mark.parametrize("mode", [0o600, 0o640, 0o400, 0o644])
def test_modes_that_nobody_else_can_write_are_fine(tmp_path: Path, mode: int) -> None:
    assert load_hosts(write_hosts(tmp_path / "h.json", {"hosts": [host()]}, mode))


def test_32_hosts_is_the_limit(tmp_path: Path) -> None:
    ok = {"hosts": [host(id=f"h{i}") for i in range(32)]}
    assert len(load_hosts(write_hosts(tmp_path / "ok.json", ok))) == 32
    too_many = {"hosts": [host(id=f"h{i}") for i in range(33)]}
    with pytest.raises(HostsError, match="1-32"):
        load_hosts(write_hosts(tmp_path / "many.json", too_many))


# ── hosts.json: what is refused ─────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("doc,needle", [
    ("{not json", "not valid JSON"),
    ("", "not valid JSON"),
    ("[]", "nothing else"),
    ('"hosts"', "nothing else"),
    ({"hosts": [host()], "extra": 1}, "nothing else"),
    ({"hosts": {}}, "1-32"),
    ({"hosts": []}, "1-32"),
    ({"hosts": ["local"]}, "must be an object"),
    ({"hosts": [host(), host()]}, "used twice"),
    ({"hosts": [host(colour="red")]}, "unknown key"),
    ({"hosts": [{"ssh": "ops@10.0.0.5"}]}, "id must be"),
    ({"hosts": [host(id="Upper")]}, "id must be"),
    ({"hosts": [host(id="-lead")]}, "id must be"),
    ({"hosts": [host(id="a" * 33)]}, "id must be"),
    ({"hosts": [host(id="a\n")]}, "id must be"),                    # fullmatch, not match-with-$: no trailing newline
    ({"hosts": [host(label="")]}, "label"),
    ({"hosts": [host(label="x" * 65)]}, "label"),
    ({"hosts": [host(label="bad\x07bell")]}, "label"),
    ({"hosts": [host(label=7)]}, "label"),
    ({"hosts": [host(enabled="yes")]}, "true or false"),
    ({"hosts": [host(ssh_config=1)]}, "true or false"),
    ({"hosts": [{"id": "local", "ssh": "ops@10.0.0.5"}]}, "takes no ssh"),
    ({"hosts": [{"id": "local", "port": 22}]}, "takes no ssh"),
    ({"hosts": [{"id": "local", "ssh_config": True}]}, "takes no ssh"),
    ({"hosts": [{"id": "gpu-box-2"}]}, "ssh must be"),
    ({"hosts": [host(ssh=None)]}, "ssh must be"),
    ({"hosts": [host(ssh=5)]}, "ssh must be"),
    ({"hosts": [host(ssh="-oProxyCommand=touch${IFS}x")]}, "ssh must be"),
    ({"hosts": [host(ssh="ops@-host")]}, "ssh must be"),
    ({"hosts": [host(ssh="ops@host%h")]}, "ssh must be"),
    ({"hosts": [host(ssh="ops@host name")]}, "ssh must be"),
    ({"hosts": [host(ssh="ops@host\n")]}, "ssh must be"),           # a trailing newline must not slip through
    ({"hosts": [host(ssh="ops@host;reboot")]}, "ssh must be"),
    ({"hosts": [host(ssh="a" * 33 + "@host")]}, "ssh must be"),
    ({"hosts": [host(ssh="h" * 254)]}, "ssh must be"),
    ({"hosts": [host(port=0)]}, "port"),
    ({"hosts": [host(port=65536)]}, "port"),
    ({"hosts": [host(port=True)]}, "port"),
    ({"hosts": [host(port="22")]}, "port"),
    ({"hosts": [host(port=22.5)]}, "port"),
])
def test_refused_documents(tmp_path: Path, doc: Any, needle: str) -> None:
    with pytest.raises(HostsError, match=needle):
        load_hosts(write_hosts(tmp_path / "hosts.json", doc))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_numbers_are_rejected(tmp_path: Path, constant: str) -> None:
    text = '{"hosts": [{"id": "gpu-box-2", "ssh": "ops@10.0.0.5", "port": ' + constant + "}]}"
    with pytest.raises(HostsError):
        load_hosts(write_hosts(tmp_path / "hosts.json", text))


def test_duplicate_keys_in_one_object_are_rejected(tmp_path: Path) -> None:
    text = '{"hosts": [{"id": "a", "id": "b", "ssh": "ops@10.0.0.5"}]}'
    with pytest.raises(HostsError, match="appears twice"):
        load_hosts(write_hosts(tmp_path / "hosts.json", text))


def test_absurd_nesting_is_an_error_not_a_crash(tmp_path: Path) -> None:
    with pytest.raises(HostsError, match="not valid JSON"):
        load_hosts(write_hosts(tmp_path / "hosts.json", "[" * 60_000))   # RecursionError inside the JSON decoder


def test_huge_integers_are_an_error(tmp_path: Path) -> None:
    with pytest.raises(HostsError):
        load_hosts(write_hosts(tmp_path / "hosts.json", '{"hosts": [{"id": "a", "ssh": "x", "port": ' + "9" * 6000 + "}]}"))


def test_invalid_utf8_is_an_error(tmp_path: Path) -> None:
    p = tmp_path / "hosts.json"
    p.write_bytes(b'{"hosts": [{"id": "a\xff"}]}')
    p.chmod(0o600)
    with pytest.raises(HostsError, match="not valid JSON"):
        load_hosts(p)


def test_error_messages_name_the_file(tmp_path: Path) -> None:
    p = write_hosts(tmp_path / "hosts.json", {"hosts": []})
    with pytest.raises(HostsError) as e:
        load_hosts(p)
    assert str(e.value).startswith(str(p))


# ── hosts.json: the file itself must be trustworthy ─────────────────────────────────────────────────────
@pytest.mark.parametrize("mode", [0o620, 0o602, 0o666, 0o664])
def test_group_or_world_writable_is_refused(tmp_path: Path, mode: int) -> None:
    p = write_hosts(tmp_path / "hosts.json", {"hosts": [host()]}, mode)
    with pytest.raises(HostsError, match="group- or world-writable"):
        load_hosts(p)


def test_a_symlink_is_refused_even_to_a_good_file(tmp_path: Path) -> None:
    real = write_hosts(tmp_path / "real.json", {"hosts": [host()]})
    link = tmp_path / "hosts.json"
    link.symlink_to(real)
    with pytest.raises(HostsError, match="symlink"):
        load_hosts(link)


def test_a_dangling_symlink_is_refused_not_treated_as_absent(tmp_path: Path) -> None:
    link = tmp_path / "hosts.json"
    link.symlink_to(tmp_path / "nowhere.json")
    with pytest.raises(HostsError, match="symlink"):
        load_hosts(link)


def test_a_file_owned_by_someone_else_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = write_hosts(tmp_path / "hosts.json", {"hosts": [host()]})
    monkeypatch.setattr(os, "geteuid", lambda: os.stat(p).st_uid + 1)
    with pytest.raises(HostsError, match="owned by"):
        load_hosts(p)


def test_a_directory_and_a_fifo_are_refused_without_hanging(tmp_path: Path) -> None:
    d = tmp_path / "dir.json"
    d.mkdir()
    with pytest.raises(HostsError, match="regular file"):
        load_hosts(d)
    fifo = tmp_path / "fifo.json"
    os.mkfifo(fifo)
    with pytest.raises(HostsError, match="regular file"):
        load_hosts(fifo)


def test_size_cap(tmp_path: Path) -> None:
    padding = " " * (64 * 1024)
    p = write_hosts(tmp_path / "hosts.json", json.dumps({"hosts": [host()]}) + padding)
    with pytest.raises(HostsError, match="over 64 KiB"):
        load_hosts(p)
    small = write_hosts(tmp_path / "ok.json", json.dumps({"hosts": [host()]}) + " " * 1000)
    assert load_hosts(small)


def test_an_unreadable_file_is_an_error_not_a_default(tmp_path: Path) -> None:
    p = write_hosts(tmp_path / "hosts.json", {"hosts": [host()]}, 0o000)
    if os.access(p, os.R_OK):          # running as root: the mode does not stop us, so there is nothing to test
        pytest.skip("file is readable regardless of its mode")
    with pytest.raises(HostsError, match="cannot open"):
        load_hosts(p)


def test_parse_hosts_is_usable_on_its_own() -> None:
    assert parse_hosts({"hosts": [host()]})[0].ssh == "ops@10.0.0.5"


# ── the ssh command line ────────────────────────────────────────────────────────────────────────────────
OPTIONS = ["BatchMode=yes", "StrictHostKeyChecking=yes", "UpdateHostKeys=no", "VerifyHostKeyDNS=no", "IdentityAgent=none",
           "PKCS11Provider=none", "PasswordAuthentication=no", "KbdInteractiveAuthentication=no", "ControlMaster=no",
           "ControlPath=none", "ForwardAgent=no", "ForwardX11=no", "ClearAllForwardings=yes", "PermitLocalCommand=no",
           "ConnectTimeout=5", "ServerAliveInterval=5", "ServerAliveCountMax=3"]


def expected_argv(key: str, dest: str, *, port: int | None = None, with_config: bool = False) -> list[str]:
    argv = ["ssh", "-T", "-e", "none"] + ([] if with_config else ["-F", "none"])
    for o in OPTIONS:
        argv += ["-o", o]
    argv += ["-o", f"IdentityFile={key}", "-o", "IdentitiesOnly=yes"] + (["-p", str(port)] if port else [])
    return argv + ["--", dest, REMOTE_CMD]


def test_a_key_path_ssh_would_parse_as_syntax_is_refused(tmp_path: Path) -> None:
    from engine_console.domain.fleet import HostConfig
    from engine_console.services.hosts import HostsError, ssh_argv
    host = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="me@10.0.0.5")
    for name in ("with space", "quote'd", "pct%d"):
        key = tmp_path / name
        key.write_text("x")
        with pytest.raises(HostsError):
            ssh_argv(host, key, REMOTE_CMD)


@pytest.fixture
def hub_key(tmp_path: Path) -> Path:
    """A stand-in key file: ssh_argv only needs it to exist (ssh reads it later), and it is never a real key."""
    key = tmp_path / "hub-key"
    key.write_text("not a real key\n")
    return key


def test_ssh_argv_is_exactly_the_documented_command(hub_key: Path) -> None:
    h = HostConfig(id="gpu-box-2", label="GPU box 2", ssh="ops@10.0.0.5")
    assert ssh_argv(h, hub_key, REMOTE_CMD) == expected_argv(str(hub_key), "ops@10.0.0.5")


def test_ssh_argv_port_only_when_set_and_config_only_when_asked(hub_key: Path) -> None:
    h = HostConfig(id="gpu-box-2", label="x", ssh="ops@10.0.0.5", port=2222)
    assert ssh_argv(h, hub_key, REMOTE_CMD) == expected_argv(str(hub_key), "ops@10.0.0.5", port=2222)
    a = HostConfig(id="alias", label="x", ssh="edge-alias", ssh_config=True)
    argv = ssh_argv(a, str(hub_key), REMOTE_CMD)
    assert argv == expected_argv(str(hub_key), "edge-alias", with_config=True)
    assert "-F" not in argv and "-p" not in argv


def test_the_destination_follows_a_double_dash_and_is_never_an_option(hub_key: Path) -> None:
    argv = ssh_argv(HostConfig(id="a", label="a", ssh="ops@10.0.0.5"), hub_key, REMOTE_CMD)
    assert argv[-3:] == ["--", "ops@10.0.0.5", REMOTE_CMD]
    assert argv.count("--") == 1


@pytest.mark.parametrize("bad", ["-oProxyCommand=evil", "-J jump", "ops@-host", "-ops@10.0.0.5", ".ops@host", "a b", "host\n", "", "x;y", "host%h"])
def test_ssh_argv_refuses_a_destination_that_is_not_a_plain_name(bad: str, hub_key: Path) -> None:
    h = HostConfig.model_construct(id="a", label="a", ssh=bad, port=None, ssh_config=False, enabled=True)   # bypasses loader checks
    with pytest.raises(HostsError):
        ssh_argv(h, hub_key, REMOTE_CMD)


def test_ssh_argv_refuses_local_and_odd_key_paths_and_commands(hub_key: Path) -> None:
    with pytest.raises(HostsError, match="without ssh"):
        ssh_argv(HostConfig(id="local", label="l"), hub_key, REMOTE_CMD)
    h = HostConfig(id="a", label="a", ssh="ops@10.0.0.5")
    for key in ("", "-key", "k\n", "k\x00"):
        with pytest.raises(HostsError, match="key path"):
            ssh_argv(h, key, REMOTE_CMD)
    for cmd in ("", "ecnode --stdio", "/x; reboot", '/x"y', "/x --a\nb", "/x $(id)", "-/x", "/x " + "a" * 100 * 100):
        with pytest.raises(HostsError, match="remote node command"):
            ssh_argv(h, hub_key, cmd)


def test_ssh_argv_refuses_to_build_a_command_without_the_hub_key(tmp_path: Path) -> None:
    """With a missing `-i` file the real ssh only warns and then offers the user's default identity files, so the
    command must never be built: the console would log in with a key that is not restricted to the node."""
    h = HostConfig(id="a", label="a", ssh="ops@10.0.0.5")
    with pytest.raises(NoHubKey, match="hosts init-key"):
        ssh_argv(h, tmp_path / "missing" / "hub-key", REMOTE_CMD)
    (tmp_path / "a-directory").mkdir()
    with pytest.raises(NoHubKey):
        ssh_argv(h, tmp_path / "a-directory", REMOTE_CMD)


@pytest.mark.skipif(shutil.which("ssh") is None, reason="no ssh client on this machine")
def test_the_real_ssh_client_accepts_every_option_and_applies_it(hub_key: Path) -> None:
    """`ssh -G` evaluates a command line without connecting, so it proves the options are ones ssh knows (a typo is
    "Bad configuration option") and that they take effect. `-F none` keeps it from reading any user or system config."""
    h = HostConfig(id="gpu-box-2", label="x", ssh="ops@10.0.0.5", port=2222)
    argv = ssh_argv(h, hub_key, REMOTE_CMD)
    done = subprocess.run([argv[0], "-G", *argv[1:]], capture_output=True, text=True, timeout=15, env=ssh_env(), check=False)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    cfg = dict(line.split(" ", 1) for line in lines if " " in line)
    assert cfg["batchmode"] == "yes" and cfg["stricthostkeychecking"] == "true" and cfg["updatehostkeys"] == "false"
    assert cfg["identitiesonly"] == "yes" and cfg["forwardagent"] == "no" and cfg["clearallforwardings"] == "yes"
    assert cfg["port"] == "2222" and cfg["user"] == "ops" and cfg["hostname"] == "10.0.0.5"
    assert cfg["connecttimeout"] == "5" and cfg["serveraliveinterval"] == "5" and cfg["serveralivecountmax"] == "3"
    assert cfg["passwordauthentication"] == "no" and cfg["kbdinteractiveauthentication"] == "no"
    assert cfg["identityagent"] == "none" and cfg["controlmaster"] == "false" and "controlpath" not in cfg
    assert cfg["requesttty"] == "false" and cfg["escapechar"] == "none" and cfg["permitlocalcommand"] == "no"
    assert cfg["verifyhostkeydns"] == "false" and cfg["forwardx11"] == "no"
    # the hub's key is the ONLY identity ssh will offer: no default identity files are added next to it
    assert [ln for ln in lines if ln.startswith("identityfile ")] == [f"identityfile {hub_key}"]


# ── environments ────────────────────────────────────────────────────────────────────────────────────────
def test_ssh_env_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {"SSH_AUTH_SOCK": "/run/agent", "HF_TOKEN": "hf_secretsecret", "HTTPS_PROXY": "http://p", "DISPLAY": ":0",
                 "SSH_ASKPASS": "/bin/askpass", "LANG": "de_DE.UTF-8", "USER": "ops", "HOME": "/home/ops",
                 "PATH": "/usr/bin:/bin"}.items():
        monkeypatch.setenv(k, v)
    assert ssh_env() == {"HOME": "/home/ops", "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "USER": "ops"}


def test_ssh_env_falls_back_when_the_environment_is_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("HOME", "PATH", "USER"):
        monkeypatch.delenv(k, raising=False)
    env = ssh_env()
    assert env["PATH"] and env["HOME"] and env["LANG"] == "C.UTF-8"
    assert set(env) <= {"HOME", "PATH", "LANG", "USER"}


def test_local_env_never_carries_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in {"HF_TOKEN": "hf_secretsecret", "VLLM_API_KEY": "k", "ENGINE_CONSOLE_ENV_FILE": "/x", "SSH_AUTH_SOCK": "/a",
                 "HOME": "/home/ops", "PATH": "/usr/bin", "USER": "ops", "XDG_RUNTIME_DIR": "/run/user/1000",
                 "DOCKER_HOST": "unix:///run/user/1000/docker.sock"}.items():
        monkeypatch.setenv(k, v)
    env = local_env()
    assert set(env) <= set(H._LOCAL_ENV_KEYS)
    assert env["DOCKER_HOST"] == "unix:///run/user/1000/docker.sock" and env["XDG_RUNTIME_DIR"] == "/run/user/1000"
    assert not {"HF_TOKEN", "VLLM_API_KEY", "SSH_AUTH_SOCK", "ENGINE_CONSOLE_ENV_FILE"} & set(env)


def test_local_argv(make_settings: Any, tmp_path: Path) -> None:
    default = [sys.executable, "-I", "-m", "ecnode", "--stdio"]
    assert local_argv(make_settings()) == default
    assert local_argv(make_settings(node_policy=tmp_path / "policy.json")) == [*default, "--policy", str(tmp_path / "policy.json")]
    # an override is the complete command line: the policy setting is not appended to it
    assert local_argv(make_settings(node_command=["python3", "fake.py"], node_policy=tmp_path / "p.json")) == ["python3", "fake.py"]
    assert local_argv(make_settings(node_command=[])) == default


# ── the line that authorizes the hub's key on a machine ─────────────────────────────────────────────────
def test_key_line_format() -> None:
    line = key_line(PUB, "10.0.0.5", REMOTE_CMD)
    assert line == ('restrict,from="10.0.0.5",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 '
                    "AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl engine-console-hub")
    assert "user@somewhere" not in line                      # the public key's own comment is replaced, never kept


def test_key_line_defaults_and_from_variants() -> None:
    assert key_line(PUB + "\n", None).startswith('restrict,command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 ')
    for ok in ("10.0.0.0/24", "10.0.0.5,10.0.0.6", "fd00::1", "*.lab.example", "!10.0.0.9,10.0.0.*"):
        assert f'from="{ok}"' in key_line(PUB, ok)


@pytest.mark.parametrize("pub", [
    "", "ssh-ed25519", "garbage AAAAC3NzaC1lZDI1NTE5AAAAIOMq", "ssh-dss AAAAC3NzaC1lZDI1NTE5AAAAIOMq",
    "ssh-ed25519 not*base64*at*all*!!", "ssh-ed25519 AAAA", PUB + "\nssh-ed25519 " + PUB.split()[1],
    'command="x" ' + PUB, "ssh-ed25519\rAAAAC3NzaC1lZDI1NTE5AAAAIOMq\nrest",
])
def test_key_line_refuses_anything_but_one_public_key(pub: str) -> None:
    with pytest.raises(HostsError):
        key_line(pub, "10.0.0.5")


@pytest.mark.parametrize("from_", ["", '10.0.0.5"', "10.0.0.5 ", "10.0.0.5\n", 'a",command="x', "a" * 257, "10.0.0.5;x"])
def test_key_line_refuses_a_from_that_could_leave_its_quotes(from_: str) -> None:
    with pytest.raises(HostsError, match="--from"):
        key_line(PUB, from_)


@pytest.mark.parametrize("cmd", ['/x"; reboot; "', "/x\n", "relative/ecnode --stdio", "/x ; id", "/x --a=$(id)"])
def test_key_line_refuses_a_remote_command_that_could_leave_its_quotes(cmd: str) -> None:
    with pytest.raises(HostsError, match="remote node command"):
        key_line(PUB, "10.0.0.5", cmd)


# ── init_key (real ssh-keygen in a temp dir) ────────────────────────────────────────────────────────────
needs_keygen = pytest.mark.skipif(shutil.which("ssh-keygen") is None, reason="ssh-keygen is not installed")


@needs_keygen
def test_init_key_creates_a_private_key_and_returns_the_line(tmp_path: Path) -> None:
    key = tmp_path / "config" / "ssh" / "hub-key"
    line = init_key(key, "10.0.0.5", REMOTE_CMD)
    assert line.startswith('restrict,from="10.0.0.5",command="/usr/local/lib/ecnode/ecnode --stdio" ssh-ed25519 ')
    assert line.endswith(" engine-console-hub")
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700 and stat.S_IMODE(key.parent.parent.stat().st_mode) == 0o700
    assert "PRIVATE KEY" not in line
    assert key.read_text().startswith("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert (tmp_path / "config" / "ssh" / "hub-key.pub").read_text().split()[1] in line
    # the key is passphrase-less (it is used unattended): ssh-keygen can derive its public half without a prompt
    derived = subprocess.run(["ssh-keygen", "-y", "-f", str(key)], capture_output=True, text=True, timeout=15,
                             stdin=subprocess.DEVNULL, check=True).stdout
    assert derived.split()[1] in line


@needs_keygen
def test_init_key_never_overwrites_an_existing_key(tmp_path: Path) -> None:
    key = tmp_path / "ssh" / "hub-key"
    first = init_key(key, "10.0.0.5", REMOTE_CMD)
    before, before_pub = key.read_bytes(), key.with_name("hub-key.pub").read_bytes()
    again = init_key(key, "10.0.0.5", REMOTE_CMD)
    assert again == first and key.read_bytes() == before and key.with_name("hub-key.pub").read_bytes() == before_pub
    assert init_key(key, "10.0.0.9", REMOTE_CMD) != first        # same key, a different from=
    assert key.read_bytes() == before


@needs_keygen
def test_init_key_rederives_a_lost_public_half(tmp_path: Path) -> None:
    key = tmp_path / "hub-key"
    first = init_key(key, "10.0.0.5")
    key.with_name("hub-key.pub").unlink()
    assert init_key(key, "10.0.0.5") == first


@needs_keygen
def test_init_key_validates_before_touching_the_disk(tmp_path: Path) -> None:
    key = tmp_path / "new-dir" / "hub-key"
    with pytest.raises(HostsError, match="--from"):
        init_key(key, 'bad"from')
    assert not (tmp_path / "new-dir").exists()


@needs_keygen
def test_init_key_does_not_clobber_a_stray_public_key_or_follow_a_dangling_link(tmp_path: Path) -> None:
    (tmp_path / "hub-key.pub").write_text("precious\n")
    with pytest.raises(HostsError, match="without its private key"):
        init_key(tmp_path / "hub-key", "10.0.0.5")
    assert (tmp_path / "hub-key.pub").read_text() == "precious\n"
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "nowhere")
    with pytest.raises(HostsError, match="dangling"):
        init_key(link, "10.0.0.5")
    assert not (tmp_path / "nowhere").exists()


def test_init_key_without_ssh_keygen_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(HostsError, match="ssh-keygen was not found"):
        init_key(tmp_path / "k", "10.0.0.5", keygen="definitely-not-an-installed-program")
    assert not (tmp_path / "k").exists()


def test_init_key_restores_the_umask(tmp_path: Path) -> None:
    old = os.umask(0o022)
    try:
        with pytest.raises(HostsError):
            init_key(tmp_path / "k", "10.0.0.5", keygen="definitely-not-an-installed-program")
        assert os.umask(0o022) == 0o022
    finally:
        os.umask(old)
