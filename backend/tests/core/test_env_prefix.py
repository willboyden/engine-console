"""Review finding L5: the generic setting names (HOST, PORT, ALLOWED_HOSTS, TRUST_LOOPBACK, DATA_DIR) are read only as
ENGINE_CONSOLE_<NAME>, from the environment and the env file alike, so a stray `export HOST=0.0.0.0` meant for something
else no longer widens the bind. The bare names are ignored with one warning at start-up that names only the variables."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import engine_console.main as main
from engine_console.config import Settings

BARE = {"HOST": "0.0.0.0", "PORT": "9", "ALLOWED_HOSTS": '["evil.example"]', "TRUST_LOOPBACK": "true",
        "DATA_DIR": "/tmp/somewhere-else"}


def defaults(s: Settings) -> bool:
    return (s.host, s.port, s.allowed_hosts, s.trust_loopback, s.data_dir) == (
        "127.0.0.1", 8791, [], False, Settings.model_fields["data_dir"].default)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in [*BARE, *(f"ENGINE_CONSOLE_{n}" for n in BARE), "EGRESS_PROXY", "MAX_SSE_PER_KEY", "MAX_CHAT_STREAMS",
                 "ENGINE_CONSOLE_EGRESS_PROXY"]:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_bare_generic_names_in_the_environment_are_ignored(clean_env: pytest.MonkeyPatch) -> None:
    for k, v in BARE.items():
        clean_env.setenv(k, v)
    assert defaults(Settings())


def test_bare_generic_names_in_the_env_file_are_ignored(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "env"
    f.write_text("".join(f"{k}={v}\n" for k, v in BARE.items()))
    assert defaults(Settings(_env_file=f))  # type: ignore[call-arg]


def test_prefixed_names_work_from_the_env_file_and_the_environment_wins(clean_env: pytest.MonkeyPatch,
                                                                        tmp_path: Path) -> None:
    f = tmp_path / "env"
    f.write_text("ENGINE_CONSOLE_HOST=10.0.0.1\nENGINE_CONSOLE_PORT=9000\nENGINE_CONSOLE_ALLOWED_HOSTS=[\"console.example\"]\n"
                 f"ENGINE_CONSOLE_TRUST_LOOPBACK=true\nENGINE_CONSOLE_DATA_DIR={tmp_path / 'data'}\nHOST=0.0.0.0\n")
    s = Settings(_env_file=f)  # type: ignore[call-arg]
    assert (s.host, s.port, s.allowed_hosts, s.trust_loopback, s.data_dir) == (
        "10.0.0.1", 9000, ["console.example"], True, tmp_path / "data")
    clean_env.setenv("ENGINE_CONSOLE_PORT", "9100")
    clean_env.setenv("HOST", "0.0.0.0")                                  # bare in the environment: still ignored
    s = Settings(_env_file=f)  # type: ignore[call-arg]
    assert s.port == 9100 and s.host == "10.0.0.1"


def test_every_other_setting_keeps_its_documented_name(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("EGRESS_PROXY", "http://127.0.0.1:8082")
    clean_env.setenv("MAX_SSE_PER_KEY", "3")
    clean_env.setenv("MAX_CHAT_STREAMS", "5")
    s = Settings()
    assert (s.egress_proxy, s.max_sse_per_key, s.max_chat_streams) == ("http://127.0.0.1:8082", 3, 5)
    clean_env.delenv("EGRESS_PROXY")
    clean_env.setenv("ENGINE_CONSOLE_EGRESS_PROXY", "http://127.0.0.1:9")  # only the five generic names take the prefix
    assert Settings().egress_proxy is None


def test_keyword_arguments_still_use_the_field_names(clean_env: pytest.MonkeyPatch, tmp_path: Path) -> None:
    s = Settings(host="10.0.0.2", port=1234, allowed_hosts=["x.example"], trust_loopback=True, data_dir=tmp_path)
    assert (s.host, s.port, s.allowed_hosts, s.trust_loopback, s.data_dir) == (
        "10.0.0.2", 1234, ["x.example"], True, tmp_path)


def test_an_empty_host_is_refused_rather_than_listening_everywhere(clean_env: pytest.MonkeyPatch) -> None:
    clean_env.setenv("ENGINE_CONSOLE_HOST", "")
    with pytest.raises(ValidationError, match="ENGINE_CONSOLE_HOST must not be empty"):
        Settings()


def test_a_stale_name_gives_one_warning_with_the_new_names_and_no_values(tmp_path: Path,
                                                                        caplog: pytest.LogCaptureFixture) -> None:
    f = tmp_path / "env"
    from engine_console.config import legacy_env_names
    f.write_text("PORT=4242\nDATA_DIR=/srv/private-place\nHF_ENDPOINT=https://huggingface.co\n")
    environ = {"HOST": "203.0.113.9", "PATH": "/usr/bin", "ENGINE_CONSOLE_PORT": "8791"}
    assert legacy_env_names(environ, f) == ["HOST", "PORT", "DATA_DIR"]
    assert legacy_env_names({"PATH": "/usr/bin"}, tmp_path / "missing") == []
    with caplog.at_level(logging.WARNING, logger="engine_console"):
        main.warn_legacy_env(environ, f)
        main.warn_legacy_env({"PATH": "/usr/bin"}, None)          # nothing stale: nothing logged
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "ENGINE_CONSOLE_HOST" in msg and "ENGINE_CONSOLE_PORT" in msg and "ENGINE_CONSOLE_DATA_DIR" in msg
    assert "203.0.113.9" not in msg and "4242" not in msg and "private-place" not in msg


def test_a_stray_exported_host_no_longer_widens_the_bind(clean_env: pytest.MonkeyPatch, tmp_path: Path,
                                                         caplog: pytest.LogCaptureFixture) -> None:
    """The finding end to end: `engine-console` started with HOST=0.0.0.0 in its environment still binds loopback."""
    clean_env.setenv("HOST", "0.0.0.0")
    clean_env.setenv("PORT", "80")
    bound: dict[str, Any] = {}
    clean_env.setattr(sys, "argv", ["engine-console"])
    clean_env.setattr(main, "ENV_FILE", tmp_path / "no-env-file", raising=False)
    clean_env.setattr(main, "configure_logging", lambda: None)
    clean_env.setattr(main, "create_app", lambda cfg: object())
    clean_env.setattr(main.uvicorn, "run", lambda app, **kw: bound.update(kw))
    with caplog.at_level(logging.WARNING, logger="engine_console"):
        main.run()
    assert bound["host"] == "127.0.0.1" and bound["port"] == 8791
    assert any("ENGINE_CONSOLE_HOST" in r.getMessage() for r in caplog.records)
