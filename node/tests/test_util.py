"""Secret masking, argv redaction, secret-looking names, subprocess and time helpers."""
from __future__ import annotations

import sys
import threading
import time

import pytest

from ecnode import util
from ecnode.util import REDACTED, is_secret_name, parse_time, redact_argv, scrub

SECRET = "s3cr3tVALUE_abc123"
FAKE_HF = "hf_" + "Fake0" * 5         # token-shaped for the scrubber, obviously synthetic for anyone else
FAKE_NV = "nvapi-" + "Fake0" * 5


@pytest.mark.parametrize("argv", [
    ["vllm", "serve", "m", "--api-key", SECRET],
    ["vllm", "serve", "m", f"--api-key={SECRET}"],
    ["x", "--hf-token", SECRET],
    ["x", f"--token={SECRET}"],
    ["env", f"HF_TOKEN={SECRET}", "python"],
    ["env", f"MY_PASSWORD={SECRET}"],
    ["sh", "-c", f"vllm serve m --api-key {SECRET} --port 8000"],
    ["sh", "-c", f"VLLM_API_KEY={SECRET} exec vllm serve m"],
    ["x", "--client-secret", SECRET],
    ["x", f"--db-url=postgres://admin:{SECRET}@db:5432/x"],
])
def test_redact_argv_masks_every_form(argv):
    out = redact_argv(argv)
    assert SECRET not in " ".join(out), out
    assert REDACTED in " ".join(out)


def test_redact_argv_keeps_ordinary_flags_and_values():
    argv = ["vllm", "serve", "org/m", "--tokenizer", "org/tok", "--max-num-batched-tokens", "8192",
            "--port", "8000", "--tensor-parallel-size", "2"]
    assert redact_argv(argv) == argv


def test_redact_argv_bounds_items_and_length():
    out = redact_argv(["x" * 5000] * 1000, max_items=400, max_len=1024)
    assert len(out) == 400 and all(len(t) <= 1024 for t in out)


@pytest.mark.parametrize("text", [
    "Authorization: Bearer abcdefghijklmnop1234",
    f"non-default args: {{'api_key': ['{SECRET}']}}",
    f'{{"api_key": "{SECRET}"}}',
    f"api_key={SECRET}",
    f"API_KEY: {SECRET}",
    f"--api-key {SECRET}",
    f"token {FAKE_HF}",
    f"using key {FAKE_NV}",
    f"https://user:{SECRET}@example.invalid/x",
])
def test_scrub_masks_secret_values_in_free_text(text):
    out = scrub(text)
    for s in (SECRET, "abcdefghijklmnop1234", FAKE_HF, FAKE_NV):
        assert s not in out, out


def test_scrub_leaves_ordinary_log_lines_alone():
    line = "INFO 10-04 12:00:00 engine.py:300 max_num_batched_tokens=8192, tokenizer=org/tok, port=8000"
    assert scrub(line) == line


def test_scrub_is_fast_on_hostile_lines():
    t0 = time.monotonic()
    scrub("a-" * 4000 + "=" + "b" * 8000)
    scrub(("x" * 30 + ".") * 300)
    assert time.monotonic() - t0 < 2.0


@pytest.mark.parametrize("name", ["id_rsa", "id_ed25519", "id_ed25519.pub", "server.pem", "tls.key", ".env",
                                  ".env.local", "token", "TOKEN", "tokens.json", "token.txt", "credentials",
                                  "credentials.json", "stored_tokens", ".netrc", ".ssh", ".config", ".gnupg"])
def test_secret_looking_names(name):
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json", "tokenizer.model", "config.json",
                                  "model-00001-of-00004.safetensors", "special_tokens_map.json", "README.md",
                                  "keymap.txt", "blobs", "snapshots"])
def test_model_files_are_not_secret_looking(name):
    assert not is_secret_name(name)


def test_parse_time_handles_docker_timestamps():
    assert parse_time("2026-09-01T10:00:00Z") == 1788256800.0
    assert parse_time("2026-09-01T10:00:00.5Z") == 1788256800.5
    assert parse_time("2026-09-01T10:00:00.123456789Z") == pytest.approx(1788256800.123456789)
    assert parse_time("2026-09-01T12:00:00+02:00") == 1788256800.0
    for bad in ("0001-01-01T00:00:00Z", "", None, "garbage", 5):
        assert parse_time(bad) is None


def test_run_reports_missing_binaries_and_timeouts(tmp_path):
    assert util.run([str(tmp_path / "nope")], timeout=5).rc == 127
    res = util.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.3)
    assert res.rc == 124
    ok = util.run([sys.executable, "-c", "print('hi')"], timeout=10)
    assert ok.ok and ok.out.strip() == "hi"


def test_run_tail_keeps_only_the_last_bytes():
    rc, data, truncated = util.run_tail([sys.executable, "-c", "print('x' * 100000); print('END')"],
                                        timeout=10, max_bytes=1000)
    assert rc == 0 and truncated and len(data) <= 1000 and data.rstrip().endswith(b"END")


def test_guarded_call_times_out_and_skips_a_stuck_key():
    g = util.Guarded()
    gate = threading.Event()
    assert g.call("k", lambda: gate.wait(5) and 1, 0.1) is None      # stuck
    assert g.call("k", lambda: 2, 0.5) is None                       # still stuck: skipped, no new thread
    gate.set()
    time.sleep(0.2)
    assert g.call("k", lambda: 3, 0.5) == 3
    assert g.call("other", lambda: 1 / 0, 0.5) is None                 # an exception is just "no answer"


def test_fit_tail_respects_the_encoded_size():
    text = "\x1b[0m" * 100_000 + "THE END"
    tail, trimmed = util.fit_tail(text, 10_000)
    assert trimmed and tail.endswith("THE END") and util.json_size(tail) <= 10_000
    assert util.fit_tail("short", 100) == ("short", False)
