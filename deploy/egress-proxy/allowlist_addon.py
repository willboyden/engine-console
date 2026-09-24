"""Minimal allowlisting addon for mitmproxy (opt-in template for Engine Console's EGRESS_PROXY).

Every HTTPS request the console makes is decrypted by mitmproxy, then checked here against
allowlist.txt (one fnmatch host glob per line, '#' comments). Anything else gets a 403 whose body
contains "egress blocked", which the console maps to an egress-denied error. The file is re-read on
every request, so edits apply without a restart.

Run (see docs/OPERATIONS.md): mitmdump --set block_global=false -s allowlist_addon.py
Generated 2026-09-24, unverified: not run against a live mitmproxy by the author.
"""
import fnmatch
import pathlib

from mitmproxy import http

ALLOWLIST = pathlib.Path(__file__).with_name("allowlist.txt")


def _patterns() -> list[str]:
    try:
        lines = ALLOWLIST.read_text().splitlines()
    except OSError:
        return []          # unreadable allowlist: deny everything (fail closed)
    return [ln.strip().lower() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def request(flow: http.HTTPFlow) -> None:
    host = flow.request.pretty_host.lower()
    if not any(fnmatch.fnmatchcase(host, p) for p in _patterns()):
        flow.response = http.Response.make(
            403, b"egress blocked: host not in allowlist.txt\n", {"Content-Type": "text/plain"})
