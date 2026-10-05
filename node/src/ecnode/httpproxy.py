"""http.request: one HTTP/1.1 exchange with an engine on 127.0.0.1, for the hub's chat, benchmarks and probes.

Only ports that are a console-owned gateway's published port or an engine endpoint of the current inventory;
methods GET and POST; paths under the engine API prefixes of docs/FLEET-PROTOCOL.md §4; hop-by-hop and Host
headers from the hub are dropped. http.client never follows redirects. The response goes back as one event
{status, headers}, then the body as DATA under the hub's credit, then res {status, bytes}.
"""
from __future__ import annotations

import contextlib
import http.client
import re
import socket
import urllib.parse
from collections.abc import Callable
from typing import Any

from . import proto
from .errors import Cancelled, OpError, Stream

PATH_PREFIXES = ("/v1/", "/health", "/metrics", "/version", "/api/", "/get_model_info", "/get_server_info",
                 "/server_info")
HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                        "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
                        "expect"})
DEFAULT_TIMEOUT_S = 600.0
MAX_TIMEOUT_S = 3600.0
MAX_BODY = 1 << 30
CONNECT_S = 5.0
# Monitor mode is read-only: GET anything above, but POST only to inference. Ollama's /api/pull, /api/create, /api/push,
# /api/copy, /api/delete and /api/blobs change the machine; they need manage mode.
INFERENCE_POSTS = frozenset({"/v1/chat/completions", "/v1/completions", "/v1/embeddings", "/api/chat", "/api/generate",
                             "/api/embed", "/api/embeddings", "/api/show"})
_TOKEN = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]{1,128}")
_PATH = re.compile(r"/[\x21-\x7e]{0,2047}")


def check_path(path: object) -> str:
    if not isinstance(path, str) or not _PATH.fullmatch(path) or "#" in path:
        raise OpError(proto.BAD_ARGS, "the path must start with '/' and hold only printable ASCII (no spaces)")
    route = path.split("?", 1)[0]
    decoded = urllib.parse.unquote(route)
    if any(seg in (".", "..") for seg in decoded.split("/")) or "\\" in decoded or any(ord(c) < 32 for c in decoded):
        raise OpError(proto.BAD_ARGS, "the path may not contain '.' or '..' segments or control characters")
    if not route.startswith(PATH_PREFIXES):
        raise OpError(proto.POLICY_DENIED, f"path {route[:200]!r} is not an engine API path this node forwards "
                                           f"({', '.join(PATH_PREFIXES)})")
    return path


def clean_headers(headers: dict[str, str] | None) -> list[tuple[str, str]]:
    headers = headers or {}
    named = {h.strip().lower() for k, v in headers.items() if k.lower() == "connection" for h in v.split(",")}
    out: list[tuple[str, str]] = []
    for k, v in headers.items():
        lk = k.lower()
        if lk in HOP_BY_HOP or lk in named:
            continue
        if not _TOKEN.fullmatch(k):
            raise OpError(proto.BAD_ARGS, f"header name {k[:60]!r} is not a valid HTTP token")
        if any(c in v for c in "\r\n\x00"):
            raise OpError(proto.BAD_ARGS, f"header {k!r} has a line break in its value")
        out.append((k, v))
    return out


def _shutdown(conn: http.client.HTTPConnection) -> None:
    sock = conn.sock
    if sock is not None:
        with contextlib.suppress(OSError):
            sock.shutdown(socket.SHUT_RDWR)


def request(args: dict[str, Any], stream: Stream, allowed_ports: Callable[[bool], set[int]], *,
            monitor: bool = False) -> dict[str, Any]:
    """allowed_ports(refresh) -> engine endpoint ports + console gateway ports (refresh=True re-reads docker).
    monitor: the node's policy is in monitor mode, so POST is limited to INFERENCE_POSTS."""
    port = args["port"]
    if not 1 <= port <= 65535:
        raise OpError(proto.BAD_ARGS, "port must be 1-65535")
    if port not in allowed_ports(False) and port not in allowed_ports(True):
        raise OpError(proto.POLICY_DENIED, f"port {port} is not an engine endpoint or console gateway on this node")
    method = str(args["method"]).upper()
    if method not in ("GET", "POST"):
        raise OpError(proto.POLICY_DENIED, "only GET and POST are forwarded to engines")
    path = check_path(args["path"])
    if monitor and method == "POST" and path.split("?", 1)[0] not in INFERENCE_POSTS:
        raise OpError(proto.POLICY_DENIED, f"this node is in monitor mode: POST is forwarded only to inference "
                                           f"endpoints ({', '.join(sorted(INFERENCE_POSTS))}); other engine "
                                           "requests need mode \"manage\" in the node's policy")
    headers = clean_headers(args.get("headers"))
    body_len = args.get("body_len") or 0
    if not 0 <= body_len <= MAX_BODY:
        raise OpError(proto.BAD_ARGS, "body_len must be 0 to 1 GiB")
    if body_len and method == "GET":
        raise OpError(proto.BAD_ARGS, "a GET request has no body")
    timeout = args.get("timeout_s") or DEFAULT_TIMEOUT_S
    timeout = max(1.0, min(float(timeout), MAX_TIMEOUT_S))

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=CONNECT_S)
    stream.on_cancel(lambda: _shutdown(conn))
    try:
        conn.connect()
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        conn.putheader("Host", f"127.0.0.1:{port}")
        for k, v in headers:
            conn.putheader(k, v)
        if method == "POST":
            conn.putheader("Content-Length", str(body_len))
        conn.endheaders()
        sent = 0
        while sent < body_len:
            chunk = stream.read_upload()
            if not chunk:
                break
            take = chunk[: body_len - sent]
            conn.send(take)
            sent += len(take)
            if len(take) < len(chunk):
                raise OpError(proto.BAD_ARGS, f"the hub sent more than body_len ({body_len}) bytes")
        if sent != body_len:
            raise OpError(proto.BAD_ARGS, f"the hub sent {sent} of body_len {body_len} bytes")
        if body_len and stream.read_upload():          # the body is complete: what follows must be `end`
            raise OpError(proto.BAD_ARGS, f"the hub sent more than body_len ({body_len}) bytes")
        resp = conn.getresponse()
        hdrs: dict[str, str] = {}
        for k, v in resp.getheaders():
            lk = k.lower()
            if lk in HOP_BY_HOP - {"content-length"} or len(hdrs) >= 64 or len(k) > 256:
                continue
            hdrs[lk] = f"{hdrs[lk]}, {v}"[:4096] if lk in hdrs else v[:4096]
        stream.event({"status": resp.status, "headers": hdrs})
        total = 0
        while True:
            stream.check()
            chunk = resp.read1(65536)
            if not chunk:
                break
            stream.send_data(chunk)
            total += len(chunk)
        return {"status": resp.status, "bytes": total}
    except (Cancelled, OpError):
        raise
    except TimeoutError:
        if stream.cancelled:
            raise Cancelled from None
        raise OpError(proto.TIMEOUT, f"the engine on port {port} did not answer within {timeout:g} s") from None
    except ConnectionRefusedError:
        raise OpError(proto.HTTP_ERROR, f"nothing is listening on 127.0.0.1:{port} (is the engine running?)") from None
    except (OSError, http.client.HTTPException) as e:
        if stream.cancelled:
            raise Cancelled from None
        raise OpError(proto.HTTP_ERROR, f"the connection to the engine on port {port} failed "
                                        f"({type(e).__name__})") from None
    finally:
        conn.close()
