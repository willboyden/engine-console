"""Real HubClient over httpx with the ADR 8 egress allowlist. The token is only ever put in an
Authorization header (never a URL) and httpx drops it on cross-origin redirects (CDN presigned URLs)."""
from __future__ import annotations

import fnmatch
import ssl
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from engine_console.domain.errors import (
    EgressDenied,
    EgressProxyUnavailable,
    GatedModel,
    NotFound,
    ProblemError,
    UpstreamError,
)
from engine_console.domain.ports import ByteStream, HubModel, RepoFile


def host_allowed(host: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(host.lower(), p.lower()) for p in patterns)


@dataclass
class EgressConfig:
    proxy: str | None = None
    ca_bundle: Path | None = None
    require_proxy: bool = False

    @property
    def mode(self) -> str:
        return "proxied" if self.proxy else "direct"


def public_proxy(url: str | None) -> str | None:
    """Proxy URL without any userinfo, safe to show."""
    if not url:
        return None
    u = urlparse(url)
    return f"{u.scheme}://{u.hostname}{f':{u.port}' if u.port else ''}"


class _HttpStream:
    def __init__(self, resp: httpx.Response, requested_start: int) -> None:
        self._resp = resp
        length = resp.headers.get("content-length")
        self.offset = requested_start if resp.status_code == 206 else 0
        self.total = (int(length) + self.offset) if length and length.isdigit() else None

    async def chunks(self) -> AsyncIterator[bytes]:
        async for c in self._resp.aiter_bytes(1024 * 1024):
            yield c

    async def aclose(self) -> None:
        await self._resp.aclose()


class HfHttpClient:
    def __init__(self, endpoint: str, allowed_hosts: list[str], token: Callable[[], str | None],
                 client: httpx.AsyncClient | None = None, egress: EgressConfig | None = None) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._allowed = allowed_hosts
        self._token = token
        self._egress = egress or EgressConfig()
        self._injected = client
        self._built: httpx.AsyncClient | None = None

    @property
    def _client(self) -> httpx.AsyncClient:
        """Fail-closed client factory. Every network operation goes through here, so a missing CA, a missing
        proxy (when required) or an unbuildable TLS context stops the request instead of degrading to direct."""
        eg = self._egress
        if eg.require_proxy and not eg.proxy:
            raise EgressProxyUnavailable("REQUIRE_EGRESS_PROXY is set but EGRESS_PROXY is not: refusing Hugging Face traffic")
        if self._injected is not None and not eg.proxy:
            self._hook(self._injected)
            return self._injected
        if self._built is None:
            # trust_env=False in BOTH modes: ambient HTTP(S)_PROXY / SSL_CERT_FILE / netrc must never steer HF traffic
            kw: dict[str, object] = {"timeout": httpx.Timeout(30.0, read=120.0), "follow_redirects": True,
                                     "trust_env": False}
            if eg.proxy:
                if eg.ca_bundle is not None and not eg.ca_bundle.is_file():
                    raise EgressProxyUnavailable("egress proxy configured but its CA certificate file is missing: "
                                                 "refusing to connect (no direct fallback)")
                try:   # no bundle configured: the system trust store (the proxy's CA must be installed there)
                    kw.update(proxy=eg.proxy, verify=ssl.create_default_context(
                        cafile=str(eg.ca_bundle) if eg.ca_bundle else None))
                except (ssl.SSLError, OSError, ValueError) as e:
                    raise EgressProxyUnavailable(f"cannot load the egress CA bundle: {type(e).__name__}") from e
            self._built = httpx.AsyncClient(**kw)  # type: ignore[arg-type]
            self._hook(self._built)
        elif eg.proxy and eg.ca_bundle is not None and not eg.ca_bundle.is_file():
            raise EgressProxyUnavailable("egress CA certificate disappeared: refusing to connect")
        return self._built

    def _hook(self, c: httpx.AsyncClient) -> None:
        # request hooks fire on every redirect hop too, so a redirect cannot leave the allowlist
        hooks = c.event_hooks.setdefault("request", [])
        if self._check_egress not in hooks:
            hooks.append(self._check_egress)

    def _transport_error(self, e: httpx.HTTPError) -> ProblemError:
        if self._egress.proxy and isinstance(e, httpx.ConnectError | httpx.ConnectTimeout | httpx.ProxyError):
            # A TLS verification failure through an intercepting proxy is a CA problem, not a reachability one;
            # saying "unreachable" sends the operator to debug the wrong thing.
            if "CERTIFICATE_VERIFY_FAILED" in str(e) or isinstance(e.__cause__, ssl.SSLCertVerificationError):
                return EgressProxyUnavailable(
                    "TLS verification failed through the egress proxy: set EGRESS_CA_BUNDLE to the proxy's CA "
                    "certificate (PEM), or install it in the system trust store. Refusing to connect directly.")
            return EgressProxyUnavailable("egress proxy unreachable: refusing to connect directly")
        return UpstreamError(f"cannot reach Hugging Face: {type(e).__name__}")

    async def _check_egress(self, request: httpx.Request) -> None:
        if request.url.scheme != "https":   # also fires on every redirect hop: no https -> http downgrade
            raise EgressDenied("refusing a non-https request/redirect target for Hugging Face traffic")
        if not host_allowed(request.url.host, self._allowed):
            raise EgressDenied(f"egress to {request.url.host} is not on the HF allowlist")

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        h = {"User-Agent": "engine-console"}
        tok = self._token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
        h.update(extra or {})
        return h

    def _raise(self, resp: httpx.Response, repo: str | None = None) -> None:
        if resp.status_code < 400:
            return
        if resp.status_code == 403 and "egress blocked" in resp.text[:200]:   # addon_guard's 403 body
            raise EgressDenied("the egress proxy blocked this host; add it to the proxy's allowlist")
        if resp.status_code in (401, 403):
            tail = f" {repo}" if repo else ""
            raise GatedModel(f"Hugging Face denied access to{tail}: the model is gated or private. Set HF_TOKEN "
                             f"and accept the license at {self._endpoint}/{repo or ''}")
        if resp.status_code == 404:
            raise NotFound(f"not found on Hugging Face: {repo or resp.request.url.path}")
        raise UpstreamError(f"Hugging Face returned HTTP {resp.status_code}")

    async def _get(self, url: str, params: Sequence[tuple[str, str]] | None = None, repo: str | None = None) -> httpx.Response:
        client = self._client
        query: list[tuple[str, str | int | float | bool | None]] = list(params or [])
        try:
            resp = await client.get(url, params=query or None, headers=self._headers())
        except httpx.HTTPError as e:
            raise self._transport_error(e) from e
        self._raise(resp, repo)
        return resp

    async def search(self, *, q: str, task: str | None, library: str | None, quant: str | None,
                     sort: str, limit: int, token_ok: bool = True) -> list[HubModel]:
        params: list[tuple[str, str]] = [("limit", str(limit)), ("sort", sort), ("direction", "-1")]
        if q:
            params.append(("search", q))
        if task:
            params.append(("pipeline_tag", task))
        if library:
            params.append(("library", library))
        if quant:
            params.append(("filter", quant))
        params += [("expand[]", f) for f in ("downloads", "likes", "pipeline_tag", "tags", "gated",
                                             "library_name", "safetensors")]
        resp = await self._get(f"{self._endpoint}/api/models", params)
        return [_parse_model(m) for m in resp.json()]

    async def model(self, repo_id: str, revision: str | None) -> HubModel:
        rev = quote(revision or "main", safe="")
        resp = await self._get(f"{self._endpoint}/api/models/{quote(repo_id, safe='/')}/revision/{rev}", [("blobs", "true")], repo_id)
        hub = _parse_model(resp.json())
        try:
            cfg = await self._get(f"{self._endpoint}/{quote(repo_id, safe='/')}/resolve/{quote(hub.sha or revision or 'main', safe='')}/config.json",
                                  repo=repo_id)
            parsed = cfg.json()
            hub.config = parsed if isinstance(parsed, dict) else None
        except (NotFound, GatedModel, ValueError):
            hub.config = None   # gated w/o token, or a non-transformers repo: degrade, don't fail
        return hub

    async def card(self, repo_id: str, revision: str | None) -> str:
        try:
            resp = await self._get(f"{self._endpoint}/{quote(repo_id, safe='/')}/raw/{quote(revision or 'main', safe='')}/README.md", repo=repo_id)
        except NotFound:
            return ""
        return resp.text

    async def open_file(self, repo_id: str, revision: str, path: str, start: int) -> ByteStream:
        url = f"{self._endpoint}/{quote(repo_id, safe='/')}/resolve/{quote(revision, safe='')}/{quote(path)}"
        extra = {"Range": f"bytes={start}-"} if start > 0 else None
        client = self._client
        req = client.build_request("GET", url, headers=self._headers(extra))
        try:
            resp = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            raise self._transport_error(e) from e
        if resp.status_code >= 400:
            await resp.aread()
            await resp.aclose()
            self._raise(resp, repo_id)
        return _HttpStream(resp, start)

    async def aclose(self) -> None:
        for c in (self._built, self._injected):
            if c is not None:
                await c.aclose()


def _parse_model(m: dict[str, Any]) -> HubModel:
    st = m.get("safetensors") or {}
    card = m.get("cardData") or {}
    lic = card.get("license") if isinstance(card, dict) else None
    tags = [t for t in m.get("tags", []) if isinstance(t, str)]
    if not lic:
        lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
    files = []
    for s in m.get("siblings") or []:
        lfs = s.get("lfs") or {}
        files.append(RepoFile(path=s["rfilename"], size=int(s.get("size") or lfs.get("size") or 0),
                              sha256=lfs.get("sha256") or lfs.get("oid")))
    gated = m.get("gated")
    return HubModel(repo_id=m.get("id") or m.get("modelId") or "", sha=m.get("sha"), gated=bool(gated),
                    license=lic, pipeline_tag=m.get("pipeline_tag"), library_name=m.get("library_name"),
                    tags=tags, files=files, safetensors_params=dict(st.get("parameters") or {}),
                    safetensors_total=st.get("total"), config=None, downloads=m.get("downloads"),
                    likes=m.get("likes"))


def url_host(url: str) -> str:
    return urlparse(url).hostname or ""
