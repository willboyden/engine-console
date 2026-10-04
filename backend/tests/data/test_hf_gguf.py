"""GGUF quantisation grouping (the download picker's data) and the HF client's mid-download failure mapping."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from engine_console.domain.errors import UpstreamError
from engine_console.domain.ports import RepoFile
from engine_console.services.hf import HfService, gguf_quant, gguf_stem, group_gguf
from engine_console.services.hf_http import HfHttpClient

from .fakes import FakeHub


@pytest.mark.parametrize(("path", "quant"), [
    ("Qwen3-8B-Q4_K_M.gguf", "Q4_K_M"),
    ("Qwen3-8B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),                      # Unsloth's dynamic quants keep their prefix
    ("Qwen3-8B-UD-IQ2_XXS.gguf", "UD-IQ2_XXS"),
    ("model-IQ2_XXS.gguf", "IQ2_XXS"),
    ("model-IQ4_NL.gguf", "IQ4_NL"),
    ("model-BF16.gguf", "BF16"),
    ("model-F16.gguf", "F16"),
    ("model-F32.gguf", "F32"),
    ("gpt-oss-120b-MXFP4.gguf", "MXFP4"),
    ("gpt-oss-120b-MXFP4_MOE.gguf", "MXFP4_MOE"),
    ("model-Q8_0.gguf", "Q8_0"),
    ("model-Q5_K_S.gguf", "Q5_K_S"),
    ("model-Q2_K.gguf", "Q2_K"),
    ("model-Q3_K_L.gguf", "Q3_K_L"),
    ("model-TQ1_0.gguf", "TQ1_0"),
    ("qwen2.5-7b-instruct-q4_k_m.gguf", "Q4_K_M"),                    # lower case is common in official repos
    ("model.Q6_K.gguf", "Q6_K"),                                       # dot separated
    ("model.i1-Q4_K_M.gguf", "i1-Q4_K_M"),                             # imatrix quants stay apart from plain ones
    ("model-Q4_K_M-00001-of-00003.gguf", "Q4_K_M"),                    # a split set shares its quant
    ("Q4_K_M/Model-Q4_K_M-00002-of-00003.gguf", "Q4_K_M"),
    ("BF16/Big-00001-of-00010.gguf", "BF16"),                          # quant only in the folder name
    ("mmproj-F16.gguf", "mmproj-F16"),                                  # a vision projector is its own choice
    ("mmproj-BF16.gguf", "mmproj-BF16"),
    ("model.gguf", "model"),                                            # no token: labelled by its own name
    ("sub/ggml-model.gguf", "ggml-model"),
    ("mmproj.gguf", "mmproj"),
    ("Llama-3.1-8B-Instruct-Q4_K_M.GGUF", "Q4_K_M"),
    ("my-F16x-model.gguf", "my-F16x-model"),                           # a token inside a longer word is not a quant
])
def test_gguf_quant_token_from_a_repo_path(path: str, quant: str) -> None:
    assert gguf_quant(path) == quant


def test_gguf_stem_drops_the_extension_and_the_split_suffix() -> None:
    assert gguf_stem("a/b-Q4_K_M-00001-of-00003.gguf") == "a/b-Q4_K_M" and gguf_stem("c.GGUF") == "c" and gguf_stem("d") == "d"


def test_groups_keep_split_sets_together_and_sort_smallest_first() -> None:
    files = [("README.md", 10), ("config.json", 5),
             ("m-Q8_0.gguf", 800), ("m-Q4_K_M.gguf", 400), ("m-BF16-00001-of-00002.gguf", 700), ("m-BF16-00002-of-00002.gguf", 650),
             ("m-UD-Q4_K_XL.gguf", 450), ("mmproj-F16.gguf", 50)]
    groups = group_gguf(files)
    assert [(g.quant, g.total_bytes) for g in groups] == [
        ("mmproj-F16", 50), ("Q4_K_M", 400), ("UD-Q4_K_XL", 450), ("Q8_0", 800), ("BF16", 1350)]
    bf16 = next(g for g in groups if g.quant == "BF16")
    assert bf16.files == ["m-BF16-00001-of-00002.gguf", "m-BF16-00002-of-00002.gguf"]
    assert not any("README" in f or f.endswith(".json") for g in groups for f in g.files)         # only .gguf files
    assert group_gguf([("a.safetensors", 1)]) == []


def test_the_files_of_a_group_are_usable_as_allow_patterns() -> None:
    from engine_console.services.downloads import _match
    groups = group_gguf([("dir/m-Q4_K_M-00001-of-00002.gguf", 1), ("dir/m-Q4_K_M-00002-of-00002.gguf", 1), ("dir/m-Q8_0.gguf", 1)])
    g = next(x for x in groups if x.quant == "Q4_K_M")
    kept = [p for p in ("dir/m-Q4_K_M-00001-of-00002.gguf", "dir/m-Q4_K_M-00002-of-00002.gguf", "dir/m-Q8_0.gguf", "README.md")
            if _match(p, g.files)]
    assert kept == g.files and g.quant == "Q4_K_M"
    assert _match("a[1].gguf", ["a[1].gguf"])         # an exact path wins over fnmatch's reading of [..]


def test_hf_service_gguf_groups_accepts_repo_files_and_dicts() -> None:
    svc = HfService(FakeHub())
    as_files = svc.gguf_groups([RepoFile("m-Q4_K_M.gguf", 4), RepoFile("x.txt", 1)])
    as_dicts = svc.gguf_groups([{"path": "m-Q4_K_M.gguf", "size": 4, "sha256": None}, {"path": "bad", "size": "x"}, {"size": 1}])
    assert as_files == as_dicts and as_files[0].quant == "Q4_K_M" and as_files[0].total_bytes == 4


async def test_model_detail_carries_gguf_groups() -> None:
    hub = FakeHub()
    hub.add("org/m-GGUF", {"m-Q4_K_M.gguf": b"1234", "m-Q8_0.gguf": b"12345678", "README.md": b"x"})
    hub.add("org/plain", {"model.safetensors": b"1234", "config.json": b"{}"}, config={"model_type": "x"})
    svc = HfService(hub)
    detail = await svc.detail("org/m-GGUF")
    assert [(g.quant, g.files, g.total_bytes) for g in detail.gguf_groups] == [
        ("Q4_K_M", ["m-Q4_K_M.gguf"], 4), ("Q8_0", ["m-Q8_0.gguf"], 8)]
    dumped = detail.model_dump()
    assert dumped["gguf_groups"][0] == {"quant": "Q4_K_M", "files": ["m-Q4_K_M.gguf"], "total_bytes": 4}
    assert (await svc.detail("org/plain")).gguf_groups == []


# ---- a connection that dies in the middle of a file ------------------------------------------------------------------------------
MIB = 1024 * 1024


class _Dies(httpx.AsyncByteStream):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"a" * MIB          # httpx hands out whole 1 MiB chunks: what is buffered below that is lost with the connection
        raise self.error

    async def aclose(self) -> None:
        return None


@pytest.mark.parametrize("error", [httpx.ReadError("boom"), httpx.RemoteProtocolError("closed"), httpx.ReadTimeout("slow")])
async def test_a_connection_that_drops_mid_file_is_an_upstream_error_to_resume(error: Exception) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-length": "100"}, stream=_Dies(error))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: None, client=client)
    stream = await hub.open_file("org/m", COMMIT_SHA, "w.safetensors", 0)
    got: list[bytes] = []
    with pytest.raises(UpstreamError) as ei:
        async for chunk in stream.chunks():
            got.append(chunk)
    assert got == [b"a" * MIB] and ei.value.code == "upstream_error" and "resume" in ei.value.detail
    assert type(error).__name__ not in ei.value.detail.split("(")[0]          # no library exception name as the headline
    await stream.aclose()
    await hub.aclose()


COMMIT_SHA = "c0ffee00000000000000000000000000000000ff"


async def test_the_range_header_and_token_are_sent_to_hf_only() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(206, headers={"content-length": "3"}, content=b"xyz")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hub = HfHttpClient("https://huggingface.co", ["huggingface.co"], lambda: "hf_secrettoken123", client=client)
    s = await hub.open_file("org/m", COMMIT_SHA, "dir/w.safetensors", 7)
    assert s.offset == 7 and s.total == 10
    assert seen[0].headers["range"] == "bytes=7-" and seen[0].url.host == "huggingface.co"
    assert "hf_secrettoken123" not in str(seen[0].url)
    await s.aclose()
    await hub.aclose()


def _unused(_: Any) -> None:     # keeps the Any import honest for type checkers reading this file
    return None
