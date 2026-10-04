"""Read-only views over what a node last pushed (hello, sample, disks, inventory).

Everything a node says is untrusted data that has already been through `ecnode.proto.sanitize_*` (unknown fields
dropped, strings capped, wrong types turned into None), so a value can still be missing or None: every accessor
here tolerates that and returns a plain default instead of raising. Nothing here talks to a node."""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from engine_console.domain.fleet import NodeClient

GIB = float(1024**3)
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


def hello(node: NodeClient) -> dict[str, Any]:
    return node.latest("hello") or {}


def policy(node: NodeClient) -> dict[str, Any]:
    p = hello(node).get("policy")
    return p if isinstance(p, dict) else {}


def mode(node: NodeClient) -> str:
    """The node's policy mode. Anything unexpected is monitor mode: nothing may change on that machine."""
    return "manage" if policy(node).get("mode") == "manage" else "monitor"


def arch(node: NodeClient) -> str:
    host = hello(node).get("host")
    a = host.get("arch") if isinstance(host, dict) else None
    return a if isinstance(a, str) and a else "x86_64"


def link_up(node: NodeClient) -> bool:
    return node.link().state == "up"


def port_range(node: NodeClient, default: tuple[int, int]) -> tuple[int, int]:
    pr = policy(node).get("port_range")
    if isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, int) for x in pr) and 0 < pr[0] <= pr[1] <= 65535:
        return int(pr[0]), int(pr[1])
    return default


def engine_images(node: NodeClient) -> list[str]:
    return [p for p in policy(node).get("engine_images") or [] if isinstance(p, str)]


def external_control(node: NodeClient) -> str:
    v = policy(node).get("external_control")
    return v if isinstance(v, str) else "none"


def static_gpus(node: NodeClient) -> list[dict[str, Any]]:
    return [g for g in hello(node).get("gpus") or [] if isinstance(g, dict) and g.get("uuid")]


def uuid_index(node: NodeClient) -> dict[str, int]:
    return {str(g["uuid"]): int(g["index"]) for g in static_gpus(node) if isinstance(g.get("index"), int)}


def sample(node: NodeClient) -> dict[str, Any]:
    return node.latest("sample") or {}


def inventory(node: NodeClient) -> dict[str, Any]:
    return node.latest("inventory") or {}


def containers(node: NodeClient) -> list[dict[str, Any]]:
    return [c for c in inventory(node).get("containers") or [] if isinstance(c, dict) and c.get("name")]


def container_named(node: NodeClient, name: str) -> dict[str, Any] | None:
    return next((c for c in containers(node) if c.get("name") == name), None)


def disk_mounts(node: NodeClient) -> list[dict[str, Any]]:
    d = node.latest("disks") or {}
    return [m for m in d.get("mounts") or [] if isinstance(m, dict) and m.get("mount")]


def endpoints(node: NodeClient) -> list[dict[str, Any]]:
    return [e for e in inventory(node).get("endpoints") or [] if isinstance(e, dict) and isinstance(e.get("port"), int)]


def host_ports(c: dict[str, Any]) -> list[int]:
    """Host ports a container publishes that are reachable as 127.0.0.1 on its machine."""
    out: list[int] = []
    for p in c.get("ports") or []:
        if not isinstance(p, dict):
            continue
        ip, hp = p.get("host_ip") or "", p.get("host_port")
        if ip in ("", "0.0.0.0", "127.0.0.1", "::", "::1") and isinstance(hp, int) and hp not in out:  # noqa: S104
            out.append(hp)
    return out


def ports_in_use(node: NodeClient) -> set[int]:
    """Every host port the inventory says something listens on (container publishes and probed endpoints)."""
    used = {p for c in containers(node) for p in host_ports(c)}
    used |= {int(e["port"]) for e in endpoints(node)}
    return used


def model_roots(node: NodeClient) -> dict[str, dict[str, Any]]:
    roots = policy(node).get("model_roots") or []
    return {str(r["id"]): r for r in roots if isinstance(r, dict) and r.get("id")}


def scripts(node: NodeClient) -> list[dict[str, Any]]:
    return [s for s in policy(node).get("scripts") or [] if isinstance(s, dict) and s.get("name")]


def addresses(node: NodeClient) -> list[str]:
    out: list[str] = []
    for a in hello(node).get("addresses") or []:
        if isinstance(a, dict) and isinstance(a.get("addr"), str):
            out.append(a["addr"])
    return out


def num(v: Any) -> float | None:
    return float(v) if isinstance(v, int | float) and not isinstance(v, bool) and v == v else None


def gib(v: Any) -> float | None:
    n = num(v)
    return n / GIB if n is not None else None


def ok_status(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and 200 <= v < 300


def auth_status(v: Any) -> bool:
    return v in (401, 403)


def image_allowed(image: str, patterns: Iterable[str]) -> bool:
    """The node policy's `engine_images` rule, mirrored so preflight can say no before a round trip: an entry is an
    exact reference, or `repository:*` / `repository@sha256:*` with a literal repository (never a prefix match, so
    `vllm/vllm-openai:evil/x` or a different repository cannot slip through). The node enforces it again."""
    for p in patterns:
        if p == image:
            return True
        if p.endswith("@sha256:*"):          # a digest pattern: the part after the colon must be a full sha256, nothing else
            repo = p[:-9]
            digest = image[len(repo) + 8:] if image.startswith(repo + "@sha256:") else None
            if repo and "*" not in repo and digest is not None and _DIGEST.fullmatch(digest):
                return True
        elif p.endswith(":*"):               # a tag pattern: any single tag of exactly this repository
            repo = p[:-2]
            tag = image[len(repo) + 1:] if image.startswith(repo + ":") else None
            if repo and "*" not in repo and "@" not in repo and tag is not None and _TAG.fullmatch(tag):
                return True
    return False
