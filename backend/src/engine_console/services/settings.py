"""Global settings, HF token handling, and API keys / roles (ADR 7 and 9)."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine_console.config import Settings
from engine_console.domain.errors import BadRequest, NotFound
from engine_console.domain.models import KeyInfo, SettingsPatch
from engine_console.services.common import new_id, now
from engine_console.services.hf_http import public_proxy
from engine_console.services.store import Store

BOOTSTRAP_KEY_FILE = "bootstrap-admin.key"


def image_allowed(image: str, allowlist: list[str] | set[str]) -> bool:
    """Exact match against full image references. Prefix matching would let `vllm/vllm-openai:evil` through."""
    return image in allowlist


@dataclass(frozen=True)
class Principal:
    name: str
    role: str  # "admin" | "viewer"
    key_id: str | None = None


def _hash(secret: str) -> str:
    # keys are 256-bit random, so a fast unsalted hash is fine (no dictionary to attack)
    return hashlib.sha256(secret.encode()).hexdigest()


class SettingsService:
    def __init__(self, store: Store, cfg: Settings, adapter_images: Callable[[], set[str]] = lambda: set()) -> None:
        self._db = store
        self._cfg = cfg
        self._adapter_images = adapter_images

    def engine_image_allowed(self, image: str) -> bool:
        return image_allowed(image, set(self._cfg.engine_image_allowlist) | self._adapter_images())

    # -- HF token ------------------------------------------------------------------------------
    def hf_token(self) -> str | None:
        if self._cfg.hf_token:
            return self._cfg.hf_token.strip() or None
        f = self._cfg.hf_token_file
        try:
            st = f.stat()
        except OSError:
            return None
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            return None   # refuse a group/world-accessible token file; status reports why
        try:
            return f.read_text().strip() or None
        except OSError:
            return None

    def token_status(self) -> str:
        if self._cfg.hf_token:
            return f"[set, {len(self._cfg.hf_token.strip())} chars] (env)"
        f = self._cfg.hf_token_file
        try:
            st = f.stat()
        except OSError:
            return "[unset]"
        if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            return f"[unusable: {f} must be chmod 600]"
        tok = self.hf_token()
        return f"[set, {len(tok)} chars] (file)" if tok else "[unset]"

    # -- persisted overrides -----------------------------------------------------------------------
    def _get(self, key: str, default: Any) -> Any:
        row = self._db.one("SELECT value FROM settings WHERE key=?", (key,))
        return json.loads(row["value"]) if row else default

    def hf_cache_dir(self) -> Path:
        return self._cfg.hf_cache_dir   # config/env only: never changeable through the API

    def hub_dir(self) -> Path:
        return self.hf_cache_dir() / "hub"

    def idle_ttl_s(self) -> int:
        return int(self._get("idle_ttl_s", 0))

    def default_gpu_ids(self) -> list[int]:
        return [int(x) for x in self._get("default_gpu_ids", [])]

    def image_pin(self, engine: str) -> str | None:
        pins = self._get("image_pins", {})
        v = pins.get(engine) if isinstance(pins, dict) else None
        return v if isinstance(v, str) else None

    def effective(self) -> dict[str, Any]:
        return {
            "hf_cache_dir": str(self.hf_cache_dir()),
            "default_gpu_ids": self.default_gpu_ids(),
            "idle_ttl_s": self.idle_ttl_s(),
            "image_pins": self._get("image_pins", {}),
            "hf_token": self.token_status(),
            "docker_context": self._cfg.docker_context,
            "engine_network": self._cfg.engine_network,
            "egress_mode": "proxied" if self._cfg.egress_proxy else "direct",
            "egress_proxy": public_proxy(self._cfg.egress_proxy),
            "require_egress_proxy": self._cfg.require_egress_proxy,
            "port_range": [self._cfg.port_range_start, self._cfg.port_range_end],
            "hf_allowed_hosts": self._cfg.hf_allowed_hosts,
        }

    def update(self, patch: SettingsPatch) -> dict[str, Any]:
        data = patch.model_dump(exclude_none=True)
        if "image_pins" in data:
            for eng, img in data["image_pins"].items():
                if ":" not in img or img.endswith(":latest"):
                    raise BadRequest(f"image pin for {eng} must be a pinned tag (not :latest)", code="invalid_setting")
                if not self.engine_image_allowed(img):
                    raise BadRequest(f"image {img} is not on ENGINE_IMAGE_ALLOWLIST", code="image_not_allowed")
        for k, v in data.items():
            self._db.execute("INSERT INTO settings(key,value) VALUES(?,?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, json.dumps(v)))
        return self.effective()

    # -- API keys ------------------------------------------------------------------------------
    def create_key(self, name: str, role: str) -> KeyInfo:
        secret = "ec_" + secrets.token_urlsafe(32)
        kid, ts = new_id("key_"), now()
        self._db.execute("INSERT INTO api_keys(id,name,role,prefix,hash,created_at) VALUES(?,?,?,?,?,?)",
                         (kid, name, role, secret[:8], _hash(secret), ts))
        return KeyInfo(id=kid, name=name, role=role, prefix=secret[:8], created_at=ts, last_used_at=None, secret=secret)

    def list_keys(self) -> list[KeyInfo]:
        rows = self._db.all("SELECT * FROM api_keys WHERE revoked_at IS NULL ORDER BY created_at")
        return [KeyInfo(id=r["id"], name=r["name"], role=r["role"], prefix=r["prefix"],
                        created_at=r["created_at"], last_used_at=r["last_used_at"]) for r in rows]

    def revoke_key(self, key_id: str) -> None:
        cur = self._db.execute("UPDATE api_keys SET revoked_at=? WHERE id=? AND revoked_at IS NULL", (now(), key_id))
        if cur.rowcount == 0:
            raise NotFound(f"no such key: {key_id}")

    def authenticate(self, secret: str) -> Principal | None:
        h = _hash(secret)
        row = self._db.one("SELECT id, name, role, hash FROM api_keys WHERE hash=? AND revoked_at IS NULL", (h,))
        if row is None or not hmac.compare_digest(row["hash"], h):
            return None
        self._db.execute("UPDATE api_keys SET last_used_at=? WHERE hash=?", (now(), h))
        return Principal(name=row["name"], role=row["role"], key_id=row["id"])

    def ensure_bootstrap_key(self, data_dir: Path) -> Path | None:
        """First start: mint an admin key and write it 0600. Returns the file path if one was created."""
        if self._db.one("SELECT 1 FROM api_keys LIMIT 1"):
            return None
        info = self.create_key("bootstrap-admin", "admin")
        path = data_dir / BOOTSTRAP_KEY_FILE
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        os.fchmod(fd, 0o600)   # O_CREAT's mode is ignored for a pre-existing, possibly looser file
        with os.fdopen(fd, "w") as fh:
            fh.write((info.secret or "") + "\n")
        return path
