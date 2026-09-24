"""Named parameter bundles per engine/model, with YAML import/export and diffing."""
from __future__ import annotations

import json
import sqlite3
from typing import Any

import yaml

from engine_console.adapters import AdapterRegistry
from engine_console.domain.errors import BadRequest, Conflict, NotFound, Unprocessable
from engine_console.domain.models import Page, Profile, ProfileDiff, ProfileIn
from engine_console.services.common import new_id, now
from engine_console.services.secrets_store import MARKER, split_secrets
from engine_console.services.store import Store, paginate


class ProfileService:
    def __init__(self, store: Store, adapters: AdapterRegistry) -> None:
        self._db = store
        self._adapters = adapters

    @staticmethod
    def _to_model(r: Any) -> Profile:
        return Profile(id=r["id"], name=r["name"], engine=r["engine"], repo_id=r["repo_id"], params=json.loads(r["params"]),
                       description=r["description"], created_at=r["created_at"], updated_at=r["updated_at"])

    def _resolve(self, p: ProfileIn) -> dict[str, Any]:
        adapter = self._adapters.get(p.engine)
        params: dict[str, Any] = {}
        if p.preset:
            presets = adapter.presets()
            if p.preset not in presets:
                raise Unprocessable(f"unknown preset '{p.preset}' for {p.engine}", code="unknown_preset",
                                    available=sorted(presets))
            params.update(presets[p.preset])
        params.update(p.params)
        errs = adapter.validate({k: v for k, v in params.items() if v != MARKER})
        if errs:
            raise Unprocessable("invalid engine parameters", errors=errs)
        image_keys = {s.key for s in adapter.param_catalog() if s.flag == "@image"}
        if image_keys & params.keys():
            raise Unprocessable("per-instance image overrides are not allowed; use settings pins", code="image_not_allowed")
        public, _ = split_secrets(params)   # profiles are stored/exported: secret values are dropped, marker kept
        return public

    def get(self, pid: str) -> Profile:
        r = self._db.one("SELECT * FROM profiles WHERE id=?", (pid,))
        if r is None:
            raise NotFound(f"no such profile: {pid}")
        return self._to_model(r)

    def list_page(self, limit: int = 100, cursor: str | None = None, engine: str | None = None) -> Page[Profile]:
        limit, off = paginate(limit, cursor)
        where, args = ("WHERE engine=?", [engine]) if engine else ("", [])
        rows = self._db.all(f"SELECT * FROM profiles {where} ORDER BY name LIMIT ? OFFSET ?",  # noqa: S608
                            [*args, limit + 1, off])
        return Page[Profile](items=[self._to_model(r) for r in rows[:limit]],
                             next_cursor=str(off + limit) if len(rows) > limit else None)

    def create(self, p: ProfileIn) -> Profile:
        params = self._resolve(p)
        pid, ts = new_id("prof_"), now()
        try:
            self._db.execute("INSERT INTO profiles VALUES(?,?,?,?,?,?,?,?)",
                             (pid, p.name, p.engine, p.repo_id, json.dumps(params), p.description, ts, ts))
        except sqlite3.IntegrityError:
            raise Conflict(f"a profile named '{p.name}' already exists", code="profile_exists") from None
        return self.get(pid)

    def update(self, pid: str, p: ProfileIn) -> Profile:
        self.get(pid)
        params = self._resolve(p)
        try:
            self._db.execute("UPDATE profiles SET name=?, engine=?, repo_id=?, params=?, description=?, updated_at=? WHERE id=?",
                             (p.name, p.engine, p.repo_id, json.dumps(params), p.description, now(), pid))
        except sqlite3.IntegrityError:
            raise Conflict(f"a profile named '{p.name}' already exists", code="profile_exists") from None
        return self.get(pid)

    def delete(self, pid: str) -> None:
        self.get(pid)
        self._db.execute("DELETE FROM profiles WHERE id=?", (pid,))

    def diff(self, a: str, b: str) -> ProfileDiff:
        pa, pb = self.get(a).params, self.get(b).params
        return ProfileDiff(a=a, b=b,
                           changed={k: {"a": pa[k], "b": pb[k]} for k in pa.keys() & pb.keys() if pa[k] != pb[k]},
                           only_a={k: pa[k] for k in pa.keys() - pb.keys()},
                           only_b={k: pb[k] for k in pb.keys() - pa.keys()})

    def export_yaml(self, pid: str) -> str:
        p = self.get(pid)
        return yaml.safe_dump({"name": p.name, "engine": p.engine, "repo_id": p.repo_id,
                               "description": p.description, "params": p.params}, sort_keys=False)

    def import_yaml(self, text: str) -> Profile:
        try:
            doc = yaml.safe_load(text)   # safe_load: never construct arbitrary objects from an upload
        except yaml.YAMLError as e:
            raise BadRequest(f"invalid YAML: {type(e).__name__}", code="invalid_yaml") from None
        if not isinstance(doc, dict):
            raise BadRequest("profile YAML must be a mapping", code="invalid_yaml")
        try:
            p = ProfileIn(**{k: doc[k] for k in ("name", "engine", "repo_id", "description", "params") if k in doc})
        except ValueError as e:
            raise Unprocessable(f"invalid profile document: {type(e).__name__}", code="invalid_profile") from None
        return self.create(p)
