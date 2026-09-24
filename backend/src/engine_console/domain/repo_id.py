"""Validated identifier types shared by every route (defence against path/flag/mount injection)."""
from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator

from engine_console.domain.errors import BadRequest

_REPO = re.compile(r"^[A-Za-z0-9][\w.\-]{0,95}/[A-Za-z0-9][\w.\-]{0,95}$")
_REV = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def check_repo_id(v: str) -> str:
    # "--" is how huggingface_hub flattens "/" in cache folder names, so "a--b/c" could alias "a/b--c"
    if not _REPO.fullmatch(v) or "--" in v or ".." in v:
        raise ValueError("repo_id must look like 'org/name' (letters, digits, . _ -)")
    return v


def check_revision(v: str) -> str:
    if not _REV.fullmatch(v) or ".." in v:
        raise ValueError("revision may only contain letters, digits, . _ -")
    return v


def require_repo_id(v: str) -> str:
    """For path parameters and internal callers: same rule, but raises the API's problem error."""
    try:
        return check_repo_id(v)
    except ValueError as e:
        raise BadRequest(str(e), code="invalid_repo_id") from None


RepoId = Annotated[str, AfterValidator(check_repo_id)]
Revision = Annotated[str, AfterValidator(check_revision)]
