"""Write backend/crates/ec-hub/tests/data/validation_vectors.json: pydantic v2's lax answers (python mode, as FastAPI
validates query strings and decoded JSON bodies) for edge inputs of the scalar types the API uses. The Rust test
tests/test_validation_vectors.rs replays every vector against api/validate.rs.

    "$(node backend/parity/lib/python-release.mjs)/backend/.venv/bin/python" -I backend/parity/gen/validation_vectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, TypeAdapter, ValidationError

OUT = Path(__file__).resolve().parents[2] / "crates" / "ec-hub" / "tests" / "data" / "validation_vectors.json"

INPUTS: list[Any] = [
    "3", "3.0", "1.5", " 7 ", "1_000", "+5", "-0", "0x10", "1e3", "", "3.", ".5", "inf", "nan", "-inf", "1.0e2",
    "00012", "3.00", "-3.0", "+3.0", "3.0 ", " 3.0", "1__0", "_1", "1_", "-1_000.0", "true", "True", "TRUE", " true ",
    "yes", "on", "t", "Y", "off", "no", "0", "1", "0.0", "1.0", "2", "abc", "Grüße",
    True, False, 0, 1, 2, -1, 2.0, 2.5, 1.0, 0.0, -1.0, 1e20, 9007199254740993, None, [1], {"a": 1}, [],
]
TYPES: dict[str, Any] = {
    "int": int,
    "float": float,
    "bool": bool,
    "str": str,
    "int_ge1_le4096": Annotated[int, Field(ge=1, le=4096)],
    "int_ge0": Annotated[int, Field(ge=0)],
    "str_min1_max5": Annotated[str, Field(min_length=1, max_length=5)],
    "host_id": Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,31}$")],
    "literal_role": Literal["admin", "viewer"],
    "list_int": list[int],
    "dict_str": dict[str, str],
    "opt_int": int | None,
}


def main() -> int:
    out = []
    for tname, t in TYPES.items():
        ta = TypeAdapter(t)
        for v in INPUTS:
            try:
                r = ta.validate_python(v)
                rec = {"type": tname, "input": v, "ok": True, "value": r}
            except ValidationError as e:
                rec = {"type": tname, "input": v, "ok": False,
                       "errors": [{"loc": list(x["loc"]), "msg": x["msg"], "type": x["type"]} for x in e.errors()]}
            try:
                json.dumps(rec, allow_nan=False)
            except ValueError:
                rec = {"type": tname, "input": v, "ok": True, "value": "<non-finite>"}
            out.append(rec)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(out)} vectors)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
