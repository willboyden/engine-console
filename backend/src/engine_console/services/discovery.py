"""Pure helpers for engines the console did not create (contract v2).

Finding them is the node's job now: it lists running containers, classifies them by image/command signature,
redacts their argv, probes their loopback endpoints and pushes all of that in its `inventory` message. What stays
on the hub is text hygiene for names that end up in instance ids, and the generic Prometheus map for engines that
have no adapter. Everything a node says is untrusted: length-capped, control characters stripped, never executed."""
from __future__ import annotations

import re

EXT = "ext-"        # external instance ids: ext-<host>-<slug>
SCRIPT = "scr-"     # script instance ids: scr-<host>-<script name>

# generic Prometheus -> canonical keys, for engines without an adapter (missing keys are simply omitted)
GENERIC_METRICS: dict[str, dict[str, str]] = {
    "llamacpp": {"llamacpp:requests_processing": "requests_running", "llamacpp:requests_deferred": "requests_waiting",
                 "llamacpp:prompt_tokens_total": "prompt_tokens_total",
                 "llamacpp:tokens_predicted_total": "generation_tokens_total"},
}
_CTRL = re.compile(r"[\x00-\x1f\x7f]")


def clean_text(s: object, limit: int = 200) -> str:
    return _CTRL.sub("", str(s))[:limit]


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48] or "unknown"


def display_name(s: str) -> str:
    out = re.sub(r"[^A-Za-z0-9 _.:\-]", "", _CTRL.sub("", s)).strip(" .-")[:64]
    return out or "external"


def parse_generic_metrics(engine: str, text: str) -> dict[str, float]:
    """Map whatever canonical keys an adapter-less engine exposes; everything else is ignored."""
    mapping = GENERIC_METRICS.get(engine)
    if not mapping:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(" ")
        name = name.split("{", 1)[0]
        if name in mapping:
            try:
                out[mapping[name]] = float(rest.split()[0])
            except (ValueError, IndexError):
                continue
    return out
