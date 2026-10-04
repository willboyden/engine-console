"""Reads the captured `llama-server --help` (fixtures/llama-server-help.txt) into entries the catalog tests compare with."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"
_FLAG = re.compile(r"(?<![\w-])(--?[A-Za-z][\w-]*)")
_ENV = re.compile(r"\(env: ([A-Z0-9_]+)\)")
_DEFAULT = re.compile(r"default:\s*'?([^,()';=]+)'?")


@dataclass
class HelpEntry:
    flags: list[str]
    text: str                  # description lines joined with single spaces
    env: str | None
    default: str | None        # the text after the first "default:" of the entry (a literal, or a phrase such as "same as ...")

    @property
    def deprecated(self) -> bool:
        return "DEPRECATED" in self.text


def read(name: str) -> str:
    return (FIX / name).read_text(encoding="utf-8")


def parse_help(text: str) -> list[HelpEntry]:
    """An entry starts at a line that begins with '-'; its description is the rest of that line and the indented lines."""
    entries: list[HelpEntry] = []
    current: list[str] = []

    def flush() -> None:
        if not current:
            return
        head = current[0]
        # the help prints flags in the first 40 columns and the description from column 40; a longer flag list leaves
        # the description to the following lines
        same_line = len(head) > 40 and head[38:40] == "  " and head[40] != " "
        flags_part, first_desc = (head[:40], head[40:]) if same_line else (head, "")
        flags = _FLAG.findall(flags_part)
        desc = " ".join([first_desc.strip(), *[ln.strip() for ln in current[1:]]]).strip()
        env = _ENV.search(desc)
        default = _DEFAULT.search(desc)
        entries.append(HelpEntry(flags, desc, env.group(1) if env else None,
                                 default.group(1).strip() if default else None))

    for line in text.splitlines():
        if line.startswith("-"):
            flush()
            current = [line]
        elif current and line.startswith(" "):
            current.append(line)
        else:
            flush()
            current = []
    flush()
    return entries


def help_flags(text: str | None = None) -> set[str]:
    return {f for e in parse_help(text if text is not None else read("llama-server-help.txt")) for f in e.flags}


def entry_for(flag: str, text: str | None = None) -> HelpEntry:
    for e in parse_help(text if text is not None else read("llama-server-help.txt")):
        if flag in e.flags:
            return e
    raise KeyError(flag)
