"""`engine-console hosts ...`: set up and check the machines the console manages (contract §10.3).

  hosts init-key --from ADDRS   create the console's own ssh key and print the one line each machine must authorize
  hosts check                   connect to every host once and report what each node says
  hosts list                    the configured hosts, without connecting

`main.py` hands everything after `hosts` to `hosts_cli`, which returns the process exit code.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from engine_console.config import Settings
from engine_console.domain.fleet import HostSummary
from engine_console.services.common import EventBus
from engine_console.services.fleet import FleetService
from engine_console.services.hosts import HostsError, init_key, load_hosts

CHECK_TIMEOUT_S = 20.0
SETTLED = ("up", "stale", "down", "incompatible", "disabled")   # a link in any of these has finished its first attempt


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="engine-console hosts", description="Set up and check the machines the console manages.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("init-key", help="create the console's ssh key (if missing) and print the authorized_keys line")
    k.add_argument("--from", dest="from_", required=True, metavar="ADDRS",
                   help="the console machine's address as the others see it (comma-separated, e.g. 10.0.0.5): "
                        "each machine then accepts the key from there only")
    k.add_argument("--key-file", type=Path, default=None, help="where the key lives (default: the HUB_KEY_FILE setting)")
    c = sub.add_parser("check", help="connect to every host once and print its state, versions and policy mode")
    c.add_argument("--timeout", type=float, default=CHECK_TIMEOUT_S, help="seconds to wait for every host (default 20)")
    sub.add_parser("list", help="list the configured hosts without connecting")
    return ap


def _how(host_ssh: str | None, port: int | None) -> str:
    if host_ssh is None:
        return "this machine"
    return f"ssh {host_ssh}" + (f" -p {port}" if port else "")


def _cmd_init_key(cfg: Settings, from_: str, key_file: Path | None, out: TextIO) -> int:
    key = key_file or cfg.hub_key_file
    existed = key.exists()
    line = init_key(key, from_, cfg.remote_node_command)
    print(f"using the existing key {key}" if existed else f"created {key} (private half 0600, never printed)", file=out)
    print("\nAdd this ONE line to authorized_keys of the login user on every machine the console should manage\n"
          "(the user that machine's node was installed for):\n", file=out)
    print(line, file=out)
    if from_.strip() in ("*", "*.*.*.*"):
        print("\nWARNING: with --from '*' a copy of this key works from anywhere (still only to start a node). "
              "Prefer this machine's address.", file=out)
    print(f"\nThen list the machines in {cfg.hosts_file} and run: engine-console hosts check", file=out)
    return 0


def _cmd_list(cfg: Settings, out: TextIO) -> int:
    hosts = load_hosts(cfg.hosts_file)
    print(f"{cfg.hosts_file}{'' if cfg.hosts_file.exists() else ' (not present: only this machine is managed)'}", file=out)
    rows = [("ID", "LABEL", "REACHED BY", "ENABLED")]
    rows += [(h.id, h.label, _how(h.ssh, h.port) + (" (ssh_config)" if h.ssh_config else ""), "yes" if h.enabled else "no")
             for h in hosts]
    _table(rows, out)
    return 0


def _table(rows: Sequence[tuple[str, ...]], out: TextIO) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths, strict=True)).rstrip(), file=out)


async def _check(cfg: Settings, timeout_s: float, out: TextIO) -> int:
    fleet = FleetService(cfg, EventBus())
    await fleet.start()
    snaps: dict[str, HostSummary] = {}
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline and len(snaps) < len(fleet.hosts()):
            for s in fleet.summaries():
                # remember a host the moment its first attempt has an outcome: a failed link retries and flips back to
                # `connecting`, and the report must show why the first attempt failed
                if s.id not in snaps and s.link.state in SETTLED:
                    snaps[s.id] = s
            await asyncio.sleep(0.1)
        for s in fleet.summaries():
            snaps.setdefault(s.id, s)
    finally:
        await fleet.stop()
    rows = [("HOST", "STATE", "NODE", "ARCH", "GPUS", "POLICY")]
    detail: list[str] = []
    bad = 0
    for h in fleet.hosts():
        s = snaps[h.id]
        up = s.link.state in ("up", "stale")
        bad += 0 if up or s.link.state == "disabled" else 1
        rows.append((h.id, s.link.state, (s.node.version if s.node else None) or "-", (s.node.arch if s.node else None) or "-",
                     str(len(s.gpus)) if up else "-", (s.policy.mode if s.policy else None) or "-"))
        if s.link.error:
            detail.append(f"  {h.id}: {s.link.error}")
        elif not up and s.link.state == "connecting":
            detail.append(f"  {h.id}: no answer yet")
        detail += [f"  {h.id}: note: {n}" for n in s.link.notes]
        if s.policy and s.policy.error:
            detail.append(f"  {h.id}: node policy: {s.policy.error}")
    _table(rows, out)
    for line in detail:
        print(line, file=out)
    return 1 if bad else 0


def hosts_cli(argv: list[str], *, cfg: Settings | None = None, out: TextIO | None = None) -> int:
    """Run one `engine-console hosts` command; the return value is the exit code. `cfg` and `out` exist so tests can
    point it at a temporary directory and capture the text; the entry point passes neither."""
    out = out or sys.stdout
    try:
        args = _parser().parse_args(argv)
    except SystemExit as e:          # argparse exits on a usage error (2) and after --help (0)
        return e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    cfg = cfg or Settings()
    try:
        if args.cmd == "init-key":
            return _cmd_init_key(cfg, args.from_, args.key_file, out)
        if args.cmd == "list":
            return _cmd_list(cfg, out)
        return asyncio.run(_check(cfg, args.timeout, out))
    except HostsError as e:
        print(f"engine-console: {e}", file=sys.stderr)
        return 1
