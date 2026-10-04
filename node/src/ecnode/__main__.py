"""ecnode command line.

  ecnode --stdio [--policy PATH]     speak the hub protocol on stdin/stdout (what the hub starts)
  ecnode check [--policy PATH]       print what this machine would report (read-only)
  ecnode policy-template             print a starter monitor-mode policy (writes nothing)
  ecnode version
"""
from __future__ import annotations

import argparse
import os
import sys

MIN_PYTHON = (3, 10)


def _stdio(policy_path: str | None) -> int:
    from . import proto
    from .agent import Agent
    from .policy import PolicyStore
    from .server import Server

    # The protocol owns the original stdin/stdout. Everything else (a stray print, a child process that forgets
    # to redirect) gets /dev/null as stdin and stderr as stdout, so it can never corrupt the frame stream.
    proto_in = os.fdopen(os.dup(0), "rb")
    proto_out = os.fdopen(os.dup(1), "wb")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    try:
        agent = Agent(PolicyStore(policy_path))
    except Exception as e:      # say why in-protocol, then go
        proto_out.write(proto.MAGIC + proto.encode_json({"type": "fatal", "error": f"the node could not start: "
                                                                                   f"{type(e).__name__}"}))
        proto_out.flush()
        return 2
    return Server(agent, proto_in, proto_out).run()


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < MIN_PYTHON:
        sys.stderr.write(f"ecnode needs Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer\n")
        return 2
    from . import __version__

    ap = argparse.ArgumentParser(prog="ecnode", description="Engine Console node agent (one per machine).")
    ap.add_argument("--stdio", action="store_true", help="speak the hub protocol on stdin/stdout")
    ap.add_argument("--policy", metavar="PATH", help="policy file (default: /etc/ecnode/policy.json, then "
                                                     "~/.config/ecnode/policy.json)")
    sub = ap.add_subparsers(dest="cmd")
    chk = sub.add_parser("check", help="print what this machine would report to a hub (read-only)")
    chk.add_argument("--policy", metavar="PATH", dest="check_policy")
    sub.add_parser("policy-template", help="print a starter monitor-mode policy (writes nothing)")
    sub.add_parser("version", help="print the version")
    args = ap.parse_args(argv)
    if args.stdio:
        if args.cmd:
            ap.error("--stdio takes no command")
        return _stdio(args.policy)
    if args.cmd == "version":
        print(f"ecnode {__version__}")
        return 0
    if args.cmd == "check":
        from .cli import check
        return check(getattr(args, "check_policy", None) or args.policy, sys.stdout)
    if args.cmd == "policy-template":
        from .cli import policy_template
        return policy_template(sys.stdout)
    ap.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
