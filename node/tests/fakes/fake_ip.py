"""A fake `ip -j addr`: interfaces from $FAKE_IP_STATE (JSON) or a small default."""
from __future__ import annotations

import json
import os
import sys

DEFAULT = [
    {"ifname": "lo", "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8}]},
    {"ifname": "eth0", "addr_info": [{"family": "inet", "local": "10.0.0.5", "prefixlen": 24},
                                      {"family": "inet6", "local": "fe80::1", "prefixlen": 64},
                                      {"family": "inet6", "local": "fd00::5", "prefixlen": 64}]},
    {"ifname": "docker0", "addr_info": [{"family": "inet", "local": "172.17.0.1", "prefixlen": 16}]},
]


def main():
    if sys.argv[1:] != ["-j", "addr"]:
        return 2
    path = os.environ.get("FAKE_IP_STATE")
    data = DEFAULT
    if path and os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    print(json.dumps(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
