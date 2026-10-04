"""A fake `docker` CLI for the node's tests: state in $FAKE_DOCKER_STATE (JSON), every call appended to
$FAKE_DOCKER_LOG as {"argv": [...], "env": {NAME: value for each bare `-e NAME`}}. Only what the node uses."""
from __future__ import annotations

import json
import os
import secrets
import sys
import time

PS_FORMAT = "{{.ID}}\t{{.Names}}\t{{.State}}\t{{.Status}}"


def load():
    with open(os.environ["FAKE_DOCKER_STATE"]) as f:
        return json.load(f)


def save(state):
    with open(os.environ["FAKE_DOCKER_STATE"], "w") as f:
        json.dump(state, f)


def log(argv):
    env = {}
    for i, a in enumerate(argv):
        if a == "-e" and i + 1 < len(argv) and "=" not in argv[i + 1]:
            env[argv[i + 1]] = os.environ.get(argv[i + 1])
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as f:
        f.write(json.dumps({"argv": argv, "env": env}) + "\n")


def find(state, ref):
    for c in state["containers"]:
        if c["Name"].lstrip("/") == ref or c["Id"] == ref or (len(ref) >= 6 and c["Id"].startswith(ref)):
            return c
    return None


def opt(args, flag):
    return args[args.index(flag) + 1] if flag in args else None


def container_doc(name, image, labels, running, args, extra=None):
    cid = secrets.token_hex(32)
    doc = {"Id": cid, "Name": "/" + name, "Image": "sha256:" + "ab" * 32, "Created": "2026-09-01T10:00:00.123456789Z",
           "Path": args[0] if args else "", "Args": args[1:],
           "State": {"Status": "running" if running else "created", "Running": running, "Pid": 4242 if running else 0,
                     "ExitCode": 0, "StartedAt": "2026-09-01T10:00:01Z"},
           "Config": {"Image": image, "Labels": labels, "Env": ["SHOULD_NEVER_BE_READ=value"]},
           "HostConfig": {"NetworkMode": "bridge"}, "NetworkSettings": {"Ports": {}}, "Mounts": []}
    doc.update(extra or {})
    return doc


def parse_run(args):
    """(name, labels, image, container argv) from `run`/`create` flags."""
    takes = {"--name", "--network", "--network-alias", "--restart", "--cap-drop", "--security-opt", "--pids-limit",
             "--gpus", "--shm-size", "--ipc", "--label", "-v", "-e", "-p", "--tmpfs", "--user", "--entrypoint",
             "--pull"}
    i, labels, name = 0, {}, None
    while i < len(args) and args[i].startswith("-"):
        a = args[i]
        if a in ("-d", "--read-only"):
            i += 1
            continue
        if a in takes:
            v = args[i + 1]
            if a == "--label":
                k, _, val = v.partition("=")
                labels[k] = val
            if a == "--name":
                name = v
            i += 2
            continue
        raise SystemExit(f"fake docker: unknown flag {a}")
    return name, labels, args[i], args[i + 1:]


def main():
    argv = sys.argv[1:]
    log(argv)
    state = load()
    if argv[:1] == ["--context"]:
        argv = argv[2:]
    if not argv:
        return 1
    verb, args = argv[0], argv[1:]
    if state.get("daemon_down") and verb not in ("context",):
        print("Cannot connect to the Docker daemon at unix:///run/user/1000/docker.sock. Is the docker daemon "
              "running?", file=sys.stderr)
        if verb == "info":
            print(json.dumps({"ServerErrors": ["Cannot connect to the Docker daemon. Is the docker daemon running?"]}))
        return 1
    if verb == "context" and args[:1] == ["show"]:
        print(state.get("context", "default"))
        return 0
    if verb == "context" and args[:1] == ["ls"]:
        for name in state.get("contexts", ["default"]):
            print(name)
        return 0
    if verb == "info":
        print(json.dumps(state["info"]))
        return 0
    if verb == "ps":
        assert opt(args, "--format") == PS_FORMAT, "the node's ps format changed"
        label = opt(args, "--filter")
        for c in state["containers"]:
            if "-a" not in args and not c["State"]["Running"]:
                continue
            if label:
                k, _, v = label.removeprefix("label=").partition("=")
                if c["Config"]["Labels"].get(k) != v:
                    continue
            status = "Up 3 minutes" if c["State"]["Running"] else "Exited (0) 1 minute ago"
            print(f"{c['Id']}\t{c['Name'].lstrip('/')}\t{c['State']['Status']}\t{status}")
        return 0
    if verb == "container" and args[:1] == ["inspect"] or verb == "inspect":
        refs = args[1:] if verb == "container" else args
        found, missing = [], []
        for r in refs:
            c = find(state, r)
            (found.append(c) if c else missing.append(r))
        print(json.dumps(found))
        for r in missing:
            print(f"Error: No such container: {r}", file=sys.stderr)
        return 1 if missing else 0
    if verb == "image" and args[:1] == ["inspect"]:
        fmt = opt(args, "--format")
        refs = [a for a in args[1:] if a not in ("--format", fmt)]
        out, missing = [], []
        for r in refs:
            img = next((i for i in state["images"]
                        if r in i.get("RepoTags", []) + i.get("RepoDigests", []) or i["Id"] == r), None)
            (out.append(img) if img else missing.append(r))
        if fmt == "{{.Id}}":
            for i in out:
                print(i["Id"])
        else:
            print(json.dumps(out))
        for r in missing:
            print(f"Error response from daemon: No such image: {r}", file=sys.stderr)
        return 1 if missing else 0
    if verb == "image" and args[:1] == ["ls"]:
        for i in state["images"]:
            print(i["Id"])
        return 0
    if verb == "images":
        for i in state["images"]:
            for d in i.get("RepoDigests", []):
                repo, _, digest = d.partition("@")
                print(f"{repo}@{digest}")
        return 0
    if verb == "network":
        sub, rest = args[0], args[1:]
        nets = state.setdefault("networks", {})
        if sub == "inspect":
            name = rest[-1]
            if name not in nets:
                print(f"Error response from daemon: network {name} not found", file=sys.stderr)
                return 1
            print("true" if nets[name]["internal"] else "false")
            return 0
        if sub == "create":
            nets[rest[-1]] = {"internal": "--internal" in rest}
            save(state)
            print(secrets.token_hex(32))
            return 0
        if sub == "connect":
            return 0 if find(state, rest[1]) else 1
    if verb in ("run", "create"):
        if state.get("fail", {}).get(verb):
            print(state["fail"][verb], file=sys.stderr)
            return 125
        name, labels, image, cargs = parse_run(args)
        if find(state, name):
            print(f'docker: Error response from daemon: Conflict. The container name "/{name}" is already in use '
                  'by container "abc".', file=sys.stderr)
            return 125
        if not any(image in i.get("RepoTags", []) + i.get("RepoDigests", []) for i in state["images"]):
            print(f"docker: Error response from daemon: No such image: {image}", file=sys.stderr)
            return 125
        doc = container_doc(name, image, labels, verb == "run", cargs)
        state["containers"].append(doc)
        save(state)
        print(doc["Id"])
        return 0
    if verb in ("start", "stop", "rm"):
        ref = args[-1]
        c = find(state, ref)
        if c is None:
            print(f"Error response from daemon: No such container: {ref}", file=sys.stderr)
            return 1
        if verb == "rm":
            state["containers"].remove(c)
        else:
            c["State"]["Running"] = verb == "start"
            c["State"]["Status"] = "running" if verb == "start" else "exited"
        save(state)
        print(ref)
        return 0
    if verb == "logs":
        c = find(state, args[-1])
        if c is None:
            print(f"Error response from daemon: No such container: {args[-1]}", file=sys.stderr)
            return 1
        for line in state.get("logs", {}).get(c["Name"].lstrip("/"), []):
            print(line, flush=True)
        if "-f" in args:
            while True:
                time.sleep(0.2)
        return 0
    if verb == "pull":
        for line in ("Pulling from library/x", "Digest: sha256:" + "cd" * 32, "Status: Downloaded newer image"):
            print(line, flush=True)
        state["images"].append({"Id": "sha256:" + "ef" * 32, "RepoTags": [args[-1]], "Size": 1, "Created": "",
                                "Architecture": "amd64"})
        save(state)
        return 0
    print(f"fake docker: unsupported {argv}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
