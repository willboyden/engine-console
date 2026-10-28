# Regenerate (from the repo root, with the Python reference of backend/parity/lib/python-release.mjs):
#   PYREL=$(node backend/parity/lib/python-release.mjs); "$PYREL/node/.venv/bin/python" -I backend/crates/ecnode/tests/gen/wsd_discovery_vectors.py "$PYREL/node/src" \
#       > backend/crates/ecnode/tests/data/wsd_discovery_vectors.json
# Golden vectors for discovery.py (Python reference, seeded, deterministic): classify, parse_params, container_view
# over hostile and ordinary inspect documents, PortProbe heuristics, build_endpoints with a scripted prober.
import json, random, sys
sys.path.insert(0, sys.argv[1])
from ecnode import discovery as d
from ecnode.policy import Root

rng = random.Random(20261027)
images = ["vllm/vllm-openai:v0.11.0", "VLLM/VLLM-OPENAI:X", "lmsysorg/sglang:latest", "ghcr.io/ggml-org/llama.cpp:server",
          "llama-cpp:1", "x/llama-server:2", "nvcr.io/nvidia/tensorrt-llm/release:1", "trtllm:1", "ollama/ollama:0.12",
          "ai-dynamo/x:1", "nvcr.io/dynamo:1", "vllm/vllm-router:1", "litellm/litellm", "custom:1", "", "myvllm:1",
          "a/vllm", "a/vllm-x", "gateway/vllm:1", "x" * 400 + "vllm"]
cmds = ["", "vllm serve m", "python -m vllm.entrypoints.openai.api_server", "python3 -m sglang.launch_server",
        "sglang serve", "ollama serve", "/app/llama-server -m x", "trtllm-serve serve m", "python -m tensorrt_llm",
        "python -m dynamo.frontend", "dynamo-run", "vllmserve", "x" * 8190 + " vllm serve m"]
classify = [[i, c, d.classify(i, c)] for i in images for c in cmds[:4]] + [["custom:1", c, d.classify("custom:1", c)] for c in cmds]
argvs = [["vllm", "serve", "m", "--tp=2", "--x", "--y", "1"], ["serve"], ["serve", "-x"], ["--"], ["---a", "b"],
         ["--A-B.c=1"], ["--a", "--b", "--c=", "--d"], ["--x", "serve"], ["serve", "m", "serve", "n"],
         ["--k" + str(i) for i in range(70)], ["--Ünï", "\x1bv"], ["--a=" + "z" * 300]]
params = [[a, d.parse_params(a)] for a in argvs]

def ctr(**kw):
    doc = {"Id": "a" * 64, "Name": "/c1", "Image": "sha256:" + "ab" * 32, "Created": "2026-09-01T10:00:00Z",
           "Path": "vllm", "Args": ["serve", "m", "--api-key", "sk-test-not-a-real-key-0000"],
           "State": {"Status": "running", "Running": True, "Pid": 77, "ExitCode": 0, "StartedAt": "2026-09-01T10:00:01Z"},
           "Config": {"Image": "vllm/vllm-openai:v1", "Labels": {}, "Env": ["SECRET=x"]},
           "HostConfig": {"NetworkMode": "bridge", "DeviceRequests": []}, "NetworkSettings": {"Ports": {}}, "Mounts": []}
    for k, v in kw.items():
        if v is None:
            doc.pop(k, None)
        else:
            doc[k] = v
    return doc

docs = [
    ctr(),
    ctr(Name="c2", Path=5, Args="x", Config={"Image": 7, "Entrypoint": ["/bin/sh", "-c", "vllm serve m --token=abc"], "Cmd": "x"}),
    ctr(Path=None, Args=None, Config={"Image": "x:1", "Cmd": "ollama serve"}),
    ctr(Config={"Image": "x:1", "Labels": {"engine-console": "1", "engine-console.role": "engine", "engine-console.instance": "i",
                                          "engine-console.engine": "BAD ID", "k\x01": "v\x02", "pw": "password=hunter2"}}),
    ctr(Config={"Image": "nginx:1", "Labels": {"engine-console": "1", "engine-console.role": "gateway", "engine-console.instance": "i"}}),
    ctr(Config={"Image": "x:1", "Labels": {"engine-console": "2", "engine-console.role": "engine", "engine-console.engine": "vllm"}}),
    ctr(State={"Status": "exited", "Running": False, "Pid": 0, "ExitCode": 137}),
    ctr(State={"Status": "exited"}),
    ctr(State={"Status": "exited", "ExitCode": 1.5, "Pid": 9.0, "Running": "yes"}),
    ctr(State="bad", HostConfig=[1], NetworkSettings="x", Mounts={"a": 1}),
    ctr(NetworkSettings={"Ports": {"8000/tcp": [{"HostIp": "0.0.0.0", "HostPort": "18000"}, {"HostIp": "::", "HostPort": "18000"}],
                                   "8001/udp": [{"HostIp": "10.1.1.1", "HostPort": "9"}], "x": None, "8002": None,
                                   "8003/tcp": [{"HostPort": 1.9}, {"HostPort": True}, {"HostPort": False}, {"HostPort": 0},
                                                {"HostPort": "65536"}, {"HostPort": " 12 "}, {"HostPort": "1_2"}, {"HostPort": [1]}]}}),
    ctr(NetworkSettings={"Ports": {}}, HostConfig={"NetworkMode": "host", "PortBindings": {"9/tcp": [{"HostIp": "127.0.0.1", "HostPort": "19"}]}}),
    ctr(NetworkSettings={"Ports": {f"{i}/tcp": [{"HostPort": str(20000 + i)}] for i in range(30)}}),
    ctr(Mounts=[{"Source": "/data/hf/hub/x", "Destination": "/m", "RW": False}, {"Source": "/data/hf", "Destination": 5},
                {"Source": "/data/hfx/y"}, {"Source": "rel/x", "RW": None}, {"Source": None}, {"Source": "/real/hf/z/", "RW": "no"},
                {"Source": "/data//hf/./a"}] + [{"Source": f"/m{i}"} for i in range(40)]),
    ctr(HostConfig={"DeviceRequests": [{"Count": -1}, {"Count": -1.0, "DeviceIDs": []}, {"DeviceIDs": ["0", " 1 ", "GPU-x", "MIG-y", "nvidia.com/gpu=0", "nvidia.com/gpu=all", 3, "GPU-x", "foo"]},
                                       "junk", {"DeviceIDs": {"a": 1}}, {"Count": True}]}),
    ctr(HostConfig={"DeviceRequests": [{"DeviceIDs": [f"GPU-{i}" for i in range(20)]}]}),
    ctr(Name=None), ctr(Name="/"), ctr(Name="//x\ny"), ctr(Id=None, Image=None, Created="garbage"),
]
roots = [(Root("hf", "/data/hf"), "/real/hf"), (Root("deep", "/data/hf/hub"), "/data/hf/hub")]
by_index = {"0": "GPU-zero", "1": "GPU-one"}
views = []
for di, doc in enumerate(docs):
    for status in (None, "Up 2 minutes", ""):
        try:
            v = d.container_view(doc, roots=roots, gpu_by_index=by_index, status=status)
        except Exception as e:      # Python crashes on these; the Rust node reads around the malformed field
            views.append([di, status, {"raises": type(e).__name__}])
            continue
        views.append([di, status, None if v is None else {"info": v.info, "pid": v.pid, "running": v.running,
                      "host_network": v.host_network, "loopback_ports": v.loopback_ports,
                      "published": [[k, x] for k, x in v.published.items()]}])

def P(status=None, body=b""):
    return d.Probe(status, body)
bodies = [b"", b"{}", b'{"data": []}', b'{"data": "x"}', b'[1]', b'{"data": [{"id": "m1", "owned_by": "library"}, 5, {"id": 3}, {"id": "\\u0001m2"}]}',
          b'{"data": [' + b",".join(b'{"id": "m%d"}' % i for i in range(30)) + b']}', b"\xef\xbb\xbf{\"data\": [{\"id\": \"bom\"}]}", b"<html>",
          b'{"data": [{"owned_by": "library"}]}']
mets = [b"", b"# vllm:x 1", b"sglang:y", b"tensorrt_llm", b"dynamo_x", b"llamacpp:z", b"trtllm"]
probes = []
for port in (8000, 11434):
    for st_m in (None, 200, 401):
        for body in bodies:
            for st_x in (None, 200, 403):
                for met in mets[:3]:
                    pp = d.PortProbe(port, P(st_m, body), P(st_x, b""), P(200 if met else None, met))
                    probes.append([port, st_m, bodies.index(body), st_x, met.decode(),
                                   {"any": pp.any_answer, "names": pp.model_names(), "hint": pp.hint(),
                                    "listing": pp.models_listing(), "qualifies": pp.qualifies()}])
# build_endpoints over the container views above (running ones) with a scripted prober
def scripted(port):
    k = port % 5
    if k == 0:
        return d.PortProbe(port, P(200, b'{"data": [{"id": "m"}]}'), P(200), P(200, b"vllm:x"))
    if k == 1:
        return d.PortProbe(port, P(401), P(None), P(None))
    if k == 2:
        return d.PortProbe(port, P(200, b"<html>"), P(200), P(None))
    if k == 3:
        return d.PortProbe(port, P(None), P(None), P(200, b"sglang:x"))
    return d.PortProbe(port, P(None), P(None), P(None))
eps_views = []
ep_docs = []
for doc in docs:
    try:
        v = d.container_view(doc, roots=roots, gpu_by_index=by_index)
    except Exception:
        continue
    if v is not None:
        eps_views.append(v)
        ep_docs.append(doc)
endpoints = d.build_endpoints(eps_views, [8000, 8000, 11434, 18000, 30000, 30001, 30002, 30003, 30004], scripted, 3, scripted)
json.dump({"docs": docs, "bodies": [b.hex() for b in bodies], "classify": classify, "params": params, "views": views, "probes": probes,
           "roots": [[r.id, r.path, real] for r, real in roots], "by_index": by_index, "endpoints": endpoints,
           "ep_docs": [docs.index(x) for x in ep_docs]}, sys.stdout, ensure_ascii=True, default=str)
