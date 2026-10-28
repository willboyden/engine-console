# Regenerate (from the repo root, with the Python reference of backend/parity/lib/python-release.mjs):
#   PYREL=$(node backend/parity/lib/python-release.mjs); "$PYREL/node/.venv/bin/python" -I backend/crates/ecnode/tests/gen/wsd_httpproxy_vectors.py "$PYREL/node/src" \
#       > backend/crates/ecnode/tests/data/wsd_httpproxy_vectors.json
# Golden vectors for httpproxy.check_path / clean_headers (Python reference, seeded, deterministic).
import json, random, sys
sys.path.insert(0, sys.argv[1])
from ecnode.httpproxy import check_path, clean_headers
from ecnode.errors import OpError
rng = random.Random(20261027)
alphabet = list("/v1abc.%2eE5fF?#\\ \t\x00\x7f~:@!-_") + ["%2e", "%2E", "%2f", "%5c", "%00", "%0a", "..", "/v1/", "/api/", "/health", "%c3%a9", "%ff"]
paths = ["/v1/models", "/v1/../x", "/v1/%2e%2e/x", "/v1/%2E./x", "/health?x=../y", "/metrics#frag", "/version",
         "/api/chat", "/get_model_info", "/server_info/x", "/v1", "/v1/a%5cb", "/v1/a%00b", "/v1/%c3%a9", "/v1/%ff",
         "/v1/%", "/v1/%2", "/v1/./x", "/healthz", "/" + "a" * 2047, "/" + "a" * 2048, "", "v1/models"]
for _ in range(400):
    n = rng.randint(1, 12)
    s = "".join(rng.choice(alphabet) for _ in range(n))
    if rng.random() < 0.6:
        s = rng.choice(["/v1/", "/api/", "/health", "/"]) + s
    paths.append(s)
out_paths = []
for p in paths:
    try:
        out_paths.append([p, {"ok": check_path(p)}])
    except OpError as e:
        out_paths.append([p, {"code": e.code, "message": e.message}])
hdrs = [{}, {"A": "1"}, {"Connection": "close, X-Y", "x-y": "1", "Keep-Alive": "5", "X-Ok": "v"},
        {"Bad Name": "x"}, {"X": "a\rb"}, {"X": "a\nb"}, {"Ä": "x"}, {"connection": " Foo ,BAR", "foo": "1", "Bar": "2", "baz": "3"},
        {"Host": "h", "content-length": "5", "Expect": "100-continue", "TE": "x", "X-Auth": "Bearer t"},
        {"x" * 128: "1"}, {"x" * 129: "1"}]
out_h = []
for h in hdrs:
    try:
        out_h.append([h, {"ok": clean_headers(h)}])
    except OpError as e:
        out_h.append([h, {"code": e.code, "message": e.message}])
json.dump({"paths": out_paths, "headers": out_h}, sys.stdout, ensure_ascii=True)
