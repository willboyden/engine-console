# Security

Basis: reading the code (re-read 2026-09-24 after the hardening pass) and the project's security principles (below). No penetration test,
fuzzing or independent review has been done.

**What was exercised on real hardware:** a live self-test on real rootless Docker verified (1) the internal engine
network, (2) an engine with no egress and no published ports, (3) the gateway forwarding on 127.0.0.1. It found and
fixed one real bug (nginx needed `daemon off;`).
**Not verified:** GPU engines under `--cap-drop ALL` or tensor parallelism, real proxy TLS interception and CA trust,
real Hugging Face downloads through the proxy. Everything else below is from code reading and offline unit tests,
marked *unverified* where a claim depends on runtime behaviour.

## Trust boundaries

```
[browser / curl on the host] --loopback--> [engine-console process, user uid]
    Host allowlist, Origin check, CSRF header, bearer/loopback auth, body + SSE caps
                                             |-- docker CLI (rootless) --> [gateway sidecar per instance, 127.0.0.1:18000-18099]
                                             |                                 |-- engine network (--internal) --> [engine container: no ports, no egress]
                                             |-- HTTPS via EGRESS_PROXY --> [allowlisting proxy :8082] --> huggingface.co + CDNs
                                             |-- gRPC, plaintext ---------> [OTel collector localhost:4317]
                                             `-- files: data dir (SQLite, secrets/, admin key), HF cache, hf_token
[other container, e.g. an LLM router] --opt-in: operator joins the engine network--> engine (internal_endpoint)
[externally started engines] <--read-only GET, loopback + container metadata-- [discovery (ADR-0013)]
```

1. Client to console: Host allowlist, then Origin/CSRF checks, then bearer key or loopback trust.
2. Console to Docker: the console's uid owns the rootless socket. Not defended (ADR-0002); mitigated only by label
   verification (the `engine-console` label, plus role and instance labels for gateways) and image allowlisting.
3. Console to engines: HTTP to the gateway on loopback; the gateway forwards TCP into the internal network.
4. Engine to anything else: no route (internal network, no ports). Reads weights from the read-only cache mount.
5. Console to Internet: Hugging Face only, through the proxy when `EGRESS_PROXY` is set (ADR-0012), plus the in-app allowlist.
6. Model-derived content (chat output, model cards, logs) is untrusted data rendered in the UI.

## Controls added in the hardening pass

| Control | Where | Notes |
|---|---|---|
| Host header allowlist, 421 on mismatch | `api/security.py` | Loopback names + configured port; extend with `ALLOWED_HOSTS` |
| Same-origin `Origin` check on non-GET (403 `cross_origin`; `null` refused) | `api/security.py` | |
| `X-Engine-Console: 1` required on non-GET without a bearer key (403) | `api/security.py` | Bearer clients exempt |
| Body caps: 1 MiB, 32 MiB chat, 256 KiB profile import (413), import content-type check (415) | `api/security.py` | |
| SSE cap (8 connections) and 300 s idle timeout | `api/deps.py`, config | |
| Audit stores metadata only for chat/arena/conversations/prompts; actor is the key id | `api/security.py` | |
| `hf_cache_dir` from environment only; validated absolute, no `: , newline`, not a symlink | `config.py`, `settings.py` | Prevents `-v` syntax injection |
| Engine mounts of the cache read-only by default | `config.py` (`hf_cache_readonly=True`) | |
| `RepoId`/`Revision` validated types; `--` and `..` refused | `domain/repo_id.py` | |
| Downloader: malformed sha256 rejected, sha256 verified when provided, symlink targets resolved and contained, queue cap 50 | `services/downloads.py` | Path-escape tests are unit-level only |
| Secret params stored as `[set]`; real value in a 0600 per-instance file | `services/secrets_store.py` | Value is plaintext on disk |
| Image allowlist by prefix (`vllm/vllm-openai:`, `lmsysorg/sglang:`, the pinned gateway digest) | `services/settings.py` | Prefix only: a tag is not forced to the pinned one |
| Container label verification before stop/rm/logs | `services/docker.py` | Refuses with `not_owned` |
| `--cap-drop ALL`, `no-new-privileges`, `--pids-limit 4096` (engines); gateway also read-only, uid 101 | `services/docker.py` | GPU engines under cap-drop unverified |
| Engines on `--internal` network, no published ports, gateway on 127.0.0.1 | ADR-0011 | Live-verified for CPU-visible behaviour |
| Console egress via an allowlisting proxy, fail-closed 503 | ADR-0012 | Interception unverified |

## Threat model (STRIDE)

| Threat | Scenario | Mitigation present | Residual risk |
|---|---|---|---|
| Spoofing | Web page in the operator's browser calls the API as loopback admin (cross-site, DNS rebinding) | Host allowlist (421), Origin check, CSRF header requirement | Closes the earlier gap for browsers (*unverified* against a real browser). |
| Spoofing | Another local process calls the API as loopback admin | Bearer keys hashed with SHA-256; `TRUST_LOOPBACK=false` | A local process can set the CSRF header itself, so default settings still make any local process admin. |
| Spoofing | Reverse proxy without `X-Forwarded-For` makes remote clients look like loopback; a proxy that rewrites Host may need `ALLOWED_HOSTS` | Loopback distrusted when the header exists | Proxy misconfiguration is a full auth bypass. |
| Tampering | Viewer key mutates state | Role check except chat/arena/conversations/fit | Prefix allowlist (`VIEWER_POST`): a new route under those prefixes is viewer-writable. |
| Tampering | Crafted params or repo ids inject docker flags, mounts or paths | Argv lists, no shell; `RepoId`/`Revision` types; validated `hf_cache_dir`; typed params; image allowlist | Not fuzzed (*unverified*). Image allowlist is by prefix, so `vllm/vllm-openai:<any tag>` runs. |
| Tampering | Malicious hub response writes outside the cache | Resolved-path containment checks; sha256 validation; `revision` validation | Unit-tested only. |
| Repudiation | Actor denies a change | Audit of every mutating call with key id, role, method, path, status, redacted params | Same SQLite file an admin can edit; not tamper-evident. |
| Information disclosure | Token or key in logs, API, audit | Redaction; secret params `[set]`; JSON logs drop tracebacks; chat bodies audited as metadata | Secret param values sit in plaintext 0600 files in `<data_dir>/secrets/`; pattern-based redaction can miss free text; chat content is plaintext in SQLite. |
| Information disclosure | Token visible to the egress proxy | Proxy is on loopback and operator-run | the proxy's flow log stores request bodies and headers: treat as sensitive. |
| Information disclosure | XSS via model output or card | Bespoke `markdown.js` (not audited here) | No CSP header set by the backend (none found). |
| Denial of service | Request or SSE flood | Body caps, 8 SSE connections, idle timeout, download queue cap | No rate limiting; not load-tested. |
| Denial of service | Disk exhaustion | Download disk pre-check, queue concurrency 2 | Unbounded SQLite growth (rollups, audit, usage): no retention job found. |
| Elevation of privilege | Compromised console starts arbitrary containers | Label verification, image allowlist, `cap-drop ALL`, internal network | An admin still controls what runs among allowed image prefixes; console compromise is still the user's container control. |
| Elevation of privilege | Engine container escape or lateral movement | Rootless Docker, `cap-drop ALL`, `no-new-privileges`, pids limit, internal network, read-only cache | Kernel/runtime escape not addressed; no seccomp profile beyond Docker's default; GPU passthrough widens the surface. |
| Elevation of privilege | Engine pivots via the router once opted in | Opt-in only, documented as a human-approved change | An opted-in engine can reach the router container and whatever it exposes on that network. |
| Exfiltration | Engine or compromised model sends data out | No route out (internal network) | Live-verified for the test container; not verified for GPU images. |
| Spoofing / Tampering / Information disclosure | A hostile local service or a compromised external engine feeds crafted names, metrics or command-line text to the dashboard (external engine discovery, ADR-0013) | Read-only, loopback-only, GET-only probes with short timeouts, 1 MB cap and no redirects; container environment never read; argv secrets redacted; all discovered strings length-capped and control-stripped, and escaped when rendered; lifecycle actions return 409 `instance_not_managed`; benchmarks need `confirm_external`; forced refresh is admin-only | Live-verified on three real engines (no secret in the JSON); hostile-data handling is covered by unit tests with fakes. A discovered engine sits outside the console's isolation model (it may have egress or other published ports), and a local service can masquerade as an engine. |
| Availability | Egress proxy down | Fail closed (503); no direct fallback when the proxy is configured | Default `REQUIRE_EGRESS_PROXY=false`: with `EGRESS_PROXY` unset the console runs direct and health reports a warning. |

## New residual risks (after this pass)

1. Egress enforcement is opt-in: default is `direct`. Set `EGRESS_PROXY` and `REQUIRE_EGRESS_PROXY=true`.
2. Secret param files are plaintext on disk (0600).
3. Image allowlist is prefix-based and does not enforce the pinned tag.
4. Local processes remain trusted as admin by default.
5. The gateway has no authentication of its own; loopback reachability is the only control.
6. `hf_cache_readonly` protects the cache from engines; the console itself still writes it, and a bug in the downloader is the remaining path.
7. The proxy's own log can hold HF credentials and is not encrypted.
8. Opting the router in widens its blast radius by design.

9. Discovery reads `docker inspect` metadata (image, command line with secrets redacted, ports, labels, name) of containers the console did not create, though never their environment. Anyone who can view the dashboard sees those names and images.

## Host memory monitoring

Read-only: the console reads `/proc/meminfo`, `/proc/<pid>/cgroup` (and process status as a fallback) and the container's
`memory.current` and `memory.stat` under `/sys/fs/cgroup`. It writes nothing there. The pid comes from `docker inspect`. The
cgroup path read from `/proc/<pid>/cgroup` is treated as untrusted text: it must match a strict character set, contain no
`..` segment, and its resolved directory must lie under `/sys/fs/cgroup`, otherwise it is ignored. Live-verified on real
engines 2026-09-24; the path-validation and RSS-fallback branches are covered by unit tests with fake `/proc` and cgroup trees only.

## Egress

- Console: `HF_ALLOWED_HOSTS` (`huggingface.co`, `cdn-lfs*.huggingface.co`, `cas-bridge.xethub.hf.co`, `*.hf.co`)
  checked per request and redirect hop; and the proxy's allowlist (the example in `deploy/egress-proxy/allowlist.txt` lists
  `huggingface.co`, `*.huggingface.co`, `*.hf.co`; glob matching means `cas-bridge.xethub.hf.co` matches `*.hf.co`, by reading
  the addon code, not by a live request). The two lists must be kept in step.
- OTLP exporter: `localhost:4317`, not proxied.
- Engines: none.

## Secrets handling

- `HF_TOKEN` env var or `~/.config/engine-console/hf_token` (umask 077, mode 0600). Never in the systemd unit.
- Token is sent only in the `Authorization` header of allowlisted requests.
- API keys: 256-bit random, shown once, stored as SHA-256. First-start admin key in `<data_dir>/bootstrap-admin.key` (0600).
- Engine secret params: `[set]` everywhere except the 0600 file; passed to the container by env var name.
- Backups of `<data_dir>` contain key hashes, secret files, conversations and audit: encrypt them .

## Security principles this project follows

| Principle | How the console relates |
|---|---|
| Deny egress by default | Engines have no route out; console egress goes through an allowlisting proxy and fails closed when configured; still opt-in by default. |
| Isolate what you run | Engines on an internal network, no ports, `cap-drop ALL`; the console only touches containers it labelled. |
| Never read or leak credentials | Reads only its own token file and environment; discovery never reads container environment values; no path touches SSH, cloud or browser credential stores. |
| No host-level changes without the operator | systemd unit, Grafana dashboard, running the proxy and joining another container to the engine network are all operator actions; nothing self-applies. |
| Observable from the first request | OTLP, `/metrics`, JSON logs, audit (ADR-0010). |
| Secrets never on disk unprotected | 0600 files, `[set]` markers; backups must be encrypted (OPERATIONS.md). |
| Do not claim what was not verified | Live-verified items are listed at the top; everything else is marked unverified. |

## Recommended posture

Loopback bind; `EGRESS_PROXY` plus `REQUIRE_EGRESS_PROXY=true`; `TRUST_LOOPBACK=false` on shared hosts; the hardened
unit; never expose port 8791 without a TLS proxy that sets `X-Forwarded-For` and whose hostname is in `ALLOWED_HOSTS`.
