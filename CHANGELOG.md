# Changelog

## 0.2.1
- Optional per-user env file (`~/.config/engine-console/env`) so settings survive restarts.
- Egress: a missing proxy CA is now reported as a CA problem (with the fix), not "proxy unreachable"; `/health` warns when `EGRESS_PROXY` is set without `EGRESS_CA_BUNDLE`.
- Local library lists models whose `refs/main` points at a missing snapshot folder instead of hiding them.
- Layout: below 1024 px the navigation is a compact sticky top bar (it used to stretch to half the screen); added a small-screen tier for filters and tables.

## 0.2.0
- **External engine discovery.** Engines already running before the console starts (containers and `127.0.0.1` port
  probes) appear as monitor-only instances with metrics from discovery onward; lifecycle actions return 409
  `instance_not_managed`, and benchmarks need `confirm_external`. New settings `discovery_enabled`, `discovery_ports`,
  `discovery_interval_s`. Verified live on three engines (one SGLang, two vLLM); other states and engine classes are
  tested with fakes only.
- **Standalone documentation.** Docs, ADRs (now 13), deploy templates and README no longer depend on any host repository;
  a minimal example egress proxy (`deploy/egress-proxy/`) is included.
- **Breaking: renamed ownership label and engine network.** The label is now `engine-console` and the default engine
  network is `engine-console-engines`. Containers and networks created by 0.1.x carry the old names and are no longer
  recognised as the console's own: stop and remove them (and the old network) before upgrading, or relaunch them
  through the console. The default model cache is `~/.cache/huggingface`.

## 0.1.1
Fix clean-clone CI: the vLLM log fixtures were being ignored (not committed), and a test that depended on a
file outside this repository now skips when that file is absent. `v0.1.0` predates this fix and fails CI on a clean clone.

## 0.1.0
First public cut. vLLM and SGLang adapters behind one engine-agnostic core; HF search and resumable
downloads; fit estimator; schema-driven launch form; instances with logs; metrics; benchmark; chat and
arena; usage; audit. Engines run on an internal network behind a per-instance loopback gateway; console
egress goes through a mitmproxy chokepoint and fails closed.

Known gaps are listed in `docs/FEATURES.md` and `docs/SECURITY.md` (residual risks). Not yet verified on
real hardware: GPU engines under `--cap-drop ALL` with tensor parallelism, and full model downloads through
the proxy.
