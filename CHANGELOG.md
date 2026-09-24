# Changelog

## 0.1.1
Fix clean-clone CI: the vLLM log fixtures were being ignored (not committed), and the allowlist tests now
skip when run outside the lab repo. `v0.1.0` predates this fix and fails CI on a clean clone.

## 0.1.0
First public cut. vLLM and SGLang adapters behind one engine-agnostic core; HF search and resumable
downloads; fit estimator; schema-driven launch form; instances with logs; metrics; benchmark; chat and
arena; usage; audit. Engines run on an internal network behind a per-instance loopback gateway; console
egress goes through a mitmproxy chokepoint and fails closed.

Known gaps are listed in `docs/FEATURES.md` and `docs/SECURITY.md` (residual risks). Not yet verified on
real hardware: GPU engines under `--cap-drop ALL` with tensor parallelism, and full model downloads through
the proxy.
