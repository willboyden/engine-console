# ADR-0007: Secrets by name, never by value

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1); amended 2026-09-24 after the hardening pass

## Context

The console handles an HF token and engine API keys, and it writes audit records and logs. Project principle: secrets never reach logs, URLs, API output or unprotected files.

## Decision

`HF_TOKEN` comes from the environment or a 0600 file (`~/.config/engine-console/hf_token`) and is attached only to allowlisted HF requests. The API reports token status only. Secret env vars (`HF_TOKEN`, `VLLM_API_KEY`) are passed to containers by name. A `redact()` step scrubs audit bodies and log lines; JSON logs drop tracebacks because they can contain secrets. The first-start admin key is written to a 0600 file, and its path (not value) is logged.

## Consequences

- Reduces the leak surface to the process and one file.
- Cost: redaction is pattern/key based; a secret embedded in a free-text field could slip past. Passing by name relies on the container runtime reading the console's environment.
- API keys are stored as unsalted SHA-256 of 256-bit random secrets; adequate for high-entropy keys, not for passwords.

## Alternatives considered

- OS keyring: rejected, not available headless.
- Secrets in the SQLite settings table: rejected, would put them in backups.

## Amendment (2026-09-24)

Secret engine parameters (any key matching key/token/secret/password) are stored in the database, API
responses, profiles and audit as the literal marker `[set]`. The real value lives in a per-instance 0600 JSON
file under `<data_dir>/secrets/` and is passed to the container as an env var name with the value in the child
process environment. Consequences: the value is plaintext on disk (protected by file mode only, so backups of
`<data_dir>` must be encrypted); a profile export cannot carry the secret, so restoring an instance needs it
re-entered. The `hf_cache_dir` is env-only and cannot be changed through the settings API.
