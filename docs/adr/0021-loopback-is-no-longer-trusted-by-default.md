# ADR-0021: Loopback is no longer trusted by default

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 21); implemented in
`backend/src/engine_console/config.py` (`trust_loopback: bool = False`), `api/security.py` and `api/routers/system.py`.
Replaces the default recorded in ADR-0009 (loopback clients are admin); the rest of ADR-0009 (bearer keys, two roles,
hashed storage) stands.
**Verified live:** against a running console, a request without a key gets 401, and the CSRF header is enforced.
**Not verified:** the sign-in form of the user interface against the real backend (the frontend is still being
finished), and whether a given browser's password manager offers to save the key.

## Context

ADR-0009 trusted loopback clients as admin so a single-user, single-machine console needed no setup. Host, Origin and CSRF
checks were added later, which closed the hole for web pages in the operator's browser but not for other local software:
any process that can open `127.0.0.1:8791` can set the CSRF header itself. In 0.3 such a process could already start
engine containers as the console's user on that one machine. In 0.4 the console can start containers on every machine of
the fleet, run the operator's scripts, delete models and copy files between machines, so the same trust is a much larger
grant.

## Decision

- `TRUST_LOOPBACK` is `false` by default. Every call except `GET /api/v1/health` and the static app shell needs
  `Authorization: Bearer <key>`. That includes `GET /metrics` and the OpenAPI JSON (`/api/v1/openapi.json`).
- Keys are `admin` or `viewer`, stored as SHA-256 of a 256-bit random secret and shown once when created. The first start
  writes an admin key to `<data_dir>/bootstrap-admin.key` (mode 0600) and logs only its path; later starts log a reminder
  to copy it somewhere safe and delete the file.
- The user interface shows a sign-in form for the key, built so that a browser's password manager can save it.
- The Host allowlist (HTTP 421), the Origin check (403 `cross_origin`) and the CSRF header (403 `csrf_header_required`, for
  requests that carry no bearer key) stay. Bearer clients are exempt from the CSRF header by design.
- `TRUST_LOOPBACK=true` is still possible for a single-user machine. `/api/v1/health` then carries a warning that any local
  process is admin and that the console can now start containers on every machine.

## Consequences

- A local process needs a key like any other client. `curl`, scripts and a Prometheus scrape must send the header; a scrape
  needs a `viewer` key.
- Anyone upgrading from 0.3 must sign in once. The bootstrap key file is a credential on disk: copy it into a password
  manager and delete the file, or mint a named key with `POST /api/v1/keys` and revoke the bootstrap key.
- The viewer role is still narrow: it may call `POST` only on chat, fit, conversations and arena routes (matched exactly),
  and read everything else, including model names, engine arguments and logs of managed engines.
- The application shell is public, so the sign-in page loads without a key; it contains no data.
- A key is a bearer secret: it is only as protected as the place it is stored (browser, shell history, scrape config).

## Alternatives considered

- **Keep loopback trust and rely on the Host, Origin and CSRF checks** (the state at 0.3.0). A local account or any
  software running as it would drive every machine of the fleet.
- **Trust loopback for reads and require a key only to change things.** Reads expose model names, engine command lines and
  logs, which the operator may consider private on a shared machine.
- **A Unix-domain socket with a peer-credential check instead of a TCP port.** Would remove the need for a key for the
  owner, but the browser interface needs TCP, so it would not replace the key for the UI.
- **OpenID Connect or another single sign-on.** More than a single-operator console needs, and it would add a service to
  run.
