# ADR-0005: SQLite in WAL mode for all state

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1)

## Context

Single node, single writer process, modest write rate. State: profiles, downloads, instances, metric rollups, benchmarks, conversations, prompts, arena, usage, audit, API keys, settings.

## Decision

One SQLite file at `<data_dir>/console.db` with ordered SQL migrations (`migrations/0001_init.sql`...). No ORM. Access through `services/store.py`.

## Consequences

- Zero ops; backup is a file copy (see OPERATIONS.md, use SQLite's online backup for consistency).
- Cost: single-writer, so this design does not scale to multiple console replicas; no HA.
- Conversations and audit are stored in the same file as keys; the data dir must be protected as a unit.
- Metric rollups and audit have no retention job documented in code; growth is unbounded until one is added (unverified, grep found none).

## Alternatives considered

- PostgreSQL: rejected, operational overhead for one user.
- Files/JSON per entity: rejected, no transactions or querying.
