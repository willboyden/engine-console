# ADR-0004: Zero-build, zero-npm frontend

Status: accepted (contract v1, see `docs/ARCHITECTURE.md` section 1)

## Context

The audience values auditability and offline operation. oMLX vendors its dependencies for the same reason. A typical SPA pulls hundreds of transitive npm packages.

## Decision

Hand-written ES modules and web components under `frontend/`, served as static files by the backend. No bundler, no runtime dependency, no CDN reference (the only external-looking URL in `index.html` is an inline data: favicon). Charts are hand-rolled SVG. Pure logic modules are tested with `node --test`.

## Consequences

- Nothing to scan, pin or update in the browser supply chain; works offline.
- Cost: no type checking, no component ecosystem, more hand-written DOM code. Markdown rendering and charting are bespoke and will be less capable than mature libraries.
- Only `en` ships; the i18n mechanism exists (flat keys, plurals) but oMLX-style eight-language coverage does not.
- UI behaviour is covered by unit tests of pure modules, not by browser end-to-end tests (none exist).

## Alternatives considered

- React/Vue/Svelte with a build: rejected (supply chain, build step).
- Vendored htmx/Alpine: viable, rejected to keep zero third-party JS.
