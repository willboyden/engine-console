# ADR-0020: Learn mode: plain-language explanations as data, with a coverage gate

Status: accepted (contract v2, `docs/ARCHITECTURE.md` section 10.1, decision 20, and section 10.7); implemented in
`backend/src/engine_console/learn/`, `domain/learn.py`, `GET /api/v1/learn` and `backend/tests/learn/`.
**Offline tests:** the coverage gate and the content rules run in the normal backend test suite.
**Not verified:** that every explanation is *true* for the pinned engine versions (the gate checks that an explanation
exists and obeys the style rules, not that it is correct; the style guide asks authors to source facts from each pinned
image's own `--help` and docs), and the user-interface parts (the info popovers, Beginner mode and the Glossary page)
against the real backend: the frontend is still being finished.

**Status note (0.5.0).** Since 0.5.0 the content is `backend/crates/ec-hub/learn/` (embedded in the binary at build time), the types are in `domain/learn.rs`, and the coverage gate is `backend/crates/ec-hub/tests/test_learn_content.rs` with its sibling `test_learn_*.rs` files. The decision stands. See ADR-0022.

## Context

Five engines expose a very large number of settings, and the console adds metrics, states, link states, launch checks and
terms of its own. The audience includes people who have never run an inference engine and who need to know what a knob
does, whether to touch it, and what breaks if they do.

## Decision

- The explanations are **data, not code**: JSON files in `engine_console/learn/` (`glossary.json`, `metrics.json`,
  `concepts.json`, `params/<engine>.json`, `presets/<engine>.json`), loaded once and served as one bundle by
  `GET /api/v1/learn`. `GET /api/v1/engines/{engine}/params` also carries each parameter's `explain` and `level`.
- An `Explain` has a `summary` (one plain sentence, at most 160 characters), `details` (900), `when_to_change` (400),
  `risks` (400), `example` (200), `terms` (glossary keys it relies on) and a `level` (`basic`, `advanced` or `expert`). A
  glossary `Term` has a title, a short sentence, a long text and `see_also` links. `learn/README.md` is the style guide:
  everyday words first, then the consequence in numbers, say when to leave it alone, name the failure.
- Concept keys come in families: `state.*`, `link.*`, `management.*`, `verdict.*`, `confidence.*`, `check.*` (every launch
  check code), `format.*`, `root.*`, `alert.*`, `download.*`, `mode.*`, `policy.*`, `page.*` and `field.*`.
- A **coverage gate** (`tests/learn`) fails the build when a parameter or preset of any registered engine has no
  explanation, when a state, link state, management class, verdict, preflight or compatibility code that appears in the code
  has none, when an explanation names something that no longer exists, when a text breaks the length limits, is empty, has
  placeholders, promotional or vague wording, or when a text contains a personal path, e-mail address, token-looking string
  or the name of the account that runs the tests. Adding an engine, a check, a state or a page requires its explanation in
  the same change.
- In the UI an info button sits next to every parameter, metric, state badge, verdict and column header and opens a popover
  of the explanation. A Beginner mode setting shows the summaries inline and starts launching as a guided wizard. A Glossary
  page lists every term with search.

## Consequences

- No runtime generation: the help works offline, is the same every time, and costs nothing to serve.
- The content must be maintained with every engine bump and catalog refresh. The gate makes an *absent* explanation a build
  failure; it cannot detect a stale or wrong one.
- English only. A second language means translating every file.
- Writing is hand work that has to be checked against each pinned engine version, so the explanations are only as accurate as
  that check.
- Nothing in the console depends on the content being right: it is help text, and no behaviour reads it.

## Alternatives considered

- **Show the engines' own `--help` strings.** Jargon-heavy, uneven, and silent about risks and when to change a value. The
  `docs_url` of each parameter still links to the engine's documentation.
- **Generate explanations with a language model when asked.** Needs a running model, is not repeatable, and can be wrong with
  confidence in the one place a beginner trusts.
- **Link to external documentation only.** Not plain language and not available offline.
- **Prose documentation outside the application.** Not at the point of use, and it drifts from the settings it describes
  because nothing fails when it does.
