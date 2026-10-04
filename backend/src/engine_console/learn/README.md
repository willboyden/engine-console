# Learn-mode content

Every ⓘ in the UI reads from these files. They are data, not code: the console loads them once and serves them as
one bundle (`GET /api/v1/learn`, shape `domain/learn.py: LearnBundle`). The coverage test in `tests/learn` fails
the build when something the UI shows has no explanation.

## Files

| file | keys | value |
|---|---|---|
| `glossary.json` | term key (`snake_case`, e.g. `kv_cache`) | `Term`: `title`, `short`, `long`, `see_also[]` |
| `metrics.json` | canonical metric key (ARCHITECTURE §5) and host metric key (`vram_used_gib`, `host_ram_gib`, ...) | `Explain` |
| `concepts.json` | concept key (ARCHITECTURE §10.7: `state.ready`, `link.stale`, `management.script`, `verdict.tight`, `check.gpu_shared`, `format.gguf`, `root.hf-home`, `alert.host_ram_low`, `page.library`, `field.host`, ...) | `Explain` |
| `params/<engine>.json` | that engine's `ParamSpec.key` | `Explain` |
| `presets/<engine>.json` | that engine's preset name | `Explain` |

Every file is `{"version": 1, "<section>": {key: value}}` where section is `terms`, `metrics`, `concepts`,
`params` or `presets`. UTF-8, two-space indentation, keys sorted.

`Explain` (from `adapters/base.py`):

| field | limit | content |
|---|---|---|
| `summary` | 160 chars, required | One plain sentence: what this is. No jargon unless it is in `terms`. |
| `details` | 900 chars | How it works. An everyday analogy where it helps. |
| `when_to_change` | 400 chars | The situations where a beginner should touch it, or "Leave this alone unless …". |
| `risks` | 400 chars | What goes wrong if it is set badly: crash at start-up, out of memory, slower, worse answers. |
| `example` | 200 chars | A concrete value and its effect ("0.85 leaves about 14 GB of a 96 GB card for other programs"). |
| `terms` | keys | Glossary keys the text relies on. Each must exist in `glossary.json`. |
| `level` | `basic` / `advanced` / `expert` | `basic`: most people should know it (model, context length, GPU memory share). `advanced`: tuning. `expert`: debugging or very specific hardware. |

`Term`: `title` (display name), `short` (160 chars), `long` (1500 chars), `see_also` (glossary keys).

`summary`, `short`, `title` and `example` are one line; `summary` and `short` are one sentence ending in a full stop.
`details`, `when_to_change`, `risks` and `long` may hold several paragraphs: separate them with a blank line (two
newlines), because the UI shows each paragraph on its own.

## How to write them

The reader is clever but has never run an inference engine. They want to know what the knob does, whether to
touch it, and what breaks.

1. **Say what it is in everyday words first.** "How much of the graphics card's memory this engine may use" before
   "fraction of VRAM for weights, activations and KV cache".
2. **Then the consequence in numbers.** Memory in GB, speed as "about twice as fast", context in pages of text
   (1,000 tokens ≈ 750 English words ≈ 1.5 pages).
3. **Tell them when to leave it alone.** Most parameters are fine at their defaults; say so.
4. **Name the failure.** "If it is too high the engine refuses to start because another program already uses part
   of the card." Not "may cause issues".
5. **One idea per sentence, no marketing, no hedging stacks.** Short sentences. Active voice.
6. **Be true for the pinned engine version.** Source facts from the engine's own `--help` of the pinned image and
   its docs (`ParamSpec.docs_url`), not from memory. If you are unsure of a number, leave the number out.
7. **Jargon goes in the glossary.** Use the word, add its key to `terms`, and make sure the term exists.
8. **No secrets, no personal paths, no lab-specific names.** This repository is public.

### Example

```json
"gpu_memory_utilization": {
  "summary": "How much of each graphics card's memory this engine may claim.",
  "details": "At start-up the engine reserves this share of every GPU it uses: room for the model's weights, and the rest becomes the KV cache, the working memory that holds the conversations it is serving. It takes the whole share at once, even when idle, much like booking a whole room rather than a seat.",
  "when_to_change": "Lower it when another program or a second engine shares the card. Raise it a little when you need longer conversations or more users at once and the card is otherwise free.",
  "risks": "Too high: start-up fails because the memory is already in use, or other programs on the card run out of memory. Too low: the model may not fit, or only very short conversations do.",
  "example": "0.85 on a 96 GB card reserves about 82 GB and leaves about 14 GB for other programs.",
  "terms": ["vram", "kv_cache", "weights"],
  "level": "basic"
}
```

## Metrics and concepts

The same fields, read a little differently. For a metric, `details` says how it is worked out and what a healthy value
looks like, `when_to_change` says what to do when it is off, `risks` says what a bad value means. For a state, link
or check, `when_to_change` is "what to do about it". Every metric fills all five text fields; every `state`, `link`,
`check`, `alert`, `verdict` and `download` concept fills `when_to_change`.

`concepts.json` keys are `<family>.<name>`. Where the code owns the list, the test reads it from the code, so a new
value without an explanation fails the build, and so does an explanation for a value that no longer exists (every
family below except `policy`, `page` and `field`):

| family | keys | list comes from |
|---|---|---|
| `state.*` | every instance state | `InstanceState` in `domain/models.py` |
| `link.*` | every link state | `LinkState` in `domain/fleet.py` |
| `management.*` | managed, script, external | `Management` in `domain/refs.py` |
| `verdict.*` | fits, tight, wont_fit, unknown, and `ok` (the host-RAM verdict) | `Verdict`, `HostRamFit` |
| `confidence.*` | high, medium, low | `Confidence` |
| `check.*` | every launch check code: `PreflightCheck(code=...)`, an adapter's `Compat(code=...)` or `add(level, code, ...)`, and the codes raised while a launch is built | found in the source with `ast`; two dynamic ones are listed in the test |
| `format.*` | hf, gguf, ollama, plain, other | `LibraryFormat` |
| `root.*` | auto, hf-home, hf-hub, plain, gguf, ollama, other | `docs/FLEET-PROTOCOL.md` 6.1 |
| `alert.*` | every code `services/hostmem.py` raises, plus `gpu_hot` and `vram_full` | found in the source |
| `download.*` | every download state, `kind_hf`, `kind_copy` | `DownloadState`, `Download.kind` |
| `mode.*`, `policy.*`, `page.*`, `field.*` | monitor/manage; the node policy switches; the pages; the form fields that are not engine parameters | ARCHITECTURE 10.7, FLEET-PROTOCOL 6 |

The keys the frontend asks for are listed in `frontend/js/learn-keys.js`; the test checks that file too.

## What the test checks

`backend/tests/learn/test_learn_content.py` runs offline. It fails when:

- a parameter or preset of any registered engine has no entry (the message lists the missing keys), or an entry names
  one that no longer exists;
- a required glossary term, a metric, a concept key or a launch check code is missing;
- text is over the limits above, a summary is empty, has more than one sentence or lacks a full stop;
- a `terms` or `see_also` key does not exist, or a text in `glossary.json`, `metrics.json` or `concepts.json` uses a
  glossary word (KV cache, VRAM, NVLink, ...) without listing it (for the per-engine files this is advice, shown as a
  warning);
- a file is not sorted, has duplicate keys, an unknown field, a placeholder (TODO), promotional wording or vague
  "may cause issues" wording;
- a personal path, e-mail address, IP address outside the documentation ranges, token-looking string or the name of the
  account running the tests appears anywhere in this directory.

When you add an engine, a launch check, an instance state or a page, add its explanation in the same change. A new
glossary word goes in `glossary.json` with its `see_also` links.
