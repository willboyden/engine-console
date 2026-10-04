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
