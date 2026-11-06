# Scenario promotions

Provenance log for scenarios added to the live suite (`scenarios/*.json`).
Generation is automated (`scripts/gen_scenarios.py`, staged into
`scenarios/staged/`); promotion is **manual review only** and recorded here.

## 2026-10-25 — first promotion (4 scenarios)

Source: `scripts/gen_scenarios.py`, authored at 2026-10-24 15:25–15:43Z by
`glm-5.3-flash-next-local` (the pinned `gpt-oss-120b` was 404 upstream).
Reviewed and promoted by the orchestrator session.

| scenario | theme | why it is harder than the original 8 |
|---|---|---|
| `warehouse_bin_relocation` | warehouse_relocation | requires arithmetic from a tool result (capacity − stock = 40) plus an explicit distractor (bin D-11) that must be ignored and must not appear in a `note` |
| `bank_transfer_limit_split` | bank_transfer_limits | error-recovery: attempt 6500, read the rejection, retry as two 3250 transfers (3 calls, sequence enforced) |
| `hotel_cancel_refund` | hotel_cancel_refund | two-step lookup→act with a parameter that must be ABSENT (`promo_code`) |
| `playlist_merge_dedupe` | playlist_merge_dedupe | union + dedupe + order over two tool results, then create with a computed track list |

### Review changes applied before promotion

1. **Position semantics fixed.** The generator wrote `"position": 1` meaning
   "second call overall", but `harness/judge.py::_nth_call` resolves position
   WITHIN calls of the same tool name (the original 8 only ever use
   `"first"`). All four scenarios failed "no call to X" despite correct
   model behaviour. Criteria were converted to `params_exact_any` (single-call
   tools) and `position: "first"` (absence checks). `bank_transfer_limit_split`
   keeps 0/1/2 positions because all three calls use the same tool.
2. **Unfair criterion dropped.** `bank_transfer_limit_split` required no
   `note` param, but its prompt never asks for one; removed.
3. **Harness bug fixed.** `_num_variants("3250")` emitted `"3,250.0"` (trailing
   `.0` from the float string), so an answer written `"3,250"` never matched.
   Now emits `"3,250"`; covered by the `bank_transfer_limit_split` answer check.

### Effect (measured, 3 live models, 1 rep, 2026-10-25T01:08Z)

| model | before (8 scenarios) | after (12 scenarios) |
|---|---|---|
| GLM-5.3-Flash, served locally | 100.0 | 100.0 (12/12) |
| a routed pair (small judge model first, GLM for hard turns) | 100.0 | 98.1 (11/12) |
| Nemotron Lightning on a 32 GB card | 100.0 | 90.4 (10/12) |

The suite previously saturated at 100 for every model and could only detect
outages; it now separates models by 9.6 points.
