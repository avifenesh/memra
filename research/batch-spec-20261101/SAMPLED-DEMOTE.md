# Sampled spec demotion: the Philox counter handoff

Date: 2026-11-01. Lane: batch-spec. Code: `demoted_philox_base`, `plain_draw_ctr` and
`spec_demote_eligible` in `crates/memra-server/src/worker.rs`.

## What changed

The spec-gate tick demotion used to skip sampled MTP and DFlash2 sessions, so a sampled session
admitted on a quiet box kept bursting serially after load arrived. Sampled sessions now demote.
Constrained sessions still do not, and GLM5 still demotes greedy only.

## Why the old exclusion existed, and the exact handoff

Plain device sampling draws generated token `i` as one gumbel event at Philox `(seed, stream_pos=i)`.
The sampled spec routes draw from the same Philox space (same seed, same kernels) but count events
with the session counter `sctr`: every draft, residual, bonus and boundary draw. After `g` emitted
tokens `sctr = S`, usually well above `g`. If the demoted row kept the plain counters literally, it
would draw tokens `g+1..S` with noise vectors the spec prefix already used, some on drafts the
client saw. That continuation is not an exact sample.

The handoff keeps the whole plain program and moves only the counter origin. The handed token (MTP
`next_pred`, DFlash2 `last`) is already a draw and has not been emitted yet. The first plain draw
then lands on `S`, the spec session's first unused event, and no event is reused. A session that
never demotes keeps `philox_base = 0`, which is the old rule bit for bit.

## GPU cell (`sampled_demote_cell.py`)

- Rig: RTX 5090 Laptop, driver 615.71.09.
- Model: q9 (`Qwen3.5-9B-NVFP4-MTP-GGUF.gguf` with the owntrim draft head).
- Request: temperature 0.8, seed 1234, max_tokens 160, thinking off.
- Server binary sha256: `24eacca052f0beef07a749717a157da5593abfdbf08f968094e9ad49cd91c33f`.
- Each arm ran on its own boot.

| Arm | Env | Spec rounds / drafted / accepted |
|---|---|---|
| demote_a | `MEMRA_SPEC_DEMOTE_AT=40` | 29 / 87 / 36 |
| demote_b | `MEMRA_SPEC_DEMOTE_AT=40` | 29 / 87 / 36 |
| spec | none (c=1, no demotion) | 75 / 225 / 82 |

Results:

- **Demotion fired on the target.** The log line was `generated 67, sampled, plain draws from
  philox event 119`. That is the first burst boundary at or past 40. Under the old rule, the plain
  draws for tokens 68..118 would have reused 51 spec events.
- **The output is deterministic.** demote_a and demote_b produced identical text (780 chars) across
  two boots.
- **The pre-demotion prefix is unchanged.** demote and spec agree on the first 331 chars (about 68
  tokens at this stream's 4.9 chars/token), which is the 67 emitted tokens plus the handed token.
  They diverge after that, because the continuation draws fresh counters. No arm errored.

Before this change the same `DEMOTE_AT` arm never demotes a sampled session: the old sweep skipped
`!is_greedy()`. The unit test `demoted_sampled_rows_draw_from_fresh_philox_events` fails against
the old base-0 rule.

## Not changed here

The sampled admission laws still assume sampled rows cannot demote: the SOLO sampled-restore
watermark and the "one non-demotable dspark row" block. Relaxing them is a policy change, and it
gets an A/B in the GB10 batch-spec receipts.
