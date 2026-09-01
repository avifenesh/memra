# Gemma-4-31B batched decode arm — results (lane/gemma-batched, 2026-08-16)

THE aggregate gate for the perfection lane. gemma4 served eager-only ("no batched
decode arm" boot notice); c1→c8 was FLAT (~55 agg, per-stream collapse) because the
arm was missing, not because of quantization. This lane builds it.

## What shipped

`gemma4_decode_batch` + `gemma4_batch_attn` in `decode_batch.rs`, behind
`MEMRA_GEMMA4_BATCH=1` (default OFF — eager stays the default until the serving-battery
gate is green). Dense 31B only; E4B keeps its dedicated decode and the refusal beneath.

Shape (the step35 law on gemma4's geometry): embed+scale, attn_norm+q8_1, wq/wk/wv,
fused q/k-RMSNorm + weightless-V-norm + dual rope (`rms_norm_qkv_rope`), post_attn_norm,
the layer-scale GEGLU tail (`gemma4_layer_tail_add_nq`), output norm, softcapped head —
ALL at m=B (one weight stream, B rows: decode is weight-BW-bound, that is the win).
Every one of these is the SAME batch-capable function the PROVEN verify trunk
(`gemma4_verify_trunk`) runs at width t, so the arm inherits verify's numerics wholesale.
KV append + `fa_decode_kvmod` stay a per-session loop (each session's own len drives its
SWA-window/global view — no cross-session attention, identical to eager per session).

Two module-private gemma4 helpers (`gemma4_layer_tail_add_nq`, `gemma4_suppress`) lifted
to `pub(crate)` for the decode_batch module; no behavior change.

## Exactness gate — ALL GREEN (5090, gemma-4-31B Q4_0 QAT, MEMRA_GEMMA_ROWS_W=0)

`decode-batch-gate --mode config`, MEMRA_GEMMA4_BATCH=1, MEMRA_SERVE_B1FAST=1:

| batch | gate1 (B=1 argmax vs eager decode_step_h) | gate2 (B=N isolation, bit-checked) | gate3 (sampling+lean) |
|---|---|---|---|
| B=4, 12 steps, 6 seeds | PASS (all steps) | PASS | PASS |
| B=8, 10 steps, 6 seeds | PASS (all steps) | PASS | PASS |

gate2 is the serving contract at full bit strength: a batchmate cannot change your
stream. gate1 confirms B=1 reproduces the eager decode's argmax stream. The house law
for a batched arm is NOT bitwise-vs-eager (the batched path is a documented distinct-but-
valid FP composition — see decode_batch_gate.rs header); it is isolation + argmax
calibration, exactly what step35/generic pass. Met.

## Aggregate scaling — the flat line is BROKEN

`decode-batch-bench --steps 64 --reps 3 --batches 1,4,8` (5090, Q4_0 QAT, median of 3):

| B | aggregate tok/s | per-seq tok/s | scale vs B=1 |
|---|---|---|---|
| 1 | 37.4 | 37.4 | 1.00× |
| 4 | 93.4 | 23.4 | 2.50× |
| 8 | 109.3 | 13.7 | 2.92× |

Previously gemma4 aggregate was FLAT c1→c8. The arm scales **2.92× at B=8** — the c8
non-scaling was the missing arm, now closed. Per-seq drops under batching (expected:
shared weight BW, higher per-seq latency); aggregate is the serving-economics number and
it scales.

### Read these numbers honestly
- **5090, not Japan PRO 6000; Q4_0 QAT, not the NVFP4 product artifact.** These prove
  the SCALING SHAPE and exactness, not the product's absolute serving throughput.
- B=1 here (37.4) is the generic BATCHED body at B=1 (B1FAST pinned off for the gate);
  the eager B1FAST path measured higher (72.4 tok/s single-stream on Japan-450W in the
  acceptance receipts). The bench isolates the batched arm's scaling, not single-stream.
- The n_vocab=262144 lm_head is a large per-row cost (host-sample note: greedy 223us/row)
  — a head-side batching/perf increment would lift the scaling curve further.

## Remaining (sized, not started)

1. **Worker serve route (the served-aggregate gate).** `worker.rs` routes gemma4 to the
   eager per-session loop and never calls `decode_step_batch` (the 2026-08-07
   panic-avoidance). To produce a SERVED c1/c8/c16 aggregate (worker chunking B sessions
   into decode_step_batch groups), that routing must send dense-31B gemma4 to the batched
   arm when MEMRA_GEMMA4_BATCH=1. Days-class-small: a routing predicate + the eager-vs-
   batched serve-stream identity check. This is what turns the engine-bench 2.92× into a
   served number on the NVFP4 artifact at 450W.
2. **Per-session fast attention arms.** v1 routes every session through `fa_decode_kvmod`
   (correct, matches the eager fallback). The `fa_decode_rows`/`rows_w` per-session fast
   arms (global-hd512 past the fa512 floor, windowed past `win`) are a perf increment
   behind their own seam — lifts the curve, no correctness change.

## Recommendation

The arm is exact (all gates green, both widths) and fixes the aggregate scaling. Keep
the seam DEFAULT-OFF until the worker serve route lands and a served c8/c16 aggregate on
the NVFP4 artifact confirms the number at 450W — then the default-on flip is a release
decision with a served receipt behind it. Do not flip on engine-bench alone.
