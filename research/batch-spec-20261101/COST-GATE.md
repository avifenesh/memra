# Cost-model spec gate

Step 4 of the batch-spec lane. The LOW/HIGH band decided "speculate or batch" from concurrency
alone, using a 5090 measurement of a short greedy shape. Pooled verify changed that trade, and the
right answer differs by device. The gate now prices both options from measurements taken on the
serving box.

## Mechanism

- **Boot calibration** (`HybridModel::spec_cost_calibrate`, `run_spec_cost_calibration`). Every
  MTP-served model gets a `SpecCostTable` at depth 2048, logged as `[spec-cost]`:
  - verify forward over 1..16 rows (measured at 1, 2, 3, 4, 6, 8, 12 and 16 rows, interpolated
    between);
  - plain lean batched step over 1..16 sessions (measured at 1, 2, 4, 8 and 16);
  - per-session round overhead beyond the verify, from a real 96-token session burst.

  The calibration takes about 11 s on q9 and runs before the admission probe, so the admission
  probe is still the last warm-up before serving. In the first ordering (calibration after the
  probe) the model arm was 2-4% slower at c=1/4/8 with the same verdicts as the band. The
  reordered run below is at parity.
- **Acceptance.** The model's recent per-position acceptance comes from the `SpecMetricState`
  window. Expected tokens per round are `1 + sum_{j<K} accepted_j / offered_j`. A model has no
  verdict until it has 64 rounds; at that count the standard error of each rate is at most 0.0625.
- **Verdict** (`spec_cost_best_k`). For N sessions it picks the K in `0..=rung` (0 = plain) that
  maximizes predicted aggregate tok/s. Speculation predicts `N * E[tokens | K] / (pooled verify
  chunks of <= 16 rows + N * overhead)`, or serial verifies where the model does not pool. Plain
  predicts `N / plain ticks chunked at the serving wave cap`.
- **Where the verdict applies:**
  - admission K (`choose_spec_k`), with reason `cost-model`;
  - tick demotion (`spec_demote_due`);
  - sampled prefix-cache restores (`sampled_restore_load_admits`).
- **Sampled admission laws, relaxed.** Sampled rows demote now, so a sampled restore follows the
  verdict instead of the SOLO watermark. For dspark widening, only a constrained row blocks
  (`dspark_blocks_greedy_widening`).
- **Where the band still decides:** operator pins, the PP-2 default, explicit LOW/HIGH overrides,
  DFlash2-served models, and models without a verdict yet. `MEMRA_SPEC_GATE_MODEL=0` restores the
  band everywhere; it is the A/B seam.

## RTX 5090 Laptop, q9, `battery/gate_ab.py`

Aggregate tok/s. Three reps per arm, interleaved in alternating order, one fresh boot per run.
Each boot runs a c=1 warm-up request first, then waves at c=1, 2, 4 and 8. Waves arrive together
and mix greedy with seeded T=0.7 requests, 256 tokens each, with serving defaults (admission yield
on).

| c | band runs | model runs | band median | model median | ratio |
|---:|---|---|---:|---:|---:|
| 1 | 195.9 / 198.3 / 207.4 | 197.6 / 198.0 / 207.4 | 198.2 | 198.0 | 1.00x |
| 2 | 232.4 / 224.5 / 222.2 | 245.3 / 239.4 / 235.6 | 224.5 | 239.4 | **1.07x** |
| 4 | 333.4 / 332.6 / 329.7 | 328.6 / 336.9 / 332.0 | 332.6 | 332.0 | 1.00x |
| 8 | 429.3 / 430.0 / 427.6 | 427.9 / 433.9 / 436.7 | 429.3 | 433.9 | 1.01x |

What the model chose:

- **Verdicts:** K=3 at n=1, K=2 at n=2 and n=3, plain at n>=4. Predicted at n=2: about 240 vs
  plain 184 tok/s. At n=4, predicted spec equals predicted plain.
- **Calibration line:**
  - verify_ms: 9.0 (1 row), 11.2 (4), 16.4 (8), 33.6 (16);
  - plain_ms: 9.5 (1 session), 12.9 (4), 18.4 (8), 35.6 (16);
  - round overhead 3.31 ms.
- **Why plain wins above c=4 on this card:** verify gets no cheaper per row here, so plain
  batching is the right call. The band reached the same answer from its fixed watermark; the
  model reaches it from measurement.
- **Why c=2 gains:** the band runs K=3 at c=2, and the model picks K=2 with pooled verifies.

GB10, where verify over 1-8 rows costs the same as one row (GB10-CR.md), is where the verdicts
move. Those receipts are measured on the Spark against the vLLM/SGLang agentic references.

Lib tests: memra-engine 675, memra-server 1143 pass. The cost-model unit tests cover:
- expected tokens computed from acceptance;
- verify chunking at 16 rows;
- on the GB10 table, speculation at c=1..4 and plain at c=16;
- no verdict without acceptance history;
- the restore-verdict law.

## Realized-rate pricing (2026-11-02, follow-up)

The tables are measured at depth 2048 in an engine-only burst. Serving adds token emission,
admission and whatever depth the traffic has. A first follow-up also measured the per-session
overhead on a pooled 4-session cohort (`pooled_overhead_ms`, 0.72 ms against 2.78 ms alone).
With that number the model predicted pooled K=1 beating plain at c=4 on the 5090. Measured, that
was **0.78x and 0.81x** of the band.

Two fixes:

- **Realized pricing.** Every pair of consecutive ticks in the same steady regime gives one
  aggregate tok/s sample for (sessions, K) (`spec_cost_tick_regime`, `spec_cost_observe`). A
  regime here means one model, everything live either plain or on MTP spec at one K. After 3
  samples the realized EMA replaces the prediction. Until then, predictions for an unobserved
  (n, K) are scaled by the realized/predicted ratio of the closest observation of the same kind:
  first the same n, then the same K at the nearest n, then any K at the nearest n.
- **A token-accounting bug found by that sampler.** Pooled cohort members counted their progress
  from the length they had after the cohort ran. Their tokens never reached `n_tokens_out`, the
  lane accounting or the tick traces. They now count from the length before the cohort.

**Learning within one boot** (`battery/waves_once.py`, c=4 four times after a c=1 warm-up), in
tok/s:

| wave 1 | wave 2 | wave 3 | wave 4 |
|---:|---:|---:|---:|
| 300.5 | 326.6 | 326.0 | 325.9 |

- Wave 1 explores pooled K=1 and realizes 293.5 tok/s against a plain prediction of 310.7, so the
  session demotes mid-wave.
- Plain then realizes 356.8 tok/s, and the model stays on plain from that point.

**3-rep A/B, final binary** (`battery/gate-ab-final/summary.json`). Each c is the boot's first
wave at that concurrency, so this is the worst case for learning:

| c | band | model | ratio |
|---:|---:|---:|---:|
| 1 | 191.8 | 189.3 | 0.99x |
| 2 | 234.0 | 247.3 | **1.06x** |
| 4 | 323.8 | 292.9 | 0.90x (one exploring wave) |
| 8 | 405.1 | 414.3 | 1.02x |

The c=4 cost is the one exploring wave. A serving box that sees a concurrency repeatedly pays it
once. On GB10 the agentic cells run 48 turns per concurrency, so the gate's prices settle inside
each cell.

**Battery on the final binary.**

- With `MEMRA_SPEC_GATE_MODEL=0` (`battery/compare-int-band.txt`), every stream is byte-identical
  to the round-split reference. The only difference is in the concurrent greedy pair's per-burst
  acceptance lines: a prime burst with a live same-model peer now ends after one token (see
  POOLED-ROUND.md), which shifts burst boundaries but not tokens.
- With the model on (`compare-int-model.txt`), greedy streams are identical. Two sampled streams
  differ: the model picks K=2 where the table says K=3, and K changes which Philox events a
  sampled stream draws.
