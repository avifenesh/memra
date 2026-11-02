# The unified ragged step (v2)

Step 6 of the batch-spec lane. Same-model plain decoding sessions ride the pooled MTP cohort's
lockstep forwards as one-row segments, so a mixed tick runs one forward per round over spec
verify rows and plain decode rows together.

## 1. Epilogue split (pure refactor)

`decode_batch_epilogue` (decode_batch.rs) is now two stages that the batched step runs back to
back, in the old device order:

- `decode_batch_sample_stage`: pristine parks, device penalties, grammar masks, the device draws
  and the `[B]` token readback (skipped for the pending step). Returns a `SampledStack`.
- `decode_batch_sample_rows`: the lean logits parks and the host rows.

The `pos` bump and the pending-step handoff stay in `decode_batch_epilogue`. The split lets a
caller sample any `[n, n_vocab]` logits stack with the batched step's own epilogue, without the
`pos` bump (the pooled walk already advanced each cache).

### Receipts (RTX 5090 Laptop, q9 + owntrim draft)

Reference: the 3cd42279ee binaries (`~/.cache/memra-batchspec-bins/int`). Hashes in
`unified/refactor/binaries.sha256`.

- Serial battery (`battery/battery.sh`, `MEMRA_SPEC_GATE_MODEL=0`) against `int-band`:
  **BYTE-IDENTICAL** on every cell, including the 56 + 4 + 4 `[spec-acc]` lines
  (`unified/refactor/compare-battery-band.txt`).
- MTP pool cell (`battery/pool_cell.py`) against `pool-int-a`: **14/14 streams identical**
  (`unified/refactor/compare-pool-cell.txt`). Single run; its tok/s are not a speed claim.

The committed tree differs from the measured binaries only by two clippy fixes in the split
(an `allow` and a needless borrow).

## 2. v2a: plain rows ride the pooled MTP lockstep

### What runs

- **Who rides** (`plain_rider_candidate`, worker.rs). A plain decoding session that phase (c)
  would serve through the batched step with a device sample: prompt primed, unconstrained, no
  vision input, no fork, `devsample_meta` is `Some` (greedy, temperature, filtered, device
  penalties), on a model whose plain rows the pooled walk reproduces
  (`HybridModel::plain_rider_capable`: the pooled-verify family with a dense FFN, the qualified
  generic batched rewrite, no hyper trunk, no fixed-solo B=1 arm, no width-cap door, full
  precision off). Constrained, host-sampled (greedy + penalties), vision, eager-only and
  other-model sessions keep phase (c).
- **Cohorts.** Phase (a) attaches every rider of a model to that model's MTP cohort. A cohort
  now also forms with ONE spec member when riders exist; it runs through the same lockstep driver,
  which for one member is the serial burst program (same begin/gate/draft/verify/finish/end; the
  serial verify when its forward has no other segment).
- **Engine** (`generate_spec_sessions_unified`, spec.rs; `generate_spec_sessions_pooled` is the
  rider-free wrapper). Per round, after the members' drafts, `PooledPlainRiders::round_begin`
  runs each live rider's emission step (`advance_sample_emit`: emit the previous step's token,
  stop battery) and returns the token it feeds plus its `devsample_meta`. The verify chunk walk
  then fills each pooled forward's free rows with riders, up to 16 rows when every matmul has the
  exact b16 class (8 otherwise, the plain step's own tier). Rider rows are one-row segments
  (`host_embed`: the plain step's host gather even when the forward has the resident embed
  table). A forward with riders pools even with one member. The riders' rows of the logits stack
  are sampled by the plain batched step's own epilogue (`decode_batch_sample_logits`, from the
  refactor), lean, with no second `pos` bump. Riders that no forward has room for, or whose
  forward did not pool, take the plain batched step itself (`rider_plain_step`).
- **Worker bookkeeping** (`PlainRiderSet`, `RiderLedger`). A rider's step result is the batched
  tick's: `last_logits`, `device_next`, `fed`; an error takes the batched step's error arm. A
  rider leaves when it stops, errors or the cohort ends. Every rider served this tick is skipped by
  phase (c), so no session is served twice in a tick. Riders' tokens count in `n_tokens_out`,
  lane accounting and the interactive TPOT marker. Receipt line: `[spec-pool] unified step
  engaged: S spec + P plain sessions`.
- **Rollback seam**: `MEMRA_UNIFIED_STEP=0` keeps every plain session in phase (c).

### Exactness (RTX 5090 Laptop, q9 + owntrim draft)

1. **Unified forward vs per-session verify + the serving plain step**
   (`decode-batch-bench --unified SHAPE --plain P --kinds K`, `unified_step_probe`, raw
   `unified/v2a/bench-unified-cells.log`). One pooled forward over spec segments and plain
   one-row segments, the plain rows sampled by the epilogue, against each spec session's own
   verify (resident embed table, as the serving round) and `decode_step_batch_sampled_lean_masked`
   over the plain sessions (lean, no outer scope, as phase (c)). Compared bitwise: spec logits
   and hidden rows, the plain rows' drawn tokens, their returned rows and parked pristine logits,
   and one follow-up step per session over the state each path wrote. **10/10 cells BITWISE
   EQUAL**:

| spec rows | plain rows | kinds | depth |
|---|---:|---|---:|
| 4,4 | 4 | greedy, T=0.8, filtered, penalized | 512 and 2048 |
| 4,4 | 8 | + host-sampled row | 2048 |
| 4 | 12 | mixed | 2048 |
| 3,2 | 6 | + host-sampled, below the fa vec floor | 40 |
| 2,2,2,2 | 8 | mixed | 200 |
| 1 | 1 | greedy | 2048 |
| 8 | 8 | all penalized | 2048 |
| 4,4,4 | 4 | T=0.8, filtered | 2048 |
| 2 | 3 | T=0.8 | 2048 |

2. **Serial battery** (`MEMRA_SPEC_GATE_MODEL=0`, against `int-band`): **BYTE-IDENTICAL**
   on every cell, including the 56 + 4 + 4 `[spec-acc]` lines (`unified/v2a/compare-battery-band.txt`).
3. **MTP pool cell** (every request speculates, no riders exist): **14/14 streams identical** to
   `pool-int-a` (`unified/v2a/compare-pool-cell.txt`).
4. **Mixed cell** (new, `battery/mixed_cell.py`). The band gate at LOW=2 / HIGH=16 admits the
   first two requests of a wave to MTP spec and the rest to plain decode, and nothing demotes.
   Waves c = 3, 4, 6, 8, staggered 0.25 s, `MEMRA_ADMIT_YIELD=0`, 200 tokens:
   - r0 greedy and r1 T=0.8 seeded (spec);
   - riders: r2 greedy, r3 T=0.8 seeded, r4 T=0.7 + top_k 40 + top_p 0.9, r5 T=0.8 + presence
     1.5 (device penalties);
   - phase (c) only: r6 greedy + frequency 0.5 (host-sampled), r7 greedy json_schema.

   The A/B below is 6 boots, 3 per binary. All **21/21 streams are identical across all six**
   (`unified/v2a/mixed-ab/compare-streams.txt`). The v2a boots logged unified cohorts of
   2 spec + 1..4 plain and 1 spec + 1..4 plain.
5. **Mixed cell with demotion** (`mixed_cell.py ... demote`, `MEMRA_SPEC_DEMOTE_AT=40` inside the
   concurrent waves). Not a valid byte gate: three boots of the reference binary disagree on
   one sampled stream (c3/r1, `unified/v2a/mixed-demote/compare-streams.txt`). The door is
   documented as solo-only: inside concurrency the demotion point follows burst boundaries, which
   follow arrival timing. The v2a boot matches the reference on 14 of 15 streams. The 15th
   (c8/r1, sampled) is where the reference boots agree with each other but the v2a boot's first
   serial burst starts one position later (ctx 40 vs 39), and it demotes at generated 65 instead
   of 68. Every greedy stream is identical in all four boots.

### Speed

Per-forward (`unified/v2a/bench-unified-timing.log`, depth 256, median of 9). It compares the
unified forward with the v1 serving shape, which is the spec rows' own pooled forward plus the
plain batched step:

| spec rows | plain | v1 shape ms | unified ms | gain |
|---|---:|---:|---:|---:|
| 4 | 1 | 17.25 | 10.88 | 1.59x |
| 4 | 3 | 19.52 | 13.90 | 1.40x |
| 4 | 4 | 21.01 | 15.93 | 1.32x |
| 2,2 | 4 | 21.19 | 16.03 | 1.32x |
| 4,4 | 2 | 22.80 | 20.80 | 1.10x |
| 3,3 | 4 | 22.70 | 22.11 | 1.03x |
| 4,4,4 | 4 | 32.47 | 31.30 | 1.04x |
| 4,4 | 4 | 24.44 | 24.61 | 0.99x |
| 4,4 | 6 | 26.77 | 28.16 | 0.95x |
| 4,4 | 8 | 28.96 | 30.98 | 0.93x |

On this card a forward costs about the same per row above 8 rows, in the pooled walk and in the
plain step. Riders win when they fill a forward that is still in the m <= 8 tier, and stop
paying off once they push an 8-row spec forward into the 9..16 tier.

Served, the mixed cell, interleaved v1 (3cd42279ee) / v2a in both orders (int, v2a, v2a, int,
int, v2a), N=3, fresh boot per run. GPU 82-87 C, SM 1545-1800 MHz at run boundaries (laptop
thermal regime, `gpu-*.txt`). Aggregate tok/s per wave (`unified/v2a/mixed-ab/summary.txt`):

| c | v1 runs | v2a runs | v1 median | v2a median | ratio |
|---:|---|---|---:|---:|---:|
| 3 | 148.6 / 152.5 / 153.2 | 171.3 / 164.0 / 165.7 | 152.5 | 165.7 | **1.09x** |
| 4 | 183.7 / 183.5 / 184.5 | 202.4 / 189.9 / 192.8 | 183.7 | 192.8 | **1.05x** |
| 6 | 241.2 / 242.5 / 242.5 | 248.9 / 239.7 / 235.3 | 242.5 | 239.7 | 0.99x |
| 8 | 269.1 / 269.2 / 272.4 | 253.2 / 257.3 / 250.8 | 269.2 | 253.2 | **0.94x** |

Median end-to-end seconds by request class:

| c | spec v1 / v2a | riders v1 / v2a | phase (c) only v1 / v2a |
|---:|---|---|---|
| 3 | 2.16 / 2.40 | 3.43 / 3.12 | - |
| 4 | 2.19 / 2.44 | 3.72 / 3.43 | - |
| 6 | 2.27 / 2.76 | 4.08 / 3.94 | - |
| 8 | 2.30 / 2.94 | 4.50 / 4.43 | 2.93 / 3.34 |

- **What moves.** In v1 a plain session gets one token per tick, and a tick holds a whole
  32-token spec burst, so plain rows wait while spec sessions finish first. In v2a riders take a
  token every round. Each round's forward then carries their rows, so spec rounds get slower and
  riders get faster.
- **Why c=6 and c=8 do not gain.** At c >= 6 the forward runs 2 x 4 spec rows plus 4 riders, the
  12-row shape the table above prices at 0.99x. The time riders gain is time spec sessions lose.
- **The c=8 loss** is the tail. The two phase-(c)-only requests still get one token per tick, and
  a tick now lasts longer because the cohort's rounds carry riders. r6, the host-sampled
  request, ends the wave about 0.35 s later.
- **Where it should pay.** GB10, the lane's target, verifies 1-8 rows for the price of one
  (GB10-CR.md), so there rider rows are close to free. No GB10 receipt here: the Sparks were out
  of scope for this step.

## Binaries

sha256 in `unified/{refactor,v2a}/binaries.sha256`. Reference: the 3cd42279ee build in
`~/.cache/memra-batchspec-bins/int` (`memra-server` `bc1269892e63e678`). The v2a tree adds the
`MEMRA_UNIFIED_STEP` read over the measured v2a binary. Its default is the measured behavior.

## Checks (v2a tree)

fmt `--check`, `git diff --check`, clippy `-D warnings` (memra-engine, memra-server, all
targets): clean. Lib tests: memra-engine 676 pass, memra-server 1144 pass. One memra-server run
in between failed `darklane::tests::repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown`
on its 30 s guard with the child in state D, while builds were loading the box (load 12.8). The
rerun passed; nothing in this lane touches that path.
