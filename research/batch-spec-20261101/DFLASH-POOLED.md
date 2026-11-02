# Pooled verify on the DFlash2 (dspark) route

Step 5 of the batch-spec lane: same-model DFlash2 sessions on a continuation burst run their rounds
in lockstep, and each round's verifies share pooled target forwards. The MTP twin is
`POOLED-ROUND.md`.

## What runs

- **Round split** (pure refactor, its own commit). `dspark_spec_session_burst_inner` is a serial
  driver over `dspark_burst_begin`, `dspark_burst_vg_engage`, `dspark_round_prepare`,
  `dspark_round_verify`, `dspark_round_finish` and `dspark_burst_end`. Fixed inputs and running
  totals live in `DsparkBurst`; a round paused at its verify boundary is a `DsparkRoundPrep`
  (candidates, proposal record, draft logits, snapshot, tap-sink ownership). The worker's
  `step_dspark_spec` tail is split the same way (`dspark_burst_plan`, `DsparkBurstEmit`,
  `dspark_burst_post`), and the eager-first-token census now pins those pieces.
- **Pooled verify with drafter taps** (`verify_pooled_dev`, spec.rs). A session whose cache has an
  armed `dflash_taps` sink gets its rows' post-layer residuals copied into it after every tapped
  layer, in `dflash_tap`'s per-row copy form. `dspark_verify_pooled` wraps it for dspark rounds:
  host-token embed (the dspark verify has no resident table), one `VerifyCkpt` per round when the
  stash arm is on, and each session's own `[vt, n_vocab]` logits rows back.
- **Engine driver** (`dspark_spec_sessions_pooled`, dflash.rs). It holds the model's verify-graph
  mutex once for the cohort, begins one burst per member, then per round:
  1. every live member runs its own `dspark_round_prepare`, with a round-owned tap buffer;
  2. consecutive non-deferred rounds form a forward of at most 16 rows, pooled verify, and that
     forward's rounds finish before the next forward runs;
  3. greedy rounds read the per-row device argmax walk, and the logits arms (sampled and
     greedy-penalised) read raw logits, exactly as their serial verify does.
- **Worker** (`dspark_pool_candidate`, `step_dspark_pooled`). Phase (a) groups DFlash2 sessions
  on a continuation burst by model, beside the MTP cohorts. A session qualifies with its session
  installed, no prime walker, an empty prefill queue, room in its request, and a DFlash2
  drafter. Cohorts of two or more run pooled; results feed the same per-session handling.

## Decisions

- **Graphs: option (a), the eager pooled walk.** On a DFlash2 drafter the serve route never
  replays the dspark verify graphs. Only the deferred greedy-chain arm (`_am_ckpt_dev`) does, and
  DFlash2 rounds take the selector proposal, so they always verify through
  `dspark_verify_t_am_ckpt` / `_logits_ckpt` (eager t-parallel walk, cols checkpoint,
  `round_slab = false`). For DFlash2 the pool is only a per-vt tap-buffer cache. Pooled rounds own
  their tap buffers, because the cohort holds several sinks at once. The verify rewrites the
  buffer completely, so ownership moves no byte. The commit reads the same cols checkpoint it
  reads today. A non-DFlash2 drafter never enters the cohort, and the driver refuses it too.
- **No per-session vt cap.** A 16/B cap on vt would make a sampled session's stream depend on
  its cohort size, because vt decides which drafts the rejection walk consumes. That breaks one
  numeric program per request. Members are chunked instead: each forward stays at 16 rows or
  fewer, and every session keeps its serial vt.
- **VRAM headroom guard.** The first pooled build kept every member's checkpoint alive until the
  round's finish loop. On the 27B a checkpoint holds the state of each linear layer for each
  verify column, about 157 MB per column (48 layers x 3.27 MB), so up to about 1.1 GB per round.
  Admission charges one round of transients, not eight. At c=8 on the 5090 that run died with
  `CUDA_ERROR_OUT_OF_MEMORY` in the pooled verify, dev0 free 68 MB, and 4 requests got 503
  (`cells/cell-pooled-oom-v1/`). The fix has two parts:
  1. finish each forward's rounds before the next forward;
  2. admit a member to a forward only while the extra memory (the other members' checkpoints and
     logits rows) fits `pooled_verify_headroom`. That is driver free + pool cached, minus
     `CAPTURE_HEADROOM_FLOOR` (the admission transient-floor class).

  A member the guard refuses verifies serially, which gives the same bytes. The guard fired once
  per process at c=8 (`761 more MB needed, 623 MB above the transient floor`).
- **MTP head not loaded** (`MEMRA_MTP_SKIP=1`) in every DFlash2 cell and bench, per the owner rule.
- **Spec gate for the cell.** `MEMRA_SPEC_GATE=0` admits DFlash2 solo only
  (`dspark_load_admits_with`: gate off admits at `n_active == 0`). A first run with the MTP pool
  cell's settings therefore served one DFlash2 row per wave and pooled nothing (`gate-off/`). The
  cell runs the gate on at LOW=8 and HIGH=16: every request rides DFlash2 and demotion stays out
  of reach.

## Exactness (RTX 5090 Laptop, Qwen3.8-27B NVFP4-Q5K GGUF + DFlash2 drafter q4, MTP head skipped)

1. **Pooled verify vs per-session verify, tap rows included** (`decode-batch-bench --pooled
   SHAPE --taps 5,19,33,47,61`, the drafter's tap layers). All shapes are bitwise equal on every
   logit row, on a follow-up step that reads the written state, and on every tap row. Shapes:
   8+8, 4x4, 3+5+8, 2+6 and 1+7+8 at depths 40 (below the fa vec floor), 200 and 2048
   (`bench-pooled-taps.log`), re-run on the final binary (`bench-pooled-taps-final.log`).
2. **Served streams** (`battery/dspark_pool_cell.py`, `cells/compare.txt`). Waves c = 1, 2, 4, 8
   with 0.25 s staggered arrivals and four arms by request index: greedy, sampled T=0.8 seeded,
   greedy + presence penalty, and sampled + frequency penalty + top_p. 200 tokens each,
   `MEMRA_ADMIT_YIELD=0`. Every request rode DFlash2 (spec rounds 88-105 each).
   - The serial reference is the pre-lane binary (`13ec038086e7700a`, the MTP-pool build).
   - Serial, serial rerun, pooled and pooled rerun match on **15/15** streams.
   - The pooled boots logged DFlash2 cohorts of 2, 3, 4, 6, 7 and 8 sessions.
   - All four runs show the same 2 recovered step-OOM parks at the c=8 prime. They predate the
     lane: the serial reference shows them too, at dev0 free 68 MB.
3. **The refactor alone.** The c=1 stream and every round outside a cohort run the split serial
   path. c=1 is identical to the pre-lane binary in every run above and in the gate-off run
   (`gate-off/compare.txt`: 15/15, where 11 of 15 requests went plain and 4 ran solo DFlash2).
4. **MTP serial battery** (`battery/battery.sh` on the final binaries vs the reference
   `memra-batchspec-battery/pool`, `compare-mtp-battery.txt`): **BYTE-IDENTICAL**, 23/23 cells,
   including the 56 + 4 + 4 `[spec-acc]` lines.

fmt `--check`, `git diff --check`, clippy `-D warnings` (memra-engine and memra-server, all
targets): clean, on both commits. Lib tests: memra-engine 674 pass, memra-server 1140 pass, on
both commits.

## Speed (RTX 5090 Laptop, the cell's aggregate tok/s, 200-token requests)

Verify forward alone (`bench-pooled-taps-final.log`, median of 5, taps armed in both arms):

| shape | depth | serial ms | pooled ms | gain |
|---|---:|---:|---:|---:|
| 8+8 | 40 | 88.2 | 79.6 | 1.11x |
| 4x4 | 40 | 104.6 | 81.1 | 1.29x |
| 3+5+8 | 40 | 97.4 | 77.4 | 1.26x |
| 8+8 | 2048 | 83.4 | 76.9 | 1.08x |
| 4x4 | 2048 | 112.3 | 86.0 | 1.31x |
| 3+5+8 | 2048 | 102.2 | 82.4 | 1.24x |

Served, the exactness boots (two per arm, `cells/compare.txt`):

| c | serial | serial rerun | pooled | pooled rerun |
|---:|---:|---:|---:|---:|
| 1 | 60.43 | 59.66 | 59.59 | 59.83 |
| 2 | 60.42 | 60.07 | 69.52 | 69.61 |
| 4 | 57.47 | 58.72 | 67.01 | 67.54 |
| 8 | 51.54 | 52.97 | 57.27 | 57.60 |

Interleaved same-window A/B, both orders (S P P S S P, N=3 per arm, fresh boot per run,
`dflash-pooled/ab/`). All 6 boots are 15/15 byte-identical. GPU 75-87 C over the six runs, SM
clock at run end 1387-1597 MHz (laptop thermal regime, `ab/*.gpu-*.txt`). Medians:

| c | serial runs | pooled runs | serial median | pooled median | gain |
|---:|---|---|---:|---:|---:|
| 1 | 59.75 / 59.73 / 59.59 | 59.72 / 59.82 / 59.55 | 59.73 | 59.72 | 1.00x |
| 2 | 60.19 / 60.01 / 60.05 | 67.76 / 67.30 / 69.51 | 60.05 | 67.76 | 1.13x |
| 4 | 56.48 / 56.81 / 56.75 | 64.97 / 64.41 / 66.70 | 56.75 | 64.97 | 1.14x |
| 8 | 51.59 / 51.01 / 53.39 | 55.56 / 54.72 / 56.33 | 51.59 | 55.56 | 1.08x |

c=1 never forms a cohort, so it is the control. Its two arms match within 0.2%.

- **Why the served gain is smaller than the verify gain.** Only the verify is shared. The rest of
  each round stays serial per session: the drafter block forward, the selector walk and its host
  readback, the commit and the drafter KV ingest. At c=8 the headroom guard also splits forwards:
  the boot logs show about 2 GB of effective free memory with the 27B and eight sessions resident.
- **Context.** The gate-off run served 11 of 15 requests plain, at 114 tok/s on c=8. Serial
  DFlash2 at c=8 gives 52 tok/s, so on this card plain batching still beats one-verify-per-session
  speculation at high concurrency. Pooling narrows the gap but does not close it. The batched
  drafter forward is the next lever.

## What stayed serial, and why

- **The drafter block forward, the selector walk and the KV ingest.** Batching the draft across
  sessions was optional. It is not bitwise-trivial: the drafter's windowed SDPA over per-session
  KV rings and its m-dependent matmul tier would need their own pins. It is the next step.
- **Deferred greedy-chain rounds** (non-DFlash2 drafters with a resident embed). They replay the
  captured verify graphs, which bake the pool's tap-buffer address, so the cohort never takes
  them.
- **Prime and resume bursts.** These keep the serial per-session step, as the MTP cohort does.

## Finding for the MTP driver (not changed here)

`generate_spec_sessions_pooled` keeps every member's verify output until the finish loop: all
chunks' logits and hidden rows, plus each member's `VerifyCkpt` columns. The checkpoint memory
grows with the cohort the same way. It is small on q9, but a dense 27B MTP cohort would hit the
same OOM class. The same per-forward finish and headroom guard would apply.

## Binaries (`binaries.sha256`)

- `memra-server`: final `9ae083e8eac5eeb7`; first pooled build (OOM) `f4a13fb1db159abe`; serial
  reference `13ec038086e7700a`.
- `decode-batch-bench`: final `b1760abad7167493`.
- `run-spec`: final `0127886d38e2ff9c`.

## Rerun on the integrated tree (2026-11-02)

`battery/dspark_pool_cell.py` ran twice on the integrated batch-spec binary (3cd42279ee, the tree
that lands). Both runs were compared with the step-5 serial and pooled references
(`dflash-pooled/compare-integrated.txt`). The result is **BYTE-IDENTICAL** on all 15 streams across
all four runs, with the same cohort sizes and the same VRAM-headroom cap event.

Aggregate tok/s:

| c | serial | step-5 pooled | integrated a | integrated b |
|---:|---:|---:|---:|---:|
| 1 | 60.43 | 59.59 | 60.20 | 59.82 |
| 2 | 60.42 | 69.52 | 69.93 | 68.26 |
| 4 | 57.47 | 67.01 | 69.17 | 66.72 |
| 8 | 51.54 | 57.27 | 57.48 | 56.37 |
