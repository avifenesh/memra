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

## 3. v2b: captured pooled forwards

### What runs

- **Pool** (`PooledGraphs`, on `HybridModel::pooled_graphs`). One CUDA graph per row structure
  (`PooledGraphKey`): each segment's row count and whether it keeps a checkpoint, in stack order,
  plus the attention rung. A graph serves every forward of that shape, whatever sessions it
  carries. Everything session-scoped that the captured body reads is device data at a baked
  address, refreshed before each replay:
  - the input residual and the row positions;
  - per segment and linear layer, the `[conv, ssm, alt]` state pointers;
  - per attention layer and row, the `(k, v)` cache pointers;
  - per segment slot and linear layer, the checkpoint snapshot slabs the scans write.
- **Body**. The eager walk's own layer functions in graph mode (`PooledGraphIo`): table slices
  instead of per-layer uploads, slab snapshots instead of per-forward buffers, and one attention
  launch over every row sized at the rung (the dspark full-verify graph's sizing argument: the rung
  sets stride and grid only, and per-row T_kv derives in-kernel from the positions). The body
  refuses what one launch cannot express, so the per-row fallback is never captured.
- **Replay bookkeeping** (`pooled_graph_forward`). After the launch, the host does what the eager
  walk does inline: per segment, the parity swap of every linear layer (odd row count), the KV
  len bump of every attention layer, the checkpoint columns copied out of the slabs (the same
  copies the eager walk makes) and the `pos` bump.
- **Policy**:
  - Shapes the body cannot express stay eager: a row below the fa vec floor, rows on two split
    steps, drafter taps (every DFlash2 round), more than 16 segments.
  - A shape is captured on its second sighting. The first eager walk has loaded its kernels, and
    a one-off shape never pays a capture.
  - The capture runs without warmups (`capture_graph_retained_nowarm`), so it executes nothing on
    live state.
  - Before a capture: `capture_headroom_refusal` with the slab growth plus the body's widest
    transients. Before each launch: `graph_launch_headroom_ok`.
  - The pool freezes at 256 graphs. A slab growth drains the stream and drops the graphs, which
    bake the slab addresses.
  - A failed capture leaves its shape eager.
- **Door**: `MEMRA_POOLED_GRAPH=1`, default off (see the verdict below).

### Exactness (RTX 5090 Laptop, q9; graphs forced on in every cell below)

1. **Graph vs eager walk** (`decode-batch-bench --unified`, raw `unified/v2b/bench-graph-cells.log`).
   Spec segments keep checkpoints as in serving. Each cell runs the captured graph twice from
   the restored states: the capture's first launch, then a pure replay. Both are compared bitwise
   against the eager unified walk on logits, hidden rows, plain draws and rows, every checkpoint
   column, parked logits and follow-up steps.
   - 10 graph shapes: **BITWISE EQUAL** on both passes. They are 4,4+4; 4,4+8 (with a
     host-sampled row); 4+12; 1+1; 4+3; 2,2+4; 4,4+2; and spec-only 4,4, 3,4,2 and 4,4,4,4.
   - 2 cells the pool declines: 3,2+6 at depth 40 (below the vec floor) and 2,2,2,2+8 at depth
     200 (rows on two split steps). They stay eager, also bitwise equal.
   - The eager unified rows also match per-session verify + the plain step in all 12 cells.
2. **Serial battery** on the graph-default build (`MEMRA_SPEC_GATE_MODEL=0`):
   **BYTE-IDENTICAL** to `int-band` (`unified/v2b/compare-battery-band.txt`).
3. **Served A/B, the same six boots as the speed table**:
   - pool cell **14/14** identical to `pool-int-a` on all six (`unified/v2b/served-ab/compare-pool-cell.txt`);
   - mixed cell **21/21** identical to the v1 reference on all six (`compare-mixed-cell.txt`).
   - The graph boots logged 1-4 captured shapes per boot: spec cohorts, and 1-spec-plus-rider
     shapes such as `[(4, true), (1, false), (1, false)]`.

### Speed

Per forward: eager walk vs graph replay from the same restored states, ABBA order, 36 pairs
per shape (`UNIFIED-GRAPH-TIME` lines). The ratio is **0.985x to 1.021x**, with overlapping
p10/p90 on every shape. Examples: 4,4+4 23.62 vs 23.52 ms; 1+1 9.33 vs 9.14 ms; 2,2+4 16.03 vs
15.73 ms; 4,4,4,4 33.59 vs 34.12 ms.

Served, v2a (eager) vs v2b (graph default), interleaved in both orders (v2a, v2b, v2b, v2a,
v2a, v2b), N=3, fresh boot per run, pool cell and mixed cell per boot. GPU 82-86 C, SM
1575-1770 MHz at run boundaries (`unified/v2b/served-ab/summary.txt`):

| cell | c | v2a median | v2b median | ratio |
|---|---:|---:|---:|---:|
| mixed | 3 | 168.9 | 166.9 | 0.99x |
| mixed | 4 | 195.6 | 194.4 | 0.99x |
| mixed | 6 | 243.3 | 240.9 | 0.99x |
| mixed | 8 | 252.9 | 252.7 | 1.00x |
| pool | 2 | 164.2 | 163.8 | 1.00x |
| pool | 4 | 172.3 | 172.7 | 1.00x |
| pool | 8 | 177.6 | 176.9 | 1.00x |

**Verdict: exact, and flat on this card.** A q9 pooled forward on the 5090 is GPU-bound at every
shape served here: the host enqueues the eager walk faster than the device runs it, so
removing launches buys nothing. The door stays default-off with one named missing gate, a GB10
served A/B (ARM host, 27B, longer forwards). If GB10 is flat too, the door, the pool and the
graph-mode layer branches are deleted.

## Binaries

sha256 in `unified/{refactor,v2a,v2b}/binaries.sha256`. Reference: the 3cd42279ee build in
`~/.cache/memra-batchspec-bins/int` (`memra-server` `bc1269892e63e678`).

- The v2a tree adds the `MEMRA_UNIFIED_STEP` read over the measured v2a binary. Its default is
  the measured behavior.
- The v2b tree flips the measured binary's graph default to opt-in. Every v2b receipt above ran
  with graphs on.

## Checks (v2a tree)

fmt `--check`, `git diff --check`, clippy `-D warnings` (memra-engine, memra-server, all
targets): clean. Lib tests: memra-engine 676 pass, memra-server 1144 pass. One memra-server run
in between failed `darklane::tests::repeated_stop_resume_has_fresh_acknowledgements_and_stopped_shutdown`
on its 30 s guard with the child in state D, while builds were loading the box (load 12.8). The
rerun passed; nothing in this lane touches that path.

## Checks (v2b tree)

fmt `--check`, `git diff --check`, the flags census, clippy `-D warnings` (memra-engine,
memra-server, all targets): clean. Lib tests: memra-engine 676 pass, memra-server 1144 pass.
The tree differs from the measured v2b binary by the default flip and one clippy type alias in
the bench probe.

The binary built from this tree (`unified/v2b/final/binaries.sha256`) ran the mixed cell twice:
at its defaults (unified step on, graphs off), and with `MEMRA_UNIFIED_STEP=0`. Both runs are
21/21 identical to the v1 reference (`unified/v2b/final/compare-mixed-cell.txt`). The default
run logs the unified cohorts and no graph capture. The seam run logs neither.


## Riders priced by the cost model; the seam is gone (2026-11-02)

v2a shipped with c=8 at 0.94x on the 5090. That cannot be a global default, so the rider decision
now lives in the cost model (`rider_forward_caps`, worker.rs). A rider joins a pooled forward only
while each row it adds costs the forward no more than the marginal plain-step row at this tick's
rider count (a lone rider pays a whole plain step), measured from the serving box's
`SpecCostTable`.

| device | where riders stop |
|---|---|
| 5090, q9 | about 8 rows; above that a verify row costs what a plain row does |
| GB10, 27B | at 8 rows, where the b16 tier starts charging about 7 ms a row |
| no table (the band owns the gate) | no riders |

`MEMRA_UNIFIED_STEP` is deleted (FLAGS.md row removed): the decision now has a measured basis, so
the seam has no reason to stay.

Why no table means no riders, measured on the q9 mixed cell (band forced to LOW=2/HIGH=16,
`battery/mixed_cell.py`, aggregate tok/s, `cap2-*` and `iso-*` boots):

- **c=8 with riders capped at 8 rows:** 283.2 / 276.3 / 282.7 / 275.3 / 279.0, against the
  3cd42279ee binary's 308.6 / 286.6 / 309.9 / 291.4 / 291.9 (about 0.94x).
- **c=8 with the epilogue refactor alone:** 304.8 / 289.6, at parity.
- **c=8 with no riders whenever a decoding plain row must stay in phase (c):** still lost. The loss
  is riders advancing at round cadence beside spec rows the band put there, not the forward's
  width.
- **c=3 and c=4:** riders gain, 1.07-1.10x and 1.00-1.05x.

So riders are a cost-model decision. Under the default cost-model gate on the 5090, the model keeps
c>=4 plain and no cohort forms. On GB10, the agentic receipts measure the cost-model arm with
riders priced from its table.

**Mixed-cell stream note.** c6/r1 is admitted to spec or plain depending on arrival timing, and
it differs between boots of the reference binary itself (int2 vs int1/int3). So it is not a byte
gate. All other streams match across every boot.
