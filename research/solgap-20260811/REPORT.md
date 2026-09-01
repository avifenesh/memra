# SOL gap audit — Step-3.7 PP-2 serving path (box1 / PRO 6000 pair)

Date: 2026-08-11. Lane: `lane/solgap` (read-only research; no GPU work, no code edits).
Method: existing receipts + arithmetic only (`sol-model.py` in this directory, runnable).
Confidence marking: every number below is either **receipted** (file cited) or
**model-based, not profiled** (derived from the cost model; needs an on-box ncu/nsys pass
to confirm the mechanism split).

## Headline

Decode on the PP-2 pair runs at **~27-29% of single-card weight-streaming SOL at every
width B=1..8**, while the same engine's dense Q8_0 decode on the same 188-SM silicon class
measured **91% of achievable** (`research/q27-deepdive-20260805/RESULTS.md` §1b). The gap is
not bandwidth per se — it is (a) **serial PP stages** (one card idle at all times: an
automatic 2x ceiling loss), (b) a **host-synchronous sigmoid router** (42 full-stream D2H
round-trips per chunk per token-step), and (c) the **per-row B>1 attention walk** (the
exact launch class eagerpar just removed at B=1 for +4.4%). Prefill sits at ~3.2% of
one card's int8-TC reference — but is Amdahl-secondary for serving throughput; TTFT
already met the bar (5.227 s 4k on the new box, `research/newbox-bench-20260811/RESULTS.md`).

## Cost model output (verbatim, `python3 sol-model.py`)

```text
=== Step-3.7 active-parameter bill (per token, B=1) ===
  attention (all layers)         4.52 B params   2.41 GB @4.26bpw
  dense FFN x3                   0.42 B params   0.22 GB @4.26bpw
  routed experts 42L x top8      5.28 B params   2.81 GB @4.26bpw
  shared experts                 0.66 B params   0.35 GB @4.26bpw
  router                         0.05 B params   0.03 GB @4.26bpw
  lm_head                        0.53 B params   0.28 GB @4.26bpw
  TOTAL active                  11.46 B params   6.10 GB/token

=== Decode weight-streaming SOL vs measured (serial PP-2: one card active at a time) ===
  B  GB/chunk  uniqExp  SOL ms  SOL tok/s meas tok/s   %SOL
  1      6.10      8.0    3.41        293       85.0  29.0%
  2      8.77     15.6    4.90        408      119.9  29.4%
  4     13.96     30.3    7.80        513      144.7  28.2%
  8     23.51     57.5   13.13        609      165.6  27.2%

=== Achieved per-stage bandwidth at B=1 (specpp2 anatomy stage times) ===
  stage0: ~2.85 GB / 5.557 ms = 512 GB/s = 28.6% of card
  stage1: ~3.26 GB / 6.179 ms = 527 GB/s = 29.4% of card
  reference: q27 Q8_0 dense decode on the same 188-SM class achieved 88-96% per class,
  91% aggregate (research/q27-deepdive-20260805)

=== KV bytes per token (q8_0 K 34B/32, q5_1 V 24B/32) ===
  per layer per token: K 1088 B + V 768 B = 1856 B
  read at depth   512:   42.8 MB/token (0.7% of the B=1 weight bill)
  read at depth  4096:  122.6 MB/token (2.0% of the B=1 weight bill)

=== Prefill compute SOL (grouped, pp4096 solo) ===
  ~22.9 GFLOP/token active -> measured 692.7 tok/s = 15.9 TFLOP/s aggregate
  int8-TC card reference (SM-scaled from measured 82-SM 219 TFLOP/s): 502 TFLOP/s
  fraction of ONE card's int8 peak: 3.2%
  per-expert prefill GEMM shape at 4096 tok: m~114 rows, n=1280, k=4096 (small-m regime)

=== Launch/sync bill per token, B=1 (step35_decode_batch_layers walk) ===
  ~15+9 launches/layer fused-pair (43 layers), ~15+34 on the 2 clamped layers
  -> ~1130 kernels/token + lm_head + sampler epilogue
  q27 dense reference: 1015 launches/token = 7.5% launch-gap tax at 92.5% busy
  step35 adds 42 PER-LAYER HOST ROUTER SYNCS/chunk (sigmoid host oracle:
  e.dtoh(router logits) in moe_route_cfg) — the launch queue drains every MoE layer

=== Per-chunk host sync count at c=64 (8 serial B=8 chunks/tick) ===
  42 router D2H syncs x 8 chunks = 336 full-stream syncs per 381.7 ms tick
  + 1 [B]-u32 token readback per chunk (receipted flat to defer)

=== B>1 per-row attention walk (decode_batch.rs:1446-1481) ===
  B=8: 45 layers x 8 rows x 4 launches = 1440 per-row launches per chunk
  (the eagerpar class, B>1 edition)

=== PP boundary at decode (receipted) ===
  B=1: 16 KB f32 peer copy, tx 0.013 + rx 0.014 ms = 0.23% of the token walk
```

Model caveats (honesty): the SOL rows assume the whole active bill streams from HBM once
per chunk. Grouped/fused dispatch at B>1 reads each *unique* expert once (the uniqExp
column uses the balanced-routing expectation `E(1-(1-1/E)^(B*topk))`; real routing is
skewed, so the B=8 bill is an upper bound and 27.2% a lower bound on %SOL). L2 residency
of hot experts would raise the effective ceiling further. These refinements move the
number by single digits, not the ~3.3x class of the gap.

## Anchor receipts used

- **Round anatomy / stage times**: `research/specpp2-20260810/` — T=1 stage0 5.557 ms +
  stage1 6.179 ms, boundary tx/rx 0.027 ms total; verify = 95.13% of a spec round.
- **Decode ceiling + chunk cap**: `research/throughput-20260810/` — grouped-on c=8..64
  flat 128.4->129.7 agg (steady step 165.6->167.7); cap 8 is the exactness boundary
  (`research/chunkcap-20260810/`, KEEP verdict).
- **B=1 copy-launch class exists and pays**: `research/eagerpar-20260810/` — removing 90
  arithmetic-free D2D launches/token = **+4.381% c=1** (85.041), +5.496% c=2, +3.718% c=4.
- **Grouped dispatch decode win**: `research/throughput-20260810/` — grouped ON is
  +25.9% across the whole c-curve at decode (t=B>1 selects grouped; FLAGS.md's
  "prefill-only" description is stale vs `hybrid_forward.rs` t>1 dispatch).
- **Prefill**: `research/grouped-serve-20260810/` 692.7 tok/s pp4096 solo (+62.3% over
  ungrouped); `research/concprefill-20260808/` — loaded/concurrent 568-580 tok/s,
  SATURATED verdict (no hidden second compute lane).
- **Residency**: serving logs in throughput/eagerpar raw — both stages RESIDENT
  (`experts 45.72GB dev0 / 55.35GB dev1 vs ~94.9GB budget`); decode B>1 rides
  `moe_ffn_grouped_resident_q8` (2 fused kernels + router per MoE layer), B=1 rides the
  slab fused pair. No SLRU staging on this pair — expert H2D is NOT the decode limiter.
- **New box**: `research/newbox-bench-20260811/` — 600W pair: c=1/2/4/8 = 99.0/137.1/
  161.3/177.0, short TTFT 0.133 s, 4k 5.227 s. Gate medians supplied by the coordinator
  (98.30 / 158.45 / 173.62 at c=1/4/8, N=5) are consistent with that single-run ladder.
- **Coordinator-supplied ncu27b anatomy** (not read by this lane; recorded as given):
  qmatvec 68.8% of decode, per-kernel ceilings 4.83 / 7.09 / 3.46% — consistent with the
  "one weight-bound family owns the tick" shape from q27-deepdive.

## Closed items — do NOT re-propose (receipted dead)

| item | receipt |
|---|---|
| decode chunk cap > 8 | `chunkcap-20260810` — exactness boundary, not a knob |
| spec-on-PP2 (any K, any placement) | `specpp2-20260810` HOLD; K=1 -18.8% c=1, -42.8% c=2 |
| cross-stage token microchunk for verify | `specpp2-20260810` — bounded -4.0% |
| two-session spec pipeline (increment 1) | `specmech-20260810` — +0.27% vs serial spec, -53% vs plain |
| optimistic c=1 round pipeline | `optipipe-20260810` DESIGN — EV break-even at measured q |
| concurrent-prime scheduler work | `concprefill-20260808` SATURATED; `primemech-20260810` |
| deferred per-tick token readback | worker.rs inc3 note — measured FLAT, killed |
| bigger prefill tick | already promoted (2048) in serving config |

## Ranked tune candidates (expected % of end-to-end serving throughput if closed)

Confidence on all mechanism splits: **model-based, not profiled** unless a receipt is cited.

### 1. Device-side sigmoid router — kill the 42 host D2H syncs per layer-walk
- **Question**: how much of the ~71% decode SOL gap is the launch queue draining at every
  MoE layer? `moe_route_cfg` does `e.dtoh(router logits)` + host top-k for every
  sigmoid-router arch (step35/M3/Hy3) — a full stream sync per MoE layer, 42/token at
  B=1, **336 per c=64 tick**. The softmax archs already have the fused device router;
  step35 needs the sigmoid+bias+norm variant (small kernel; the host oracle stays as the
  bit-identity reference).
- **Expected**: +5-15% decode at all widths (the q27 launch-gap analogue was 7.5% with
  NO syncs; 42 hard syncs should be worse — the most launch-shaped gap on the board).
- **Gate**: routing bit-identity vs the host oracle (sel/w exact), one-hash serving
  golden, run-gen/run-spec. Effort: **M**. (A lane is already spawned per coordinator.)

### 2. Batched-decode per-row attention walk — the eagerpar win, B>1 edition
- **Question**: at B=2..8 the FA loop issues per-row `q_row`/`a_row` D2D copies plus a
  per-row `fa_decode_kvmod` launch: ~1440 per-row launches per B=8 chunk
  (`decode_batch.rs:1446-1481`). eagerpar removed the B=1 twin of exactly this class and
  banked +4.4%. Attack: drop the two copies per row via strided views first; a multi-row
  FA entry second.
- **Expected**: +2-6% at c>=2 (copy removal ~+2-3% by the eagerpar analogue; a batched-KV
  kernel more). Effort: **M** copies, **L** fused kernel. Gate: decode-batch-gate 0
  differing bits + the b1fix one-hash matrix.

### 3. Dual-active PP decode — stop paying the idle-card 2x
- **Question**: plain decode runs stage0 then stage1 serially; each card idles ~50%
  (stages already balanced at 5.557/6.179 ms). Chunk-level pipelining — stage0 of chunk
  N+1 under stage1 of chunk N — was priced as "the remaining ~2x" and explicitly cut to a
  follow-up in `pp-leverb-20260807/PROGRESS.md` (step 6); the pipelined engine arm
  measured **1.87-1.88x at N=2/4 sharded** (`m2-pp8-20260802`) and passed its x100
  cross-device quarantine-lift soak. At c=64 the tick already holds 8 independent B=8
  chunks — the schedule exists; the worker just issues them serially.
- **Expected**: up to +80-90% aggregate at c>=16 (bounded by the 1.88x receipt), ~+40-60%
  realistic first cut. THE largest single gap on the board.
- **Gate**: ppsplit bit-identity; the m2-pp8 flake history makes event ordering the risk —
  cross-device only, serial rollback seam. Effort: **L** (boundary double-buffering
  already exists: `pp.rs` slots + `MEMRA_PP_OVERLAP` seam).

### 4. Decode kernel %SOL spike — profile the fused expert pair on-box
- **Question**: even inside one stage, 512-527 GB/s achieved vs ~1557 GB/s the q27 dense
  receipt hit on this silicon class. After #1/#2 remove sync/launch tax, is the residual
  in `moe_gate_up_silu8_dev_q8_rows`/`moe_down8_fma_dev_q8_rows_g` (pointer-indirect,
  n=1280 skinny shapes) or the attention/glue tail? One ncu pass on box1; the 5090
  DECODE-GEMV-SOL playbook ports directly (DRAM %, long_scoreboard, L2 sector hit).
- **Expected**: unknown until profiled; q27 precedent warns the kernels may be near-wall
  and the gap all scheduling — do NOT bank kernel rewrites before the spike. Effort: **S**.

### 5. Sampler/epilogue + T=1 boundary riders
- Per-row `argmax_token_device_col` loop -> one batched kernel; overlap detokenize/SSE
  with next-chunk issue. Boundary copy is 0.23% (receipted), token readback deferral
  measured flat. **Expected**: <1-2% combined. Take only as riders. Effort: **S**.

## What this changes about the 632 tok/s breakeven

The receipted c=64 full-window ceiling is 129.7 (20.5% of target). The live stack:
#3 (up to ~1.9x) x #1/#2 (+8-20% combined) projects the ~260-320 agg class on box1-like
pairs, plus the new box's +31-40% silicon uplift on top — a path to ~50-60% of breakeven
from mechanism work alone; the remainder is capacity (more pairs), consistent with the
concprefill saturation verdict. These are model-based projections, not measurements; each
lane's own N=5 interleaved A/B is the truth.
