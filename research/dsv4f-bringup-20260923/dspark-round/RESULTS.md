# The TP/EP DSpark round: where it goes, and the first cut (memra #710, 2026-09-26)

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`. Hardware: a
second 2x RTX PRO 6000 Blackwell Server Edition pair. Route: `MEMRA_DSV4_DRAFTER=dspark` on the
TP/EP default, with PDL and the vocab-parallel head on (`../levers-20260926/`).

## Anatomy

With the lever stack, DSpark served 93.6 tok/s greedy c1 decode against 74.6 plain on this pair.
Per round it drafted 4.9 tokens and committed 3.46. The NVTX ranges of the round, from an nsys trace
over two 256-token greedy requests (148 rounds), with PDL off so that kernel durations are the
kernels' own (`raw/se2-anatomy-pdl0/`):

| phase | wall per round | GPU kernel sum | note |
|---|---|---|---|
| round | 41.7 ms | | |
| 2. verify (T about 5 rows) | 29.0 ms | 26.3 / 27.0 ms (dev0 / dev1) | GPU-bound |
| 1. drafter forward | 9.3 ms | 1.2 / 6.8 ms | host and dev1 |
| 3. commit and rollback | 2.7 ms | 0.48 / 0.49 ms | host-bound: 358 memcpy launches |
| 4. ring writes | 0.5 ms | 0.33 ms (dev1) | |

**Verify, kernel sum per device:**
- MoE stream visitor (multi-row): 7.25 ms. The expert union of about 5 rows x 6 experts.
- Dense FP8 GEMV (multi-row): 5.85 ms.
- The expert one-shot reduce: 2.18 ms, 43 launches at 50 us. It carries `t x topk x hidden`.
- Dots: 1.4 / 2.2 ms.
- Row gathers: 0.84 / 1.1 ms.
- Indexer scores: 1.0 ms, 123 per-row launches.
- memcpy D2D: 0.9 ms, 930 per round.
- HC finish: 0.69 ms.

**Drafter (dev1):**
- f64 island dots: 2.18 ms, 1.73 ms of it the Markov bias GEMV, 10 x 173 us.
- FP4 GEMMs: 1.0 ms.
- The reference sink attention: 0.9 ms, 3 x 300 us.
- cuBLAS BF16 GEMMs: 0.65 ms.
- The drafter's plane reduce: 0.52 ms.

The same trace with PDL on (`raw/se2-anatomy-pdl1/`) attributes about 38 ms of kernel time to a
27.5 ms verify. That is because with PDL a kernel starts early and waits at its entry, so its
duration includes its predecessor's. A cost table needs PDL off.

## First cut (`lane/dsv4-dspark-round-20260926`)

- **Rollback in one launch.** A partially accepted round's compressor rollback was 2 +
  2 n_commit + 2 x blocks memcpy nodes per compressor. It is now one launch per compressor:
  snapshot restore, the committed rows' slot writes and the overlap half shifts, replayed per
  element in position order (`dsv4_cmp_rollback_kernel`).
- **Row placement in one launch.** A verify or prefill transaction's rows into their pending
  slots were 2 t memcpy nodes per compressor. They are now one launch per run of rows between two
  block boundaries (`dsv4_cmp_rows_to_slots_kernel`).

Both are pure bit movement.

**Correctness** (`raw/se2-round-s2e/`):
- the long gate's `PROGRAM_SHA256` is unchanged (`fbce1a0492d69635`, the prefix prefill included);
- `dsv4_kv_split_gate` passes;
- the DSpark TP/EP gate gives the snap binary's proposal shas on the same pod, with batched ==
  sequential over 3206 cache classes.

**Served DSpark**, one boot per row, order C E F F E C C E F F E C
(`raw/se2-round-s2e/q-s2e.summary`):

| arm | greedy c1 agg | sampled c1 agg |
|---|---|---|
| C, the lever stack | 87.31 (85.83..87.46), N=3 | 73.86 |
| E, the round lane | 92.20 (91.74..92.43), N=4, +5.60% | 77.85, +5.40% |
| F, E plus the two drafter doors | 91.13, N=4, -1.16% against E | 77.10 |

## The two drafter doors, one at a time

`MEMRA_DSV4_DSPARK_MARKOV=rowblk` and `MEMRA_DSV4_DSPARK_CHAIN=device` were default-OFF doors from
an earlier iteration, bit-identical by construction and never measured on TP/EP. One binary, order
E G H H G E E G H, N=3 (`raw/se2-doors-s2f/`):

| arm | greedy c1 agg | sampled c1 agg |
|---|---|---|
| E, both off | 92.26 | 77.54 |
| G, rowblk Markov GEMV | 90.48, -1.93% | 76.74, -1.03% |
| H, device chain | 92.22, -0.04% | 77.60, +0.08% |

Negative and flat, so both doors are deleted, with every kernel reachable only through them
(`docs/FLAGS.md`, removed doors 2026-09-26).

## What the anatomy leaves

These are the next cuts, largest first. Each keeps the target program's bits.
1. The multi-row dense GEMV (5.85 ms for about 5 rows, where 1 row takes about 3.1 ms): weight
   reads should amortize across rows.
2. The expert reduce's plane (`t x topk x hidden`): a slot gather would carry only each rank's own
   slots.
3. The drafter's reference kernels (sink attention, f64 dots) and its host routing.
4. Per-row indexer and top-k launches in verify.

Changing the drafter's numerics moves its proposals, not the target's tokens.
