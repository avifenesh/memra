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

## Prefill TTFT

The row placement kernel also serves the chunked prefill: a prime chunk's rows used to go into
their pending slots as 2 t memcpy nodes per compressor. Plain route, snap binary (C) against the
round lane (E), one boot per row, order C E E C C E, N=3 (`raw/se2-prefill-s2g/`):

| cell | C TTFT p50 | E TTFT p50 | delta |
|---|---|---|---|
| c2, 1500-word context | 10385 ms (9590..11236) | 8154 ms (8094..8492) | -21.5% |
| c1, 12000-word context | 41795 ms (38863..44856) | 34265 ms (33620..34732) | -18.0% |
| c1 decode p50 | 74.55 tok/s | 74.45 tok/s | -0.13% |

Every E row is under every C row in both TTFT cells. C drifted down across its three boots (11236,
10385, 9590 ms at 2k) and E did not; the cause is not measured. The plain c1 decode and aggregate
(71.60 against 71.35) sit 0.1% to 0.3% lower on E in every row, although the plain step runs none
of the changed code: the replay step appends rows through `memra_dsv4_replay_copy_row` and the
one-row eager step projects straight into its slot. Recorded as measured, not explained. Every
greedy text is identical across both arms.

## Rebased on main (push joins merged)

The lane was rebased onto `d04d9b817`, which carries the push joins, and re-gated on the second SE
pair (`raw/se2-rebased-s2h/`):
- the long gate's `PROGRAM_SHA256` is `fbce1a0492d69635`, as before;
- `dsv4_kv_split_gate` passes;
- the DSpark TP/EP gate's proposal shas equal the pre-rebase E and C shas.

Served DSpark against main at the same base, one boot per row, order M R R M R M, N=3:

| arm | greedy c1 agg | sampled c1 agg |
|---|---|---|
| M, main `d04d9b817` | 89.16 (87.78..89.57) | 75.84 (73.50..76.01) |
| R, the rebased lane | 93.17 (92.93..93.86), +4.50% | 78.65 (78.48..79.05), +3.71% |

Every R row is above every M row in both cells, and every greedy and sampled text is identical
across both arms.

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
