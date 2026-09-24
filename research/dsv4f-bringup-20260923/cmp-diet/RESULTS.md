# DSv4 compressor diet on the served PP-2 program (memra #695)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 2026-09-24. Served program: PP-2, matrix expert
program, host sampler, one serving lane, memra-server otherwise naked. One scored campaign at a
time under `/tmp/memra-gpu.lock`, 250 ms telemetry per row. Lane commit `bec39b449` on main
`25bbb91f5`; both binaries built on the box from those trees
(`raw/q-pair2.summary` has the tree SHAs and binary hashes).

## What changed

Two halves, one lane.

1. **Copy diet.** A one-row round always commits its row, so outside TP/EP nothing reads the
   pending-state rollback snapshot and nothing replays the row record. Plain decode now skips both
   snapshot copies per compressor (256 KB each on ratio-128 layers) and projects the kv/score rows
   straight into the pending slot. The overlap compressor's cur->prev shift copies the disjoint
   upper half over the lower half in one copy instead of four through a scratch. TP/EP keeps the
   snapshot for its zero-row refusal rollback. A rollback with no live snapshot now refuses
   instead of restoring stale bytes.
2. **BF16 weight storage.** The checkpoint stores all 62 compressor projections (41 attention, 21
   indexer) as BF16. The engine widened them to f32 islands and streamed about 1.16 GB per token of
   them. Every dots kernel the compressor reaches widens BF16 exactly and runs the f32 branch's
   product order, so the lane keeps the BF16 plane and passes `w_is_bf16 = 1`: the same bits at
   half the bytes, and about 290 MB less device memory per card.

## Correctness

- Kernel boundary: `tests/dsv4_island_bf16_gpu.rs` holds all five dots entries the compressor
  reaches to bit identity between the BF16 plane and the f32 widening, at the three compressor
  shapes and 1, 2, 6 and 33 rows, with a red arm: `DSV4_ISLAND_BF16 EXACT cases=60 entries=5
  shapes=3 rows=1,2,6,33 red_arm=1` (`raw/cmp/island/gate.log`).
- Served DSpark identity gate on the lane binary: `GPU DSPARK GATE [PASS]`
  (`raw/cmp/dspark-served/gate.log`).
- Served text: every request's text hash is identical across all 20 plain rows and all 20 DSpark
  rows of both arms, and the DSpark greedy hashes equal the plain greedy hashes (first request
  `aea6e69e` on both routes). Same bits in, same bits out.

## Served A/B

One boot per row, order `L B B L L B B L L B` (N=5 per arm), cells `raw/cells-spec.txt`
(greedy c=1 x8, sampled c=1 x8, greedy c=1 ignore-eos x4), 250 ms telemetry. Raw rows:
`raw/cmp/plain/r*-{lane,base}/`, `raw/cmp/spec/s*-{lane,base}/`.

### Plain decode

| cell | lane decode p50 | base decode p50 | delta | lane TPOT p50 | base TPOT p50 | TTFT p50 lane / base |
|---|---|---|---|---|---|---|
| greedy-c1 | 53.51 | 50.93 | **+5.07%** | 18.69 ms | 19.64 ms | 210 / 213 ms |
| sampled-c1 | 48.19 | 46.16 | **+4.41%** | 20.75 ms | 21.67 ms | 212 / 214 ms |
| greedy-c1-ignore-eos | 53.45 | 50.88 | **+5.07%** | 18.71 ms | 19.66 ms | 210 / 213 ms |

Row spread is tight: lane greedy 53.49..53.53, base 50.92..50.94. TPOT p99 is within 0.03 ms of
p50 on every row. Thermal, samples at >= 20% utilization: lane power median 218..219 W, base
215..216 W, SM clock 2610..2857 MHz on both, max temperature 52 C.

### DSpark

| cell | lane decode p50 | base decode p50 | delta |
|---|---|---|---|
| greedy-c1 | 67.70 | 66.90 | **+1.19%** |
| sampled-c1 | 54.85 | 54.36 | **+0.91%** |
| greedy-c1-ignore-eos | 67.53 | 66.78 | **+1.12%** |

DSpark gains less because its verify rows are multi-row, and multi-row rounds keep the snapshot
(they can refuse). The BF16 half still applies there.

## Anatomy (nsys, served greedy, 510 steps)

`raw/prof/plain-{main,cmp}/ana.txt`, captured with `raw/prof-served.sh` under the same lock:

| | main `25bbb91f5` | lane `bec39b449` |
|---|---|---|
| step span under nsys | 20.99 ms | 20.00 ms |
| `cuMemcpyDtoDAsync` per step | 414 (13.35 MB, 3.08 ms API) | 141 (0.56 MB, 0.86 ms API) |
| memcpys per step, dev0 / dev1 | 229 / 237 | 97 / 96 |
| `dsv4_dense_fast_dots_kernel`, dev0 / dev1 | 0.535 / 1.217 ms | 0.367 / 1.034 ms |
| kernel launches per step | 2854 | 2854 |

The copy diet removes 273 device copies and 12.8 MB per step. The BF16 plane cuts the dots
kernel's device time by 0.17 to 0.18 ms per card per step. The launch count is unchanged; the
step remains launch- and bubble-bound (the plain-now rows in `raw/prof/`).

`raw/prof/{dspark-main,plain-now,dspark-now}` are the same campaign's anatomy of the DSpark route
on this lane's base and of the then-current tree on both routes, kept for the ceiling write-up.
The "now" binary is `lane/dsv4-brow-serve-20260924` `46c259a2a` (main `a8455d29f` with sink and HC2,
plus the #699 pipelining and B-row lanes) run as the serial route: `MEMRA_DSV4_SESSIONS=1`, B-row
off.

## Decision

Default, no door: the change is bit-identical at the kernel boundary and in served text, and it
wins on both routes. Nothing to flip and nothing to keep.
