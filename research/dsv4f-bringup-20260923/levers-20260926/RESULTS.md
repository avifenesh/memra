# TP/EP decode levers: PDL chain, vocab-parallel head, snapshot kernel, push joins (memra #710, 2026-09-26)

Model: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`. Program: the
served TP/EP default (exact attention TP2, expert-id EP pair, full-token replay graphs, four lanes
with B-row graph steps, position-split C4 store). Hardware: 2x RTX PRO 6000 Blackwell Server
Edition (SE pair). One scored campaign at a time under `/tmp/memra-gpu.lock`.

These four lanes stack, and each is measured against the one below it in a single interleaved
window. Every lever is a scheduling or placement change: each keeps the kernels' arithmetic, so
the gates compare bits, not tolerances.

## What changed

- **PDL chain** (`MEMRA_DSV4_PDL`, `lane/dsv4-pdl-20260926`).
  - Every DSv4 decode-chain kernel is launched through `memra_chain_launch`, which adds the
    programmatic stream serialization attribute.
  - Every such kernel opens with `griddepcontrol.wait`, then `launch_dependents`. The wait holds
    each thread until the previous grid has completed and flushed. The next kernel's launch then
    overlaps this one's run, and the kernel's own work is unchanged.
  - `tools/check-pdl-chain.py` (CI) refuses a chain-launched kernel without the entry.
- **Vocab-parallel head** (`MEMRA_DSV4_VOCAB_HEAD`, `lane/dsv4-vocab-head-20260926`).
  - Rank r computes the logits of vocab rows `[r V/2, (r+1) V/2)` from its own copy of the
    replicated final residual and head.
  - One row gather lays both halves out in vocab order on both ranks.
  - It covers the one-row eager and replay steps and the B-row eager and graph steps. Before, rank
    1 alone paid about 0.6 ms of head per step while rank 0 waited.
- **Snapshot kernel** (`lane/dsv4-snap-copy-20260926`, no door). A compressor checkpoint's two
  snapshots are one PDL-chained copy kernel instead of two memcpy nodes: 62 compressors per rank
  per step, and a memcpy node breaks the launch chain.
- **Push joins** (`MEMRA_DSV4_AR_PUSH`, `lane/dsv4-ar-push-20260926`): one fabric crossing per
  join and no end barrier (section below).

## Correctness

The long gate is `dsv4_tp_replay_long_gate`: 304 replayed steps from a 400-token prefix, compared
with the eager program on every step's token, logits bits and cache and hidden digests. It now also
prints `PROGRAM_SHA256`, one hash over all of that. On the SE pair (`raw/se-v5k/levers-v5k/`):

| arm | binary | doors | PROGRAM_SHA256 |
|---|---|---|---|
| Z | vocab-head lane | PDL=0, VOCAB_HEAD=0 | `fbce1a0492d69635` |
| A | vocab-head lane | PDL=1 | `fbce1a0492d69635` |
| B | vocab-head lane | PDL=1, VOCAB_HEAD=1 | `fbce1a0492d69635` |
| C | snapshot lane | PDL=1, VOCAB_HEAD=1 | `fbce1a0492d69635` |

- **Other rigs.** The same hash on the WS pair, PDL on and off (`raw/ws-pdl-50722e199/`), and on a
  second SE pair.
- **Other gates on C.** `dsv4_rows_gate` TP/EP and `dsv4_kv_split_gate` are bit-identical, and the
  DSpark TP/EP gate keeps its proposal shas.
- **Served text.** Every greedy request of every row gives one sha list per cell across all four
  arms.

## Served (SE pair, one boot per row, order Z A B C C B A Z repeated, N=5 per arm)

`raw/se-v5k/`, cells `cells-pdl.txt`. Medians with the range over five boots, aggregate tok/s:

| cell | Z | A (PDL) | B (+vocab head) | C (+snapshot kernel) | C vs Z |
|---|---|---|---|---|---|
| greedy c1 | 67.53 (67.50..67.91) | 70.90 (70.87..71.30), +4.99% | 72.63 (72.48..72.95), +2.44% | 73.31 (73.07..73.55), +0.94% | +8.56% |
| sampled c1 | 68.65 | 72.00, +4.88% | 73.94, +2.69% | 74.45, +0.69% | +8.45% |
| greedy c2 | 93.74 | 97.92, +4.46% | 99.73, +1.85% | 100.41, +0.68% | +7.12% |
| greedy c4 | 123.39 | 128.62, +4.24% | 129.56, +0.73% | 131.18, +1.25% | +6.31% |
| c2, 1500-word context | 16.89 (TTFT 12.08 s) | 17.49 (11.52 s) | 17.54 (11.52 s) | 17.58 (11.50 s) | +4.09% |

**By run order** (greedy c1, the gain within each ABCD block):

| lever | forward blocks | reverse blocks |
|---|---|---|
| PDL | +5.04% | +4.92% |
| vocab head | +2.31% | +2.48% |
| snapshot kernel | +0.82% | +0.94% |

At c1 each arm's range clears the one below it.

**WS pair.** It carried the first PDL rows: +3.2% greedy c1 at N=2 per arm
(`raw/ws-pdl-50722e199/`). Then its PCIe link failed mid-window. The pod read 49,181 and 26,349
replays (18 rollovers) against 0 on the SE pairs, and every later boot failed the P2P byte-integrity
probe, so the WS rows stop at N=2. A WS confirmation row is each door's decide-by gate.

## Push joins

`MEMRA_DSV4_AR_PUSH`, `lane/dsv4-ar-push-20260926`, `cu/tp_ar.cu`.

**What changed.**
- **The pull join it replaces** crosses the PCIe fabric three times per join: a start barrier, a
  non-posted peer read of the operand, and an end barrier that keeps each rank's input alive until
  the peer has read it.
- **Push join.** Each rank writes its operand into the peer's output buffer (posted writes),
  fences, and raises one flag in the peer's signal block. It then waits on its own local flag and
  reads only local memory. That is one crossing per join, with no end barrier.
- **Arithmetic.** The reduce computes `in_rank0 + in_rank1` on the same values, and the gathers
  move the same bits.
- **Why no end barrier is needed.** Each walk join (wo_a gather, attention-out gather, expert
  reduce, vocab head gather) writes its own persistent buffer, and every consumer of that buffer
  runs before the next join. `TpEpArState` refuses a push join that writes the previous push
  join's output. Joins on temporary buffers (the DSpark EP combine) stay pull.
- **The race the first cut had.** The peer can finish join k and raise join k+1's flag in the same
  slot before this rank has read join k's. The wait is therefore "at least this epoch", as a
  signed difference so the counter can wrap. A later epoch still means this join's push has
  landed, and the peer cannot run two joins ahead. The first cut waited for an exact epoch and
  refused `[40043, 0]` in the long gate's back-to-back timing phase (`raw/se2-push-s2c/`, the
  voided run).

**Correctness.**
- On the second SE pair, `PROGRAM_SHA256` equals the pull joins' (`fbce1a0492d69635`), through the
  long gate's timing phase.
- `dsv4_rows_gate` TP/EP and `dsv4_kv_split_gate` are bit-identical.
- The DSpark TP/EP gate gives the same proposal shas as the pull control on the same pod.
- Every greedy served text is identical in both A/Bs.

**Served.**

| pair, protocol | cell | pull | push | delta |
|---|---|---|---|---|
| second SE pair, C D D C x5 boots (`raw/se2-push-s2c/`) | greedy c1 | 71.64 (71.23..71.81) | 77.19 (77.09..77.73) | +7.75% (forward +8.37%, reverse +7.57%) |
| | sampled c1 | 71.51 | 77.33 | +8.14% |
| | greedy c2 | 95.99 | 104.42 | +8.78% |
| | greedy c4 | 125.95 (125.60..127.56) | 130.44 (105.47..133.60) | +3.56% |
| SE pair, one binary, P Q Q P x4 boots, 5 c4 cells per boot (`raw/se-push-c4-v5p/`) | greedy c1 | 73.20 | 77.56 | +5.96% |
| | greedy c2 | 100.16 | 106.29 | +6.13% |
| | greedy c4, 20 cells per arm | 130.75 (129.51..131.40) | 135.10 (134.51..135.71) | +3.32% |

**Unexplained.** On the second SE pair, two of five push boots ran c4 about 20% slow (105.47 and
111.26 aggregate, TPOT 36 and 34 ms against 28), with c1 and c2 in the same boots normal and SM
clocks unchanged. It did not reproduce in 20 push c4 cells over four boots on the SE pair, and the
pull arm never showed it. Cause unknown; recorded here, not explained away.

The long gate's replay timing on the second SE pair agrees: 13.46 ms per token with pull and 12.41
with push.

## Profiling note

With PDL on, a kernel starts while its predecessor still runs and waits at its entry. An nsys
kernel duration then includes the predecessor's run, so the per-kernel attribution of a trace
taken with the door on is not a cost table. Anatomies run with `MEMRA_DSV4_PDL=0`.
