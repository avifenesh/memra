# memra#1327's four-card take on the landing head, 2e42b5549830

Scoped extras for the landing of memra#1315, memra#1327, memra#1308 and memra#1326. They are not a cell record; the gate does not read them. The push evidence for `crates/memra-engine/src/dsv41_gpu/tp.rs` (numeric and perf cells) is the device records next to this directory; the device battery's roster holds no DeepSeek-V4.1 model and runs one card, so the four-rank rule memra#1327 changes is shown here.

## Why on the landing head

memra#1327 (memra#1311: the split MoE keeps the bf16 gather while a rank's stores fit a push, and gathers codes above 9 rows) passed its four-card confirm on its own binary. The landing head also holds memra#1315, which changes `cu/dsv41_gpu.cu` and the same `tp.rs`, so that binary does not stand for the head. A pair does not reach the rule: on two ranks the path is main's.

## What ran

All four cards of an RTX PRO 6000 Blackwell Max-Q workstation, 2026-11-18, 01:51:46 to 01:56:59 local time, all four cards held for the whole window. DeepSeek-V4.1-Flash, 40 layers, `MEMRA_FULL_PREC=1`, `MEMRA_DSV41_TP_ISSUE=parallel`. The commands are in `land27-four/inner.sh`; the timeline is in `land27-four/runs.txt` and `w11-l27/summary.txt`.

- Arm A: main 5a6ae2438e. Arm B: the landing head 2e42b55498. Both built on the box that ran them, CUDA 13.4, sm_120a (`land27-four/arms.txt`).
- Gates of B first (untimed): the kernel gate, then the four-rank `tp` gate under each transport.
- Then eight timed runs of `dsv41-tpbench` on the four cards, in the order A B B A B A A B, each with calls of 4, 8, 9, 10, 12 and 16 rows and one-row decode.

## Results

| gate (arm B) | result |
|---|---|
| `dsv41-gate kernels` | 1,027 checks, 0 failed |
| `dsv41-gate tp`, default transport | 1,056 checks, 0 failed |
| `dsv41-gate tp`, peer-pull | 1,054 checks, 0 failed |
| `dsv41-gate tp`, host-canonical | 1,054 checks, 0 failed |
| `stress/push_gather` | 20,000 gathers on four devices, 0 words differ, 0 ranks flagged |

Steady call time, median ms, from `land27-four/steady.md` (calls from 0.5 s into a run, 44 calls per row count and arm, 1,008 to 1,012 for the one-row decode):

| call | A, main | B, head | B / A | the confirm's criterion |
|---|---:|---:|---:|---|
| decode, 1 row | 9.640 | 9.617 | 0.998 | 1.000 +/- 0.003 |
| 4 rows | 13.055 | 13.082 | 1.002 | 1.000 +/- 0.003 |
| 8 rows | 16.617 | 16.623 | 1.000 | 1.000 +/- 0.003 |
| 9 rows | 18.904 | 18.892 | 0.999 | 1.000 +/- 0.003 |
| 10 rows | 20.680 | 19.623 | 0.949 | at most 0.953 |
| 12 rows | 21.865 | 20.638 | 0.944 | at most 0.950 |
| 16 rows | 24.733 | 23.573 | 0.953 | at most 0.959 |

Every row is inside its criterion. The decoded tokens are one stream across the eight runs (`decode tokens: decode-c1: 1 distinct over the runs`). The token files themselves are left out; their digests are in `land27-four/BINARIES.sha256`.

## Recomputed without the disturbed runs

A CPU build of this landing ran on the box until 01:54:10: through timed run 1 (arm A, 01:53:20 to 01:53:48) and the first 22 s of run 2 (arm B, 01:53:48 to 01:54:18). Run 1 is 0.3 to 0.6% slow at every call; run 2 is in line with the other B runs. The take was not rerun. `recompute.py` recomputes the ratios from the per-call records (`land27-four/runs/*/cells.jsonl`) with chosen runs left out: the pooled median per arm over the kept runs, the first round of each run dropped, as `steady.md` does. Over all eight runs it gives `steady.md`'s figures exactly (`recompute-all.txt`).

| call | criterion | all 8 runs | run 1 (A) left out | runs 1 (A) and 2 (B) left out |
|---|---|---:|---:|---:|
| decode, 1 row | 1.000 +/- 0.003 | 0.9976 | 0.9990 | 0.9989 |
| 4 rows | 1.000 +/- 0.003 | 1.0021 | 1.0021 | 1.0022 |
| 8 rows | 1.000 +/- 0.003 | 1.0004 | 1.0008 | 1.0007 |
| 9 rows | 1.000 +/- 0.003 | 0.9994 | 0.9999 | 0.9997 |
| 10 rows | at most 0.953 | 0.9489 | 0.9495 | 0.9491 |
| 12 rows | at most 0.950 | 0.9439 | 0.9443 | 0.9443 |
| 16 rows | at most 0.959 | 0.9531 | 0.9537 | 0.9535 |

Every criterion holds in all three columns (`recompute-drop1.txt`, `recompute-drop12.txt`, with the per-run medians; `RECOMPUTE.md` is the note that came with them). memra#1327's own four-card confirm on 2026-11-17, on its own binary, read 0.949, 0.944 and 0.954 at 10, 12 and 16 rows.

## Conditions

- Power limits of the four cards: 270, 266, 281 and 276 W. Peak draw 217 to 246 W per card, peak temperature 59 to 71 C (`land27-four/lease.telemetry.csv`).
- No thermal slowdown: the counters of every card are equal before and after each step (`land27-four/slowdown.txt`, 12 runs, 0 flagged).
- Host load (1-minute) 3.21 at the start and 4.66 at the end. A CPU build of this landing ran on the box until 01:54:10, on 16 host threads: through timed run 1 (arm A) and the first 22 s of run 2 (arm B). The other six runs had no build beside them; the section above gives the ratios without those runs.
- The absolute times are times under these cards' power caps on this build; the comparison is the interleaved ratio.
