# The 16-row decode tick on GB10: exact limit reached, non-exact options

For the owner's decision, 2026-11-04. Model: Qwen3.8-27B NVFP4 + Q5_K, one GB10. Depth 12288 is the served agentic cell's median turn. Receipts: `LEVER-LEDGER.md` rows 1-33, `B16-MATMUL-DESIGN.md`, `GB10-PLAIN-DECODE.md`.

## Where it stands (exact)

| 16-row tick, depth 12288 | start of day | now |
|---|---:|---:|
| tick | 173.3 ms | 161.7 ms |
| matmul | 91.5 | 86.1 |
| attention + combine | 48.5 | 48.1 |
| GDN state (fp32) | 22.4 | 22.7 |
| memset + gaps | 7.2 | 1.0 |

The 16-row matmul is 14 ms over the 8-row tier for the same 15.7 GB of weights:
- singles 61.9 vs 53.1 ms;
- the group door 17.7 vs 13.9;
- the Q5_K head 6.6 vs 5.0.

Every exact lever has been measured:
- 8 dp4a schedules;
- an activation tile in smem;
- compile-time width with 1/2/4/8-column load groups (2 adopted: -3.4%);
- the group door twins;
- the tensor-core integer part at 16 rows with 1/2/4 chains.

The rest are flat or slower. The cause, from ncu:
- At 16 columns the exact float chain (one `partial`/`acc` update per row, column and 32-element group, in lane order) is most of the instructions.
- Its registers cap residency at about 19 warps per SM.
- Hiding more latency costs residency, and the tensor-core form adds instructions at this width.

## Non-exact options

Each changes the numeric program, so each needs the quality panel (HumanEval 164 + GSM8K 100 after a 131K session) and a new token-fingerprint baseline. The "16-row tick" column is the expected change at depth 12288. It is an estimate from the measured part times, not a measurement.

| option | what changes numerically | expected 16-row tick | expected served effect | cost / risk |
|---|---|---|---|---|
| A. bf16 GDN state (built, `research/combo-tc-gdnbf16`) | state stored as bf16, math fp32 | -11 ms (measured: 173.5 -> 162.1 before the other levers) | served, 2 runs each against fp32: c16 +2.9%, c32 +2.6%, c8 -3.2% (60.2 vs 62.3); quality 160/164, 97/100 vs fp32 158/164, 97/100 | ready |
| B. matmul with a different reduction order (e.g. a K-split across warps or 4 rows per warp with a tree reduce) | float sums per (row, column) reassociated | up to -14 ms (to the 8-row tier's time) | c16 up to about +9%; makes speculation pay at c8 by the cost model (16-row tick about 148 vs the 1.49 x 110 = 164 break-even) | a new kernel; batch-invariance across widths is lost unless every width uses the same order |
| C. attention with a smaller tile / different split | online-softmax summation order | up to -11 ms (core at 138 GB/s; KV + partials at 226 GB/s is about 31 ms) | grows with depth: about -25 ms at depth 32768, 8 rows | moderate; one kernel family |
| D. FP8 or smaller KV format | KV values | attention bytes down by the format ratio (q8_0/q5_1 today) | largest at long context | changes every model's KV path; acceptance and quality rows per model |
| E. tensor-core GEMM tier (MMQ) for 16..32 rows (measured, rejected) | GEMM order | +68 ms at 16 rows (slower) | none | dropped |

The exact ceiling for the 16-row tick is about 162 ms. With A + B + C together the estimate is about 125 ms, against vLLM's implied 16-row cost on this box: 121 tok/s at c16 is about 132 ms per 16 tokens of plain decode.

## Recommendation

- **A** is built and measured: it trades about 3% at c8 for about 3% at c16 and c32.
- **B** is the largest remaining lever on the matmul and the one that unlocks speculation at c8. Build it next only if you accept a non-batch-invariant decode matmul, or one order shared by all widths.
- **C** is the long-context lever.

## Measured: option C, attention partition (2026-11-05)

Arm: the existing `MEMRA_FA_SPLIT` knob forced to 256 keys per split at every depth, on the default
binary (655a07cb82), default untouched (the ladder is 64 keys up to depth 16,384 and 128 above).
Receipts `gb10-optC/`, branch `research/attn-split-c`.

Decode tick, GB10, one binary (`gb10-optC/sweep/ab.txt`; 16 x 12288 in 2+2 runs per split, both orders):

| cell | default | split 128 | split 256 | split 512 |
|---|---:|---:|---:|---:|
| 16 sessions, depth 12288 | 161.74, 161.62 | 149.56, 149.06 | **145.24, 145.29** | 144.82, 147.21 |
| 8 sessions, depth 32768 | 133.02, 133.27 | | **127.24, 127.50** | 125.79, 125.91 |
| 32 sessions, depth 12288 (dp4a b32) | 325.15 | | **291.20** | |

256 was kept over 512: equal at depth 12288, and a fixed 512 leaves 4 splits per head at depth 2,048.

Served agentic cell, fresh boot, c1..c32, then the 264-task quality panel:

| c | default now (fp32, v5 gate, 2 runs) | split 256 | change |
|---:|---:|---:|---:|
| 1 | 22.65 | 22.81 | +0.7% |
| 2 | 28.65, 27.83 | 30.81 | noise band |
| 4 | 39.52, 38.55 | 40.60 | +3% |
| 8 | 62.12, 62.40 | 63.11 | +1.3% |
| 16 | 77.28, 77.33 | **83.70** | **+8.3%** |
| 32 | 83.75, 83.17 | **90.65** | **+8.6%** |

Quality: long HumanEval 156/164 (1 truncated), long GSM8K 97/100, inside the default's band
(156-159, 96-97). Divergence against exact: greedy coherence probes identical on 5 of 8 prompts
(the rest diverge after 560-940 characters); 48-token greedy decode streams identical on 13/16
sessions (16 x 12288), 5/8 (8 x 32768) and 23/32 (32 x 12288). The change is a different
partition of the same online-softmax sums, so divergence is rounding-order only.

## Measured: option B, decode matmul on the FP4 tensor-core MMA (2026-11-05)

Arm: `MEMRA_W4A4_DECODE=1` on `research/w4a4-decode` (v1 13831108a3, v2 0a2dc378a1). Rows 2..32 of
every NVFP4 decode matmul requantize the activations to FP4 per 16-element block (ue4m3 scale) and
run `mma.sync kind::mxf4nvf4 m16n8k64`. This is a different numeric program, not a reordering:
the activations lose precision, not only the summation order. Receipts `gb10-optB/`.

Kernel accuracy against the exact dp4a chain (27B shapes): relative L2 error 10.4-10.6%, cosine
0.9945. Batch invariance: one MMA order serves every width 2..32, and a column's output is
bit-identical at widths 8, 13 and 16 (`local-5090/`). The 1-row path stays on the exact kernel, so a
session's tokens change when it moves between 1 row and 2+ rows: invariant across widths 2..32,
not across 1 vs many.

Matmul bench (3 tensors of the 27B FFN, median ms, exact -> W4A4 v2):

| rows | RTX 5090 | GB10 |
|---:|---:|---:|
| 8 | 2.26 -> 1.42 | 4.71 -> 6.73 |
| 16 | 3.34 -> 1.34 | 5.82 -> 6.22 |
| 32 | 6.80 -> 2.33 | 9.94 -> 6.46 |

On the 5090 it is 1.6-2.9x faster. On GB10 it loses at 8 and 16 rows and wins only at 32.

Decode tick, GB10, one binary, flag off/on (`ticks-v1/ab.txt`, `ticks-v2/ab.txt`):

| cell | exact | W4A4 v1 | W4A4 v2 |
|---|---:|---:|---:|
| 16 sessions, depth 12288 | 161.66, 162.05 | 199.03, 199.41 | 186.92, 188.14 |
| 8 sessions, depth 32768 | 133.44 | 186.84 | 179.29 |
| 32 sessions, depth 12288 | 325.08 | 300.09 | 277.29 |

nsys at 16 x 12288 (v2, `ticks-v2/p-w1-kern.csv`): the W4A4 matmul takes 102.65 ms per step
against 86.1 ms for the exact matmul; the requantize kernel adds 1.76 ms. On GB10 the FP4 MMA
path is bound by the scale and fragment loads, not by the MMA, and 16 columns do not fill an m16n8
tile pair enough to pay for them.

Served agentic cell (v1 kernel, cb33, fresh boot, c1..c32), then the 264-task quality panel:

| c | default now (fp32, v5 gate) | W4A4 v1 | change |
|---:|---:|---:|---:|
| 1 | 22.65 | 17.18 | -24% |
| 2 | 28.65, 27.83 | 24.30 | -14% |
| 4 | 39.52, 38.55 | 31.53 | -19% |
| 8 | 62.12, 62.40 | 43.58 | -30% |
| 16 | 77.28, 77.33 | 60.11 | -22% |
| 32 | 83.75, 83.17 | 76.23 | -9% |

c1 drops too because spec verify runs at 2+ rows: the boot calibration measured verify at 2 rows
103.2 ms against 77.0 ms at 1 row (`served/server-lines.txt`). The v2 served run was stopped before
its first cell when b23d went to the qualification run; the v2 ticks above say it would stay below
exact at every c up to 16.

Quality: long HumanEval 155/164 (1 truncated), long GSM8K 96/100, at the bottom of the default's
band (156-159, 96-97). Divergence against exact: coherence probes identical on 3 of 16 records
(first difference after 24-423 characters); 48-token greedy streams identical on 8/16 sessions
(16 x 12288), 1/8 (8 x 32768) and 14/32 (32 x 12288).

Verdict: rejected on GB10. Slower at every served c, larger divergence than C, quality at the
band's floor. The kernel is a real lever on the 5090 (16 rows 2.5x faster); keep the branch for
an RTX arm, where it needs its own served run and quality panel.
