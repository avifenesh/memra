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
| A. bf16 GDN state (built, `research/combo-tc-gdnbf16`) | state stored as bf16, math fp32 | -11 ms (measured: 173.5 -> 162.1 before the other levers) | served c16 +2.0%, c32 +3.5% (one run); quality 160/164, 97/100 vs fp32 158/164, 97/100 | ready; c8 depends on the gate fix being verified now |
| B. matmul with a different reduction order (e.g. a K-split across warps or 4 rows per warp with a tree reduce) | float sums per (row, column) reassociated | up to -14 ms (to the 8-row tier's time) | c16 up to about +9%; makes speculation pay at c8 by the cost model (16-row tick about 148 vs the 1.49 x 110 = 164 break-even) | a new kernel; batch-invariance across widths is lost unless every width uses the same order |
| C. attention with a smaller tile / different split | online-softmax summation order | up to -11 ms (core at 138 GB/s; KV + partials at 226 GB/s is about 31 ms) | grows with depth: about -25 ms at depth 32768, 8 rows | moderate; one kernel family |
| D. FP8 or smaller KV format | KV values | attention bytes down by the format ratio (q8_0/q5_1 today) | largest at long context | changes every model's KV path; acceptance and quality rows per model |
| E. tensor-core GEMM tier (MMQ) for 16..32 rows (measured, rejected) | GEMM order | +68 ms at 16 rows (slower) | none | dropped |

The exact ceiling for the 16-row tick is about 162 ms. With A + B + C together the estimate is about 125 ms, against vLLM's implied 16-row cost on this box: 121 tok/s at c16 is about 132 ms per 16 tokens of plain decode.

## Recommendation

- **A** is built and measured: adopt it when its c8 is confirmed.
- **B** is the largest remaining lever on the matmul and the one that unlocks speculation at c8. Build it next only if you accept a non-batch-invariant decode matmul, or one order shared by all widths.
- **C** is the long-context lever.
