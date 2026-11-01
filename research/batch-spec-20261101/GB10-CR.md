# GB10 C(R) and pooled verify: Qwen3.8-27B, owner GGUF

Date: 2026-11-02, 00:29-00:55 Jerusalem.

- **Device:** one DGX Spark (GB10, sm_121a), whole box, under the batch-spec slot.
- **Model:** `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` (memra's NVFP4 + Q5_K trunk).
- **Binary:** `decode-batch-bench`, built on the box from batch-spec at c1cd81c343 plus the bench
  commits, sha256 `3abe957d9f7df2be86be0f1ada5b9665f31798e6bbafd513d4d1f0d1dd19646a`.
- **Raw logs:** `gb10-cr/`.
- **Stopped early:** the run ended when the owner paused the memra Spark effort. Not measured:
  B=12/16 plain batching, verify at 16k depth, and the NVIDIA safetensors arms.

## Verify cost over R rows, one session, depth 2048 (`--verify-t`, median of 9)

| R | ms | x R=1 |
|---:|---:|---:|
| 1 | 75.16 | 1.000 |
| 2 | 74.34 | 0.989 |
| 3 | 74.46 | 0.991 |
| 4 | 75.54 | 1.005 |
| 5 | 76.18 | 1.014 |
| 6 | 77.28 | 1.028 |
| 8 | 80.72 | 1.074 |
| 10 | 94.88 | 1.262 |
| 12 | 107.81 | 1.434 |
| 14 | 121.98 | 1.623 |
| 16 | 135.97 | 1.809 |
| 17 | 1184.48 | 15.76 |

Reading the curve:

- Up to 8 rows cost the same as one row: the 27B step is bound by weight reads.
- Rows 9-16 (the b16 tier) add about 7 ms each. That tier is the next kernel lever on GB10.
- At 17 rows the verify falls to the rowwise arm, so it must never be reached.

## Plain batched decode, depth 2048 (`--batches`, median of 3)

| B | aggregate tok/s | ms/step |
|---:|---:|---:|
| 1 | 13.5 | 74.1 |
| 2 | 25.9 | 77.2 |
| 4 | 48.9 | 81.8 |
| 8 | 84.9 | 94.2 |

## Pooled cross-session verify vs serial per-session verify (`--pooled`)

Every shape is BITWISE EQUAL to per-session verify. The check covers every logit row, plus one
follow-up step that reads the state each path wrote.

| sessions x rows | R | depth | serial ms | pooled ms | gain |
|---|---:|---:|---:|---:|---:|
| 2 x 2 | 4 | 2048 | 148.86 | 79.00 | 1.88x |
| 4 x 4 | 16 | 2048 | 303.95 | 145.95 | 2.08x |
| 5 x 3 | 15 | 2048 | 375.64 | 142.45 | 2.64x |
| 1,2,3,4,5 | 15 | 2048 | 379.36 | 143.65 | 2.64x |
| 8 x 2 | 16 | 2048 | 600.02 | 160.72 | 3.73x |
| 2 x 8 | 16 | 2048 | 161.83 | 138.12 | 1.17x |
| 4 x 4 | 16 | 16380 | 367.56 | 208.28 | 1.76x |

## What it means for the gate

On GB10 a pooled verify with R <= 8 rows costs about one plain decode step (75-81 ms against
74-94 ms). Pooling therefore turns every accepted draft token into nearly free throughput.

- 4 sessions at K=1 (8 rows) cost about 81 ms per round and produce 4 x (1 + acceptance)
  tokens. Plain batching at B=4 produces 4 tokens in 82 ms.
- Past 8 rows the b16 tier charges about 7 ms per row. A cost-model gate should fill R up to 8
  first, then weigh rows 9-16 against acceptance.

The serial spec path that serves today is the "serial" column above. Pooling removes most of
that cost.
