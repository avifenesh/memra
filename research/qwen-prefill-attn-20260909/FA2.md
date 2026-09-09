# Qwen FA2 prefill qualification

`MEMRA_PRIME_ATTN_FA2` is a default-OFF experiment on sm_120a, decide-by
2026-09-23. It preserves the 24 Q / 4 KV / d256 geometry and shares staged K/V
across the six query heads. The numerical program changes through direct FP32
PV accumulation and the BF16-rounded softmax denominator. No external kernel
is linked or shipped.

Standalone attention reaches 144.09 TF/s at 1024 x 32768 and 146.15 TF/s at
1024 x 131070, against 111-112 TF/s for the existing kernel. The 8k shape reaches
139.24 TF/s, accepted by the owner. See `fa2/CHECKPOINT.md` for the original
microbench and its numerical boundary.

The owner-selected chunk-1024/512 final-chunk calibration is byte-identical.
A second isolation control retains main's denominator and scale but changes
only PV accumulation order: max/RMS logit delta 13.7485/1.6584, alongside FA2's
13.8527/1.5964. The new staging/layout with main's complete arithmetic order is
byte-identical through final logits. These controls support accumulation-order
amplification; serving quality is decided by the independent gates.

The first integration failed cold/restored greedy identity because t<128 and
widened tails fell back to the old numerical class. The corrected dispatch
covers all qualified prefill chunks, including t=16..1039. Decode is unchanged.
The corrected ON program passes the four-turn cold/restored greedy twin and
cold/restored boundary capture. Its frozen-prompt margin gate has zero flips.
The remaining paired timing, evaluation and cache receipts are being collected.
No default flip is proposed while those gates are pending.

All GPU gates and builds run on the designated non-serving 5090, architecture
120a, with the canonical lock and an empty compute list before each job.
RUSTC_WRAPPER is empty for CUDA rebuilds. Local rig gates are prohibited by the
owner: pushes set MEMRA_SKIP_PERF_CI=1; hosted CI still gates the merge.
