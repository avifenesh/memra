# Standalone FA2 checkpoint, 2026-09-09

Historical checkpoint before serving integration; current qualification is in `../FA2.md`. The native grouped-head candidate clears
140 TF/s at 32k and 131k, but misses at 8k. Real final-chunk logit deviation needs
diagnosis before a quality decision.

| Shape | Current ms | Candidate ms | Candidate TF/s | Offline FlashInfer bar ms |
| --- | ---: | ---: | ---: | ---: |
| 1024 x 8192 | 1.780320 | 1.388160 | 139.24 | 1.198048 |
| 1024 x 32768 | 7.300192 | 5.633632 | 144.09 | 4.916352 |
| 1024 x 131070 | 29.242752 | 22.480703 | 146.15 | 20.000160 |

Nine interleaved CUDA-event samples after three warmups, one process on sm_120a.
FlashInfer is the previously measured offline bar. These are isolated kernel
times, not TTFT. Native candidate: 24 Q / 4 KV / d256, shared GQA tile, BF16 MMA
with FP32 accumulation, 32-key online softmax, MMA denominator, direct exponential,
causal block skipping and three rotating cp.async staging planes. Occupancy is
two CTAs/SM, 255 registers, 49152 shared bytes and 40 local bytes per thread.

The offline probe uses baseline prefix state and changes only the final 1024 rows
at depth 32768, with 16 attention dispatches. All 248320 logits are finite.
Maximum logit delta is **13.8527069092**, RMS **1.5964384079**. The final argmax
matches, but this one wide-margin position does not pass the shared margin gate.
The baseline twin is byte-identical and the captured first attention Q/K/V bytes
match across arms. On those real inputs, the standalone candidate takes 5.627584 ms
(144.25 TF/s), maximum attention-output delta 0.00209737, relative L2 0.00010465.

Candidate fatbin SHA256:
`04236d8165f8be0bd7e5f8face19fe8430fab8c36b6f693c482428d9998ef1a4`.
All 99 baseline global entries remain; six research entries are added.
Nsight Compute counters are unavailable on the box (ERR_NVGPUCTRPERM).
No FA2 serving door or dispatch is integrated. The numerics-changing serving
protocol remains pending. No local rig gates ran, and no push has occurred yet.

Standalone sources and the offline snapshot patch are banked in the companion research archive; only the selected implementation and reusable gates remain in this PR. Detailed receipts are
banked in the companion research lane at commit `d03a9c2ae`. The GPU compute list
is empty and the shared lock is free. Scratch and worktrees are retained for the
checkpoint decision.
