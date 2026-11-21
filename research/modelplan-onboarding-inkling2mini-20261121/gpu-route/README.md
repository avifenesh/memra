# Inkling 2 Mini: native GPU route, checkpoint gates (2026-11-21)

The route is `crates/memra-engine/src/inkling2_gpu.rs` with the kernels of `cu/inkling2_gpu.cu` and the
act-quant expert kernels of `cu/inkling2_actq.cu` (memra#1396). Runner: `ink2-gpu`; kernel cells:
`ink2-gpu-gate` (also in `kernel-check`). The oracle tapes are scratch evidence outside the repo
(transformers PR 51050 model code on CPU, the checkpoint's expert bytes dequantized exactly; the act-quant
arm is a CPU simulation of the declared program). Each directory holds the batch script, its log, the
receipts and the sha256 of the binaries that ran.

Card: RTX 5090 Laptop GPU, 24 GB, sm_120a, driver 615.71.09. Weights resident as stored: 17.594 GiB
(BF16 trunk, NVFP4 experts on layers 0 to 19, MXFP8 on 20 to 23); 18.86 GiB with the per-slot state at
context 2048, 19.84 GiB at 8448. The drafter file is read through the pack's auxiliary files and its 54
tensors are bound, not uploaded. Vision is not loaded.

## Verdicts

| Gate | Class | Result | Receipt |
|---|---|---|---|
| Kernel cells (16): BF16 / NVFP4 / MXFP8 GEMM and GEMV, embed and norms, depth mix, controller, head epilogues, static and dynamic conv with carried state, sliding and full attention with the sink, router ties, SwiGLU, sums, softcap head, declared expert block | both | ALL OK. The declared expert block is bit-identical to `memra_reference::inkling2_actquant::expert_block_*` (2304/2304 bf16 outputs, NVFP4 and MXFP8) | `b4/gate.txt` |
| p1 (26-token chat, 57 positions) vs `exact_f32` | float | worst slot residual rel L2 8.2e-7, logits KL mean 2.0e-12, argmax 57/57, 32/32 greedy ids | `b3/p1-float.json` |
| p2 (1500 tokens, past the 1024 window and extent) vs `exact_f32` | float | worst slot 8.4e-4, KL mean 2.2e-7, argmax 1531/1531, greedy 32/32 | `b3/p2-float.json` |
| p3 (8260 tokens, crosses the 8192 log-scale floor) vs `exact_f32` | float | slots of pass 1 at or under 5e-4, drift grows through passes 2 and 3 to 0.11 on the last slots; KL mean 8.4e-4, argmax 178/179, greedy 32/32 | `b5/p3-float.json` |
| Per MoE with forced inputs (tape `moe_in`, route ids and weights) vs `actq_f32` | declared | p1: 41 of 48 slots bit-equal, worst 6.2e-4; p2: worst slot 4.7e-4, median 1.2e-4; end to end under forcing KL 2e-8, greedy 32/32 | `b4/p1-declared-forced-moe.json`, `b4/p2-declared-forced-moe.json` |
| End to end vs the float oracle | declared | KL mean p1 0.045, p2 0.39, p3 0.029; argmax 54/57, 1317/1531, 176/179; teacher-forced greedy 31/32, 32/32, 32/32. The CPU simulation of the same program reads 0.062 / 0.43 / 0.023 against the same oracle | `b4/p1-declared-vs-float.json`, `b4/p2-declared-vs-float.json`, `b5/p3-declared-vs-float.json` |
| Chunking invariance (prefill chunks vs one token per call) | float | 26 vs 1 over 57 rows and 512 vs 300 over 1500 rows: 0 slot rows, 0 logit rows differ | `b3/p1-float.json`, `b3/p2-float.json` |
| Chunking invariance | declared | 26 vs 1 and 512 vs 300: 0 differ | `b4/p1-declared.json`, `b4/p2-declared-vs-float.json` |
| Greedy decode on the device argmax (run-gen form) | float | 32/32 equal to the oracle's greedy ids, three takes | `b5/gen-float-*.json` |

Red arms (the gate must reject each):

| Arm | Against | Result |
|---|---|---|
| `dequant-bf16` (expert weights cast to BF16, the PR's stock load) | p1 / p2 `exact_f32` | worst slot 2.2e-2 / 0.33, KL 2.1e-4 / 0.21, against 8.2e-7 / 8.4e-4 for the route. Against the `bf16w_f32` tape the arm matches at 8.3e-7, so the arm is the program it claims to be |
| `conv-state` (the short convs drop their carried state between calls) | p1 `exact_f32` | worst slot 0.35, KL 2.3e-2, argmax 55/57 |
| `ties-high` (a router tie to the higher expert) | p1 `exact_f32` | not separated: the tapes hold no exact tie (smallest route margin 1.6e-6 on p1, 1.3e-7 on p2). The constructed-tie kernel cell `ink2g-route` separates it (3/3 tied rows moved, no untied row) |
| wrong log-scale floor (the route at the shipped floor 8192 against the `p2_floor512` tape) | p2_floor512 `exact_f32` | slots 0 to 4 equal (7e-7 or less), slot 5, the first full-attention layer, jumps to 1.7e-2; KL 8.6e-2 |
| forced MoE input on a tape that keeps a subset of positions | p3 `actq_f32` | the runner refuses (179 of 8291 positions). The first p3 forced run (`b4/p3-declared-forced-moe.json`) ran unforced on most chunks and is void |

## The declared-class bug this lane fixed

`memra_ink2_actq`'s `Rows` argument selects the row a block reads and writes; it does not gather into
the block's slot. The route passed a pair-to-token gather, so pair rows past the token count kept stale
codes. Forced-MoE residual error read 4x to 42x relative L2 per slot (`prefix-bug/`). The MoE input is
now rounded to bf16 once per routed pair and quantized per pair (the copies are bit-identical), and the
`ink2g-moe-declared-*` cells hold the route's own declared path to the reference block.

## Timed decode (correctness route, not tuned)

Full clocks per `ceiling/FULL-CLOCKS.md`: P0, memory 14001 MHz on all 768 loaded samples of
`b5/smi.csv` (250 ms), no P2 or 11001 MHz sample; SM 2730-2775 MHz at the end of each take; host load
4.9. N=3 takes per class, interleaved, 32 tokens from the p1 prompt.

| Class | tok/s (3 takes) | launches per token |
|---|---|---|
| float | 4.34, 4.35, 4.34 | 1541 |
| declared | 5.38, 5.38, 5.38 | 1637 |

Weight bytes a decode token reads: 4.63 GB over the 48 slots (BF16 trunk, 8 of 128 experts per slot) plus
the 0.93 GB head, about 5.55 GB. At the measured 865 GB/s DRAM read that bounds decode at 6.4 ms a token
(about 155 tok/s). The route runs at 186 ms a token (declared; 230 ms float), 3.4% of that bound: its GEMV and attention kernels
are written for exactness (f64 epilogues and scores, one block per head row), and the launch count alone
costs about 3 ms. Tuning is the next lane.

Earlier timings in `b3/` and the first `b4/p1-declared.json` take ran with the card clamped at about
1.1 GHz and are void as timings; their correctness numbers stand.

## Open

- p3 float drift: 0.11 relative L2 on the last slots at 8k with KL 8.4e-4. No self-variance of the
  oracle at this length exists yet (cached vs monolithic was measured only on p1 and p2), so the
  long-context per-slot bound is not set. Next: a p3 cached-vs-monolithic run of the oracle, then bound.
- Multi-sequence batching (several requests in one call) is not built; batched-vs-solo equality is
  pending with it. Single-request chunked prefill vs per-token decode is bit-identical.
- The drafter (DSpark) runs in a later lane; its tensors are bound, not resident.
- Performance: fused decode kernels, CUDA graphs, BF16 GEMV at the DRAM roofline, attention with f32
  scores where a measured KL receipt allows it.
