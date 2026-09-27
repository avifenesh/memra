# Lockstep router peer-invariance A/B (#565, #562)

Verdict: the router fix does what it claims. On Hy3 NVFP4 with the `run_lockstep` harness,
main's M=4 same-prompt run diverges from its own M=1 run at generated token 18; with the fix,
all four M=4 streams reproduce the M=1 tokens exactly, all 32 tokens, and M=1 itself is
byte-identical between the two binaries. A second, router-independent peer dependence remains
in the mixed-prompt regime on both binaries (same bytes), placed below.

## Rig and artifact

One rented RTX PRO 6000 Blackwell Workstation Edition (97887 MiB, 600 W, driver 595.84), Ryzen
9 9950X, 188 GB RAM, CUDA 13.1.2 image, 2026-09-20 (UTC evening). Artifact
`Tiyuvta/Hy3-NVFP4@0af425172b7a` (99 safetensors shards, 180.9 GB) staged to local NVMe and
verified against its `SHA256SUMS` (`raw/sha-verify.log`: every weight and config file OK; only
`README.md` mismatches, the repo README was edited after the sums were written). Single-GPU
regime from `tools/run_hy3_local_5090.sh`: frozen expert residency (`MEMRA_MOE_VRAM_FRAC=0.85`,
57.19 GB of expert slots), native CPU expert companion `caf5e60c...` (`raw/binaries.sha256`),
O_DIRECT spill, freeze profile saved by a `run-gen` warmup (128 discarded tokens,
`raw/warmup-rungen.trimmed.log`). `NVIDIA_TF32_OVERRIDE=0`. One process at a time behind
`/tmp/memra-gpu.lock`. Driver: `raw/ab.sh`; box setup: `raw/provision.sh`, `raw/build.sh`.

| Arm | Tree | `run_lockstep` sha256 |
| --- | --- | --- |
| base | origin/main `dbf88d467` | `baf1d293...` |
| fix | `dbf88d467` + this lane's commit (`5cd909f67` on the box, same diff as the PR head) | `5f03e30c...` |

Prompt P0 = "Explain speculative decoding briefly." (20 chat-template tokens); the mixed file
adds three more prompts (25, 23, 26 tokens), stream 0 always P0. Greedy, 32 new tokens.

## Cells (`raw/<arm>-<cell>.log`)

| Cell | base | fix |
| --- | --- | --- |
| M=1, P0 | `...2438, 28615, 341, 68, 2693, ...` | identical to base M=1, all 32 tokens |
| M=4, all streams P0 | 4 streams identical to each other, `...28615, 13, 286, 37246, ...`: DIVERGES from M=1 at token 18 | 4 streams identical to each other AND to M=1, all 32 tokens |
| M=4, mixed prompts, stream 0 = P0 | stream 0 `...28615, 13, 286, 37246, ...` (= base M=4 same) | stream 0 `...28615, 13, 286, 37246, ...` (= base, differs from M=1) |
| M=2, mixed prompts, stream 0 = P0 | not run | stream 0 == M=1, all 32 tokens (`raw/fix-m2-mixed.log`) |
| M=3, mixed prompts, stream 0 = P0 | not run | stream 0 == M=1, all 32 tokens (`raw/fix-m3-mixed.log`) |

Harness stream-identity gate: PASS on every same-prompt cell of both arms (the harness only
asserts streams against each other; the M=1 comparison above is the lane's claim).

Throughput, for the record only (single run each, cold page cache on the first cell): M=1
4.20 and 4.09 tok/s, M=4 same 6.91 and 7.09 tok/s aggregate, M=4 mixed 5.63 and 5.74
(base, fix). The fix costs nothing measurable at this N.

## Reading

1. The base divergence is the mechanism the PR names: `moe_ffn_lockstep` computed router
   logits with a cuBLAS matmul over all stream rows, so `m = stream_count` selected a different
   reduction than the M=1 gemv and flipped a near-tie route. Routing through the m-invariant
   `moe_router_logits` / `router_gemv` selector removes it: fix M=4 same == M=1.
2. The residual mixed-prompt divergence is not the router (both arms produce the same bytes).
   It sits where the lockstep path treats streams jointly with different rows per expert:
   HBM-resident experts routed by more than one stream run through the grouped gather/GEMM/
   scatter path at `m_e > 1`, and experts routed by two or more streams take the companion's
   multi-row ABI. `research/moe/draft-head-and-concurrency-lanes.md` (M2 gate, 2026-07-23)
   documents that class: "grouped-lockstep tokens are NOT bitwise-identical to the per-stream
   path ... the m_e>1 batched GEMM reduces each row in a different FP order than m=1". The
   same-prompt cells do not expose it because every expert then carries all four identical
   rows. The discriminators narrow it: M=2 and M=3 mixed reproduce M=1
   exactly, only M=4 mixed diverges, and M=4 same-prompt does not. Four distinct rows are
   needed to expose it on this prompt set; the exact site (grouped `m_e > 1` GEMM order,
   multi-row companion order, or a fourth-row effect elsewhere in the lockstep walk) is open.
3. Consequence for serving: `decode_step_lockstep` has one caller, the `run_lockstep` harness;
   memra-server never runs it. The residual is tracked in its own issue, not in this PR.
