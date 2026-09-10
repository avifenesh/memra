# Qwen3.8-27B calibrated prefill activation program

**STATUS 2026-09-10: this program is banked NEGATIVE and is not a serving path.** On four
held-out windows it costs +0.1132 nats of mean paired dNLL against the served W4A8 artifact,
KL(served||A4) 0.164 to 0.817, and 87.1% top-1 agreement, against bars of 0.08 nats, 0.05 and
97%. A corrected engine re-fit of all 400 multipliers bought 0.026 nats, because the global
multiplier cancels out of `s * ue4m3(block_amax/(6*s))` and is nearly inert; SGLang's NVFP4
W4A4 checkpoint sits within 0.07 nats of our served arm on the same transcripts, so the cost is
this program's linear set rather than FP4 activations as such. Follow-up: #439. Verdict tables
live in the private companion, `research/qwen-fp4-activation-mint-20260909/VERDICT.md`.

The design below is the Phase 1 text as written on 2026-09-09, kept unedited as the record of
what was proposed and qualified against.

Phase 1, 2026-09-09. Issue #420, refs #400/#408/#409/#411. Awaiting owner steering before minting or engine changes.

The existing target has 64 trunk layers and one MTP block. The measured GGUF census is 504 NVFP4 tensors, 360 F32 tensors and two Q5_K tensors (embedding and output head). The NVFP4 block format stores four per-16 UE4M3 scales and 32 E2M1 bytes per 64 weights. No calibrated activation scales are present. The existing source importer reads ModelOpt weight_scale_2 but Qwen does not execute input_scale.

Proposed native program: 400 large trunk linears use calibrated FP4 activations in prefill. The 96 narrow GDN alpha/beta projections, MTP, output head and separate drafter retain existing arithmetic. All weight bytes remain unchanged. Add one F32 calibrated input dequantization scale per selected linear and a versioned typed activation/phase policy in the qwen35 pack, source binding, plan and execution manifest. No family-wide support claim or new environment door.

Quantization: calibrated per-linear global dequant multiplier s=amax/(6*448), dynamic RN UE4M3 scale per row and 16-value K block, RN saturated E2M1 elements, native FP4 block-scale MMA, explicit global-scale epilogue. Exact zero/subnormal/rounding/saturation behavior is oracle-bound. Preserve SiLU/mul separately. Reuse the native lever-2 kernel with calibrated inputs, without adding an external runtime kernel dependency or duplicating resident weight storage.

Dispatch is explicitly Prefill versus Decode/Verify, not row-count based. All prefill tails and restored suffixes, including one row, retain A4 arithmetic. Decode and speculative verification retain W4A8 at every row count. Graph and cache identities bind the artifact/program. Missing or invalid required scales fail closed; legacy artifacts retain the existing behavior.

Generated-state reuse is an independent correctness boundary. Only canonical pre-generation prefix captures may be restored for this program. Generation-extended W4A8 state must not masquerade as all-prefill state in affinity/reuse pools. Recompute the next request suffix under the prefill program, including prior assistant text. Cold/restored boundary bytes, taps and draft state must match. This needs an implementation audit and gates, not merely a metadata field.

Calibration is planned on a non-production 80/96 GB GPU against the pinned BF16 source. There is no measured basis to assume dequantized-NVFP4 calibration is equivalent on 32 GB. Freeze source sessions and Qwen rendering before calibration. Public receipts carry hashes and aggregate statistics, not private calibration text.

Required qualification: tensor/template census; independent arithmetic and checkpoint oracle; same-artifact plain/spec K=1..8 identity; four-turn greedy restore and boundary/tap/draft equivalence; paired BFCL and Hebrew MMLU plus held-out/32k PPL and KL; vendor-default sampled DFlash2 acceptance ratio >=0.95; interleaved fresh-boot cold 8k/32k/131k, decode, eight-turn cache, four-session mix and long admission cells. Quality must be within measured noise. Cross-program margin flips are diagnostic. All gates are currently NOT RUN.

No local rig build/gate/CI. Future remote qualification uses sm_120a, the canonical shared GPU lock and an isolated target-mint directory. Pushes use MEMRA_SKIP_PERF_CI=1, disclosed in PRs; hosted CI still gates merge. No release or deployment. The private companion contains custody, corpus, pricing, full Phase 1 evidence and the owner steering report.
