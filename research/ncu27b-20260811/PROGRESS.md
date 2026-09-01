# ncu27b lane progress — 2026-08-11

## Scope

Anatomy-only Nsight Compute study of one bounded 64-token, `c=1` speculative-decode window for the local Qwen3.6-27B NVFP4 trunk + MTP + own-trim drafter on RTX 5090 (`sm_120a`). No kernel edits.

## Required output

- Attribute decode-round time across qmatvec/mmvq, `fa_decode`, drafter head, sampling, and RMS/RoPE glue.
- Report per-kernel SM throughput, memory throughput, and achieved occupancy.
- Identify the top three kernels below both suspiciousness thresholds (SM < 60%, memory < 70%), name the measured limiter, and give a tune candidate plus expected ceiling.
- Preserve raw capture/export evidence beside `RESULTS.md`.

## Guardrails

- Branch/worktree: `lane/cx-ncu27b` at `/home/avifenesh/projects/wt-cx-ncu27b` (verified).
- GPU blocks use `flock /tmp/memra-gpu.lock` and remain bounded.
- CPU work uses `nice` and bounded parallelism where applicable.
- No `cargo fmt`, kernel changes, pushes, board edits, merge, tag, or release.
- The requested inbox path, `/home/avifenesh/.lanectl/inbox/cx-ncu27b.md`, was absent at lane start.

## Status

- [x] Read `CLAUDE.md` and lane request.
- [x] Verify dedicated branch/worktree and clean starting tree.
- [x] Read relevant local kernel/speculative-decoding context.
- [x] Identify the exact 27B run command and current binary/config.
- [x] Build only `run-spec` in release mode (`nice -n 10`, four Cargo jobs).
- [x] Verify profiler/tool/GPU state and capture baseline metadata.
- [x] Capture one unperturbed 64-token Nsight Systems timing window for time weights.
- [x] Run one bounded selected-kernel 64-token Nsight Compute window.
- [x] Export and aggregate kernel metrics; validate attribution coverage.
- [x] Write `RESULTS.md` with top-three limiters and tune candidates.
- [x] Commit final report and verify the clean lane state (no push).

## Run contract

- Trunk: `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`
  (`d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`).
- Drafter: `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/draft-daily-owntrim-nvfp4head-q4blk.gguf`
  (`b445fbb139e72f9869df06f2f0f91bcaf57527ec34a24bec74d3febd719f3581`).
- Spec shape: `MEMRA_MTP_DRAFT=<drafter>`, `MEMRA_SPEC_K=3`, `MEMRA_NGEN=64`, c=1.
- Profiler isolation: `MEMRA_PROFILE_SPEC=2` starts the CUDA profiler after the spec prime;
  Nsight Compute must use profile-from-start OFF so only the round loop is eligible.
- Prompt: the tracked short code prompt at `research/e2e/prompts/p1-code-short.txt`, matching the
  second-listing/product workload rather than a synthetic token sequence.

## Log

- 2026-08-11: release build completed in 2m21s at the bounded CPU settings. The first GPU-state
  probe waited behind another lane's locked `tools/local-ci.sh --perf-quick` block and timed out
  without acquiring the lock or touching the card; it produced no measurement artifact.
- 2026-08-11: the exclusive Nsight Systems window completed with exactness PASS, 21 speculative
  rounds, and 42/63 accepted draft tokens (66.7%). Its uninstrumented-by-NCU kernel timeline is
  the time-weight source; the selected NCU replay will supply counters, not elapsed-time shares.
- 2026-08-11: the single NCU window completed with exactness PASS and the same 42/63 acceptance.
  It profiled one launch from each of 79 selected launch configs at base clocks, covering 99.50%
  of Nsys kernel time. The top suspicious unique symbols are `qmatvec_nvfp4_mmvq_b4_rpr2w8`,
  `fa_decode_f32`, and `qmatvec_nvfp4_mmvq_b4_rp`; no top-three shape has bank conflicts.
- 2026-08-11: deterministic re-aggregation reproduced the same summary hash; 79/79 raw counter
  rows, both exactness receipts, local Markdown links, scope-only diff, and generated perf-board
  check all passed. No push, tag, board update, engine edit, or kernel edit was made.
