# cx-coldhol progress

## Objective

Break the Qwen3.6-27B cold-prefill head-of-line serialization that binds the
single-card concurrency knee at c=12, without changing output exactness or
regressing the c<=12 workload.

## Success criteria

- Commit a design note before code that explains the serialization and ranks
  candidate fixes by blast radius.
- Capture the required before/after `kernel-check`, `run-gen` argmax, and
  `run-spec` K=1..8 exactness gates.
- Test the smallest scheduler change with an explicit rollback seam.
- Re-run the frozen c=8,12,16,20,24 workload on Box1 under one GPU-lock hold,
  with N>=5 interleaved samples for the before/after arms.
- Commit raw JSONL and `research/coldhol-20260812/RESULTS.md`; report the knee,
  hit-TTFT p95, and price-stated $/day delta even if the candidate is refuted.

## Status

- [x] Lane inbox, repository instructions, branch isolation, and initial GPU
  state checked.
- [x] Read the prior knee evidence and trace the scheduler/prefill mechanism.
- [x] Commit the design note before implementation.
- [x] Capture before exactness gates: Box1 GPU0, model-backed `kernel-check`
  ALL GREEN (95 cells, 13 skipped), two `run-gen` argmax MATCH checks, and
  `run-spec` self-consistency PASS at every K=1..8.
- [x] Implement the selected continuation-capable cold-chunk batch in the
  scheduler; focused test and all 212 `memra-server` unit tests pass locally.
- [x] Capture after exactness gates: Box1 GPU0, model-backed `kernel-check`
  ALL GREEN (95 cells, 13 skipped), both `run-gen` argmax comparisons MATCH,
  `run-spec` self-consistency PASS at every K=1..8, and the carried-prefix
  prime-batch gate ALL GREEN. GPU memory returned to 0 MiB on both cards.
- [x] Freeze lane-owned Box1 runners for the exact prior workload and settings,
  fixed before/after binaries, five alternating rounds, one lock hold, raw
  telemetry, and a required real-server partial-batch receipt.
- [x] Pass a candidate c=16 server smoke: 20/20 requests clean, 0 accounting
  drift, five chunk batches (four partial), no batch failure, and both GPUs at
  0 MiB with no compute processes after shutdown.
- [x] Run and analyze the frozen Box1 interleaved sweep: 10 whole-server boots
  under one lock hold, 50/50 cells and 1,100/1,100 requests clean. The formal
  knee moves c12 -> c16; c8/c12 paired throughput is +1.65%/+0.03%, pooled hit
  TTFT p95 is held, and candidate partial-batch telemetry fires in every boot.
- [x] Commit the complete evidence package and run final repository checks:
  analysis regenerates byte-for-byte, all raw manifests/sentinels verify, the
  perf board is current, both harness scripts pass `shellcheck`, and all 212
  `memra-server` unit tests pass.

## Guardrails

- Stay in `/home/avifenesh/projects/wt-cx-coldhol` on
  `lane/cx-coldhol`; preserve unrelated work.
- Box1 GPU work uses GPU 0 only, under `/tmp/memra-gpu.lock`, after a recorded
  `nvidia-smi` process/thermal snapshot.
- No origin push, merge, tag, board update, formatting sweep, rustup, nsys,
  other worktrees, or verifier bypass.
- Re-read `/home/avifenesh/.lanectl/inbox/cx-coldhol.md` between major steps.
