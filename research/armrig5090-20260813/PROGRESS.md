# RTX 5090 eager-arm cost — progress

Date: 2026-08-13
Branch: `lane/cx-armrig5090`
Pinned base: `01df75ac261c074e9ecda1b31c99615c9b5ee79c`
Rig: local NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, sm_120 / 82 SM

## Scope and invariants

- Measure one freshly built `memra-server` binary in two runtime configurations only:
  - `REPAIRED`: `MEMRA_SERVE_B1FAST` and `MEMRA_SERVE_GS` unset.
  - `EAGER`: `MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1`.
- Seal the server SHA-256 and verify every point executes that same file.
- Use Q27 NVFP4 at concurrency 1, 4, and 16. Q27 fits this 24 GB rig and exercises both eager doors; c=16 is its highest native exact decode chunk class, while wider request sets are scheduler-chunked rather than a wider numerical batch. A failed c=16 qualification invalidates the cell rather than changing its shape.
- Run a fully restored 512-token decode ladder, then the frozen 90%-hit / 10%-miss c=4 money shape.
- Use N=5 independent fresh-server launches per cell, alternate arm order by repetition, and keep all scored work in one uninterrupted `/tmp/memra-5090.lock` hold.
- Gate every cell on request success, full requested token count, `finish_reason=length`, cache-shape integrity, and cross-arm output identity before retaining throughput.
- Tee raw output before parsing; preserve per-request JSONL, server logs, environment receipts, continuous GPU telemetry, and reducer output under `raw/`.
- Report medians, min..max spreads, and paired deltas. A delta inside the observed spread is flat.
- Record the owner-imposed 210–1200 MHz clock cap and do not change or reset clocks. This rig is relative-only; no 5090-vs-PRO absolute comparison is valid.
- Stop after the lane commit: no merge, tag, push, release, or generated perf-board edit.

## Checkpoints

- [x] Confirm the clean dedicated worktree, branch, and exact base commit.
- [x] Read the numeric-program and per-hardware selection doctrine plus the repaired serving contract.
- [x] Verify the local model path, GPU identity, no compute applications, lock availability, and port preflight.
- [x] Build one binary with `TMPDIR=/home/avifenesh/tmp-lanes` and seal its SHA-256.
- [x] Validate the measurement harness and reducer with syntax checks plus a synthetic 30-point / 210-request full-hit matrix and 10-point / 200-request mixed matrix.
- [x] Commit the measurement harness and build receipt checkpoint (`b0c745a0e`).
- [x] Hold `/tmp/memra-5090.lock` and run the interleaved Q27 c=1/4/16 full-hit ladder.
- [x] Run the interleaved Q27 mixed-cache c=4 cell in the same lock and thermal window.
- [x] Reduce exactness-gated raw evidence and write `RESULTS.md`.
- [x] Confirm tenant-clean shutdown: server gone, lock released, GPU compute-idle, and ports clear.
- [x] Commit the complete lane and stop.

## Live log

- 2026-08-13: Worktree verified clean on `lane/cx-armrig5090` at `01df75ac2`, matching `main` and `origin/main` at initial inspection.
- 2026-08-13: Preflight found the RTX 5090 Laptop GPU at 49 MiB idle display use, no compute applications, 210 MHz current SM clock, and `/tmp/memra-5090.lock` free. The campaign will retain continuous telemetry to verify the fixed 1200 MHz ceiling under load; no clock-changing command will be run.
- 2026-08-13: Selected `/data/ai-ml/hf-models/qwen36-27b-nvfp4-mtp/Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf` (15,705,920,064 bytes). The full-hit ladder isolates restored decode and makes GraphSession reachable; the separate frozen mixed90 c=4 cell measures the cache-plus-prefill revenue shape.
- 2026-08-13: Fresh CUDA 13.1 / auto-sm_120a release build completed from `57ebcf8d319dc8ea9bb351b39fc1ab28d18c20db` with `TMPDIR=/home/avifenesh/tmp-lanes`. The one scored `memra-server` binary is sealed as SHA-256 `1b460e29f642b93e86ed287c144129e6c5b3a6c1cca56abf2aacff92393b809c`.
- 2026-08-13: The local driver uses a clean `env -i` server environment, fresh server per point, immediate per-pair byte-identity gates, one continuous 250 ms GPU telemetry stream, and a trap that stops the owned server before releasing `/tmp/memra-5090.lock`. No clock-changing command exists in the harness.
- 2026-08-13: Committed the sealed harness, reducer, and fresh-build receipt at `b0c745a0e`; the worktree is ready for the single-lock scored campaign.
- 2026-08-13: Attempt 1 acquired `/tmp/memra-5090.lock`, proved EAGER GraphSession activation, and qualified c=16 at 16/16 requests x 512 tokens. It then stopped at the first scored c=1 pair exactly as designed: both arms completed 512 tokens, but the pair gate emitted `ValueError: BYTE MISMATCH across A/B pair` with REPAIRED hash `210933f9d1c2e8e111b6a633829934d518f149d5c92204ffa70d1933f2e1e70e` and EAGER hash `c1491b4b22305fb2e74aa77fb111a7a4dfb6f4832c69a1239210be2831dbf1e3`. Its throughput observations are invalid and will not be reported.
- 2026-08-13: Attempt 1 cleanup was tenant-clean: driver exit 1 preserved the raw traceback, GPU returned to 49 MiB display-only / no compute applications / 210 MHz, port 18469 was clear, and `/tmp/memra-5090.lock` was independently reacquired after release.
- 2026-08-13: The next harness checkpoint will distinguish an expected research outcome (full-length byte mismatch, record the invalid cell and continue) from a protocol failure (missing/short requests, cache-shape drift, server failure, or captured CUDA error, stop). Attempt 2 will start from scratch in a new one-lock thermal window; no performance sample will cross holds.
- 2026-08-13: Updated the pair gate to return a distinct `BYTE MISMATCH` research verdict, continue the untouched matrix, and make the reducer withhold every invalid concurrency's throughput. Structural/output-length failures still stop. A synthetic N=5 matrix proved c=1 mismatch is withheld while valid c=4/c=16 and mixed90 reduce normally.
- 2026-08-13: Attempt 2 held `/tmp/memra-5090.lock` continuously from 07:41:05Z through the last scored point at 08:26:01Z. It completed all 30 full-hit points (210 requests) and all 10 mixed points (200 requests), N=5 per arm/cell, with odd/even arm-order alternation and no request errors or short completions.
- 2026-08-13: Exactness invalidated restored c=1 in 5/5 pairs, restored c=16 in 2/5 pairs (2/80 EAGER requests), and mixed c=4 because all 90/90 EAGER restored hits disagreed with their EAGER cold-seed golden. Throughput is withheld for those cells. Restored c=4 was byte-identical in 5/5 pairs and flat: REPAIRED 84.98 tok/s median [73.61..85.24], EAGER 85.17 [84.98..85.25], +0.224%, overlapping ranges.
- 2026-08-13: The campaign's initial post-run reducer rejected the telemetry `[N/A]` power-limit field after all measurements completed. The parser was fixed and the same one-hold raw window reduced successfully; no GPU work was rerun. The observed ceiling was 1200 MHz under the owner-declared 210–1200 MHz cap.
- 2026-08-13: Tenant-clean verification found the GPU at P8 / 49 MiB / 0% utilization with no compute applications, the owned server absent, port 18469 clear, and `/tmp/memra-5090.lock` free. `RESULTS.md` recommends keeping EAGER off on this 5090 class.
