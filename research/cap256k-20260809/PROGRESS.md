# cap256k admission lane — progress

## 2026-08-09 intake

- Branch/worktree: `lane/cx-cap256k` at base `b752cf2d87c61810f4f8374b5c6a937f32d28c3f`.
- Steering check: `~/.lanectl/inbox/cx-cap256k.md` is absent at intake.
- Audit contract read first: `research/code-audit-20260809/PAPER.md` findings 6.1 and 6.2.
- Scope: request-specific large-context admission cost plus reclaim-on-defer across the existing
  `reuse` and `spec_reuse` pools. Do not restructure either pool.
- Compatibility: `lane/plain-affinity` adds checkpoint and identity metadata to `ReuseEntry` and
  explicitly leaves P0-2 reclaim/right-sizing to a follow-up lane. The compatible seam is a helper
  that removes existing entries without depending on or changing their payload shape.
- Preserve: the admit-OOM lane's spec-capable `SPEC_SHRINK_RESERVE` charge and its local-CI gate.

## Planned proof

1. Trace `ctx_cap`, KV allocation geometry, current cost calibration, both parked pools, and the
   admit-OOM harness.
2. Add focused unit tests for request-specific cost scaling and oldest-across-pools reclamation.
3. Capture a bounded 5090 before receipt with a documented local GGUF under
   `flock /tmp/memra-gpu.lock`.
4. Implement the smallest admission-loop change and capture the matching after receipt.
5. Run `cargo test -p memra-server` and the admission slice of `tools/local-ci.sh`; retain raw
   JSONL/logs and summarize N and thermal regime in `RESULTS.md`.

## Status

Complete. The request-shaped admission estimator, global parked-session reclaim, unit tests,
release build, c=64 admission gate, inverted gate teeth, and matched 5090 after receipt are done.
See `RESULTS.md`.

## Steering update

- The inbox appeared after intake. Per its 2026-08-09 steering, fetched the local
  `restructure/public-split` worktree as `local-train/restructure-public-split` and merged tip
  `8e8c93af` before implementation. This brings the release pin fix and cache-salt validation;
  the expected server-test baseline is now 137.
- Read `lane/plain-affinity:research/affinity-20260809/PROGRESS.md`. That lane adds checkpoint,
  identity, and fingerprint fields to `ReuseEntry` but deliberately leaves admission reclaim and
  analytic costing here. A single park-age field plus a removal-only reclaim helper is the
  smallest compatible hook; no pool wrapper, ownership change, or checkpoint handling belongs in
  this lane.
- Local measurement artifact selected: Qwen3.5-9B NVFP4 GGUF plus its own-trim draft under
  `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/` (5,657,607,424-byte trunk and
  936,838,816-byte draft). It fits the 24 GB local RTX 5090 while leaving enough room to expose
  the difference between 8k calibration and 128k/256k admission.

## Pre-fix 5090 receipt

- Raw block: `raw/20260809T113429Z-before/`, N=1, exclusive
  `/tmp/memra-gpu.lock`, P0, 68--76 C over seven one-second samples. All 4 burst streams
  completed and no CUDA/OOM/panic line was captured.
- Sequence was one 8k calibrator, two sequential 256k parks, then a simultaneous c=4 128k
  burst. The 8k allocation was satisfied from CUDA-pool cached blocks, so the old
  `session_vram_cost` map remained unset. The first 256k admit then froze the global scalar at
  4,899 MB; every later 128k request inherited that full 256k-class charge.
- The burst recorded 10 `admission_vram_defers`. Its gate line reported two parked spec
  sessions on every shortfall and reclaimed none. TTFB service order was 0.038, 0.306, 0.463,
  and 0.853 seconds (0.815-second span); requests were short, so this is the same serialization
  mechanism as the 25-second Step receipt at a smaller wall-time scale.
- This is Bug A's opposite-order arm (large first measurable admit over-gates later smaller
  contexts) plus Bug B's direct defer-with-parked-state mechanism. The post-fix run must use the
  identical sequence and show a distinct request-scaled 128k estimate plus reclaim before any
  remaining defer.

## Completion update

- Preserved the inherited four-file patch first as `4a8ffd10`, then merged the exact local-train
  tip `abe318dc` as `f9f1d0f2`. The affinity-era `ReuseEntry` checkpoint, affinity, and fingerprint
  fields remain intact; this lane adds only park age and removal-side reclaim.
- The matched after block is `raw/20260809T123952Z-after/` at `c0cfc0eb`. It logged distinct
  request costs (256k spec 4,968 MB; 128k spec 2,536 MB; 128k plain 2,292 MB), reclaimed one
  parked spec session before admission, and finished with zero admission VRAM defers.
- `cargo test -p memra-server`: 152 passed (148 merged baseline + 4 new). `cargo build --release`:
  PASS. The local-CI c=64 admission cell completed 64/64 with a live, clean worker; its 16 MB
  reserve teeth failed 18/64 streams with quoted CUDA OOM errors, so the inverted gate remains
  sensitive.
