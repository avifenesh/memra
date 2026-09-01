# Q27 + Q35 per-card capacity progress

Lane: `lane/cx-percard`

Started: 2026-08-12

Starting revision: `250ba819e83f868d395c01c6f315a4c6344f54cb`

## Objective

Measure Qwen3.6-27B and Qwen3.6-35B-A3B serving one model per RTX PRO 6000 while both cards are
active on the london pair. Retain exactness, artifact, concurrency, latency, throughput,
thermal, and cross-interference receipts for the owner-facing capacity decision.

## Fixed protocol

- Build the bundled local `main` source on the remote pair and run `kernel-check` to `ALL GREEN`
  before capacity work.
- Locate existing tuned GGUF and MTP/drafter artifacts; do not create or requantize model bytes.
- Pin every staged artifact with a full SHA-256 manifest.
- Run Q27 on CUDA device 0 and Q35 on CUDA device 1 as simultaneous, distinct server processes.
- Establish each model's `run-gen` argmax receipt, external-drafter `run-spec` K=1..8
  self-consistency receipt, and ten-run fixed-prompt serve golden before the capacity sweep.
- Sweep per-model concurrency `1,2,4,8,12,16,24`, 128 output tokens, N=3 windows per width while
  both servers remain resident; retain raw request and GPU telemetry.
- Rotate width and condition order without artificial cooldown. For each width, compare the paired
  window against a same-width control with the peer server resident but idle.
- Define the saturation knee before viewing the curve as the lowest measured width reaching 95% of
  the best paired median aggregate output rate among widths whose median window p99 TTFT is at most
  15 seconds. Also retain the throughput-optimal width and the largest width under that tail bound.
- At the selected knee, call peer-card interference material only if the paired median loses more
  than 5% aggregate output rate, or p99 TTFT rises by both more than 10% and more than 100 ms,
  relative to the same-width peer-resident-but-idle control. Retain per-repetition deltas and all
  widths even when the threshold is not crossed.
- Copy spot-host receipts back into this worktree and commit checkpoints frequently.

## Constraints

- Remote GPU work uses `/tmp/memra-gpu.lock`; long work launches detached immediately.
- No merge, tag, push, generated perf-board update, formatting pass, or hook bypass.
- Published medians state N and thermal regime; failures use captured error text or remain
  cause-unknown.
- Final output is `RESULTS.md` plus raw evidence under `research/percard-20260812/`.

## Status

- [x] Verified dedicated branch/worktree and clean starting revision.
- [x] Read lane steering and repository instructions.
- [x] Committed this progress ledger before any remote mutation (`0579a9777`).
- [x] Bundled local `main`, cloned it on eu-west, and completed the release build from the
  byte-identical buildable v0.78 runtime parent.
- [x] Inventory and stage artifacts.
- [x] Build and pass remote kernel gate.
- [x] Pass per-model exactness and freeze serve goldens.
- [x] Complete simultaneous-serving capacity and interference sweeps.
- [x] Write and verify `RESULTS.md` and retained raw receipts.

## Log

- 2026-08-12: Lane initialized. No remote mutation or GPU command has been performed yet.
- 2026-08-12: Created `/tmp/memra-gpu.lock` on eu-west before GPU work. The pair was idle and
  reported two RTX PRO 6000 Blackwell Server Edition cards with 97,887 MiB each; NVMe had 3.2 TiB
  free.
- 2026-08-12: Bundled local `main` `250ba819e83f868d395c01c6f315a4c6344f54cb` as
  SHA-256 `29a03553...4f5d1e`, copied it to eu-west, and cloned the complete history.
- 2026-08-12: The exact bundled `main` build stopped before CUDA compilation: Cargo 1.97.1
  rejected workspace package version `0.78.0` against the still-pinned `=0.77.0` path
  dependencies. Tag `v0.78.0` contains the same one-line incomplete bump. The scored build uses
  its parent `8b2ba8c883152fdbb9f9bbd800a055ad03fe80c4`; `crates/` and `tools/` are byte-identical
  between that commit and bundled `main`. The failed build receipt is retained rather than
  silently patching the checkout.
- 2026-08-12: The buildable v0.78 runtime tree compiled `kernel-check`, `run-gen`, `run-spec`, and
  `memra-server` with CUDA 13.2 / auto-detected sm_120a. Binary SHA-256 values are retained in
  `raw/setup/build-runtime-class.log`.
- 2026-08-12: Local artifact inventory found all four required existing bytes, so no download or
  quantization is needed: Q27 target `d8d71c7e...42d517` (15,705,920,064 B), Q27 draft
  `b445fbb1...9f3581` (1,242,867,296 B), Q35 target `df27a780...1f7adf`
  (18,209,036,576 B), and Q35 draft `ae5b7797...870b6a` (944,118,560 B).
- 2026-08-12: Resumable transfer completed: 36,101,942,496 bytes in 21 minutes. Remote sizes and
  full SHA-256 values matched all four local sources exactly; `raw/setup/artifact-manifest.txt`
  retains the receipt. The remote GPUs remained idle throughout staging.
- 2026-08-12: Detached gate block held `/tmp/memra-gpu.lock` from 00:08:19Z through 00:14:13Z.
  Physical GPU 0 passed `ALL GREEN (94 cells, 13 skipped)`; physical GPU 1 independently passed
  the same full checker. The skips are explicit non-candidate fixture or optional-input skips in
  the raw logs, not hidden failures.
- 2026-08-12: Both `run-gen` logs contain prefill/decode and batched-prime/tokenwise argmax
  `MATCH`. Both external-drafter `run-spec` logs contain eight K=1..8 self-consistency PASS rows
  and the overall `SELF-CONSISTENCY PASS` verdict. The remote manifest verified after copying the
  complete gate directory back to `raw/gates/`.
- 2026-08-12: Both persistent one-card servers loaded simultaneously with their named regime
  drafters and passed fixed-prompt serve self-consistency 10/10. Q27 froze 516 bytes at SHA-256
  `11cffe49...0cf5c55`; Q35 froze 514 bytes at SHA-256 `6220c20b...891a47`. Every request
  returned 128 completion tokens with `finish_reason=length` and zero cached prompt tokens. The
  goldens and their complete per-repeat JSON receipts were copied home before the sweep continued.
- 2026-08-12: Capacity replicate 1 completed all 21 scored windows (28 per-target points across
  paired and peer-idle controls) with every request clean. The stable `r1-*` directories were
  copied and committed locally while replicate 2 continued on the spot host.
- 2026-08-12: Reverse-order replicate 2 brought the retained checkpoint to 42 clean scored
  windows and 56 per-target points. Its stable `r2-*` directories were copied and committed while
  the rotated third replicate continued.
- 2026-08-12: The detached campaign finished with `PERCARD_CAMPAIGN_PASS`: 63/63 warmups and
  63/63 scored condition windows passed, yielding 84 per-model condition points and 102,912 exact
  completion tokens with zero request errors. The complete 656-file remote manifest verified
  after transfer, both servers stopped, and both GPUs returned idle with no compute process.
- 2026-08-12: Deterministic reduction selected Q27 c=16 at 287.72 output tok/s and Q35 c=8 at
  606.66 output tok/s. At those knees, peer-card load changed Q27/Q35 p99 TTFT by +1.5/-0.2 ms
  and throughput by -0.10%/+0.06%; neither crossed the frozen interference rule. The local
  summary reproduced SHA-256 `90905b2e...1345be` byte-for-byte.
- 2026-08-12: `RESULTS.md` records the complete N=3 curves, cross-interference receipts,
  exactness/spec state, effective-price capacity ceiling, thermal regime, artifact identities,
  and the v0.78 package-metadata build caveat.
