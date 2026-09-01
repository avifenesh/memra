# PP peer byte-integrity probe — progress

Date: 2026-08-11
Branch: `lane/cx-peerprobe`
Base: `1592253f`

## Scope

- Probe every PP device boundary once at bring-up, immediately after peer access is enabled.
- Copy deterministic bytes with `cudaMemcpyPeerAsync`, read them back, and compare at a legacy
  16 KiB preflight plus real boundary slots at 1/8/16-token widths.
- Fail closed with the exact mismatch count unless `MEMRA_PP_HOST_BOUNCE=1`; retain
  `MEMRA_PEER_PROBE=0` as an explicit diagnostics escape hatch.
- Unit-test injected readback corruption without peer hardware.
- Build and run `kernel-check` on the local RTX 5090 under `/tmp/memra-gpu.lock`.
- Defer the native two-card clean pass and PP-2 golden-hash receipt to the Vast window.

## Initial state

- Worktree clean at `1592253f` on the required dedicated branch.
- The existing PP path enables peer access in `crates/memra-engine/src/pp.rs`; exact insertion
  point and runtime boundary geometry remain to be inspected.
- Prior Vast evidence recorded successful peer API/capability results alongside corrupt bytes;
  this lane treats the byte comparison, not capability, as the gate.

## Live checklist

- [x] Confirm the PP bring-up seam, boundary-slot geometry, and existing host-bounce selection.
- [x] Add deterministic multi-size peer byte-integrity probing and boot-time policy handling.
- [x] Add hardware-free injected-corruption unit coverage for fail-closed and bounce override.
- [x] Document both flags and cite the NVIDIA R570 release-note rationale.
- [x] Run focused tests and CPU-capped release build.
- [x] Run `kernel-check` ALL GREEN under the local GPU lock.
- [x] Complete `RESULTS.md`, with the native two-card clean pass explicitly pending.
- [x] Probe real stream-ordered `BoundarySlot` memory at 1/8/16-token widths after pool grants.
- [x] Revoke diagnostic pool access and disable probe-enabled peers before host-bounce serving.
- [x] Re-run exact-final-source tests, release build, and locked GPU gates after follow-up fixes.
- [x] Commit only this lane's intended changes; do not merge, tag, push, format, or move boards.

## Work log

- 2026-08-11: Read the lane brief and project instructions; confirmed the clean dedicated
  worktree and recorded the acceptance gates before implementation.
- 2026-08-11: Wired a fixed 16 KiB bidirectional check at the immediate post-peer-enable seam
  and a model-width `[n_embd] f32` check before the first weight upload. Both passes cover every
  adjacent cross-device stage boundary and use deterministic source bytes plus inverse poison.
- 2026-08-11: Added exact mismatch policy: native transport returns a count-bearing startup
  error; preselected host bounce logs the count and proceeds. `MEMRA_PEER_PROBE=0` warns and
  skips only the boot diagnostic.
- 2026-08-11: Hardware-free injected-readback test PASS (1/1); focused engine test build and
  `cargo check -p memra-engine --tests` PASS. Design/flag docs cite NVIDIA's R570 known issue
  without attributing the independent Blackwell Vast failure to the Ada-and-older mechanism.
- 2026-08-11: CPU-capped full release build PASS (2m33s); engine library tests PASS (63/63,
  one pre-existing CUDA-only ignored); flag inventory reports no new drift.
- 2026-08-11: Local RTX 5090 `kernel-check`, under `/tmp/memra-gpu.lock` with both required
  manifests, reports `ALL GREEN (101 cells, 1 skipped)`; the skip is the optional external
  sigmoid-router replay capture. Locked single-device transport smoke also PASS with byte diff 0.
- 2026-08-11: `RESULTS.md` and hashed raw receipts complete. Native two-card clean pass,
  sub-100-ms probe timing, and PP-2 golden hash remain explicitly pending for the Vast window.
- 2026-08-11: Final review corrected the fallback summary label so an unavailable peer
  direction reports `SKIP`/`PARTIAL`, never `PASS`. The then-current engine tests, release
  rebuild, locked full `kernel-check`, and locked single-device transport smoke all re-passed.
- 2026-08-11: Re-read the appended orchestrator follow-up before commit and found the
  legacy-allocation-only gate was insufficient. Added the required production-memory-class
  ladder through shared boundary TX/RX code and explicit bounce diagnostic teardown; exact-final
  validation must be repeated before commit.
- 2026-08-11: Follow-up exact-source receipts PASS: injected corruption 1/1, focused policy 4/4,
  engine library 63 passed / 0 failed / 1 pre-existing CUDA-only ignored, CPU-capped release build
  31.45s, locked transport smoke byte diff 0, and locked `kernel-check` `ALL GREEN (101 cells,
  1 skipped)`. The optional external sigmoid-router replay remains the sole skip.
- 2026-08-11: Final staged scope, whitespace, and all seven raw receipt hashes verified; commit
  contains only the PP peer-probe implementation, flag/design/results documentation, and evidence.
