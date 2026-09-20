# Day 7 — native transfer source retirement and device hand-back

## Scope and revision

Repository: `avifenesh/memra`, lane `lane/spill-a-20260919`.
Native source: `92d332b9` (full SHA in `native/source.json`), plus the two
module/bin registration fragments in `../CUDA-TRANSFERS.md`. No shared manifest,
contract, numerical program, kernel, environment read, or default changed in this
lane. E's v1.3 additive-contract wording lives in that document.
Native binary SHA-256: `22c3c389ea3b3bd20a7c5da567775130906f2d63947f7de103595e0f11aa99e0`.

Target: rented RTX 5090 **development** substrate correctness, N=1 per schedule
and size. Observed power cap **400 W / 600 W maximum**. Both GPU cells ran through
`tools/tier-battery.py` under `/tmp/memra-5090.lock`, with empty compute-app
snapshots before and after and raw 250 ms GPU samples. The collector verdict is
`executed-not-qualified`, `qualification: false`; no serving, production,
performance, NVMe, P2P, or model-format claim follows from these cells. This is
not a scored or thermally balanced throughput comparison.

## APIs

- `take_device(&DeviceLease)` returns the exact `CudaSlice<u8>` allocation after
  owner-stream synchronization, with no live leases/bindings and removal of its
  registry/governor charge. Busy preserves ownership; a second take is foreign.
  The caller assumes accounting. The gate asserts **pointer identity**, not just
  byte equality.
- `retire_source(ticket)` drops only source-side retention after observed DMA.
  D2H permits subsequent explicit `release_device` while the destination host
  image remains owned/readable/charged. H2D releases source host backing/charge
  without releasing destination ownership. Whole-ticket `retired` stays false
  until the original consumer/graph/destination contract completes.
- Busy refusals cover a consumer bound to the D2H source, graph retention, H2D
  destination binding and retained device leases. Unknown observation returns
  Quarantined. Taking a destination is still exactly once.

## Native conformance — verbatim

```
PASS v1 transfer_cancel native CUDA
PASS v1.1 transfer_complete_cancel native CUDA
PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention
PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input
PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication
PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence
PASS additive source retirement Busy while source consumer bound; host destination survives source release
PASS native governor zero after controlled drain
```

## Native byte roundtrips

All six sizes passed: **4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB**.
Each checks D2H source registry/device bytes freed while the host remains live and
hash-equal, H2D source host charge freed, destination handed back at the same CUDA
pointer, exact readback bytes and SHA-256, and final governor zero. Raw verdicts
with both hashes: `native/roundtrip/command.log`. The run is one sweep, N=1 per
size, not a median or throughput result.

Both capture descriptors' raw-log/telemetry/snapshot byte counts and hashes were
recomputed locally (`receipt-audit.json`). Native changed-file and contract hashes
match the local source. Raw build logs, binary hash, preconditions, lock evidence,
all verdicts, and failed verification attempts are retained under `native/`.

## CPU and compiler checks

Final checks in `mac-final/checks.json`, each actually executed and exit 0:
workspace fmt; explicit native-module/bin rustfmt (registration remains a lead
fragment); Mac and Linux-target memra-tier/memra-kv checks; memra-tier offline
no-fail-fast tests (**181 passed**, across unit/integration/doc suites); CPU
crates clippy all-targets `-D warnings`; native module/bin standalone clippy;
`git diff --check`; flags census.

Native release engine + gate build: PASS. Native module/bin scoped clippy:
PASS on Rust 1.98.1. Full native engine clippy: **FAIL**, dependency crates report
20 `chunks_exact_to_as_chunks` lint errors (`memra-tier` 2, `memra-gguf` 18).
Mac CPU crates pass on Rust 1.97.1. These are distinct scopes; the scoped green is
not a full-workspace green. No lint suppression or unrelated repair was made.

Initial native clippy was unavailable; installed the standard toolchain clippy
component. Initial standalone offline native check lacked the Mac lockfile's
`cfg-if 1.0.5`; retry using the workspace Cargo.lock passed without downloading
new project dependencies. Those failed attempts remain in `native/`.

Standalone native typecheck reproduction: in ignored
`target/spill-a-native-typecheck`, create an independent workspace package with
library name `memra_engine`, a lib module path
`../../../crates/memra-engine/src/tier_transfer.rs`, and bin path
`../../crates/memra-engine/src/bin/tier_transfer_gate.rs`; depend on memra-tier
at `../../crates/memra-tier` and existing cudarc `=0.19.8`, default-features=false,
features `std,cuda-13010,dynamic-loading,driver`. Seed its Cargo.lock from the
workspace before offline clippy. This only checks native module/bin code; it
cannot replace the actual native engine build or full native engine clippy.

## Handoff and budget

B can now integrate exact native operand hand-back and source-demotion retirement;
E should lift the additive schedules into v1.3. Lead owns registration fragments
and integrated runtime qualification. Full native dependency-clippy repair remains
with the integration owner; no main merge or release occurred here.

This relaunched session used approximately **0.4 agent-hours**, below its 3-hour
stop limit. This is the day-7 milestone of the seven-day lane; prior sessions'
cumulative agent-hours were not reconstructed or invented. Initial GitHub timeout
resolved on one bounded retry; missing ControlMaster resolved by its owner.
