# integ21 self-review (lead, 2026-09-21)

Read in full: `crates/memra-engine/src/tier_transfer.rs` day-14 diff (`PinnedKind::for_device`, the `pinned_default`
field resolved once in `CudaTransfers::new`, `alloc_host` delegating with it, `pinned_default()` accessor, the three
cells and the source-text cell), `crates/memra-engine/src/parallel.rs` (`from_device_name` made `pub(crate)`),
`crates/memra-engine/src/bin/tier_transfer_gate.rs` (`PINNED-DEFAULT` lines), `docs/decisions/PINNED-DESTINATIONS.md`,
TESTING; both cards' receipts spot-checked.

## Findings
1. **A per-device default keyed the way the engine already keys hardware.** `HardwareTarget::from_device_name` is the
   existing per-device key; compute capability could not separate the two classes (both 12.0). The default is
   resolved exactly once (a source-text cell pins it), `alloc_host_kind` stays the measurement seam, no env read exists.
2. **The 5090 verdict is honest.** The pre-registered rule needed D2H no worse at every pair; it lost 5/10 by 70 us on
   a 7.3 ms DMA while the host read gained 39x. Inconclusive per the rule, so the 5090 keeps write-combined; the
   replay tool agrees with the binary's verdict and its FAIL summary counts the rule clauses, with all integrity checks
   green. Ruling 23 records what would move it.
3. **Byte exactness through the new default on both cards.** Conformance 13 PASS and roundtrips `byte_exact=true` at
   six sizes with the driver flags matching the resolved kind (2 on the PRO, 6 on the 5090).
4. **Nits (not blocking).** The unknown-name arm defaults to write-combined, which is today's behaviour; a future card
   class gets its own cell before it moves. The gate's `PINNED-DEFAULT` line prints the device name from the driver;
   the boundary policy treats card names as measurement conditions, not identifiers.

## Verification this review relied on
integ21 CPU battery (`integration-day12/integ21-cpu-battery/`): fmt, portable suites, memra-server suite, memra-engine
CPU lib tests, clippy, censuses, collector pytest, A's 5090 replay, perf board, diff-check; local 5090 serve-smoke on
this tree (`integ21-serve-smoke-5090/`). A's conformance and roundtrip cells on both cards. Pushed in the announced
development mode (engine source in range); no qualification claimed.
