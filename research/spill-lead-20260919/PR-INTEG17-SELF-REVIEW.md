# integ17 self-review (lead, 2026-09-21)

Read in full: C day 15 `crates/memra-server/src/worker.rs` diff (`HostPlaneBytes`, `HostTierLedger`,
`host_kv_planes_through_contract`, the five `take()` callers, the receipt line, the quarantine latch),
`worker/host_glm.rs` (borrow follow-through), the door doc "Option B" section, TESTING rows, `verify-day15.py`; receipts
spot-checked.

## Findings
1. **OFF untouched, ON is the contract's own sequence.** Every changed statement is behind the door; the ON path is
   the `kv_tier_gate/active.rs` sequence with one `submit_batch` per entry, the contract's checksums and completion,
   and the same K and V slices returned to the same slots through `take_plane`/`into_pooled`: no second copy program,
   no placeholder, no device byte (ruling 14 honoured; the census names the alternative that would have been a D2D).
2. **Same bytes.** `MEMRA_KV_HOST_VERIFY=1` digests equal OFF and ON on every promote; the bind hash equals the
   receipt checksum (and the failure gate's injected difference is named by the door, then `VERIFY FAILED` as before).
3. **Ledger adapter.** `HostTierLedger` puts the server's mutex governor behind the engine's `BudgetGovernor` trait;
   one ledger, two handles, `Quarantined` typed on a poisoned lock; `inflight` sized to the plane count so the first
   `submit_batch` is admitted.
4. **Gates.** Identity and failure gates equal OFF/ON under the default spec env; lane A's tenant-reclaim fix arm and
   lane B's two gates unchanged under ON; serve-smoke byte-equal. N=1 each, executed-not-qualified.
5. **Findings kept, not claimed.** Write-combined destinations and the single-observation demote/promote deltas are
   recorded as Option C's first cell, not as a result; the allocation flag stays engine territory.
6. **Nits (not blocking).** C merged lane A's branch before #597 landed; the commits coincide with main, no divergence.
   `HostPlaneBytes::Contract` keeps a `CudaPinnedLease` alive per plane for the entry's host lifetime, which is the
   contract's charge model; the decide-by review should read the pinned budget under ON against OFF.

## Verification this review relied on
integ17 CPU battery (`integration-day12/integ17-cpu-battery/`): fmt, portable suites, memra-server suite, clippy,
censuses, collector pytest, `verify-day15.py`, perf board, diff-check; local 5090 serve-smoke on this tree with the door
unset (`integ17-serve-smoke-5090/`). C's target-card OFF/ON gates. This rig cannot run the model gates.
