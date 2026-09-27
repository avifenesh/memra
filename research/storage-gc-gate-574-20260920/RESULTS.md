# Temporary GC gate lifetime — issue 574

The ordinary Linux storage suite failed with `Busy` during final eviction and store reopen on
separate frozen `2d89127d` and `5fc676e0` preparation attempts. Those failures and their bounded
follow-ups remain separate evidence in [issue 574](https://github.com/avifenesh/memra/issues/574).
Neither attempt produced a GPU qualification result. No test was skipped or made tolerant of Busy.

## Causal mechanism and correction

The temporary `.ownership-gc` lock relied on closing its `File` at the end of the critical
section. A copied descriptor shares the open-file description and can keep that lock alive
past the end of the section. Concurrent process creation has such a copy before close-on-exec.
The exact operating-system interleaving of the original failures was not traced; the regression
below deterministically reproduces this product lifetime defect and the same reopen refusal.

A private `GcGate` now explicitly unlocks on release/drop. Successful open releases it only
after shared lifetime ownership is established; GC releases it only after shared ownership is
restored. An unlock error retains the descriptor rather than discarding an unknown fence.
The existing failed-restore path still forgets the gate and retains its lock. Shared lifetime
ownership, active leases, transaction ownership, genuine Busy refusals and deletion validation
are unchanged. There is no production retry/backoff, environment flag, unsafe code, or test-tolerance change.

## Deterministic CPU process control

The regression gives a real bounded child copies of the GC-gate descriptor and the separate
shared-lifetime descriptor. This safely prolongs the descriptor-copy window without adding
unsafe fork hooks to the crate, which retains `forbid(unsafe_code)`. It checks that:

- Admission remains Busy while the GC critical section is active.
- Admission succeeds after that section ends even while the child retains its copied descriptor.
- GC remains Busy while the child's independent shared-lifetime ownership survives.
- GC can proceed to the expected missing-object result after the child exits.

Child reaping and directory cleanup precede every assertion, including the failing control.
The original gate fails the second assertion with `Busy`; the corrected gate passes all four.
The existing upgrade-window/live-lease negative test remains intact.

## Evidence

Linux controls used base `5fc676e0221e4af26d65c1b92d8ad5978b54857b`, the final regression patch,
and then the minimal production patch. Exact source and raw-log hashes are recorded per command.

| Check | Result |
|---|---|
| Old implementation plus regression | Expected failure: 0 passed, 1 failed, Busy after completed scope |
| Corrected implementation, identical regression | 1 passed, 0 failed |
| Complete ordinary release-profile memra-tier suite | Passed, including all 53 storage tests with default scheduling |
| memra-tier all-target clippy, warnings denied | Passed |
| macOS regression, full tier suite and clippy | Passed after the same correction |

The [raw bank](raw/COMPLETE.json) preserves command arguments, exit codes and source hashes.
Logs and patch captures are losslessly gzip-compressed; decompress before checking canonical hashes.
This is CPU filesystem-lifecycle evidence, not a GPU, model, throughput or full release result.
