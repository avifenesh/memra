# Worker refusal diagnostic witness

At frozen `0ea0aaa2`, the selected phase passed ten cases, then case eleven
failed its error-reason assertion. The repaired admission/usage setup and actual
`advance_sample_emit` positive passed. The drift invocation correctly reported
its stale executable mapping through the operator diagnostic while returning
production's deliberately redacted `Event::Error`. Full cache and numerical/token
state snapshots before and after that invocation are byte-identical. The probe
incorrectly expected the private producer text in the public message. It failed
before the restored-origin retry, and sixteen later cases, including T16 fallback,
remain unrun. The selected index was quarantined and the failed tree resealed.

The repair preserves production's error logging, classification and public
redaction. A `cfg(test)` hook records the actual producer's class and message only
inside a thread-local scope around the actual caller invocation. The probe pairs
those records one-for-one with correctly classified, redacted public errors and
requires one exact causal refusal. Direct returned errors retain their existing
path. Earlier calls, other threads, unrelated generic errors, missing/orphaned
records and successful events cannot satisfy the witness. A scope clears on
normal exit or unwind; nested capture fails without replacing the outer scope.
No model validation, new execution guard or snapshot refresh precedes the actual
caller after drift. Existing state comparisons, fault-name checks and restored
same-origin refusal stay mandatory.

`../run-worker-diagnostic-tests.py` executes four CPU controls using the actual
production constructor/redaction/channel and the exact diagnostic module. Only
unused `SpecUsage` payload data is a stand-in. Disabling the producer hook in an
isolated copy fails two controls. Strict Linux-target server all-targets clippy,
format and diff checks pass. These CPU results do not qualify native callers.
Fresh immutable-source review, build and all selected native cases remain required.

Original stdout/stderr and CPU logs are compressed here without changing their
bytes; the evidence file records both raw and compressed hashes. The complete
failed native archive is separately hash-bound. No earlier failure is relabeled.

The resumed CPU follow-up preserves the capture implementation and all native
caller bodies from reviewed `b135038ee`. It adds malformed public-field,
missing/duplicate/orphaned record, separate restored-phase cause, and early-return
controls. All six CPU tests pass; removing the hook causes three required failures.
All 21 caller-runner controls also pass, retaining the 27-case schedule. Fresh
Linux-target server clippy, formatting and whitespace checks pass. The earlier
four-test evidence above remains unchanged; `resume-cpu-evidence.json` binds the
expanded controls and their raw logs.

`archive-reconciliation-20260921.json` records a fresh read-only verification of
the complete `ec1307fd` archive: 404 files, all six phase manifests and closed
leases, including 243 selected-caller payloads. The selected phase is still
failed with ten passing cases, one failure and sixteen unrun cases. Its index
remains quarantined and its before/after cache and state bytes still match.
This reconciliation issues no new native receipt.
