# Mirrored prefetch signed-counter regression

The first `bdd377a0` controls all exited zero, but review found a material counter defect: a
mirrored projection is charged once, creates two jobs, and the worker decrements once per job.
The existing public statistic clamps negative values to zero, masking underflow and weakening
future admission. The raw mirrored path contained the same latent defect. The original result
bank is retained in `attempt-1-bdd377a0/` with source GO explicitly withheld.

The new regression leaves production accounting unchanged so it can reproduce the defect. A
test observer holds only alternate halves while three primary halves complete and finish their
worker bookkeeping. It reads the actual signed counter, attempts another three projections at
cap=3, releases the alternates, drains every submitted job, then asserts the charge stayed three,
excess admission was zero and the final signed balance is exactly zero. Successful retry follows.
A second control truncates the retained alternate after primary completion to inject EOF and
checks exact balancing and no annex publication on the error path. Assertions occur after drain.

Observer exports and callbacks are test-only. Production builds still exclude them. The pending
minimal repair moves the decrement into the final projection_pending completion branch, after
both halves finish, for both successful and failed projections. Red and green must use identical
regressions on distinct frozen source revisions; earlier logs are not relabeled.

## Reproduced red at 291c1ef5

[Issue #586](https://github.com/avifenesh/memra/issues/586), finding
`ARCH541-CPU-MIRROR-01`, reproduced at immutable source
`291c1ef5c8a610131748645ac1d1666a121c71ef`. Three warning-clean builds passed;
all four signed regressions failed at the intended assertion under pipeline 0 and 1.
With alternates held, the success control observed `blocked_counter=0`, admitted three
extra projections despite cap=3, and drained to `final_counter=-6`. The alternate-EOF
control observed `blocked_counter=0` and drained to `final_counter=-3`. Both expected
three charges while alternates remained blocked. All jobs were drained before assertion.

Exact raw logs, commands, source and binary hashes are preserved losslessly under
`counter-red-291c1ef5/`; 914 source files matched their manifest before and after the run.
The repair now decrements once inside the last-half completion branch, after annex
success or abort. Regression source is unchanged. Its distinct frozen revision must
pass the identical controls and the prior mirrored/single-reader controls before review
can close the finding. This does not authorize root activation or model/GPU qualification.

## Green at 8852faac

The identical regression sources pass at
`8852faac7a88dba2493cc7d784deff06afbb3e0f`. Both pipeline settings retain
`blocked_counter=3`, admit zero extra projections at cap=3 and drain to exactly zero.
The success arm retries the same keys; the alternate-EOF arm publishes no annex data.
All 25 build/case commands pass, including the prior mirrored positive and refusal
controls, aligned buffered/direct reads, buffered/direct detached prefetch, original
ABI parity and legacy-extension refusal. No observer symbols appear in the production
library. The 914-file snapshot remains byte-identical after execution; its only source
delta from the red snapshot is the one-line decrement move in `memra_cpu_experts.cpp`.

Exact logs and manifests are in `counter-green-8852faac/`. The production library SHA256
is `550c2383582091098b83e836b356ddc2545a9d4c8b278e4d0848699cc0e20824`; the
observer library SHA256 is
`17e8f50df328de9af8935f1759d39d23f2e8fc70f5e43438b03b35d9c3d9511f`.
The original finder independently closed ARCH541-CPU-MIRROR-01 on exact `8852faac7`
with bounded source GO; the report and verification proofs are preserved losslessly in
`counter-review-8852faac/`. These are bounded CPU synthetic correctness
receipts: predictor routing, root activation, model and GPU qualification remain separate.
The historical `bdd377a0` result bank remains source_GO=false.
