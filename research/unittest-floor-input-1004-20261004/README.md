# Shared unittest admission input closure

The ten dynamic CPU contracts invoked `tools/unittest-floor.sh` without declaring
that shell or its Python sibling as inputs. Missing sibling source could pass
selected-input preflight and fail only when the shell executed.

The correction adds both files to all ten input lists and explicitly retains full
selection for edits to either shared wrapper. Commands, floors, presence and
required flags, Python requirements, and native obligations are unchanged.

`witness.json` records one actual Git integration on clean commit
`cb62bb34aada07528faf80aaa8e2a6adfd9170ab`. Its 1,509 fixture files come from
immutable legacy commit `71a91c6fce00fda25a2562387c1bf521e259c23a`, with blob,
SHA-256 and Git mode recorded for each. Planner imports and metadata are pinned.
Git configuration, hooks, templates, signing and automatic services are isolated.

The witness executes original preflight acceptance, the original shell failure,
all ten corrected missing-sibling refusals, Git restoration, and the real shell
passing two owned fixture predicates. It also checks all ten missing-shell
refusals, 20 coherent omitted-input mutants, both full-selection guard mutants,
ordinary scoped selection, unknown-input full selection, and 60 field/floor drift
refusals. Fixture analysis must be valid before a selection control can pass.

The registry covers all 14 workspace packages. Final service runtime was 62.813
seconds, with 60.813 CPU seconds, 410.3 MiB peak memory and no swap. The scoped
witness runs no unrelated collector suites or native compilation.

Earlier cohorts are retained separately. The initial fixture failed. The second
passed against its recorded uncommitted diff. Neither is labelled clean execution.
Public logs normalize absolute local paths and omit local process-unit metadata;
each records the SHA-256 of its original raw log.

CPU tooling evidence only. No model, runtime, serving, or native qualification.
Publicity: skipped: maintenance release.

Current-head CI then found two failed subtests in one existing collector method:
it assumed every declared input selected only the collector. The shared wrapper
inputs deliberately retain full validation. `review-repair.json` retains that
failed hosted cohort and the corrected named method passing through the real
wrapper at clean `becd3160b524653c933de6435ae0e33f156ee353` (0.274 seconds).
Every original collector-specific assertion is preserved. The two new shared
cases assert full mode, every job, CUDA requirement, full native scope and no
qualification. The planner and Git witness remain byte-identical to `cb62`.
