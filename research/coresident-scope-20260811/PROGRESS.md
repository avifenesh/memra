# Co-resident model refactor scoping progress

- Status: complete
- Date: 2026-08-11
- Lane: `lane/cx-coresident`
- Scope: read-only enumeration and sizing of the code surfaces required for N independent models, each with its own CUDA-owner worker, in one `memra-server` process.
- Constraints: no source or config edits; no implementation decision; no merge, tag, push, `cargo fmt`, or performance-board edit.
- Evidence target: current main checkout at `/home/avifenesh/projects/bw24`.
- Deliverable: `SCOPE.md` with current file:line citations for config parsing, worker ownership, engine ownership, health/model/metrics handlers, scheduler behavior, and tests.

This file was created before implementation-source inspection, as required by the lane brief.

## Result

- Evidence freeze: the current source checkout and remote `main` both resolved to
  `96afb32e197e973b256ba61a733bb185cf767302` at final validation. The lane remains rooted at
  `4e7a4a3343b8d3dffaa2170ee9eea5fca6a4d910`; every source path cited in `SCOPE.md` is unchanged
  between those commits.
- Sized surfaces: config parser **S**; worker spawn/run **L**; engine/PP ownership **L**;
  health/models/metrics **M**; scheduler **L**; tests/gates **M**.
- Consolidated size: **L, upper end**. This is enumeration only, not an implementation verdict.
- Highest-risk seams recorded without selecting a solution: process-global PP runtime state;
  worker startup/failure/routing ownership; cross-worker scheduler signals and observability.
- Validation: every cited path and line range was checked against the frozen source checkout;
  the lane contains only this progress record and `SCOPE.md` for the task.
- Deliberately not run: formatter, build, tests, GPU gates, benchmarks, or board generation. These
  would not add evidence to a read-only code-surface inventory and were outside the brief.
