# FA3/FA4 sm_120a feasibility progress

Status: source-only feasibility complete.

## Scope

- Compare FA3/FA4 mechanisms with memra's current flash-attention kernels.
- Classify each mechanism as portable, portable only in reduced form, or hardware-blocked on
  `sm_120a` with `mma.sync` and `cp.async` but without `wgmma` or `tcgen05`.
- Separate likely decode relevance from prefill relevance before recommending any kernel lane.
- Rank a gated `memra_fa3_overlap` lane and define its smallest correctness-first increment.

## Evidence contract

- Existing-kernel claims will cite repository file and line numbers.
- Upstream mechanism claims will be checked against current primary sources.
- This lane makes no GPU or performance measurements and adds no kernel code.

## Checklist

- [x] Read the lane inbox, project instructions, and prior hand-port hypothesis.
- [x] Read the current memra flash-attention path and its dispatch/build gates.
- [x] Read the H100 compile-gating precedent and the decode SOL-gap report.
- [x] Verify FA3/FA4 mechanisms and sm_120a instruction boundaries against primary sources.
- [x] Write `FEASIBILITY.md` with a ranked go/no-go recommendation.

## Guardrails

No kernel code, GPU use, measurements, formatting sweep, perf-board edit, merge, tag, or push.
