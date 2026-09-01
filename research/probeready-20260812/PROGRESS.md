# cx-probeready progress

Status: complete

Branch: `lane/cx-probeready`
Base: `584ed0af05e5f8a29b318a410a8e76ed8a08292f`

## Scope

- Expose peer-probe integrity as an advisory `/readyz` detail without changing readiness.
- Soft-refuse new speculative admissions after the configured deferral bound while allowing
  existing speculative sessions and all plain admissions to continue.
- Clear the refusal only after a completed peer-integrity probe clears the deferral streak.
- Keep single-device serving and the no-force-probe-under-live-UVA invariant unchanged.
- Document the bound (default `4`) and retain raw verification receipts in this directory.

## Gates

- [x] `/readyz` detail covered for healthy, deferred, and degraded states.
- [x] Speculative admission refusal engages at the bound and clears after a completed probe.
- [x] Plain admissions remain unaffected.
- [x] Single-device serving never engages peer-probe refusal.
- [x] `cargo test` passes.
- [x] Local-CI correctness stage passes.
- [x] `RESULTS.md` and raw receipts are complete.
- [x] Intended lane changes are committed without pushing.

## Activity

- 2026-08-12: Confirmed clean isolated worktree at the requested base and recorded lane scope before
  runtime edits.
- 2026-08-12: Added shared peer-integrity health state, advisory `/readyz.peer_probe_integrity`,
  startup-resolved `MEMRA_PEER_PROBE_DEFERRAL_BOUND`, and a spec-only per-admission gate. A real
  deferred probe publishes `deferred_<n>` and latches `degraded` at the bound; a completed probe
  clears both state and gate. No probe is forced under live UVA.
- 2026-08-12: Focused receipts green: runtime re-probe tests 5/5, readyz advisory 1/1, and explicit
  single-device/no-PP state 1/1 (`raw/focused-*.log`).
- 2026-08-12: Full `cargo test` green: 439 passed, 0 failed, 2 CUDA-only ignored.
- 2026-08-12: Locked `tools/local-ci.sh` exited 0: correctness stage GREEN, kernel-check 106 cells
  green (one optional skip), spec K=1..8 8/8, argmax/depth gates green, serve smoke 0 failed,
  c=64 stress 64/64, and acceptance 1/1. GPU compute applications were empty before and after.
- 2026-08-12: Checkpointed the compiled, fully gated implementation and raw receipts as
  `bb047e2d6`; completed `RESULTS.md` and the raw hash manifest for the closing evidence commit.
