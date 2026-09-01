# API honesty cluster — progress

Lane: `lane/cx-honesty`, base `b752cf2d`.

Status: complete at `5c585a16`. No GPU/model run was required or performed. No origin push,
merge to train, or tag.

The requested steering file `~/.lanectl/inbox/cx-honesty.md` was absent at lane start. It appeared
during the implementation test block with a required `cx-pinfix` train sync. The work was stashed,
the local `restructure/public-split` tip `04babb07` was fetched/rebased, the implementation was
restored cleanly, and every focused/full gate was rerun on the combined tree.

## Scope and invariants

Three server/engine serving-contract defects are in scope, implemented as one coherent change
where their paths overlap:

1. Clamp speculative emission and billing at the request budget while preserving every
   engine-committed token and cache row. Surplus speculative tokens remain continuation state;
   engine session-mode commit semantics do not change.
2. Emit one `Event::Token` for every in-budget speculative token id, with that token's own
   incremental text delta. Terminal token snapshots remain an authoritative receipt, not a
   substitute for the per-event contract.
3. Account successful speculative and legacy round-robin output on a per-emitted-token basis,
   including per-token step samples and lane/starvation timestamps. A speculative round is not
   one token; rejected candidates and budget-surplus commits are not client output.

No GPU kernels, model bytes, quantization paths, generated perf boards, or product docs are in
scope.

## Plan and gates

- [x] Add focused unit tests reproducing budget overshoot, coalesced speculative token events,
      missing speculative accounting, and missing round-robin accounting.
- [x] Implement the shared emission/accounting helpers needed by those tests.
- [x] Run `cargo test -p memra-server` (current train baseline 138 + 4 = 142 PASS).
- [x] Run `cargo test -p memra-engine` (55 passed, 2 GPU-required ignored, 0 failed).
- [x] Run `cargo build --release` (PASS, optimized profile).
- [x] Record exact before/after receipts and gate outcomes in `RESULTS.md`, then stop.

Success means the public completion count never exceeds the requested maximum, speculative
token events are 1:1 with the generated-token receipt, and all successful scheduler paths move
the same output/timing counters in the same per-token units.
