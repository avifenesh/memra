# MTP round split (pure refactor)

The MTP spec round in `generate_spec_inner2` (spec.rs) is now four steps that a caller can
pause at the verify boundary:

- `spec_round_gate`: the context-edge room guard (memra#659), graph-launch headroom, profiler mark.
- `spec_round_prepare`: snapshot, draft, pending carry, adaptive K, p-min cut, `verify_tokens`,
  checkpoint. Returns `SpecRoundPrep`, including `poolable` (no verify-graph pool this round, no
  ROUND-STREAM, no PP split).
- the target verify, unchanged: `decode_step_t_core_vg`.
- `spec_round_finish`: greedy or sampled accept walk, `commit_verified_prefix`, pending stash,
  telemetry, budget clamp, round flush.

Loop-carried state lives in `SpecRoundState`, fixed inputs in `SpecRoundCfg`. Only the
ROUND-STREAM burst stays inline in the loop: it runs M rounds with no host readback, which has no
verify boundary to pause at. The memra#659 census test now pins the same properties on the new
shape (the loop's first statement is the gate, the gate's first check is the room guard, the
stream arm's M-round bound precedes its engagement).

## Battery (`battery/`, RTX 5090 Laptop, driver 615.71.09, q9 + owntrim draft)

Base = batch-spec at 8458a31fb8 (spark-sm121 dc73d7b362 + the lane), refactor = base + this
commit. Same window, one binary pair:

| binary | run-spec | memra-server |
|---|---|---|
| base | `31e642120d0d914a…` | `1e12785e63a9005b…` |
| refactor | `6f302b60880fa8e1…` | `559b914819a7ffa4…` |

Cells: run-spec greedy and sampled (T=0.8, seed 1234) K=1..8, an adaptive-K + p-min + zero-draft
cell, the legacy replay cell; served streams (each a fresh boot): greedy and sampled short, greedy
and sampled long (multi-burst, pending carry), sampled with penalties and top_p, greedy
json_schema (constrained spec), a concurrent greedy pair, and `MEMRA_SPEC_DEMOTE_AT=40` twice
(sampled + greedy). Verdict `battery/compare-round-split.txt`: **BYTE-IDENTICAL** on every cell,
including the 56 + 4 + 4 per-burst `[spec-acc]` lines.

clippy `-D warnings` (memra-engine, memra-server, all targets): clean. Lib tests: memra-engine
674 pass, memra-server 1136 pass.
