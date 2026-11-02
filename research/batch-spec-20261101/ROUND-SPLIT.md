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

# Burst split (pure refactor)

`generate_spec_inner2` is now a serial driver over three steps, so a worker can hold several
sessions' bursts at once and drive their rounds in lockstep:

- `spec_burst_begin`: the setup (prime, boundary token, init feed, draft setup, draft-KV fill,
  round-stream and verify-graph arms). Returns a `SpecBurst`. Every early exit of the setup is an
  error, so there is no "done before the loop" variant.
- `spec_burst_round`: one iteration of the old loop body (gate, ROUND-STREAM arm, prepare, verify,
  finish). Returns whether the burst is still live.
- `spec_burst_end`: the tail (final flush, stats lines, session tail).

The driver checks `SpecBurst::live()` (on_commit's verdict and `out.len() < max_new`) before the
first round, as the old `while` did. `SpecBurst` holds `rs: SpecRoundState` (with the fixed inputs
in `rs.cfg`), `on_commit`, `keep_going`, the round-stream buffers and the tail's session slots.
The single-shot call's cache, draft scratch and device d2t map live in a `SpecBurstStore` in the
driver's frame, because the round state borrows them. The prompt hiddens and the init-feed hidden
stay held to the tail, so the rounds see the same device memory as before.

## Battery (`battery/`, RTX 5090 Laptop, same harness and cells as above)

Reference = the round-split binaries (`6f302b60880fa8e1…` / `559b914819a7ffa4…`), refactor =
this commit:

| binary | run-spec | memra-server |
|---|---|---|
| burst | `04243a5dc649882b…` | `b1a9373b16aa0cc3…` |

Verdict `battery/compare-burst-split.txt`: **BYTE-IDENTICAL** on every cell, including the
56 + 4 + 4 per-burst `[spec-acc]` lines.

clippy `-D warnings` (memra-engine, memra-server, all targets): clean. Lib tests: memra-engine
674 pass, memra-server 1136 pass.
