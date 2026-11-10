# The session gate compared the spec stream against a class the model does not serve (memra#1146, 2026-11-10)

Verdict: `session-gate` failed turn 2 at K=3 with 48 tokens per turn on Qwen3.8-27B (GB10 and RTX
5090) because its oracle was the eager `generate()`. The qwen35 (GatedDeltaNet) family serves
plain decode on the batched program, B=1 included, and its verify forward runs that class. The
eager step and a verify row differ in all 248,320 logits (max 0.12, measured at the failing
turn's 243-token history); batched plain steps and the verify rows over the same tokens are the
same bits (0 of 1,986,560 logits over four columns and two rounds, at a 211-token prompt). With
the oracle in the serving class the gate passes at K=3/48 and at its defaults on both devices.
Nothing in serving was wrong. `run-spec` now runs that bit comparison (`STEP-IDENTITY`) before
its K sweep and the coverage check requires the line.

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, the fast path (defaults). The diagnosis
ran on the engine of main 53eb765421 (RTX 5090 Laptop; GB10: one DGX Spark, the same tree built
for sm_121a). The landed form ran on this change rebased on main fcd66f8963.

## The failure on main

```
MEMRA_SPEC_K=3 MEMRA_NGEN=48 session-gate <model.gguf>          (rtx5090/sg-5090-k3-48.log)
turn 1: 50 tok generated, committed=177 -> MATCH
turn 2: 50 tok generated, committed=292 -> MISMATCH
  session:   [220, 15, 220, 15, 220, 15, 220, 15, 220, 15, 220, 15]
  reference: [220, 201, 248044, 248044, 2, 188, 188, 188, 188, 188, 188, 188]
turn 3: 49 tok generated, committed=375 -> MATCH
turn 4: 48 tok generated, committed=423 -> MATCH
```

The defaults (K=2, 24 tokens) pass (`rtx5090/sg-5090-def.log`). GB10's failing run is
`research/gb10-decode-20261109/eos-unit2/session-gate-k3-*.log`: the same turn, other tokens.

## Not K, not the session, not the resume (`rtx5090/routes.log`, `probes/routes_probe.rs`)

Turn 2's history is 243 tokens: turn 1's 128 ids, its 50 generated tokens, turn 2's 65 ids.

| route | first tokens |
|---|---|
| `generate()` plain greedy, the dc decode route | 220, 201, 248044, 248044, 2, 188, ... |
| `generate()` with `MEMRA_QWEN_DC=0` (`routes-qwen-dc-0.log`) | the same |
| `generate_spec` cold at K=1, K=2, K=3 | 220, 15, 220, 15, ... |
| a fresh session, no prime split, K=1, 2, 3 | 220, 15, 220, 15, ... |
| a fresh session, prime stop at 224 | 220, 15, 220, 15, ... |
| the turn-1 session rewound to its on-grid checkpoint (96), prime stop at 224 | 220, 15, 220, 15, ... |
| the turn-1 session resumed at its parked end (what the gate did) | 220, 15, 220, 15, ... |

Every speculative route agrees with every other. With `MEMRA_FAST=0` all routes agree, plain
included (`routes-fast-0.log`; that run's history is 242 tokens, turn 1 generated 49).

## The first differing value (`rtx5090/logits.log`, `probes/logits_probe.rs`)

Twin caches primed with the 243 tokens by `prime_cache` (the two primes: 0 of 248,320 logits
differ). The next step, for token 220:

| step | id 201 | id 15 | argmax |
|---|---:|---:|---|
| eager `decode_step` | 8.977869 | 8.956169 | 201 |
| verify forward, 1 row, in the exact scope | 8.956566 | 8.967237 | 15 |
| verify forward, 4 rows, row 0 | 8.956566 | 8.967237 | 15 |
| verify forward, 1 row, outside the exact scope | 8.956566 | 8.967237 | 15 |

- Eager against verify: 248,320 of 248,320 logits differ in bits, max |diff| 0.122.
- Verify over 1 row against row 0 of 4 rows: 0 differ.
- A second eager step from each cache: 248,319 differ, max 0.166. The two classes also leave
  different state.

The top two candidates are 0.02 apart, inside the gap between the classes. The engine already
names this: `batched_serving_numeric_class` in `spec.rs` ("eager-class verify logits drift from
batched-class serving logits, 1 ULP at layer 2 to 2.3e-1 logit maxdiff at the head, and the GDN
recurrence carries the drift until a near-tie flips"), and run-spec's oracle has been the serving
class since 2026-08-14. The session gate was not moved with it.

This probe has no batched row. The batched step against the verify forward is the
`STEP-IDENTITY` line of the runs below, at another prompt (211 ids, off the prime grid).

## With the oracle in the serving class

Three forms ran, each named by its log:

- `rtx5090/gate-candidate-*.log`: `probes/gate_candidate.rs`, the gate with run-spec's oracle,
  the served resume and a one-row step-identity check written into the binary (193 ids primed).
  At turn 2 of its K=3 run it also printed that the eager `generate()` leaves the serving-class
  oracle at token 1: the token the old gate failed on.
- `gb10/gate-candidate-*.log`: a later build of the same probe (211 ids primed, one row). Its
  source was not kept; the landed form below replaces it.
- `*/gate-head-*.log`, `*/run-spec-head-*.log`: the landed form, built from this change on main
  fcd66f8963. The oracle and the check are engine functions (`generate_spec_oracle`,
  `served_step_identity`); the check compares K+1 plain steps with one K+1-row verify forward,
  two rounds, from a 211-id prompt; `run-spec` runs it at four rows.

<!-- head rows -->
| device | model | run | step identity | verdict | log |
|---|---|---|---|---|---|
| RTX 5090 | Qwen3.8-27B (qwen35) | session-gate 3 / 48 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 4 of 4 turns MATCH, 2 rewound | `rtx5090/gate-head-q27-k3-48.log` |
| RTX 5090 | Qwen3.8-27B (qwen35) | session-gate 2 / 24 | 0 of 1,489,920 differ (3 columns, 2 rounds) -> MATCH | 4 of 4 turns MATCH, 2 rewound | `rtx5090/gate-head-q27-def.log` |
| RTX 5090 | Qwen3.8-27B (qwen35) | run-spec K=3 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 1 of 1 K pass, SELF-CONSISTENCY PASS | `rtx5090/run-spec-head-q27-k3.log` |
| RTX 5090 | Qwen3.8-27B (qwen35), `MEMRA_FAST=0` | run-spec K=3 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 1 of 1 K pass, SELF-CONSISTENCY PASS | `rtx5090/run-spec-head-q27-k3-fast0.log` |
| RTX 5090 | Ornith-1.5-35B-A3B (qwen35_moe) | session-gate 3 / 48 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 4 of 4 turns MATCH, 2 rewound | `rtx5090/gate-head-ornith-k3-48.log` |
| RTX 5090 | Ornith-1.5-35B-A3B (qwen35_moe) | run-spec K=3 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 1 of 1 K pass, SELF-CONSISTENCY PASS | `rtx5090/run-spec-head-ornith-k3.log` |
| RTX 5090 | Qwen3.6-35B-A3B UD-IQ4_XS with its external draft (qwen35_moe), as `tools/local-ci.sh` runs it | run-spec K=1..8 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 8 of 8 K pass, SELF-CONSISTENCY PASS | `rtx5090/run-spec-head-q36moe-k1-8.log` |
| GB10 | Qwen3.8-27B (qwen35) | session-gate 3 / 48 | 0 of 1,986,560 differ (4 columns, 2 rounds) -> MATCH | 4 of 4 turns MATCH, 2 rewound | `gb10/gate-head-q27-k3-48.log` |
| GB10 | Qwen3.8-27B (qwen35) | session-gate 2 / 24 | 0 of 1,489,920 differ (3 columns, 2 rounds) -> MATCH | 4 of 4 turns MATCH, 2 rewound | `gb10/gate-head-q27-def.log` |

Queued on GB10 at this head, to be added: Qwen3.8-27B run-spec K=3, Qwen3.5-9B session-gate 3 / 48, Qwen3.6-35B-A3B UD-IQ4_XS session-gate 3 / 48.
<!-- end head rows -->

## What changed

- `HybridModel::generate_spec_oracle`: run-spec's `plain_oracle`, moved into the engine. Plain
  greedy on the batched step for the families whose verify forward runs the batched class
  (Step35, GatedDeltaNet trunks); for the rest eager `generate`, or the eager step on a
  pipeline-split target.
- `HybridModel::served_step_identity`: the bit comparison above, with the plain step chosen by
  `served_decode_is_eager` (what the server decodes the model with), not by the oracle's
  predicate. The two primes are compared first, so a nondeterministic prime reports as itself.
  `run-spec` prints its `STEP-IDENTITY` line before the K sweep; on the families whose verify
  forward the engine declares in the serving class (Step35, GatedDeltaNet) a mismatch fails the
  run there, before the sweep and its banner. `tools/release-coverage.py` (the release battery,
  `tools/cell_evidence.py`) requires exactly one such line ending `-> MATCH`, or the named
  `n/a`, in a run-spec log. `session-gate` prints it before the turns, under the same rule.
- `session-gate`: the serving-class oracle; on a recurrent trunk the resume the served engine
  takes (rewind to the on-grid turn checkpoint with a grid stop, else a cold prime) in place of
  the parked-end resume the server refuses since the grid law (memra#1091, memra#1093). That
  resume was not the cause here (route table, last two rows), but it was not a served shape.
  The gate fails if no turn rewound, and its continuation burst is compared with the
  continuation of the previous turn's oracle.

## Scope

Which speculative routes the check covers, and what it says:

<!-- scope rows -->
- qwen35 (GatedDeltaNet, dense): Qwen3.8-27B on the RTX 5090 and GB10: `MATCH`. Qwen3.5-9B on
  GB10 is queued at this head.
- qwen35_moe (GatedDeltaNet, routed MoE): Ornith-1.5-35B-A3B and Qwen3.6-35B-A3B (with its
  external draft, the form `tools/local-ci.sh` runs) on the RTX 5090: `MATCH`. Qwen3.6-35B-A3B
  on GB10 is queued at this head.
- `MEMRA_FAST=0` on Qwen3.8-27B (RTX 5090): `MATCH`. The check holds on the oracle path too.
<!-- end scope rows -->

Not measured here:

- Step35: its verify forward is in the batched class by code (`step35_verify_batch_layers`). It
  needs the RTX PRO 6000 pair its gates run on; the first run-spec there prints the line.
- Hy3 and MiMo-v2 (a full-attention trunk with an MTP head, batch-served, verify forward still
  in the eager class): by reading, the line is expected to say `MISMATCH`. It is printed there
  and not enforced (`step_identity_enforced`), pending memra#1157, a decision about those
  families.
- The gemma program and a hyper-connection trunk have no generic verify forward to compare;
  the check returns `n/a` for them. The DSpark, DFlash2 and GLM routes have no MTP head and
  keep their own gates (`dspark_q38_gate` carries its own copy of the batched oracle).
- The verify-graph replay (the default verify of the GatedDeltaNet + routed-MoE family), the
  pooled verify and the device-token verify: the check runs the generic verify forward only.
  The K sweep against the serving-class oracle is their gate.
