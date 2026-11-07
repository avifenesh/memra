# Path coverage of the fix-stack tip on the RTX 5090 (2026-11-06)

A gate covers a change only if its model and its environment execute the changed code. A gate that exists, or that touches the same file, does not count. This directory records how the fix stack's tip, 99c65ab0b2, was checked against that rule on one RTX 5090 Laptop (82 SMs, 24 GB). The gate-script commits after the tip change no runtime code.

Paths in the scripts and logs are placeholders: `<work>` is the scratch checkout and logs, `<models>` and `<hf-models>` are the model stores.

## Method

1. A path review listed every changed program region between a1e9b5a9a9 and the tip (493 items) with the model, environment and device that execute it, and a proposed cell (`raw/path-review.txt`, 76 rows).
2. The first pass ran the proposed RTX 5090 cells (`cells/gpu-1106.sh`, `gpu-1106b.sh`, `gpu-1106c.sh`; statuses in `raw/`).
3. A second pass mapped each of the 29 rows marked runnable here to the log line that proves its code ran, and an adversarial verifier rechecked each claim (`raw/rows-vs-cells.json`): 10 covered, 17 partial, 2 not covered.
4. The owed cells ran in `cells/gpu-1106d.sh` (`raw/logs-gpu-1106d.status.txt`). The serving cells are scenarios of `cells/serve_cells.py`; each prints `CELL <name> PASS|FAIL` lines (`raw/*cells.txt`).

## Results of the second pass

| Row | What ran | Result |
|---|---|---|
| A6 | a host-sampled plain row beside a pooled MTP cohort | left behind: no unified step forms; a control boot forms one; every row equals a plain boot |
| A6, C4 | sampled cohort under `MEMRA_SPEC_K=3`, rows 0.5 s apart, two boots | row 0 equals solo; all rows equal across boots |
| A7 | rider cell with no cost table, twice | 7 unified steps per run, 21 of 21 streams equal run to run |
| A8 | band demotion (HIGH=3) and DFlash2 demotion (HIGH=4) | sampled rows hand off at their philox event; greedy rows equal solo |
| A12 | 32-row wave with the tensor-core tier off | 24 concurrent rows equal solo |
| B11 | choice-group queue, wait and deadline sheds; the body-admission pool | every `[shed]` line and 429 code fires |
| B15 | memra-probe built for 121a and 100a | 121a refuses a 12.0 card before the fatbin load; 100a warns and builds 120a |
| C2 | riders under a K pin (calibration skipped) | riders engage; greedy rows equal a plain boot |
| C4 | Eager pooled-draft arm (penalised members) | bitwise equal, all sampled and mixed |
| C5 | DFlash2 pool cell against its stored serial run | 15 of 15 streams SAME; cohorts of 2 to 8; the VRAM headroom cap |
| C9 | spec-cost calibration without pooling | `pooled=false pooled_overhead=None`; cost verdicts follow |
| C12 | DFlash2 tap rows of a pooled verify; the verify-T sweep | tap rows bitwise equal at 4x4 and 8+8 |
| D2, D4 | 27B at 24 rows, tensor-core tier on and off | the grouped batched kernels engage; gate2 passes |
| D14 | the vision tower from its directory's own checkpoint | loads; texts equal the earlier run |
| D23 | local-ci correctness | the GPU-only library tests run and report their skips by reason |
| D24 | install.sh on a 12.1 device (stubbed) | maps to 121a |

A sampled cohort sent all at once differs from solo and between boots: arrival order is a race, and a row that primes beside a live peer ends its prime burst after one token (one boundary draw moves by design).

## Not covered here, and why

- The DSpark restorable-hit preference and the retained-plan route (rows A1, A2, A17): they need the cold-drafter capture fix, which is on its own branch.
- The sampled restore guard: under load, admission sets K to 0 first (`source=concurrency`), so the restore verdict never runs.
- The 27B 32-row tier and GLM-5 cells: they need a 96 GB card or weights not on this box.
- Rows that need the PRO 6000 Max-Q pair or GB10 run there.

## Found on main

- A default boot of a MoE model with an MTP head (Ornith-1.5-35B-A3B) panics at worker init: the boot spec-cost calibration measures 16 plain rows on a model whose decode cap is 8 (exact 16-row MoE decode is opt-in). `raw/logs-gpu-1106d_serve-orn.cells.txt`.
- The boot wave probe leaves its memory mapped: a default 27B boot loses its prefix cache, and with DFlash2 refuses every request; a successful probe also inflates the floor when no calibration trims after it.
- The cold-drafter conversion publishes no tail entry, and its gate could measure another process's server on its fixed port.

## 2026-11-07: the restacked tip, the PRO 6000 Max-Q pair and the quiet slot

Main moved six times on 2026-11-07 and the stack was restacked onto each move; every layer's delta equaled main's delta. The runs below name the stack commit they built. `<cache>` is the scratch root, `<pair-kit>` and `<maxq-kit>` are the kit directories on the PRO 6000 box, `<box-work>` its work root and `<box-home>` its home; `<tid>` replaces thread ids in panic lines.

### RTX 5090 (`cells/gpu-1107.sh`, `gpu-1107n.sh`, `gpu-1107p.sh`, `gpu-1107s.sh`)

| Row | What ran | Result |
|---|---|---|
| A1, A17 | retained-plan route: agent workload, one stream, T=0, the DSpark doors on and off (a0a6816ece) | the plan admits a DFlash2 route over the restored prefix; 29 of 29 turns byte-equal on and off; 11 DSpark restores |
| A2, A17 | restorable-hit preference (`serve_cells.py restorable`) | not covered: the boundary capture runs (a 2592-token tail entry), then reclaim-on-defer evicts that entry and spares only the plain entry, so the preference never runs; at 4096 context the same (finding below) |
| | cold-drafter gate with the port guard and the listening-line check | 4 of 4 identical; the cache-preferred route and one cold-drafter restore |
| | wave-probe floor, the 2026-11-06 binary against the tip | the old binary refuses a 161-token request on an idle server; the tip admits it |
| | default boot of a MoE model with an MTP head, old binary against the tip | the old binary panics at worker init; the tip serves (plain cap 8) |
| | DSpark pair cell and the drafter host-spill identity gate, the three doors (1a6b6eb671) | warm equals cold on turns 2 and 3 with two DSpark restores; identity ALL GREEN |
| | GPU probe recovery gate | fail before (a0a6816ece): the boot's wave probe spent the scripted hangs before readiness, recover FAIL; pass after the gate arms its fake after readiness (1a6b6eb671): recover, latch and fatal PASS |
| | kernel-check FULL with the F8F4 v2 bit cells; local-ci full; the release battery; serve-smoke | ALL GREEN (113 cells); local-ci rc=0; battery PASS; 47 ok |
| | fast-gate named probes (1a6b6eb671) | 21 of 22 PASS; accept skips (no cell artifact on this card) |
| | argmax-margin gate as merged with the stack's decode arm (1dfbaec5e7) | amargin, amarginb, amarginbc, amarginc PASS; battery PASS |

### PRO 6000 Max-Q pair (`cells/stage6.sh`, `pair-cells6.sh`, `pair-extra6.sh`, `pair-extra6b.sh`)

| Row | What ran | Result |
|---|---|---|
| B0 | TP/EP placement on two cards, both placements (TP_EP-5, REPL-3) | PASS |
| B1 | the split-rows branch (TP_EP-7 kvsplit) | PASS |
| B2 | full-token replay (FIX-6); the four sibling gates (SIB-7) | FIX-6 PASS. The AR epoch fix holds in all four gates (the 2026-11-05 run stopped at an epoch assertion). dense-exact-tail PASS. The other three stopped at count constants written before exact attention TP2, the fused HC finish and the vocab-parallel head; the gates now derive those counts from the loaded program and count a fused pair node as two launches (`raw/pair-out6*.census.txt`). Their rerun with both fixes is owed to the next pair window |
| | 22 regression cells; ep-pair with the drafter; serving on TP/EP | all rc=0 |

### Quiet slot, one card of the pair (`cells/quiet-1107.sh`, `cells/power-cell.sh` at 300 W)

| Row | What ran | Result |
|---|---|---|
| D21 | spec-cache-mixed gate, fixed (prefix budget for the six repeats, a one-stream window) | fail before (pair window, old script): no repeat took the cache; pass after: ALL GREEN, every repeat took the cache, 6 spec-engaged hit rows byte-identical to spec-off (`raw/*spec-cache-mixed.verdicts.txt`) |
| | DSpark serve smoke, fixed (the LOW=1 tooth decided from the admit and decision order) | fail before: the tooth counted a row decided before its peer arrived; pass after: both rows decided after both admits, plain; r1 to r3 identical |

The card ran both gates at 300 W, peaked at 88 C with no thermal slowdown, and returned to its 280 W limit after (`raw/quiet-outq.power-*.txt`). The kv-vmm timed half that ran first in this slot is banked with its lane.

### Found on main (2026-11-07)

- Reclaim-on-defer evicts the restorable tail entry and keeps only the longest-match plain entry, so the restorable-hit preference cannot run on a 24 GB card with Qwen3.8-27B and DFlash2. A fix is on its own branch.
- With one DSpark door (prefix restore only), turn 3 of the pair cell differs from a cold prime on main's own binary: turn 2 is served plain from the boundary entry, its prime splits at the prompt end, off the GDN grid, and turn 3 resumes those rows. The three doors pass.
- fast-gate writes its logs under `/tmp`, not `$TMPDIR`.

## 2026-11-07 evening: the stack on main 72f0e26fba (#1082, #1083, #1086, #1087)

### RTX 5090 at 79d7f2c125 (`cells/gpu-1107v.sh`)

| Row | What ran | Result |
|---|---|---|
| A2, A17 | restorable-hit cell at context 8192 and 4096, greedy and sampled (`serve_cells.py restorable`) | covered: r3's admission reclaim spares the restorable entry (2592 tokens, draft tail), drops the longer plain entry (3104 tokens) and r3 restores DFlash from the restorable one (2592 of 3161 tokens, 569 to prime); r3 equals the cache-off reference, greedy and sampled. Fail before: the same cell at 1a6b6eb671 (before #1082), where the reclaim evicted the restorable entry |
| | DSpark pair cell under the one restore switch (`MEMRA_DSPARK_PREFIX_RESTORE=1`) | warm equals cold on turns 2 and 3, two strict-prefix restores |
| | the same pair cell at the RTX 5090 default (switch OFF) | turn 3 differs from cold (finding below) |
| | host-spill identity gate, drafter arm, one switch; serve-smoke; the release battery | ALL GREEN; 47 ok; PASS (kernel-check 114 cells) |

The restorable cell's `preferred` and `restore` rows read FAIL in `raw/logs-gpu-1107v_serve-*.cells.txt`: the cell then looked only for the hit-site swap line. With #1082 the reclaim drops the plain entry first, so the lookup finds the restorable one directly (`raw/logs-gpu-1107v_serve-8192.*-arm.lines.txt`). `cells/serve_cells.py` now credits that route.

### Found on main: a decode-extended recurrent session resumes off the prime grid

The DSpark pair cell on main's own binary (05a8a63503 plus #1085, which touches neither path), at the RTX 5090 default:

| Park compaction | Turn 2 `max_tokens` | Turn 3 resume | Turn 3, warm against cold |
|---|---|---|---|
| off | 96 | 3072 of 3144 (the parked session's cap is too small for turn 3) | equal |
| on | 96 | 3113 of 3144 (compaction grows the parked session) | 373 against 372 characters |
| off | 256 | 3113 of 3144 (the parked cap now covers turn 3) | 373 against 372 characters |

Both orders, twice each (`raw/parkcompact-status*.txt`). The parked session holds turn 2's prompt and four rows decode wrote; 3113 is off the 32-token prime grid, so the suffix prime runs a second numeric program. The continuation-pool resume checked only the cap (`plain_resume_cap_admits`). Park compaction widens the exposure: #1062 makes it the RTX 5090 default, so #1062 is blocked on the fix.

The fix (memra#1091, PR #1092) rewinds a recurrent session's extension hit to its grid checkpoint. On its binary all four arms equal cold, each rewinding 3113 committed rows to 3072 (`raw/parkcompact-status3.txt`).
