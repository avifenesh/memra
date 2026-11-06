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
