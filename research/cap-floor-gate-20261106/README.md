# Spec-cost cap for MoE boots, the admission floor after the wave probe, and the cold-drafter gate port (2026-11-06)

Verdict: a default boot of a MoE model with a draft head panicked before it listened, because the boot spec-cost calibration measured a 16-session plain batched step on a model whose decode wave cap is 8. The calibration now measures plain steps only up to the model's own wave cap, and both MoE models boot and serve. The inflated admission floor after the wave probe (5017 MB on a pinned-K 9B boot on main) is already fixed by the wave probe's own pool trim from memra#1075 (1177 MB measured on the landing tip, the 1536 MB static floor serves); this lane adds no code for it. The cold-drafter gate measured a foreign server that held its port; it now refuses a port it does not own. Twenty log and error messages had lost a line continuation and printed indentation runs; they are fixed and a source census keeps them fixed.

## 1. Spec-cost calibration past the decode wave cap

Mechanism:

- `decode_step_batch` refuses B above the decode batch cap unless the model has the exact-16 tier (`decode_batch_exact16_ok`). A MoE FFN refuses that tier (the predicate requires a non-MoE trunk; `MEMRA_EXACT16_WHY=1` names the refusing layer), so `chunk_cap_for` gives a qwen35_moe trunk a wave cap of 8: both MoE boots log `decode wave cap 8; scheduler tick cap 8`.
- `spec_cost_calibrate` measured plain points `[1, 2, 4, 8, 16]` whatever the cap: 9fe005013c introduced fixed points up to 16, and 547771a2ed made the table follow the cap upward (32 on the b32 tier) through `b <= plain_cap.max(16)` and `interpolate_n(&plain, plain_cap.max(16))`. The `max(16)` kept the table at the 16 entries it always had; no boot with a cap below 16 ran the calibration before, so nothing caught it.
- Every MTP-served model on the default env runs the calibration at boot, so a MoE with an embedded MTP head (Ornith-1.5-35B-A3B) or an attached NextN draft (Qwen3.6-35B-A3B + owntrim) panicked in the B=16 step.

Change (`fix(engine,server)`):

- `plain_points(cap)` measures the fixed points below the cap and the cap itself, and the plain table has exactly `cap` entries: 8 on a MoE trunk, 15 on an mHC trunk (measured instead of held flat from 8), 16 or 32. The verify table is one session's t-parallel walk, not `decode_step_batch`, and keeps its 16 rows.
- Downstream readers of `plain_ms`: `spec_plain_ms` already clamps its chunk to the table length. `rider_forward_caps` indexed `plain_ms[min(riders, 16) - 1]`, which panics on an 8-entry table at 9 or more riders; it now prices the marginal plain row inside one full wave, `min(riders, 16, table length)`, which is unchanged for tables of 16 or more.
- `tools/serve-smoke.sh` arm 13 boots Qwen3.6-35B-A3B + its owntrim draft on the default env and requires the listening line, a `[spec-cost]` line with `plain_cap=8` and 8 plain entries, one greedy answer and no panic.

Tests (CPU host, no GPU; `raw/unit/`):

| test | before the fix | after |
|---|---|---|
| `spec_cost::tests::plain_points_stop_at_the_decode_wave_cap` | FAILED: cap 1 measured `[1, 2, 4, 8, 16]` (`red-engine.log`) | ok (`green-engine-unit.log`) |
| `worker::tests::rider_caps_read_a_table_shorter_than_16_rows` | FAILED: `index out of bounds: the len is 8 but the index is 8` (`red-server.log`) | ok (`green-server-unit.log`) |
| `spec_cost::tests::calibration_measures_the_plain_table_at_the_wave_cap` (source anchor, added on review) | FAILED on the landing tip's `spec_cost.rs`: no `plain_points(plain_cap)` (`red2-engine.log`) | ok (`cpu-suites.txt`) |

serve-smoke arm 13 on the RTX 5090 (`raw/serve-smoke/`): the landing tip's hashed cell build with this lane's `serve-smoke.sh` (`red-landing-tip/serving.log`) against the lane head's cell-evidence serving run.

Boot cells (RTX 5090, `raw/boot/`): see Results.

## 2. Admission floor after the wave probe

The report: a 9B boot with `MEMRA_SPEC_K=3` ran the wave probe, skipped the spec-cost calibration (the band owns a pinned K) and logged `post-probe pool trim: released 5120MB` and `transient floor 5017MB ... peak-mapped 13280MB`, against 1536 MB on a default boot. The same holds for every boot where nothing trims between the probe and the admission calibration: a K pin, a LOW/HIGH override, `MEMRA_SPEC_GATE_MODEL=0` and DSpark models.

The log that showed it was built from main (99c65ab0b2; its wave-probe line has no "released" suffix). memra#1075 (`fix(server): hand the boot wave probe's pool back before boot reads free memory`) trims after `decode_wave_probe` before either arm returns, and `wave_probe_releases_its_pool_before_boot_reads_free_memory` pins that order. The admission calibration resets its high-water mark to the pool's reservation and measures the peak against the trimmed rest, so a trimmed pool at the reset is all the floor needs.

Measured on the 5090 (`raw/boot/before/`): main 5017 MB measured (peak-mapped 13312 MB, rest 8192 MB, `calibration-complete` trim 5120 MB); the landing tip with #1075 1177 MB measured, the 1536 MB static floor serves (`wave-probe` trim 6176 MB, `calibration-complete` trim 1280 MB). The landing tip's post-probe trim releases only the probe's own 1280 MB, so no other boot step leaves the pool mapped. Decision: no code here. Rejected: a second trim before the admission calibration's high-water reset, which has nothing to release on any measured boot.

## 3. Cold-drafter gate port

On 2026-11-06 the gate's reference boot never came up while a foreign server held 8093: `waitfor --http` passed on that server and `ref.jsonl` recorded its answers (x0 `prompt_tokens=4075`, 8-token hex strings). The gate now sources `tools/port-guard.sh`; every boot runs `memra_port_guard cold-drafter-gate "$PORT" GATE_PORT` first, and after the ready wait `memra_port_owned cold-drafter-gate "$PORT" "$SP"` and a check that the boot's own log carries `[server] listening on http://127.0.0.1:<port>`. `GATE_PORT` (default 8093) replaces `GATE_ADDR`. `tools/dspark-cold-drafter-gate.py` boots nothing (it talks to a URL), so it needs no guard.

`tools/test_dspark_cold_drafter_gate.py` gained three cases (`raw/gate-port/`): a foreign server already on the port, a separate process that takes the port after the pre-flight check, and a responder that never logs the listening line. Against the gate before this change (with only `GATE_PORT` swapped in so the test can steer it) all three fail, and the occupied-port case reports `GATE PASS` on the foreign server's texts (`red-before.log`, 3 of 7 failed). With the guard all 7 pass (`green-after.log`).

On review: a refused boot writes no server log, so a rerun into the same out dir read the previous run's `serve-arm.log` and could print `SHAPE PASS` under a failing verdict. The gate now removes both serve logs with the run files and names a missing arm log as a failure. The occupied-port case pre-seeds a passing stale `serve-arm.log`: it fails against the first guarded gate (`red-stale-log.log`) and passes now (`green-after-review.log`, 7 of 7).

## 4. Lost line continuations

`dspark_prefer_restorable_hit`'s receipt printed 22 spaces inside "plain hit {plain_len}" and before "prime": the backslash that continues a string line was gone. The same shape was in nineteen more messages (eighteen in engine and server code, two in probe binaries). `string_continuation_tests` in memra-server is the census. It lexes every crate's `src/` outside `src/bin/` (probe binaries align columns on purpose): no non-raw string literal may hold a run of eight or more spaces between two of its characters on one source line before the literal's first newline. Comments, raw strings, char literals and lifetimes are skipped, a literal is followed across `\` continuations, and text after an escaped or real newline inside a literal is multi-line fixture text (aligned /proc/meminfo lines). The first version read one line at a time; red with 16 lines on a tree that had two already fixed (`raw/unit/red-lint.log`). The review found its blind spots: a run on the second line of a continued literal, and a run before a `\n` escape on the same line. The lexer version's cases fail against the per-line census (`red2-server.log`) and pass now; the tree has no hit.

## Conditions

RTX 5090 Laptop GPU, 24463 MiB, driver 615.71.09, local rig, 2026-11-06 to 07, under the 5090 lease. Each boot is a single run with one greedy chat request (23 prompt tokens, 32 max tokens). `boot_cells.py` strips inherited `MEMRA_*`, sets only `MEMRA_COMPAT=openai`, `MEMRA_ADDR` and the cell's env, refuses an occupied port and checks that the listener is its own child.

| arm | source | memra-server sha256 |
|---|---|---|
| base | main 5d70118b99 (cell-evidence build) | `505ffce123ab3589e2e2c8cb54cef23fb27d01de29ff33dd1b3b5985dcc446ba` |
| land | landing tip 0a34ce3444 (cell-evidence build) | `9a4a0f78d2c809866e8b19125641457aa740793156736aa096c602ab749ee9ce` |
| fix | 8b5f51db31, a pre-rebase state of this lane; its non-test engine and server code equals the lane head's | `c81fc06e79a158ed58d92a8d0a71ddb2e9c67fa0fa566e9bda19b1bba354e734` |

## Results

Cells (`raw/boot/before/cells.txt`, `raw/boot/after/cells.txt`; one JSON row per boot in `rows.jsonl`):

| cell | arm | result | spec-cost | measured transient (floor) |
|---|---|---|---|---|
| orn | land | exited 1 before listening: `decode_step_batch: B=16 > cap 8 with no exact tier` at 38.4 s | panicked in the plain table | none |
| orn | fix | listening at 38.1 s, request admitted | `plain_cap=8`, `plain_ms[1..]=[9.4 12.5 15.2 18.0 20.4 22.8 25.3 27.7]`, 10.2 s; K=3 from the cost model | 84 MB (1536 MB) |
| q36-spec | land | exited 1 before listening: the same panic at 33.7 s | panicked in the plain table | none |
| q36-spec | fix | listening at 34.0 s, request admitted | `plain_cap=8`, `plain_ms[1..]=[11.8 14.2 16.8 19.4 22.2 25.0 27.8 30.5]`, 10.5 s; K=3 from the cost model | 982 MB (1536 MB) |
| q9-pinned | base | admitted | skipped (pinned K) | 5017 MB (5017 MB): peak-mapped 13312, rest 8192 |
| q9-pinned | land | admitted | skipped (pinned K) | 1177 MB (1536 MB): peak-mapped 9408, rest 8128 |
| q9-pinned | fix | admitted | skipped (pinned K) | 1177 MB (1536 MB) |
| q9-default | land | admitted | `plain_cap=32`, 32 entries from points 1, 2, 4, 8, 16, 32 | 1179 MB (1536 MB) |
| q9-default | fix | admitted | `plain_cap=32`, 32 entries from the same points | 1179 MB (1536 MB) |

The exact-16 9B measures the same six plain points before and after, so its tables and verdicts are unchanged by construction; only a cap below 16 (or between the fixed points) changes what is measured. Both MoE models report `pooled=false`, so no pooled-overhead burst ran.

## Files

- `boot_cells.py`: the boot harness (pre-flight port check, ownership check, one greedy request per boot).
- `serve_smoke_red.py`: runs this lane's `serve-smoke.sh` against the landing tip's hashed cell build, the way `cell_evidence.py` runs serving.
- `raw/boot/{before,after}/`: per-boot logs (seconds since spawn on every line), `rows.jsonl`, `cells.txt`, `run.out`, binary hashes.
- `raw/unit/`: red and green logs of the Bug 1 unit tests, the calibration source anchor and both census versions; `cpu-suites.txt` is the CPU suite summary.
- `raw/serve-smoke/`: serve-smoke with arm 13 on the landing tip (red).
- `raw/gate-port/`: the cold-drafter gate test against the unguarded gate (`red-before.log`) and the guarded one (`green-after.log`).
