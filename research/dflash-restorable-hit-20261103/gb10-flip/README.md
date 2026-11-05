# Restore doors on by device default: GB10 served A/B (2026-11-05)

Change (branch `dspark-restore-default-on`, measured binary 6561499516): with `MEMRA_DSPARK_SPEC=1`, `MEMRA_DSPARK_PREFIX_RESTORE`, `MEMRA_DSPARK_PARTIAL_RESTORE` and `MEMRA_DSPARK_BOUNDARY_CAPTURE` default ON on GB10 and the RTX 5090 and OFF elsewhere. `1` or `0` still decides. `MEMRA_PREFIX_STABLE_BOUNDARY` is unchanged.

Verdict: on GB10, doors on by default beats doors off at c1 (+17.4% across units, +17.1% in a same-unit ABBA of 2+2 runs) and c2 (+5.9%) and is flat at c4..c32 (within 0.9%). Exactness holds with the doors unset. The flip is adopted for GB10. The RTX 5090 default rests on its exactness gate; no served doors-on vs doors-off cell ran on that card (its agentic shape logs 0 vetoes, `../5090-cold-drafter-ab/`).

## Exactness

`tools/dspark-cold-drafter-gate.py` with the doors unset (`DOORS=default`), so the arm runs on the device default:

| run | a greedy | a sampled | b greedy | b sampled | conversions | boot line |
|---|---|---|---|---|---|---|
| GB10, spark-cb33 (`../cold-drafter-gate/gb10-flip-6561/`) | identical | identical | identical | identical | 4 | `default ON on "NVIDIA GB10" (gb10=true)` |
| RTX 5090 Laptop (`../cold-drafter-gate/5090-flip-6561/`) | identical | identical | identical | identical | 4 | `default ON on "NVIDIA GeForce RTX 5090 Laptop GPU"` |

## Served arms

GGUF Qwen3.8-27B (NVFP4 + Q5_K) with the DFlash2 drafter, FR-Spec trim, `MEMRA_MTP_SKIP=1`. The agentic cell of `research/batch-spec-20261101/GB10-AGENTIC.md`: 48 turns, `max_tokens` 768, T=1 seed 1234, thinking on, fresh boot per arm, one run per arm, aggregate generated tok/s.

| c | 1: #1031 ea5ef13e8b, doors on (launcher) | 2: batch-spec 655a07cb82, doors on (launcher) | 3: flip 6561499516, doors unset (default ON) | 4: #1031 ea5ef13e8b, doors off | 3 vs 4 |
|---:|---:|---:|---:|---:|---:|
| 1 | 14.39 | 12.32 | **14.28** | 12.16 | **+17.4%** |
| 2 | 23.30 | 23.58 | **23.55** | 22.23 | **+5.9%** |
| 4 | 40.27 | 40.66 | 40.43 | 40.50 | -0.2% |
| 8 | 62.41 | 62.82 | 62.66 | 62.95 | -0.5% |
| 16 | 77.54 | 77.79 | 77.77 | 78.35 | -0.7% |
| 32 | 83.25 | 84.38 | 84.19 | 84.98 | -0.9% |

| arm | unit | bench client | `MEMRA_PREFIX_STABLE_BOUNDARY` | vision tower | vetoes / cold-drafter / DSPARK restores | receipts |
|---|---|---|---|---|---|---|
| 1 | spark-cb33 | off the unit, ssh tunnel | 1 | loaded (`MEMRA_VISION_DIR` set by that launcher) | 56 / 15 / 12 | `../gb10-served/arm1-1031-doors-on-cb33/` |
| 2 | spark-b23d | off the unit, ssh tunnel | 1 | not loaded | 56 / 0 / 12 | `../gb10-served/arm2-batchspec-doors-on-b23d/` |
| 3 | spark-b23d | on the unit | 0 | not loaded | 56 / 15 / 11 | `arm3-flip-default-b23d/` |
| 4 | spark-cb33 | on the unit | 0 | not loaded | 58 / 0 / 0 | `arm4-1031-off-cb33/` |

Arms 1 and 2 ran 07:52-09:25; arms 3 and 4 ran in parallel 13:46-15:09 on the two units.

Reading:
- c1: the doors turn 15 vetoed plain hits into cold-drafter DFlash primes. Every doors-on arm with the #1031 code sits at 14.28-14.39, every arm without conversions at 12.16-12.32.
- c2: arm 4 is the only arm without restores, and the only one under 23.3.
- c4..c32: arms 3 and 4 differ by 0.2-0.9%, single runs on two units. Arms 1 and 2, the same doors on two units, differ by up to 1.3% at these cells. I read c4..c32 as flat.

## Harness notes

- Arms 3 and 4 run the bench client on the unit (`b23d_arms_local.sh`, `arms_local2.sh`), with the same serve script and bench steps as arms 1 and 2. The bench host's Wi-Fi went to 150-350 ms round trips to the units at 12:57 and dropped the first arm-3 attempt (`void-arm3-1256-tunnel/`): the warm step's `/tokenize` timed out, then the ssh tunnel died 14 minutes into c1 ("server not responding"; every later request failed with curl exit 7). No number from that attempt is used.
- Arm 4 on cb33 used a copy of the b23d serve script (`serve-batchspec-novision.sh`), so it does not set `MEMRA_VISION_DIR` and matches arm 3. Arm 1 used cb33's own launcher, which does.
- `b23d_stop.sh` stopped the b23d chain after arm 3, so the unit could go to other evidence work. The c1 repeats run on cb33 (`cb33_tail.sh`, below).

## c1 repeats (cb33, ABBA)

Arm 4's c1 is the first A (off); the repeats follow on the same unit, launcher and client in the order on, on, off (`cb33_tail.sh`, log `cb33-chain.log`), fresh boot each.

| order | doors | run | c1 tok/s | wall s |
|---:|---|---|---:|---:|
| 1 | off | arm 4, c1 of the full run (`arm4-1031-off-cb33/`) | 12.16 | 1609.2 |
| 2 | on | `c1r-on1-cb33/` | 14.22 | 1329.3 |
| 3 | on | `c1r-on2-cb33/` | 14.23 | 1327.7 |
| 4 | off | `c1r-off-cb33/` | 12.14 | 1612.0 |

Doors on 14.22 / 14.23, doors off 12.16 / 12.14: mean 14.23 against 12.15, +17.1%, with no overlap between the arms. Arm 3's c1 on b23d (14.28) sits with the doors-on runs.
