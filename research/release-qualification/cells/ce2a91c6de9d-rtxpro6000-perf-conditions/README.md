# Card conditions around the timed RTX PRO 6000 perf cell of ce2a91c6de9d

These are receipts for the record next to this directory, `ce2a91c6de9d-rtxpro6000-perf`. That record belongs to the landing of memra#1303, memra#1332, memra#1340, memra#1339, memra#1342, memra#1344, memra#1345, memra#1346, memra#1321, memra#1347, memra#1348 and memra#1350. The receipts are not part of the record's payload manifest, and the gate does not read them.

The perf cell ran on card 3 of a four-card RTX PRO 6000 Blackwell Max-Q workstation on 2026-11-19, times in local time:

| step | time |
|---|---|
| card's lease began | 02:52:33 |
| warm-up | 02:52:33 to 02:52:59 |
| timed cell | 02:52:59 to 02:55:07 |

The card's power limit was 276 W throughout.

Cards 0, 1 and 2 ran other work at full load during the whole cell; see "The other three cards" below. They were not idle, as they were for the record of 2e42b5549830.

## Build and host

Both arms are builds made on the box that ran them, with CUDA 13.4 and rustc 1.97.1, as the record's `build.toolchain` says:

- base: 96390ef0e4, the main this landing goes onto;
- head: ce2a91c6de.

`nvcc` reports `Build cuda_13.4.r13.4/compiler.38855100_0`. The A/B uses one toolchain and one build host on both sides.

The run's processes were pinned to host CPUs 0-7 and 32-39, 16 of the box's 64 threads. The other work on the box ran on other CPUs.

## The power cap and the warm-up

On these cards the power cap steps the SM clock down after about a second of continuous load. A short run on an idle card therefore reads fast. Two things keep the comparison fair.

First, a warm-up ran inside the same lease, right before the timed cell (`warm/`). The base `run-gen` ran once on each roster model with a 512-token decode:

- Ornith: 279.40 tok/s over 1.8 s;
- Qwen3.8-27B: 79.21 tok/s over 6.5 s.

The warm-up took the card from 51 C to 65 C. The temperature kept rising through the timed cell, to 78 C.

Second, the cell is the fixed interleave A B B A A B per model, so both arms run through the same regime and the same drift. The interleave also spreads the neighbouring cards' heat and load equally over both arms.

The verdict is the ratio of the arms. The absolute rates are rates under this card's cap with these neighbours. They are not comparable with records from other cards, caps, builds or box loads.

## What was checked

A thermal slowdown during a timed arm voids the arm. The card's throttle state was read before the cell, after it, and once a second during it.

| reading | value |
|---|---|
| `sw_thermal_slowdown`, `hw_thermal_slowdown`, `hw_slowdown` | Not Active in all 129 one-second samples of the timed cell |
| software and hardware thermal slowdown counters | 0 us before and after |
| `sw_power_cap` | Active in 49 of the 129 samples. The card was at its 276 W cap, which voids nothing |
| temperature | 64 C at the start of the timed cell, peak 78 C |
| fan | 47% at most |
| power | peak 278.05 W, limit 276 W |
| SM clock | 2,355 MHz at most under load; 172 MHz between runs, while a model loads |

No arm is void. `arms.tsv` has one row per run with these columns:

- its end time;
- the samples inside it;
- the count of samples with a slowdown flag active, which is 0 in every run;
- the counter delta;
- the temperature range;
- the peak power.

The heat guard for this landing is stricter than the one for 2e42b5549830. It stops the lease at the first sample in which the card reads 92 C or more, or its fan reads 95% or more. It never fired: `heat.log` has one sample every 10 s, with a peak of 76 C and 47% fan. In the battery and serving record of this landing, `ce2a91c6de9d-rtxpro6000-battery-serving`, the same card peaked at 83 C with 54% fan, and that guard did not fire either.

## The other three cards

Cards 0, 1 and 2 belong to other work on the box, which ran for the whole cell. Each card ran one compute process, with 81,102 to 84,488 MiB used. `box-before.csv`, `box-5s.csv` (31 five-second samples) and `box-after.csv` record their state:

| card | utilization | power | temperature | fan |
|---|---|---|---|---|
| 0 | 100% in 26 of the 31 samples, 0% in 4, 82% in 1 | 30 to 270 W | 77 to 89 C | 63 to 79% |
| 1 | 100% | 227 to 266 W | 84 C | 58 to 59% |
| 2 | 100% | 227 to 281 W | 88 to 90 C | 81 to 82% |

No card showed its software thermal slowdown flag in any sample.

Card 3's own throttle readings, above, show no thermal or hardware slowdown in any timed sample. The neighbours' heat therefore did not cap card 3's clock beyond its power cap. The timed cell's two arms alternate within about two minutes under the same neighbour load.

## Result

From the record, with three runs per arm and 128 decoded tokens each:

| model | base | head | change | runs A | runs B |
|---|---|---|---|---|---|
| Ornith-1.5-35B-A3B | 274.70 tok/s | 274.59 tok/s | -0.04% | 274.70, 274.72, 274.69 | 274.24, 274.62, 274.59 |
| Qwen3.8-27B | 77.34 tok/s | 77.58 tok/s | +0.31% | 78.04, 77.34, 77.16 | 77.68, 77.58, 77.00 |

The arms overlap on Qwen3.8-27B. On Ornith, the three head runs sit 0.03% to 0.17% below the three base runs. The gate counts a regression only with complete separation and a median drop of at least 2% (`PERF_MIN_REGRESSION_PCT`). A 0.04% drop is far below that floor and below the per-run noise the floor was derived from. Neither model regressed.

## Files

- `throttle-before.csv`, `throttle-after.csv`: card 3's state right before and right after the lease: throttle flags, slowdown counters, temperature, fan, power, power limit, utilization and SM clock.
- `throttle-1hz.csv`: the same fields once a second from the start of the lease. The timed cell is the samples from 02:52:59 on.
- `box-before.csv`, `box-after.csv`, `box-5s.csv`: all four cards before the lease, after it, and every five seconds during it: utilization, memory used, power, temperature, fan and the software thermal slowdown flag.
- `arms.tsv`: the per-run table.
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
- `warm/`: the two warm-up runs' logs and their end times.
- `heat.log`: the heat guard's samples.
