# Card conditions around the timed RTX PRO 6000 perf cell of 2e42b5549830

Receipts for the record next to this directory, `2e42b5549830-rtxpro6000-perf` (the landing of memra#1315, memra#1327, memra#1308 and memra#1326). They are not part of the record's payload manifest; the gate does not read them.

The perf cell ran on card 3 of a four-card RTX PRO 6000 Blackwell Max-Q workstation on 2026-11-18. The card's lease began at 02:12:02 local time, the warm-up ran to 02:12:31, and the timed cell ran from 02:12:31 to 02:14:42. The card's power limit was 276 W throughout.

## Build and host

Both arms are builds made on the box that ran them: base 5a6ae2438e and head 2e42b55498, with CUDA 13.4 (`nvcc` reports `Build cuda_13.4.r13.4/compiler.38855100_0`) and rustc 1.97.1, as the record's `build.toolchain` says. The A/B is one toolchain and one build host on both sides.

The run's processes were pinned to host CPUs 24-31 and 56-63, 16 of the box's 64 threads. Other work on the box ran on other CPUs.

## The power cap and the warm-up

On these cards the power cap steps the SM clock down after about a second of continuous load, so a short run on an idle card reads fast. Two things keep the comparison fair:

- Inside the same lease, right before the timed cell, the base `run-gen` ran once on each roster model with a 512-token decode (`warm/`): Ornith 277.32 tok/s over 1.8 s, Qwen3.8-27B 79.43 tok/s over 6.4 s. The card went from 51 C to 57 C. The temperature kept rising through the timed cell, to 76 C.
- The cell is the fixed interleave A B B A A B per model, so both arms run through the same regime and the same drift.

The verdict is the ratio of the arms. The absolute rates are rates under this card's cap and are not comparable with records from other cards, caps or builds.

## What was checked

A thermal slowdown during a timed arm voids the arm. The card's throttle state was read before the cell, after it, and once a second during it.

| reading | value |
|---|---|
| `sw_thermal_slowdown`, `hw_thermal_slowdown`, `hw_slowdown` | Not Active in all 132 one-second samples of the timed cell |
| software and hardware thermal slowdown counters | 0 us before and after |
| `sw_power_cap` | Active in 48 of the 132 samples: the card at its 276 W cap, which voids nothing |
| temperature | 57 C at the start of the timed cell, peak 76 C |
| fan | 43% at most |
| power | peak 276.94 W, limit 276 W |
| SM clock | 2,347 MHz at most under load; 180 MHz between runs, while a model loads |

No arm is void. `arms.tsv` has one row per run: its end time, the samples inside it, the count of samples with a slowdown flag active (0 in every run), the counter delta, the temperature range and the peak power.

The heat guard stops a lease only when the card reads 92 C or more with the fan at 95% or more in three samples in a row. It never fired (`heat.log`, one sample every 10 s).

## The other three cards

Cards 0, 1 and 2 held 2 MiB and ran no compute process before the cell, in each of their 32 five-second samples during it, and after it (`box-before.csv`, `box-5s.csv`, `box-after.csv`).

Card 0 read 100% utilization for the whole window, with 2 MiB used, no compute process and 104 to 112 W. It looks like a stale reading; an earlier record on this box saw the same on another card. It was constant across the window, so it does not bias an interleaved cell.

## Result

From the record: Ornith-1.5-35B-A3B base 274.47 tok/s, head 274.46 (0.0%); runs A 274.08, 274.49, 274.47 and B 274.45, 274.92, 274.46. Qwen3.8-27B base 77.80, head 78.09 (+0.37%); runs A 78.57, 77.80, 77.65 and B 78.17, 78.09, 77.56. The arms overlap on both models. Three runs per arm, 128 decoded tokens each.

## Files

- `throttle-before.csv`, `throttle-after.csv`: card 3's throttle flags, slowdown counters, temperature, fan, power, power limit, utilization and SM clock right before the lease and right after it.
- `throttle-1hz.csv`: the same fields once a second from the start of the lease; the timed cell is the samples from 02:12:31 on.
- `box-before.csv`, `box-after.csv`, `box-5s.csv`: all four cards (utilization, memory used, power, temperature, fan, the software thermal slowdown flag) before, after, and every five seconds during the lease.
- `arms.tsv`: the per-run table. `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make it.
- `warm/`: the two warm-up runs' logs and their end times.
- `heat.log`: the heat guard's samples.
