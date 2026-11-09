# Card conditions around the timed RTX PRO 6000 perf cell of e6613b4eee22

Receipts for the record next to this directory, `e6613b4eee22-rtxpro6000-perf` (memra#1153). They are not part of the record's payload manifest; the gate does not read them.

The perf cell ran on card 0 of a four-card RTX PRO 6000 Blackwell Max-Q workstation, 250 W power limit, on 2026-11-10 from 00:28:49 to 00:30:59 local time. The other cards were quiet for the window: card 1 idle, card 2's training job paused with its memory held, card 3 with no compute process.

## What was checked

A thermal slowdown during a timed arm voids the arm. The card's throttle state was read before the cell, after it, and once a second during it.

| reading | value |
|---|---|
| `sw_thermal_slowdown`, `hw_thermal_slowdown`, `hw_slowdown` | Not Active in all 130 one-second samples |
| software thermal slowdown counter | 1237713845 us before and after, delta 0 |
| hardware thermal slowdown counter | 0 us before and after |
| temperature | 58 C at start, peak 83 C, 79 C at end |
| power limit | 250 W throughout |

No arm is void. `arms.tsv` has one row per run: its end time, the samples inside it, the count of samples with a slowdown flag active (0 in every run), the counter delta, the temperature range and the peak power.

## One oddity on the box

Card 3 read 100% utilization with 2 MiB used, no compute process and about 100 W, before and after the cell (`box-before.csv`, `box-after.csv`). It looks like a stale reading. It was constant across the window, so it does not bias an interleaved A B B A A B cell.

## Files

- `throttle-before.csv`, `throttle-after.csv`: card 0's throttle flags, slowdown counters, temperature, fan, power and utilization right before and right after the cell.
- `throttle-1hz.csv`: the same fields once a second during the cell.
- `box-before.csv`, `box-after.csv`: all four cards (index, utilization, memory used, power; the first file also has temperature and the software thermal slowdown flag).
- `arms.tsv`: the per-run table.
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
