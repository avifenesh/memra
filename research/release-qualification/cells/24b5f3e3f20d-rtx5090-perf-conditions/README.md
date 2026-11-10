# Card conditions around the RTX 5090 Laptop perf cell of 24b5f3e3f20d

Receipts for the record next to this directory, `24b5f3e3f20d-rtx5090-perf` (the `release: v0.141.0` commit, PR memra#1206). They are not part of the record's payload manifest; the gate does not read them.

The 5090's battery and serving cells are in `24b5f3e3f20d-rtx5090-battery-serving`. The perf cell ran as its own record, after a cool-down, on a quiet host, with the host build lock held so that no build could start during it.

## The card ran under a 95 W limit

As for the three landings before it, the laptop was not in its performance power mode for this cell. `nvidia-smi` reported `power.default_limit` 95 W (maximum 175 W), the platform profile read `balanced`, and the software power cap flag was active in every one of the 333 samples. The card drew at most 92.6 W.

So the absolute rates here are rates at 95 W, not this card's rates. The cell is an interleaved A B B A A B comparison of two binaries under the same limit, and that comparison is what the record carries.

## Summary

| | |
|---|---|
| window (2026-11-10, local) | 19:34:05 to 19:39:38 |
| start | 59 C, 1-minute load 2.3 |
| host during the cell | mean run queue 4.2, 8% user CPU |
| power | limit 95 W, peak draw 92.6 W, software power cap active throughout |
| Ornith-1.5-35B-A3B | +0.49% (A 121.19, 120.90, 123.39; B 122.07, 121.78, 120.66 tok/s), spread 2.3% |
| Qwen3.8-27B | +0.00% (A 31.10, 31.11, 31.21; B 31.11, 31.11, 31.21 tok/s), spread 0.4% |
| verdict | pass, no arm separation on either model |

No thermal slowdown: the software and hardware thermal slowdown flags were not active in any sample, in any run. The card peaked at 78 C.

## Files

- `throttle-before.csv`, `throttle-after.csv`: the card's throttle flags, temperature, power, utilization and clock right before and right after the cell.
- `throttle-1hz.csv`: the same once a second during the cell.
- `host-load-5s.tsv`: the host's run queue and CPU split every 5 s.
- `arms.tsv`: one row per run (end time, samples, samples with a thermal or hardware slowdown flag active, temperature range, peak power).
- `arm_conditions.py`: the script that joins the samples with the perf logs' end times to make `arms.tsv`.
