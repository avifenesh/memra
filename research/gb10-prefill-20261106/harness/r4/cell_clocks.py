#!/usr/bin/env python3
"""cell_clocks.py CAMPAIGN_DIR: per cell and target, the SM clock, power and temperature over the
request's busy window (gpu.csv at 250 ms, nvidia-smi: time, SM clock, mem clock, temperature,
power, utilization), so a cell that ran in a lower clock regime is visible beside its TTFT."""
import csv
import datetime as dt
import json
import pathlib
import sys


def load(f):
    out = []
    for r in csv.reader(open(f)):
        try:
            t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            out.append((t, float(r[1].split()[0]), float(r[3]), float(r[4].split()[0]), float(r[5].split()[0])))
        except (ValueError, IndexError):
            pass
    return out


d = pathlib.Path(sys.argv[1])
print(f"campaign {d.name}: SM clock / power / temperature over each request's busy window")
print(f"{'cell':6s} {'target':>7s} {'ttft_s':>8s} {'clk_med':>7s} {'clk_p10':>7s} {'W_med':>6s} {'C_med':>5s} {'C_max':>5s}")
for cell in sorted(p for p in d.iterdir() if p.is_dir() and (p / "pf.jsonl").exists()):
    g = load(cell / "gpu.csv")
    reqs = [json.loads(line) for line in (cell / "pf.jsonl").read_text().splitlines()]
    busy = [x for x in g if x[4] > 50]
    # requests run back to back in pf.jsonl order; walk them from the end of the busy record
    end = busy[-1][0] if busy else None
    for r in reversed(reqs):
        if end is None:
            break
        start = end - dt.timedelta(seconds=r["total_s"])
        w = [x for x in busy if start <= x[0] <= end]
        if w:
            clk = sorted(x[1] for x in w)
            pw = sorted(x[3] for x in w)
            tc = sorted(x[2] for x in w)
            print(f"{cell.name:6s} {r['target']:7d} {r['ttft_s']:8.3f} {clk[len(clk) // 2]:7.0f} {clk[len(clk) // 10]:7.0f} "
                  f"{pw[len(pw) // 2]:6.1f} {tc[len(tc) // 2]:5.0f} {tc[-1]:5.0f}")
        prior = [x for x in busy if x[0] < start - dt.timedelta(seconds=2)]
        end = prior[-1][0] if prior else None
