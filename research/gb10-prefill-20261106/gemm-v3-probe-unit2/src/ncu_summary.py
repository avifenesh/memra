#!/usr/bin/env python3
"""Summarize ncu --page raw --csv exports side by side: pipes, issue, stalls, memory.
usage: ncu_summary.py NAME=file.raw.csv ..."""
import csv
import sys


def load(path):
    rows = list(csv.reader(open(path, newline="")))
    hdr = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
    names, units, vals = rows[hdr], rows[hdr + 1], rows[hdr + 2]
    out = {}
    for n, u, v in zip(names, units, vals):
        try:
            out[n] = (float(v.replace(",", "")), u)
        except ValueError:
            out[n] = (v, u)
    return out


def main():
    reps = []
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        reps.append((name, load(path)))
    keys = set().union(*[set(r) for _, r in reps])

    def pick(pred):
        return sorted(k for k in keys if pred(k))

    groups = [
        ("time / issue / occupancy", pick(lambda k: k in (
            "gpu__time_duration.sum", "sm__cycles_elapsed.avg.per_second", "smsp__issue_active.avg.pct_of_peak_sustained_active",
            "smsp__inst_executed.avg.per_cycle_active", "smsp__warps_eligible.avg.per_cycle_active",
            "smsp__warps_active.avg.per_cycle_active", "launch__registers_per_thread",
            "smsp__thread_inst_executed_per_inst_executed.ratio", "sm__warps_active.avg.pct_of_peak_sustained_active"))),
        ("pipes (pct of peak, active)", pick(lambda k: k.startswith("sm__pipe_") and k.endswith("cycles_active.avg.pct_of_peak_sustained_active"))),
        ("instructions per pipe (sum)", pick(lambda k: k.startswith("sm__inst_executed_pipe_") and k.endswith(".sum")
                                              and "pred_on" not in k)),
        ("stalls per issued instruction", pick(lambda k: k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio"))),
        ("memory", pick(lambda k: k in (
            "lts__throughput.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "l1tex__throughput.avg.pct_of_peak_sustained_active", "lts__t_bytes.sum.per_second", "dram__bytes_read.sum.per_second",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum", "smsp__sass_inst_executed_op_local_ld.sum",
            "smsp__sass_inst_executed_op_local_st.sum", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"))),
    ]
    w = max(len(n) for n, _ in reps)
    for title, ks in groups:
        print(f"## {title}")
        for k in ks:
            vals = []
            for _, r in reps:
                v = r.get(k, ("-", ""))[0]
                vals.append(f"{v:>14.4g}" if isinstance(v, float) else f"{str(v):>14}")
            if title.startswith("stalls"):
                if all((r.get(k, (0.0,))[0] if isinstance(r.get(k, (0.0,))[0], float) else 0.0) < 0.02 for _, r in reps):
                    continue
            if title.startswith("instructions") and all((r.get(k, (0.0,))[0] if isinstance(r.get(k, (0.0,))[0], float) else 0.0) == 0.0 for _, r in reps):
                continue
            print(f"  {k[:86]:<86} " + " ".join(vals))
    print("columns: " + " ".join(n for n, _ in reps))


if __name__ == "__main__":
    main()
