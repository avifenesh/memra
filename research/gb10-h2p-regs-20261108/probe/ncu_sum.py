#!/usr/bin/env python3
"""One-line-per-metric summary of an ncu --page raw --csv export: stall shares (pc sampling), issue,
warps active/eligible, tensor pipe, registers, local spill traffic, L2 throughput, duration.

usage: ncu_sum.py RAW_CSV
"""
import csv
import sys

rows = list(csv.reader(open(sys.argv[1])))
h = rows[0]
v = rows[2] if len(rows) > 2 else rows[1]
col = {n: i for i, n in enumerate(h)}


def get(n):
    try:
        return float(v[col[n]].replace(",", ""))
    except (KeyError, ValueError):
        return None


st = []
for n in h:
    if n.startswith("smsp__pcsamp_warps_issue_stalled_") and not n.endswith("not_issued"):
        x = get(n)
        if x is not None:
            st.append((x, n.replace("smsp__pcsamp_warps_issue_stalled_", "")))
tot = sum(x for x, _ in st) or 1.0
print("stalls: " + ", ".join(f"{k} {100 * x / tot:.1f}%" for x, k in sorted(st, reverse=True)[:8]))
for key, label in [
    ("gpu__time_duration.sum", "duration (ms under ncu)"),
    ("sm__cycles_elapsed.avg.per_second", "SM clock (GHz)"),
    ("launch__registers_per_thread", "registers/thread"),
    ("launch__block_size", "block size"),
    ("smsp__warps_active.avg.per_cycle_active", "warps active/scheduler"),
    ("smsp__warps_eligible.avg.per_cycle_active", "warps eligible/scheduler"),
    ("smsp__issue_active.avg.pct_of_peak_sustained_active", "issue active %"),
    ("smsp__issue_inst0.avg.pct_of_peak_sustained_active", "no-issue cycles %"),
    ("sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", "tensor pipe active %"),
    ("sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active", "hmma inst % of peak"),
    ("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "tensor cycles active %"),
    ("smsp__inst_executed.sum", "instructions executed"),
    ("smsp__sass_inst_executed_op_local_ld.sum", "local loads (warp instr)"),
    ("smsp__sass_inst_executed_op_local_st.sum", "local stores (warp instr)"),
    ("l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum", "local load requests"),
    ("lts__t_sectors.avg.pct_of_peak_sustained_elapsed", "L2 sectors % of peak"),
    ("lts__t_sector_hit_rate.pct", "L2 hit rate %"),
]:
    x = get(key)
    if x is not None:
        print(f"{label}: {x:,.3f}")
