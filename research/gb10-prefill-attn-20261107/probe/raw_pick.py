#!/usr/bin/env python3
"""raw_pick.py RAW_CSV: print the tensor-pipe, issue, occupancy, smem and warp-stall metrics of an
`ncu --page raw --csv` export (one kernel). stdlib only."""
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
hdr, units, vals = rows[0], rows[1], rows[2]
keep = ("pipe_tensor", "issue_active", "inst_executed.sum", "inst_executed_pipe", "warps_active",
        "average_warp_latency_issue_stalled", "average_warps_issue_stalled", "gpu__time_duration",
        "sm__throughput", "l1tex__data_pipe_lsu_wavefronts_mem_shared", "smsp__cycles_active.avg",
        "sm__cycles_elapsed.avg", "l1tex__data_bank_conflicts", "lts__t_bytes.sum", "dram__bytes")
for h, u, v in zip(hdr, units, vals):
    if any(k in h for k in keep):
        print(f"{h:90s} {v:>18s} {u}")
