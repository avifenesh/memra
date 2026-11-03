#!/usr/bin/env python3
"""Average selected ncu 'details' metrics per report CSV (one row per metric per launch)."""
import csv, sys
from collections import defaultdict

WANT = [
    "Duration", "Memory Throughput", "DRAM Throughput", "L1/TEX Hit Rate", "L2 Hit Rate",
    "Compute (SM) Throughput", "Executed Ipc Active", "Issue Slots Busy", "Registers Per Thread",
    "Achieved Occupancy", "Theoretical Occupancy", "Block Limit Registers", "Block Limit Shared Mem",
    "Warp Cycles Per Issued Instruction", "Active Warps Per Scheduler", "Eligible Warps Per Scheduler",
    "No Eligible", "L1/TEX Cache Throughput", "L2 Cache Throughput", "Waves Per SM", "Grid Size",
    "Dynamic Shared Memory Per Block", "Static Shared Memory Per Block", "Mem Busy", "Max Bandwidth",
    "Mem Pipes Busy", "SM Busy",
]

for path in sys.argv[1:]:
    vals = defaultdict(list)
    units = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("Metric Name", "")
            if name in WANT:
                try:
                    vals[name].append(float(row["Metric Value"].replace(",", "")))
                    units[name] = row.get("Metric Unit", "")
                except ValueError:
                    pass
    print(f"== {path}")
    for name in WANT:
        if vals.get(name):
            v = vals[name]
            print(f"  {name:38s} {sum(v)/len(v):12.2f} {units[name]} (n={len(v)})")
