#!/usr/bin/env python3
"""memwatch.py OUT.jsonl PERIOD_MS STOP_FILE: MemFree, MemAvailable and Cached (MB) and the Normal
zone's free blocks of order 9 and up, every PERIOD_MS, until STOP_FILE exists (memra#1316). The
record around a boot says what the page allocator had when the context was created. Reads /proc
only."""
import json
import os
import sys
import time

out, period, stop = sys.argv[1], int(sys.argv[2]) / 1e3, sys.argv[3]
KEYS = ("MemFree", "MemAvailable", "Cached")
with open(out, "a") as f:
    while not os.path.exists(stop):
        row = {"t": round(time.time(), 3)}
        with open("/proc/meminfo") as m:
            for line in m:
                k, v = line.split(":", 1)
                if k in KEYS:
                    row[k] = int(v.split()[0]) // 1024
        with open("/proc/buddyinfo") as b:
            for line in b:
                p = line.split()
                if p[3] == "Normal":
                    row["hi"] = [int(x) for x in p[13:]]
                    row["lo"] = [int(x) for x in p[4:13]]
        f.write(json.dumps(row) + "\n")
        f.flush()
        time.sleep(period)
