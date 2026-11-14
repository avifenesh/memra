#!/usr/bin/env python3
"""tab_probe.py RUN_DIR... : one summary per driver-level cell (ctx_probe.py records).

Per cell: boots, boots whose first try failed and at which call, the retries a failed call needed,
MemFree before the context was created (min, median, max), the free blocks of order 9 and up
(2 MiB and larger) before it, and how many of the kernel's NVRM lines in the cell's window are a
big-page fallback, a UVM system-memory failure and a context-buffer failure. `-v` adds one line
per boot. Standard library only."""
import json
import re
import statistics
import sys


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # probe_cell.sh's exit mode writes bc's ".5" for a fraction under one second.
            line = re.sub(r'("since_exit_s": )(-?)\.', r"\g<1>\g<2>0.", line)
            d = json.loads(line)
            if "rec" in d:
                rec = d["rec"]
                rec["trial"] = d["trial"]
                rec["since_exit_s"] = float(d["since_exit_s"])
                d = rec
            if "rc" in d and "steps" not in d:
                # A churn record: one cuCtxCreate in a long-lived process.
                d["ok"] = d["rc"] == 0
                d["mem_before_ctx"] = d.pop("mem")
                d["buddy_before_ctx"] = d.pop("buddy")
                d["steps"] = [{"call": "cuCtxCreate", "rc": d["rc"], "ms": d["ms"], "retries": d.get("retries"),
                               "retry_rc": d.get("retry_rc"), "retry_ms": d.get("retry_ms")}]
                if not d["ok"]:
                    d["first_fail"] = "cuCtxCreate"
            out.append(d)
    return out


def main():
    verbose = "-v" in sys.argv
    for run in [a for a in sys.argv[1:] if a != "-v"]:
        rs = rows(f"{run}/boots.jsonl")
        fails = [r for r in rs if not r.get("ok")]
        sites = {}
        for r in fails:
            sites[r.get("first_fail", "?")] = sites.get(r.get("first_fail", "?"), 0) + 1
        free = [r["mem_before_ctx"]["MemFree_mb"] for r in rs if "mem_before_ctx" in r]
        hi = [sum(r["buddy_before_ctx"][9:]) for r in rs if "buddy_before_ctx" in r]
        retries = []
        for r in fails:
            for s in r.get("steps", []):
                if s["rc"] != 0:
                    retries.append((s["call"], s.get("retries"), s.get("retry_rc"), s.get("retry_ms")))
        try:
            nv = open(f"{run}/nvrm.log").read().splitlines()
        except OSError:
            nv = []
        print(f"== {run}")
        print(f"   boots {len(rs)}  first-try failures {len(fails)}  by call {sites}")
        if free:
            print(f"   MemFree before the context, MB: min {min(free)} median {int(statistics.median(free))} max {max(free)}")
            print(f"   free blocks of order >= 9 before the context: min {min(hi)} median {int(statistics.median(hi))} max {max(hi)}")
        print(f"   retries after a failure (call, tries, final rc, ms): {retries}")
        ff = [r["mem_before_ctx"]["MemFree_mb"] for r in fails if "mem_before_ctx" in r]
        if ff:
            print(f"   MemFree before the failed ones, MB: {sorted(ff)}")
            print(f"   free blocks of order >= 9 before the failed ones: {sorted(sum(r['buddy_before_ctx'][9:]) for r in fails)}")
        print(
            "   kernel NVRM lines: "
            f"big-page fallback {sum('retrying with default page size' in x for x in nv)}, "
            f"UVM system memory (nv_gpu_ops.c) {sum('nv_gpu_ops.c' in x for x in nv)}, "
            f"context buffer (kgrctxAllocCtxBuffers) {sum('kgrctxAllocCtxBuffers' in x for x in nv)}"
        )
        if verbose:
            for r in rs:
                m = r.get("mem_before_ctx", {})
                ms = {s["call"]: s["ms"] for s in r.get("steps", [])}
                ms.setdefault("cuDevicePrimaryCtxRetain", ms.get("cuCtxCreate"))
                print(
                    f"   {r.get('trial', '')} {r.get('since_exit_s', r.get('boot', '')):>6} "
                    f"free {m.get('MemFree_mb')} cu_free {r.get('cu_free_mb')} "
                    f"hi {r.get('buddy_before_ctx', [])[9:]} ctx_ms {ms.get('cuDevicePrimaryCtxRetain')} "
                    f"{'ok' if r.get('ok') else 'FAIL ' + str(r.get('first_fail'))}"
                )


if __name__ == "__main__":
    main()
