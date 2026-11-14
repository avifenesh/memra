#!/usr/bin/env python3
"""readme_tables.py [GB10_DIR]: the tables and totals of memra#1316's README, as markdown, from
the assembled record (default: the `gb10` directory next to `tools`).

- the driver-level cells (ctx_probe.py records): creations, how many started with MemFree under
  3,000 MB, how many answered out of memory;
- the server cells (server_boot.py records) per arm: boots / boots that started with MemFree
  under 3,000 MB / deaths (arm B) or first tries that failed and were retried (arm T);
- totals over the server cells, split at 3,000 MB and for the first boot of each round (cells
  with a holder and cells without one apart), the try and the time of every recovered boot
  (the sum of its waits where the build logged that, the time since its first try began where
  the build logs that), and the median boot time per arm of boots that needed no retry;
- the holder's own context creations that failed (`holder failed` in a cell's run.log);
- the longest stretches with no free block of order 9 and up in the cells' 50 ms memory records
  (from the first sample with none to the first sample with one).
Standard library only."""
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GB10 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "gb10")


def rows(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def main():
    probe, server, holders = [], [], []
    for unit in ("unit-a", "unit-b"):
        base = os.path.join(GB10, unit)
        for cell in sorted(os.listdir(base)):
            p = os.path.join(base, cell, "boots.jsonl")
            if not os.path.exists(p):
                continue
            rs = rows(p)
            name = f"`{unit}/{cell}`"
            log = os.path.join(base, cell, "run.log")
            if os.path.exists(log):
                n = sum("holder failed" in x for x in open(log))
                if n:
                    holders.append((name, n, max((r.get("round", 0) for r in rs), default=0)))
            if os.path.exists(os.path.join(base, cell, "env.txt")):
                held = os.path.exists(log) and bool(re.search(r"HOLD_GIB=[1-9]", open(log).readline()))
                server.append((name, rs, held))
            else:
                probe.append((name, rs))
    print("| cell | creations | started under 3,000 MB | out of memory |")
    print("|---|---|---|---|")
    for name, rs in probe:
        free, bad = [], 0
        for r in rs:
            r = r.get("rec", r)
            m = r.get("mem_before_ctx") or r.get("mem")
            if m:
                free.append(m["MemFree_mb"])
            bad += (not r["ok"]) if "ok" in r else (r.get("rc") != 0)
        print(f"| {name} | {len(rs)} | {sum(v < 3000 for v in free)} | {bad} |")
    print()
    print("| cell | MemFree at the boot, median MB | B: boots / under 3,000 MB / died | T: boots / under 3,000 MB / first try failed |")
    print("|---|---|---|---|")
    tot = {a: {"n": 0, "ev": 0, "lo": 0, "lo_ev": 0, "r0": 0, "r0_ev": 0, "p0": 0, "p0_ev": 0, "ms": []}
           for a in "BT"}
    recovered, recovered_elapsed = [], []
    for name, rs, held in server:
        rs = [r for r in rs if "skipped" not in r]
        cols = []
        for arm in "BT":
            a = [r for r in rs if r["arm"] == arm]
            t = tot[arm]
            for r in a:
                ev = r["outcome"] == "fatal" if arm == "B" else r["first_try_failed"]
                lo = r["host"]["MemFree_mb"] < 3000
                t["n"] += 1
                t["ev"] += ev
                t["lo"] += lo
                t["lo_ev"] += ev and lo
                if "round" in r and r["boot"] == 0:
                    t["r0" if held else "p0"] += 1
                    t["r0_ev" if held else "p0_ev"] += ev
                if r["outcome"] == "ready" and not r["first_try_failed"]:
                    t["ms"].append(r["ms"])
                for x in r["lines"]:
                    m = re.search(r"context created on try (\d+), after (\d+) ms( of waits)?", x)
                    if m:
                        (recovered if m.group(3) else recovered_elapsed).append((name, int(m.group(1)), int(m.group(2))))
                if arm == "T" and r["outcome"] == "fatal":
                    print(f"ARM T DIED in {name}: {r}", file=sys.stderr)
            cols.append(f"{len(a)} / {sum(r['host']['MemFree_mb'] < 3000 for r in a)} / "
                        f"{sum((r['outcome'] == 'fatal') if arm == 'B' else r['first_try_failed'] for r in a)}")
        free = sorted(r["host"]["MemFree_mb"] for r in rs)
        print(f"| {name} | {free[len(free) // 2]:,} | {cols[0]} | {cols[1]} |")
    print()
    for arm, what in (("B", "died"), ("T", "first try failed, all came up")):
        t = tot[arm]
        print(f"- arm {arm}: {t['n']} boots, {t['ev']} {what}; under 3,000 MB: {t['lo_ev']} of {t['lo']}; "
              f"at 3,000 MB and over: {t['ev'] - t['lo_ev']} of {t['n'] - t['lo']}; "
              f"first boot of a round after a holder's exit: {t['r0_ev']} of {t['r0']}; "
              f"first boot of a round with no holder: {t['p0_ev']} of {t['p0']}; "
              f"median boot without a retry {statistics.median(t['ms']):.0f} ms over {len(t['ms'])}")
    print(f"- recovered boots (cell, try that succeeded, ms of waits): {sorted(recovered, key=lambda x: (x[2], x[0]))}")
    print(f"- recovered boots of the build that logs elapsed time (cell, try that succeeded, ms since the first try began): "
          f"{sorted(recovered_elapsed, key=lambda x: (x[2], x[0]))}")
    print(f"- the holder's own context creation failed (cell, times, rounds): {holders}")
    stretches = []
    for unit in ("unit-a", "unit-b"):
        base = os.path.join(GB10, unit)
        for cell in sorted(os.listdir(base)):
            p = os.path.join(base, cell, "memwatch.jsonl")
            if not os.path.exists(p):
                continue
            start = None
            for r in rows(p):
                none_free = sum(r.get("hi", [1])) == 0
                if none_free and start is None:
                    start = r["t"]
                elif not none_free and start is not None:
                    stretches.append((round(r["t"] - start, 1), f"`{unit}/{cell}`"))
                    start = None
    stretches.sort(reverse=True)
    print(f"- longest stretches with no free block of order 9 and up (seconds, cell): {stretches[:5]}")


if __name__ == "__main__":
    main()
