#!/usr/bin/env python3
"""tab_boots.py RUN_DIR... : per arm, the boots of a memra-server cell of memra#1316.

Reads `cycles.jsonl` (serve_cycle.sh: full serving cycles) or `boots.jsonl` (server_boot.py:
boot-only loops). Per arm: boots, boots that came up, boots that died (FATAL), boots whose first
context creation failed (a `[ctx-retry]` line: with the retry these come up, without it they are
the ones that die), and for the retried ones the tries and the waits the server logged; the
same counts for the boots that started with MemFree under 3,000 MB and, in a transition cell, for
the first boot of each round. With `-v`, every boot that died or retried, with its lines.
Standard library only."""
import json
import re
import sys


def rows(run):
    for name in ("cycles.jsonl", "boots.jsonl"):
        try:
            with open(f"{run}/{name}") as f:
                return [json.loads(line) for line in f if line.strip()]
        except OSError:
            continue
    return []


def main():
    verbose = "-v" in sys.argv
    runs = [a for a in sys.argv[1:] if a != "-v"]
    total = {}
    for run in runs:
        rs = rows(run)
        skipped = [r for r in rs if "skipped" in r]
        rs = [r for r in rs if "skipped" not in r]
        if any("lines" in r for r in rs):  # boot-only cells (server_boot.py) count in the total
            for r in rs:
                t = total.setdefault(r["arm"], {"boots": 0, "died": 0, "first": 0, "floor": 0, "floor_ev": 0,
                                                 "round0": 0, "round0_ev": 0})
                ev = r["outcome"] == "fatal" or any("[ctx-retry]" in x for x in r["lines"])
                t["boots"] += 1
                t["died"] += r["outcome"] == "fatal"
                t["first"] += ev
                if r["host"]["MemFree_mb"] < 3000:
                    t["floor"] += 1
                    t["floor_ev"] += ev
                if "round" in r and r["boot"] == 0:
                    t["round0"] += 1
                    t["round0_ev"] += ev
        print(f"== {run}: {len(rs)} boots" + (f" ({len(skipped)} rounds skipped: MemFree stayed high)" if skipped else ""))
        for arm in sorted({r["arm"] for r in rs}):
            a = [r for r in rs if r["arm"] == arm]
            lines = lambda r: r.get("lines") or (r.get("ctx_retry", []) + r.get("fatal", []))  # noqa: E731
            fatal = [r for r in a if r["outcome"] == "fatal"]
            first = [r for r in a if any("[ctx-retry]" in x for x in lines(r))]
            rec = []
            for r in first:
                for x in lines(r):
                    m = re.search(r"context created on try (\d+), after (\d+) ms( of waits)?", x)
                    if m:
                        rec.append((int(m.group(1)), int(m.group(2)), "waits" if m.group(3) else "elapsed"))
            free = sorted(r["host"]["MemFree_mb"] for r in a if "host" in r)
            floor = [r for r in a if "host" in r and r["host"]["MemFree_mb"] < 3000]
            if free:
                print(
                    f"   arm {arm}: MemFree at the boot, MB: min {free[0]} median {free[len(free) // 2]} max {free[-1]}; "
                    f"boots under 3000 MB: {len(floor)}, of them died {sum(r['outcome'] == 'fatal' for r in floor)}, "
                    f"first creation failed {sum(any('[ctx-retry]' in x for x in lines(r)) for r in floor)}"
                )
            firsts = [r for r in a if "round" in r and r["boot"] == 0]
            if firsts:
                print(
                    f"   arm {arm}: first boots of a round (MemFree just back under the threshold): {len(firsts)}, "
                    f"of them died {sum(r['outcome'] == 'fatal' for r in firsts)}, "
                    f"first creation failed {sum(any('[ctx-retry]' in x for x in lines(r)) for r in firsts)}"
                )
            print(
                f"   arm {arm}: boots {len(a)}, came up {len(a) - len(fatal)}, died {len(fatal)}, "
                f"first context creation failed {len(first)}, recovered (try, ms, of waits or elapsed) {sorted(rec)}"
            )
            if verbose:
                for r in fatal + [r for r in first if r not in fatal]:
                    print(f"      {r.get('cycle', r.get('boot'))} {r['outcome']} host {r.get('host', '')}".rstrip())
                    for x in lines(r):
                        print(f"         {x[:330]}")
    if len(runs) > 1 and total:
        print("== total over the boot-only cells above")
        for arm, t in sorted(total.items()):
            print(
                f"   arm {arm}: boots {t['boots']}, died {t['died']}, first context creation failed {t['first']}; "
                f"boots under 3000 MB {t['floor']} (first creation failed {t['floor_ev']}); "
                f"first boots of a round {t['round0']} (first creation failed {t['round0_ev']})"
            )


if __name__ == "__main__":
    main()
