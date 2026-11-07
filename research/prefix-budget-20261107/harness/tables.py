"""Markdown tables for the scored 5090 arms: mixed (phase1a) and agentic (phase1b/1c) medians per
arm, plus greedy identity against the cache-off runs.

  python3 tables.py RUNS_ROOT
"""
import json, re, statistics, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1])
H = Path(__file__).parent


def med(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def summ(script, d, *extra):
    out = subprocess.run([sys.executable, str(H / script), str(d), *extra], capture_output=True, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


# ---- mixed
p1a = root / "phase1a"
if p1a.exists():
    rows = summ("summarize_mixed.py", p1a, "off-new")
    print("| arm | runs | repeats hit (each run) | repeat TTFT median ms | batch tok/s | batch TTFT p50 ms | reclaims | evict lines | device free min MiB | greedy == off |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("base", "lru", "gds", "flush", "noclamp", "off"):
        rs = [r for r in rows if r["run"].split("-")[0] == arm]
        if not rs:
            continue
        print(f"| {arm} | {len(rs)} | {', '.join(str(r['repeat_hits']) for r in rs)} | "
              f"{med([r['repeat_ttft_median_ms'] for r in rs]):.0f} | {med([r['batch_agg_tok_s'] for r in rs]):.1f} | "
              f"{med([r['batch_ttft_p50_ms'] for r in rs]):.0f} | {med([r['reclaims'] for r in rs]):.0f} | "
              f"{med([r['evict_lines'] for r in rs]):.0f} | {min(int(r['free_min_mib']) for r in rs)} | "
              f"{', '.join(r['greedy_vs_off'] for r in rs)} |")
    print()

# ---- agentic
for ph in ("phase1b", "phase1c"):
    d = root / ph
    if not d.exists():
        continue
    rows = summ("summarize_agentic.py", d)
    by = {}
    for r in rows:
        m = re.match(r"c(\d+)-(base|lru|gds|flush)-r(\d+)", r["run"])
        if m:
            by.setdefault((int(m.group(1)), m.group(2)), []).append(r)
    if by:
        print("| users | arm | runs | measured hits | cached fraction | TTFT p50 ms | TTFT p95 ms | agg gen tok/s | reclaims | device free min MiB |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (u, arm) in sorted(by, key=lambda k: (k[0], ("base", "lru", "gds", "flush").index(k[1]))):
            rs = by[(u, arm)]
            print(f"| c{u} | {arm} | {len(rs)} | {', '.join(r['hits'] for r in rs)} | {med([r['cached_frac'] for r in rs]):.3f} | "
                  f"{med([r['ttft_ms_p50'] for r in rs]):.0f} | {med([r['ttft_ms_p95'] for r in rs]):.0f} | "
                  f"{med([r['agg_tps'] for r in rs]):.2f} | {med([r['reclaims'] for r in rs]):.0f} | "
                  f"{min(int(r['free_min_mib']) for r in rs)} |")
        print()
    if (d / "exact-off").exists():
        ex = summ("summarize_agentic.py", d, "--exact", "exact-off")
        print("| exactness run | measured hits | greedy == off | differing turns |")
        print("|---|---|---|---|")
        for r in ex:
            if r["run"].startswith("exact-"):
                print(f"| {r['run']} | {r['hits']} | {r['greedy_vs_off']} | {', '.join(r['diff_turns']) or 'none'} |")
        print()
