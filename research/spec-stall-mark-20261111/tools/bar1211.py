#!/usr/bin/env python3
"""memra#1211 bar, per cell, form 1 (FORM arm) against main + #1228 (MAIN arm), from stall.jsonl rows.

Items (coordinator's bar):
  1. the long stream advances at least as much (long tokens in the window);
  2. >= 90% of newcomers admitted plain in every cell that marks;
  3. short p50 / p95 no worse than main's, or explained from the log;
  4. 0 marks and identical choices in the 1-client and 1000 ms cells.
Per cell the arms' runs are shown as ranges; items 1 and 3 compare the arms' means and print
WORSE when the form's mean is worse and its best run is also outside main's range.

usage: bar1211.py STALL_JSONL MAIN_ARM FORM_ARM
"""
import collections
import json
import statistics
import sys

path, main_arm, form_arm = sys.argv[1:4]
rows = [json.loads(l) for l in open(path) if l.strip()]
cells = collections.OrderedDict()
for r in rows:
    arm = r["arm"].rsplit("-", 1)[0]
    if arm not in (main_arm, form_arm):
        continue
    key = f"{r['clients']}:{int(r['think_ms'])}" + ("" if r["class"] == "greedy" else f" {r['class']}")
    cells.setdefault(key, {main_arm: [], form_arm: []})[arm].append(r)


def k_counts(r):
    return (r.get("census_window") or {}).get("spec_k") or {}


def plain(r):
    k = k_counts(r)
    return sum(v for key, v in k.items() if key.startswith("K0/")), sum(k.values())


def rng(xs, fmt="{}"):
    if not xs:
        return "-"
    lo, hi = min(xs), max(xs)
    return fmt.format(lo) if lo == hi else f"{fmt.format(lo)}-{fmt.format(hi)}"


print(f"| cell | long tokens main / form | short p50 s main / form | short p95 s main / form | marks form | plain/all form | items |")
print("|---|---|---|---|---|---|---|")
fails = []
for key, arms in cells.items():
    m, f = arms[main_arm], arms[form_arm]
    if not m or not f:
        print(f"| {key} | incomplete ({len(m)} main, {len(f)} form) | | | | | |")
        continue
    items = []
    lt_m, lt_f = [r["tokens_during"] for r in m], [r["tokens_during"] for r in f]
    if statistics.mean(lt_f) < statistics.mean(lt_m) and max(lt_f) < min(lt_m):
        items.append("1 WORSE")
    for q in ("short_lat_p50_s", "short_lat_p95_s"):
        a, b = [r[q] for r in m], [r[q] for r in f]
        if statistics.mean(b) > statistics.mean(a) and min(b) > max(a):
            items.append(f"3 {q[10:13]} WORSE")
    marks = [((r.get("census_window") or {}).get("stall_marks") or 0) for r in f]
    pl = [plain(r) for r in f]
    if any(marks):
        frac = [p / a if a else 1.0 for p, a in pl]
        if min(frac) < 0.9:
            items.append(f"2 plain {min(frac):.0%}")
    one_or_slow = key.startswith("1:") or ":1000" in key
    if one_or_slow:
        same = all(set(k_counts(r)) == set(k_counts(m[0])) for r in m + f)
        if any(marks) or not same:
            items.append("4 marks/choices differ")
    fails += [f"{key}: {i}" for i in items]
    print(f"| {key} | {rng(lt_m)} / {rng(lt_f)} | {rng([r['short_lat_p50_s'] for r in m])} / {rng([r['short_lat_p50_s'] for r in f])} | "
          f"{rng([r['short_lat_p95_s'] for r in m])} / {rng([r['short_lat_p95_s'] for r in f])} | {rng(marks)} | "
          f"{' '.join(f'{p}/{a}' for p, a in pl)} | {', '.join(items) or 'ok'} |")
print()
print("bar:", "PASS" if not fails else "items to explain: " + "; ".join(fails))
