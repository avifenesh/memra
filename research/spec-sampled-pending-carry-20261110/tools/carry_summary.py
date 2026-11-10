#!/usr/bin/env python3
"""Per-class view of an ab_cell.py run, base against candidate (memra#1186).

The two arms write the same texts. They do not round alike: the candidate draws the token after
a burst boundary in a round, where the base drew it at the boundary, so rounds, drafted tokens
and acceptance are per arm.
  decode tok/s     completion tokens / (total - time to first chunk), median over boots
  wins             (candidate boot, base boot) pairs the candidate decodes faster in
  round ms         1000 * tokens per round / median decode tok/s, tokens per round =
                   1 + K * acceptance per drafted token (the memra#1186 issue's figure)
  wall ms / round  1000 * (total - time to first chunk) / rounds, median over boots
  close ms         total - time of the last content chunk, median over boots
The pooled rows take the geometric mean of the per-class ratios.

usage: carry_summary.py RUNS_JSONL [BASE CAND] [--drop-boots=1,11]
"""
import json
import math
import statistics
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
drop = set()
for a in sys.argv[1:]:
    if a.startswith("--drop-boots="):
        drop = {int(x) for x in a.split("=")[1].split(",")}
rows = [json.loads(line) for line in open(args[0]) if line.strip()]
base, cand = (args[1], args[2]) if len(args) > 2 else ("base", "cand")
errors = [r for r in rows if r["error"]]
rows = [r for r in rows if not r["error"] and r["boot"] not in drop]
classes = []
for r in rows:
    if r["class"] not in classes:
        classes.append(r["class"])


def sel(c, arm):
    return [r for r in rows if r["class"] == c and r["arm"] == arm]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def tpr(r):
    return 1 + int(r["k"][0]) * (r["acceptance"] or 0)


def wall(r):
    return 1000 * (r["total_s"] - r["ttft_s"]) / r["rounds"] if r.get("rounds") and r.get("ttft_s") else None


print(f"requests: {len(rows) + len(errors)}, errors: {len(errors)}, dropped boots: {sorted(drop) or 'none'}")
print()
print("| class | K | boots base / cand | decode tok/s base | cand | ratio | wins | acceptance base | cand | rounds base | cand "
      "| round ms base | cand | change | wall ms per round base | cand | close ms base | cand | one text |")
print("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
ratios, change = {}, {}
for c in classes:
    b, a = sel(c, base), sel(c, cand)
    if not b or not a:
        print(f"| {c} | | {len(b)} / {len(a)} |" + " |" * 16)
        continue
    mb, ma = med(r["decode_tok_s"] for r in b), med(r["decode_tok_s"] for r in a)
    wins = sum((x["decode_tok_s"] > y["decode_tok_s"]) + 0.5 * (x["decode_tok_s"] == y["decode_tok_s"]) for x in a for y in b)
    shas = {r["sha"] for r in b + a}
    ks = sorted({(r["k"] or [None])[0] for r in b + a}, key=str)
    tb, ta = statistics.mean(tpr(r) for r in b), statistics.mean(tpr(r) for r in a)
    rb, ra = 1000 * tb / mb, 1000 * ta / ma
    ratios[c], change[c] = ma / mb, ra - rb
    print(f"| {c} | {','.join(str(k) for k in ks)} | {len(b)} / {len(a)} | {mb:.2f} | {ma:.2f} | {ma / mb:.3f} | "
          f"{wins:g} of {len(a) * len(b)} | {med(r['acceptance'] for r in b):.3f} | {med(r['acceptance'] for r in a):.3f} | "
          f"{med(r['rounds'] for r in b):g} | {med(r['rounds'] for r in a):g} | {rb:.2f} | {ra:.2f} | {ra - rb:+.2f} | "
          f"{med(wall(r) for r in b):.2f} | {med(wall(r) for r in a):.2f} | "
          f"{med(r.get('close_ms') for r in b):.1f} | {med(r.get('close_ms') for r in a):.1f} | "
          f"{'yes' if len(shas) == 1 else 'NO ' + str(len(shas))} |")
print()
print("| pool | classes | decode ratio, geometric mean | per class | round ms change, mean |")
print("|---|---:|---:|---|---:|")
groups = [(("t1",), "temperature 1"),
          (("chat",), "temperature 0.7, top_p 0.95"),
          (("pen",), "penalised"),
          (("t1", "chat", "pen"), "all sampled classes"),
          (("greedy",), "greedy control")]
for names, label in groups:
    cs = [c for c in classes if c.split("-")[0] in names and c in ratios]
    if not cs:
        continue
    gm = math.exp(statistics.mean(math.log(ratios[c]) for c in cs))
    dm = statistics.mean(change[c] for c in cs)
    print(f"| {label} | {len(cs)} | {gm:.3f} | {', '.join(f'{ratios[c]:.3f}' for c in cs)} | {dm:+.2f} |")
