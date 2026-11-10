#!/usr/bin/env python3
"""Per-class and pooled view of an ab_cell.py run, with the round time.

Both arms write the same texts with the same acceptance, so tokens per round is one number per
class and the decode ratio is the round-time change alone.
  tokens per round = 1 + K * acceptance per drafted token (the memra#1154 receipts' definition)
  round ms         = 1000 * tokens per round / median decode tok/s
The pooled rows take the geometric mean of the per-class ratios (after over before, on the
per-arm medians over boots). "wins" counts the (after boot, before boot) pairs the after arm
decodes faster in, of N_after * N_before.

usage: round_summary.py RUNS_JSONL [BEFORE AFTER] [--drop-boots=1,11]
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
before, after = (args[1], args[2]) if len(args) > 2 else ("before", "after")
errors = [r for r in rows if r["error"]]
rows = [r for r in rows if not r["error"] and r["boot"] not in drop]
classes = []
for r in rows:
    if r["class"] not in classes:
        classes.append(r["class"])


def sel(c, arm):
    return [r for r in rows if r["class"] == c and r["arm"] == arm]


def tpr(r):
    # tokens a round emits: the accepted drafts plus the one the verify adds
    return 1 + int(r["k"][0]) * (r["acceptance"] or 0)


print(f"requests: {len(rows) + len(errors)}, errors: {len(errors)}, dropped boots: {sorted(drop) or 'none'}")
print()
print("| class | K | boots before / after | decode tok/s before | after | ratio | wins | tokens per round | round ms before | after | change ms | one text | one acceptance |")
print("|---|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|")
ratios = {}
round_ms = {}
for c in classes:
    b, a = sel(c, before), sel(c, after)
    if not b or not a:
        print(f"| {c} | | {len(b)} / {len(a)} | | | | | | | | | | |")
        continue
    mb = statistics.median(r["decode_tok_s"] for r in b)
    ma = statistics.median(r["decode_tok_s"] for r in a)
    wins = sum(1 for x in a for y in b if x["decode_tok_s"] > y["decode_tok_s"])
    shas = {r["sha"] for r in b + a}
    accs = {r["acceptance"] for r in b + a}
    ks = sorted({(r["k"] or [None])[0] for r in b + a}, key=str)
    t = statistics.mean(tpr(r) for r in b + a)
    rb, ra = 1000 * t / mb, 1000 * t / ma
    ratios[c] = ma / mb
    round_ms[c] = (rb, ra)
    print(f"| {c} | {','.join(str(k) for k in ks)} | {len(b)} / {len(a)} | {mb:.2f} | {ma:.2f} | {ma / mb:.3f} | "
          f"{wins} of {len(a) * len(b)} | {t:.2f} | {rb:.2f} | {ra:.2f} | {ra - rb:+.2f} | "
          f"{'yes' if len(shas) == 1 else 'NO ' + str(len(shas))} | {'yes' if len(accs) == 1 else 'NO ' + str(len(accs))} |")
print()
print("| pool | classes | decode ratio, geometric mean | per class | round ms change, mean |")
print("|---|---:|---:|---|---:|")
groups = [(("t1",), "temperature 1 (pure-temperature graph draft)"),
          (("chat",), "temperature 0.7, top_p 0.95 (filtered graph draft)"),
          (("t1", "chat"), "both graph-drafted sampled classes"),
          (("pen",), "penalised (eager sampled draft)"),
          (("greedy",), "greedy control")]
for names, label in groups:
    cs = [c for c in classes if c.split("-")[0] in names and c in ratios]
    if not cs:
        continue
    gm = math.exp(statistics.mean(math.log(ratios[c]) for c in cs))
    dm = statistics.mean(round_ms[c][1] - round_ms[c][0] for c in cs)
    print(f"| {label} | {len(cs)} | {gm:.3f} | {', '.join(f'{ratios[c]:.3f}' for c in cs)} | {dm:+.2f} |")
