"""Markdown tables for the step-4 receipts (long-context layer gates, whole-trunk next token, teacher-forced texts),
read from the receipts directory, with the teacher-forced pass rule applied.

usage: tables_step4.py RECEIPTS_DIR [--json OUT]"""
import argparse
import json
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("r"); ap.add_argument("--json")
a = ap.parse_args()
L = os.path.join(a.r, "long")
out = {}


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def num(x, pat):
    m = re.search(pat, x)
    return float(m.group(1)) if m else None


print("| prompt | layer | stream out | attention out | MoE out | routes (same expert set) |")
print("|---|---|---|---|---|---|")
for case, toks in (("needle1200", "18,343"), ("needle4800", "73,520")):
    g = load(os.path.join(L, f"layer-{case}.json"))
    if not g:
        print(f"| {toks} | pending | | | | |")
        continue
    by = {c["name"]: c for c in g["checks"]}
    # a later run of some layers against their own floor replaces those layers' rows
    for extra in sorted(os.listdir(L)):
        if extra.startswith(f"layer-{case}-") and extra.endswith(".json"):
            e = load(os.path.join(L, extra))
            by.update({c["name"]: c for c in e["checks"]})
    layers = sorted({int(re.match(r"layer(\d+)/", n).group(1)) for n in by if n.startswith("layer")})
    for li in layers:
        cells = []
        for k in ("out_stream", "attn_out", "moe_out"):
            c = by.get(f"layer{li}/{k}")
            if not c:
                cells.append("-")
                continue
            r = num(c["extra"], r"rel_l2=([0-9.e+-]+)")
            f = num(c["rule"], r"floor \(([0-9.e+-]+)\)")
            cells.append(f"{r:.2e} / {f:.2e}" if f is not None else f"{r:.2e}")
        c = by.get(f"layer{li}/route")
        if c:
            m = re.search(r"(\d+)/(\d+) token sets", c["extra"]); f = re.search(r"floor's (\d+)/(\d+)", c["rule"])
            cells.append(f"{int(m.group(1)) / int(m.group(2)) * 100:.1f}% / {int(f.group(1)) / int(f.group(2)) * 100:.1f}%" if f else m.group(0))
        else:
            cells.append("-")
        print(f"| {toks} | {li} | " + " | ".join(cells) + " |")
    out[f"layers-{case}"] = {"checks": len(g["checks"]), "failed": g.get("failed")}
print()
print("| prompt | tokens | next token | top-5 | KL(vendor, native) | KL floor (global-max vendor) | max abs diff | whole trunk |")
print("|---|---|---|---|---|---|---|---|")
for case in ("needle1200", "needle4800", "needle8400", "needle16800"):
    t = load(os.path.join(L, f"trunk-{case}.json"))
    if not t:
        print(f"| {case} | pending | | | | | | |")
        continue
    fl = load(os.path.join(L, f"floor-schedule-{case}.json"))
    log = open(os.path.join(L, f"trunk-{case}.log")).read()
    secs = num(log, r"argmax \d+ logit [0-9.-]+, ([0-9.]+)s")
    print(f"| {case} | {t['tokens']:,} | {'equal' if t['argmax_equal'] else 'DIFFERS'} ({t['argmax_vendor']}) | {t['top5_overlap']}/5 | "
          f"{t['kl']:.1e} | {fl['kl']:.1e} | {t['max_abs']:.2f} | {secs:.0f} s |" if fl else
          f"| {case} | {t['tokens']:,} | {'equal' if t['argmax_equal'] else 'DIFFERS'} ({t['argmax_vendor']}) | {t['top5_overlap']}/5 | "
          f"{t['kl']:.1e} | - | {t['max_abs']:.2f} | {secs:.0f} s |")
    out[f"trunk-{case}"] = t
print()
print("| text | tokens | mean NLL vendor / native | delta | top-1 agree | moved > 0.1 | floor delta (length, schedule) | floor top-1 | floor moved | pass |")
print("|---|---|---|---|---|---|---|---|---|---|")
for case in ("tf_en_lighthouse", "tflong_he_haifa", "tf128k_en_moby", "tf64k_he_cities"):
    t = load(os.path.join(L, f"tf-{case}.json"))
    if not t:
        print(f"| {case} | pending |"); continue
    floors = [f for f in (load(os.path.join(L, f"floor-length-{case}.json")), load(os.path.join(L, f"floor-schedule-{case}.json"))) if f]
    if floors:
        fd = max(abs(f["delta_mean_nll"]) for f in floors)
        ft = min(f["top1_agree"] for f in floors)
        fm = max(f["share_nll_moved_gt_0p1"] for f in floors)
        ok = abs(t["delta_mean_nll"]) <= 2 * fd and t["top1_agree"] >= ft - 0.01 and t["share_nll_moved_gt_0p1"] <= fm + 0.02
        fdesc = ", ".join(f"{f['delta_mean_nll']:+.4f}" for f in floors)
        row = (f"| {case} | {t['tokens']:,} | {t['mean_nll_vendor']:.4f} / {t['mean_nll_rewrite']:.4f} | {t['delta_mean_nll']:+.4f} | "
               f"{t['top1_agree'] * 100:.1f}% | {t['share_nll_moved_gt_0p1'] * 100:.1f}% | {fdesc} | {ft * 100:.1f}% | {fm * 100:.1f}% | "
               f"{'PASS' if ok else 'FAIL'} |")
        t["pass"] = ok
    else:
        row = (f"| {case} | {t['tokens']:,} | {t['mean_nll_vendor']:.4f} / {t['mean_nll_rewrite']:.4f} | {t['delta_mean_nll']:+.4f} | "
               f"{t['top1_agree'] * 100:.1f}% | {t['share_nll_moved_gt_0p1'] * 100:.1f}% | - | - | - | no floor |")
    print(row)
    out[f"tf-{case}"] = t
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
