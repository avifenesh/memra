"""Markdown table of the per-position KL receipts (tf_kl.py): the native run and the vendor's own floors against the
vendor tape, per text and per subset (all positions, vendor entropy above 1 nat, top entropy decile).

Each floor is compared on its own positions: the schedule floor (the global-max realization) on every position, the
length floor (the same text as a shorter tape) on that tape's prefix, with the native run cut to the same prefix. A
subset passes on the band of the larger floor, as the teacher-forced NLL rule does: the native mean KL at most twice
that floor's, and its top-1 agreement under that floor's by at most 1 point or two binomial standard errors of the
floor's agreement on the subset, whichever is larger (a 2k text's high-entropy subset has about 160 positions, so one
position moves it 0.6 points; at thousands of positions the 1 point holds). Each floor's own verdict is printed too.
The verdict is the high-entropy subsets'.

usage: tables_kl.py RECEIPTS_DIR [--json OUT]"""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("r"); ap.add_argument("--json")
a = ap.parse_args()
L = os.path.join(a.r, "long")


def load(name):
    p = os.path.join(L, name)
    return json.load(open(p)) if os.path.exists(p) else None


def slack(f):
    # the larger of 1 point and two binomial standard errors of the floor's top-1 agreement on its subset
    q = f["top1_agree"]
    return max(0.01, 2 * math.sqrt(q * (1 - q) / f["positions"]))


def e(x):
    return f"{x:.2e}"


SUB = (("all", "all"), ("entropy_gt_1nat", "entropy > 1 nat"), ("entropy_top_decile", "top entropy decile"))
out = {}
print("| text | tokens | positions | native KL | schedule floor KL | native / floor | native KL, prefix | "
      "length floor KL | native / floor | top-1: native, schedule; native, length (prefix) | pass (per floor) |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for case in ("tf_en_lighthouse", "tflong_he_haifa", "tf128k_en_moby", "tf64k_he_cities", "tf256k_en_moby"):
    nat, sch, lng = (load(f"kl-{k}-{case}.json") for k in ("native", "schedule", "length"))
    pre = load(f"kl-native-{case}-prefix.json")
    if not nat:
        print(f"| {case} | pending | | | | | | | | | |")
        continue
    for key, label in SUB:
        n = nat["subsets"][key]
        pairs = []  # (native on the floor's positions, floor)
        row = [f"{case}", f"{nat['positions']:,}", f"{label}: {n['positions']:,}", e(n["mean_kl"])]
        if sch and sch["subsets"][key].get("positions"):
            f = sch["subsets"][key]
            pairs.append((n, f))
            row += [e(f["mean_kl"]), f"{n['mean_kl'] / f['mean_kl']:.2f}"]
        else:
            row += ["-", "-"]
        if lng and pre and lng["subsets"][key].get("positions"):
            p, f = pre["subsets"][key], lng["subsets"][key]
            pairs.append((p, f))
            row += [e(p["mean_kl"]), e(f["mean_kl"]), f"{p['mean_kl'] / f['mean_kl']:.2f}"]
        else:
            row += ["-", "-", "-"]
        each = [m["mean_kl"] <= 2 * f["mean_kl"] and m["top1_agree"] >= f["top1_agree"] - slack(f) for m, f in pairs]
        ok = any(each)
        row.append("; ".join(f"{m['top1_agree'] * 100:.1f}%, {f['top1_agree'] * 100:.1f}%" for m, f in pairs))
        row.append(("PASS" if ok else "FAIL") + " (" + ", ".join("in" if x else "out" for x in each) + ")")
        print("| " + " | ".join(row) + " |")
        out.setdefault(case, {})[key] = {"native": n, "pairs": pairs, "per_floor": each, "pass": ok}
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
