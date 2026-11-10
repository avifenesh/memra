#!/usr/bin/env python3
"""Evaluate candidate per-flip rules for the argmax margin gate on the rows `collect.py` found.

Per row the receipts hold two top-1 ids, two top-2 margins (x = the reference side, y = the
other program) and the spread at the contending ids. From those:

  gap shift S = margin_x + margin_y   at a flip (the least the two programs must disagree
                                      about the two contenders for the flip to happen)
              = |margin_x - margin_y| at an agreeing row (how far the two programs disagree
                                      about top-1 against the runner-up; exact when both
                                      programs have the same runner-up)

Sections, each printed as a Markdown table:
  1. the old predicate (spread > smaller margin) on every flip;
  2. populations: agreeing-row maxima per (model, prompt, pair, device);
  3. candidate rules on the kv-quant gate pairs (2,048 rows each);
  4. the gate's 12-row window on the same rows, old and new verdicts;
  5. how many calibration rows a floor needs;
  6. every distinct 12-row table re-judged.

usage: analyze.py rows.tsv > ANALYSIS.md
"""
import collections
import csv
import io
import sys

DEV = {"48": "gb10", "82": "rtx5090", "188": "rtxpro6000", "170": "sm170"}
GATE_PAIRS = ("prefill:batch", "prefill:eager")
CONTROLS = ("prose", "csv")


def quantile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round((len(values) - 1) * p)))]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            for k in ("margin_x", "margin_y", "spread"):
                r[k] = float(r[k])
            r["flip"] = r["flip"] == "1"
            r["pos"] = int(r["pos"])
            r["gap"] = float(r["gap"]) if r["gap"] else None
            r["dev"] = r["device"] or DEV.get(r["sm_count"], "unknown")
            if not r["prompt"]:
                r["prompt"] = "T=" + r["T"] if r["T"] else "unknown"
            r["max"] = max(r["margin_x"], r["margin_y"])
            r["min"] = min(r["margin_x"], r["margin_y"])
            r["S"] = r["margin_x"] + r["margin_y"] if r["flip"] else abs(r["margin_x"] - r["margin_y"])
            # This lane's own calibration and gate logs are not history: `rejudge.py` and the
            # floors table read those.
            if "argmax-gate-rule-20261110/" in r["source"]:
                continue
            rows.append(r)
    return rows


def row_key(r):
    return (r["pos"], r["top_x"], r["margin_x"], r["top_y"], r["margin_y"], r["spread"])


def old_budget(model, prompt, window=12):
    """tools/argmax-margin-gate.sh's flip budget rows at 031f146690."""
    per12 = 1
    if model.startswith("Qwen3.8-27B-NVFP4") and prompt == "p4-16k":
        per12 = 3
    elif model.startswith(("gemma-4-31B", "gemma-4-26B")):
        per12 = 3
    return (per12 * window + 11) // 12


def md(header, lines):
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for line in lines:
        print("| " + " | ".join(str(c) for c in line) + " |")
    print()


def f3(x):
    return "%.3f" % x


def section_old_rule(rows):
    print("## 1. The old predicate on every flip\n")
    flips = {}
    for r in rows:
        if r["flip"]:
            flips[(r["tree"], r["model"], r["prompt"], r["pair"], r["dev"]) + row_key(r)] = r
    flips = list(flips.values())
    unexplained = [r for r in flips if not r["spread"] > r["min"]]
    ratio_min = min(r["spread"] / r["min"] for r in flips if r["min"] > 0)
    ratio_mean = min(r["spread"] / ((r["margin_x"] + r["margin_y"]) / 2) for r in flips)
    lines = []
    for tree in ("tables", "kv-quant"):
        sub = [r for r in flips if r["tree"] == tree]
        if not sub:
            continue
        lines.append((tree, len(sub), sum(1 for r in sub if not r["spread"] > r["min"]),
                      f3(min(r["spread"] / r["min"] for r in sub if r["min"] > 0)),
                      f3(max(r["max"] for r in sub)), f3(max(r["S"] for r in sub))))
    md(("receipts", "distinct flips", "old rule says unexplained", "smallest spread / smaller margin",
        "largest margin at a flip", "largest margin sum at a flip"), lines)
    print(f"All {len(flips)} distinct flips: {len(unexplained)} unexplained, smallest spread / smaller "
          f"margin {ratio_min:.3f}, smallest spread / mean margin {ratio_mean:.3f} (the bound is 1).\n")


def groups_of(rows):
    groups = collections.defaultdict(dict)
    tables = collections.defaultdict(set)
    for r in rows:
        if r["tree"] == "kv-quant" and r["prompt"].startswith("repro"):
            continue
        key = (r["tree"], r["model"], r["prompt"], r["pair"], r["dev"])
        groups[key][row_key(r)] = r
        tables[key].add((r["source"], r["table"]))
    return groups, tables


def section_populations(rows):
    print("## 2. Populations: what agreeing rows show per (model, prompt, pair, device)\n")
    print("`max gap shift` and `max spread` are over agreeing rows. Identical rows from repeated "
          "runs count once.\n")
    groups, tables = groups_of(rows)
    lines = []
    for key, d in sorted(groups.items()):
        d = list(d.values())
        ag = [r for r in d if not r["flip"]]
        fl = [r for r in d if r["flip"]]
        if not ag:
            continue
        top = max(ag, key=lambda r: r["S"])
        big = len(ag) >= 500
        lines.append((key[0], key[1] or "?", key[2], key[3], key[4], len(tables[key]), len(ag),
                      f3(quantile([r["S"] for r in ag], 0.99)) if big else "",
                      f3(quantile([r["S"] for r in ag], 0.999)) if big else "",
                      f"{top['S']:.3f} (row {top['pos']})",
                      f3(quantile([r["spread"] for r in ag], 0.99)) if big else "",
                      f3(max(r["spread"] for r in ag)),
                      len(fl), f3(max((r["S"] for r in fl), default=0.0)),
                      f3(max((r["max"] for r in fl), default=0.0))))
    md(("tree", "model", "prompt", "pair", "device", "tables", "agreeing rows", "gap shift p99",
        "p99.9", "max gap shift", "spread p99", "max spread", "distinct flips",
        "largest margin sum at a flip", "largest margin at a flip"), lines)


def kv_sets(rows):
    sets = collections.defaultdict(dict)
    for r in rows:
        if r["tree"] != "kv-quant" or r["prompt"].startswith("repro"):
            continue
        sets[(r["prompt"], r["pair"])][r["pos"]] = r
    return {k: [v[p] for p in sorted(v)] for k, v in sets.items()}


def calibration_set(sets, prompt, pair):
    name = prompt if prompt.startswith(CONTROLS) else "prose-" + prompt.split("-")[1]
    return name, sets.get((name, pair))


def section_rules(rows):
    print("## 3. Candidate rules on the kv-quant gate pairs\n")
    print("Floors come from the agreeing rows of the calibration set: a control (prose, csv) "
          "calibrates itself; the JSONL sets take the prose set of the same depth and pair. "
          "Each cell is flips failing / flips. Rules: A = a margin above the spread floor "
          "(R1 with the spread statistic); B = A, or the flip row's spread above the spread floor "
          "(R1 + R2 on the flip row); C = margin sum above the gap-shift floor; D = a margin "
          "above the gap-shift floor.\n")
    sets = kv_sets(rows)
    lines = []
    totals = collections.Counter()
    for (prompt, pair), g in sorted(sets.items()):
        if pair not in GATE_PAIRS:
            continue
        cal_name, cal = calibration_set(sets, prompt, pair)
        ag = [r for r in cal if not r["flip"]]
        fl = [r for r in g if r["flip"]]
        row = [prompt, pair, cal_name, len(ag), len(fl)]
        kind = "control" if prompt.startswith(CONTROLS) else "jsonl"
        for stat, label in (("spread", "A"), ("spread", "B"), ("S", "C"), ("S", "D")):
            for qname, qv in (("p99", 0.99), ("p99.9", 0.999), ("max", 1.0)):
                floor = quantile([r[stat] for r in ag], qv)
                if label == "A":
                    fails = sum(1 for r in fl if r["max"] > floor)
                elif label == "B":
                    fails = sum(1 for r in fl if r["max"] > floor or r["spread"] > floor)
                elif label == "C":
                    fails = sum(1 for r in fl if r["S"] > floor)
                else:
                    fails = sum(1 for r in fl if r["max"] > floor)
                row.append(f"{fails} ({floor:.2f})")
                totals[(kind, label, qname)] += fails
        totals[(kind, "flips")] += len(fl)
        wide = sorted(r["S"] for r in fl)
        row.append(" ".join("%.2f" % s for s in wide) if kind == "jsonl" else "max %.2f" % wide[-1])
        lines.append(row)
    header = ["set", "pair", "floor from", "agreeing rows", "flips"]
    for label in "ABCD":
        for qname in ("p99", "p99.9", "max"):
            header.append(f"{label} {qname} (floor)")
    header.append("margin sums at the flips")
    md(header, lines)
    lines = []
    for label in "ABCD":
        for qname in ("p99", "p99.9", "max"):
            lines.append((label, qname,
                          f"{totals[('control', label, qname)]} / {totals[('control', 'flips')]}",
                          f"{totals[('jsonl', label, qname)]} / {totals[('jsonl', 'flips')]}"))
    md(("rule", "floor quantile", "control flips failed (false fails)", "JSONL flips failed"), lines)


def windows(g, w=12):
    return [g[i:i + w] for i in range(0, len(g) - w + 1, w)]


def section_windows(rows):
    print("## 4. The gate's 12-row window on the same rows\n")
    print("Each 2,048-row set cut into 170 windows of 12 consecutive rows. Floor F = the largest "
          "gap shift at an agreeing row of the calibration set (section 3). Old verdict: flips "
          "<= 1. New: a window fails on a flip whose margin sum is above F (wide flip), on an "
          "agreeing row whose gap shift is above F (wide row), or on more near ties than the "
          "budget.\n")
    sets = kv_sets(rows)
    lines = []
    for (prompt, pair), g in sorted(sets.items()):
        if pair not in GATE_PAIRS:
            continue
        cal_name, cal = calibration_set(sets, prompt, pair)
        floor = max(r["S"] for r in cal if not r["flip"])
        ws = windows(g)
        hist = collections.Counter(min(4, sum(r["flip"] for r in w)) for w in ws)
        old_fail = sum(1 for w in ws if sum(r["flip"] for r in w) > 1)
        wide_flip = sum(1 for w in ws if any(r["flip"] and r["S"] > floor for r in w))
        wide_row = sum(1 for w in ws if any(not r["flip"] and r["S"] > floor for r in w))
        either = sum(1 for w in ws if any(r["S"] > floor for r in w))
        near = [sum(1 for r in w if r["flip"] and r["S"] <= floor) for w in ws]
        lines.append((prompt, pair, cal_name, f3(floor), len(ws),
                      "/".join(str(hist[i]) for i in range(5)), old_fail, wide_flip, wide_row, either,
                      sum(1 for n in near if n > 1), sum(1 for n in near if n > 2),
                      sum(1 for n in near if n > 3)))
    md(("set", "pair", "floor from", "F", "windows", "windows with 0/1/2/3/4+ flips",
        "old rule fails", "new: wide flip", "new: wide row", "new: either",
        "near ties > 1", "> 2", "> 3"), lines)


def section_calibration_size(rows):
    print("## 5. How many calibration rows a floor needs\n")
    print("Controls only. Floor = the largest gap shift at an agreeing row inside a block of N "
          "consecutive rows; tested on every 12-row window outside the block. `expected` is the "
          "order-statistics rate 12 / (N + 12) for exchangeable rows.\n")
    sets = kv_sets(rows)
    agg = collections.defaultdict(lambda: [0, 0, 0, 0, 0.0, 0])
    for (prompt, pair), g in sorted(sets.items()):
        if pair not in GATE_PAIRS or not prompt.startswith(CONTROLS):
            continue
        for n in (12, 48, 96, 256, 512, 1024):
            for start in range(0, len(g) - n + 1, max(n, 128)):
                cal = [r for r in g[start:start + n] if not r["flip"]]
                if not cal:
                    continue
                floor = max(r["S"] for r in cal)
                a = agg[n]
                a[4] += floor
                a[5] += 1
                for w in windows(g[:start]) + windows(g[start + n:]):
                    a[0] += 1
                    a[1] += any(not r["flip"] and r["S"] > floor for r in w)
                    a[2] += any(r["flip"] and r["S"] > floor for r in w)
                    a[3] += any(r["S"] > floor for r in w)
    lines = []
    for n, a in sorted(agg.items()):
        lines.append((n, a[5], f3(a[4] / a[5]), a[0], "%.1f%%" % (100 * a[1] / a[0]),
                      "%.1f%%" % (100 * a[2] / a[0]), "%.1f%%" % (100 * a[3] / a[0]),
                      "%.1f%%" % (100 * 12 / (n + 12))))
    md(("N", "blocks", "mean floor", "test windows", "windows with an agreeing row over the floor",
        "with a flip over the floor", "either (false fail)", "expected"), lines)


def section_rejudge(rows):
    print("## 6. Every distinct 12-row table, re-judged\n")
    print("One line per distinct table (identical tables from repeated runs are merged; `runs` "
          "counts them). `needs F >=` is the smallest floor under which the table passes the two "
          "per-row clauses: the largest margin sum at a flip and the largest gap shift at an "
          "agreeing row. Old verdict: the table part of the gate at 031f146690 (flip count "
          "against the budget row; the per-flip clause never fires).\n")
    tables = collections.defaultdict(list)
    for r in rows:
        if r["tree"] == "kv-quant":
            continue
        tables[(r["tree"], r["source"], r["table"])].append(r)
    distinct = collections.defaultdict(list)
    for (tree, source, _), trows in tables.items():
        trows.sort(key=lambda r: r["pos"])
        head = trows[0]
        key = (tree, head["model"], head["prompt"], head["pair"], head["dev"],
               tuple(row_key(r) for r in trows))
        distinct[key].append(source)
    lines = []
    for key, sources in sorted(distinct.items(), key=lambda kv: kv[0][:5] + (kv[1][0],)):
        tree, model, prompt, pair, dev, rk = key
        trows = [dict(pos=k[0], top_x=k[1], margin_x=k[2], top_y=k[3], margin_y=k[4], spread=k[5])
                 for k in rk]
        flips = [r for r in trows if r["top_x"] != r["top_y"]]
        agree = [r for r in trows if r["top_x"] == r["top_y"]]
        budget = old_budget(model, prompt, len(trows))
        old = "PASS" if len(flips) <= budget else "FAIL (count)"
        flip_need = max(flips, key=lambda r: r["margin_x"] + r["margin_y"]) if flips else None
        row_need = max(agree, key=lambda r: abs(r["margin_x"] - r["margin_y"])) if agree else None
        lines.append((
            tree, model or "?", prompt, pair, dev, len(sources), sorted(sources)[0],
            f"{len(flips)} / {budget}", old,
            "" if not flip_need else "%.3f (row %d: %.3f + %.3f, spread %.3f)" % (
                flip_need["margin_x"] + flip_need["margin_y"], flip_need["pos"],
                flip_need["margin_x"], flip_need["margin_y"], flip_need["spread"]),
            "" if not row_need else "%.3f (row %d: %.3f vs %.3f, spread %.3f)" % (
                abs(row_need["margin_x"] - row_need["margin_y"]), row_need["pos"],
                row_need["margin_x"], row_need["margin_y"], row_need["spread"]),
            f3(max(r["spread"] for r in trows)),
        ))
    md(("tree", "model", "prompt", "pair", "device", "runs", "first receipt", "flips / old budget",
        "old verdict", "flip clause needs F >=", "row clause needs F >=", "max spread"), lines)


def main():
    rows = load(sys.argv[1])
    out, sys.stdout = sys.stdout, io.StringIO()
    print("# Argmax margin gate: candidate rules on existing receipts\n")
    print(f"{len(rows)} rows from `collect.py`.\n")
    section_old_rule(rows)
    section_populations(rows)
    section_rules(rows)
    section_windows(rows)
    section_calibration_size(rows)
    section_rejudge(rows)
    text, sys.stdout = sys.stdout.getvalue(), out
    print(text.rstrip("\n"))


if __name__ == "__main__":
    main()
