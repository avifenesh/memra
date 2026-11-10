#!/usr/bin/env python3
"""Re-judge every 12-row (or 24-row) probe table the tree holds under the floors table.

Reads `collect.py`'s rows and `tools/argmax-margin-floors.tsv`, and applies the gate's rule
(memra_engine::argmax_margin::judge) to each distinct table: a flip whose margin sum is above
the cell's floor is a wide flip, an agreeing row whose margin difference is above it is a wide
row, a flip in a cell with no floor row cannot be classified. Arithmetic is done in f32, as the
gate does it.

A probe table names its device only by SM count, and its prompt only by name, so the key here
is (model, prompt name, decode program, device name). The device name is taken from the SM count
for GB10 (48) and the RTX 5090 Laptop GPU (82); a 188-SM table gets the RTX PRO 6000 Max-Q name
only inside research/release-qualification/, whose records name that card. Every other table has
no floor. The gate itself also keys on the prompt's sha256.

usage: rejudge.py rows.tsv floors.tsv > REJUDGE.md
"""
import collections
import csv
import struct
import sys

DEVICE_OF_SM = {
    "48": "NVIDIA GB10",
    "82": "NVIDIA GeForce RTX 5090 Laptop GPU",
}
PRO6000_MAXQ = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"


def device_of(sm, source):
    if sm == "188" and source.startswith("research/release-qualification/"):
        return PRO6000_MAXQ
    return DEVICE_OF_SM.get(sm)


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def old_budget(model, prompt, window):
    per12 = 1
    if model.startswith("Qwen3.8-27B-NVFP4") and prompt == "p4-16k":
        per12 = 3
    elif model.startswith(("gemma-4-31B", "gemma-4-26B")):
        per12 = 3
    return (per12 * window + 11) // 12


def load_floors(path):
    floors = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            model, prompt, decode, device, floor, known = f[0], f[1], f[3], f[4], f[5], f[11]
            recorded = {}
            if known != "-":
                for entry in known.split(";"):
                    pos, _p1, mp, _d1, md = entry.split(":")
                    recorded[int(pos)] = f32(f32(float(mp)) + f32(float(md)))
            floors[(model, prompt, decode, device)] = (f32(float(floor)), recorded)
    return floors


def main():
    rows_path, floors_path = sys.argv[1:3]
    floors = load_floors(floors_path)
    tables = collections.defaultdict(list)
    with open(rows_path, encoding="utf-8") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            # This lane's calibration, timing and green-arm logs are not history; the
            # landing cells banked here are.
            lane = "argmax-gate-rule-20261110/"
            if r["tree"] == "kv-quant" or any(lane + d in r["source"] for d in ("calibration/", "timing/", "green-arm/")):
                continue
            tables[(r["tree"], r["source"], r["table"])].append(r)
    distinct = collections.defaultdict(list)
    for (tree, source, _), trows in tables.items():
        trows.sort(key=lambda r: int(r["pos"]))
        head = trows[0]
        arm = "batch" if head["arm"].startswith("batch") else "eager"
        key = (tree, head["model"], head["prompt"], arm, head["sm_count"],
               tuple((int(r["pos"]), r["top_x"], f32(float(r["margin_x"])), r["top_y"],
                      f32(float(r["margin_y"])), f32(float(r["spread"]))) for r in trows))
        distinct[key].append(source)

    print("| tree | model | prompt | decode | SMs | runs | first receipt | flips / budget | old verdict "
          "| floor | new verdict | deciding row |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    tally = collections.Counter()
    for key, sources in sorted(distinct.items(), key=lambda kv: kv[0][:5] + (sorted(kv[1])[0],)):
        tree, model, prompt, arm, sm, rk = key
        budget = old_budget(model, prompt, len(rk))
        flips = [r for r in rk if r[1] != r[3]]
        old = "PASS" if len(flips) <= budget else "FAIL (count)"
        device = device_of(sm, sorted(sources)[0])
        floor = floors.get((model, prompt, arm, device)) if device else None
        findings = []
        near = 0
        for pos, p1, mp, d1, md, spread in rk:
            flip = p1 != d1
            shift = f32(mp + md) if flip else abs(f32(mp - md))
            if floor is not None and pos in floor[1]:
                name = "KNOWN-ROW-WIDER" if shift > floor[1][pos] else "KNOWN-ROW"
                findings.append((name, pos, mp, md, shift))
            elif flip and floor is None:
                findings.append(("UNCALIBRATED-FLIP", pos, mp, md, shift))
            elif flip and shift > floor[0]:
                findings.append(("WIDE-FLIP", pos, mp, md, shift))
            elif flip:
                near += 1
                findings.append(("near-tie", pos, mp, md, shift))
            elif floor is not None and shift > floor[0]:
                findings.append(("WIDE-ROW", pos, mp, md, shift))
        bad = [f for f in findings if f[0] not in ("near-tie", "KNOWN-ROW")]
        if bad:
            new = "FAIL: " + ", ".join(sorted({f[0] for f in bad}))
        elif near > budget:
            new = f"FAIL: {near} near ties > budget {budget}"
        elif floor is None:
            new = "PASS: UNCALIBRATED (no flip)"
        else:
            new = "PASS"
        tally[(old.split()[0], new.split(":")[0].split()[0] + (" UNCALIBRATED" if "UNCALIBRATED" in new and new.startswith("PASS") else ""))] += 1
        worst = max(bad or findings, key=lambda f: f[4], default=None)
        if worst is None:
            # No finding: name the largest agreeing gap shift, the row nearest the floor.
            pos, mp, md = max(((r[0], r[2], r[4]) for r in rk), key=lambda t: abs(f32(t[1] - t[2])))
            decide = f"largest agreeing gap shift {abs(f32(mp - md)):.3f} at row {pos} ({mp:.3f} vs {md:.3f})"
        else:
            name, pos, mp, md, shift = worst
            op = "+" if name != "WIDE-ROW" else "vs"
            decide = f"{name} row {pos}: {mp:.3f} {op} {md:.3f}, gap shift {shift:.3f}"
        print(f"| {tree} | {model or '?'} | {prompt} | {arm} | {sm or '?'} | {len(sources)} | {sorted(sources)[0]} "
              f"| {len(flips)} / {budget} | {old} | {('%.4f' % floor[0]) if floor else 'none'} | {new} | {decide} |")
    print()
    print("| old verdict | new verdict | distinct tables |")
    print("|---|---|---|")
    for (old, new), n in sorted(tally.items()):
        print(f"| {old} | {new} | {n} |")


if __name__ == "__main__":
    main()
