#!/usr/bin/env python3
"""WP-A design F2 (DAY71.md section 1) reader, written before its sitting runs.

usage: f2-reading.py R [arm]   (arm: the F2 arm's directory name, default f2; a dry run maps another)
(a) every R/gates/*.exit 0 (11 gates on the f2 binary); the CPU cells and the red arm are on record (day71/).
Cells R/{demote,chain,promote}/ab/o{1,2}/bNN-{base,f2} (stall_cell.py --mode demote, promote-long, promote; five boots
    per arm per order); R/hump/bNN-x{base,f2} (4 boots, --mode demote --n 8). Complete: 20 boots per A/B cell with
    receipts, `errors` empty, 20 `STALL REPLAY: PASS` each; the hump's 4 boots with 16 demote runs each.
Steady: the second and later line of each boot. LATE: a promote published after the next tick top (publication tick
    > submission tick + 1), day64-reading.py's rule.
(b) promote: f2's late promotes at most 5% of its steady promotes (DAY64's 9 of 180), pooled over both orders; and
    f2's span fill phase (`span receipt: fill X ms`) median at most 0.5 ms, per order.
(c) promote: f2's intruder e2e median at most base's - 5.0 ms and f2's PIN (`[prefix-host] promote: .. in Y ms`) median
    at most base's + 1.0 ms, per order.
(d) demote: f2's wall t0 to publication (`demote digests landed .. wall Xms t0 to publication`) and helper (`demote
    helper split: .. (helper Y ms)`) medians at most base's + 1.0 ms, per order. Reading: the helper's copy term.
(e) every A/B cell: f2's tenant stall median at most base's + 1.0 ms; chain: f2's `chain_wall_ms` median at most
    base's + 1.0 ms; the hump (day38's HUMP per boot): f2's median at most base's + 0.15 ms. Per order.
(f) every A/B cell and order: the `[prefix-host] demote:` count equal between the arms. Reading: the last `the set holds
    N bytes charged` value per boot.
Verdict: ADOPT when (a) to (f) pass; REFUTED when (a) fails; otherwise REVERT with the failed clauses named.
"""
import glob
import json
import os
import re
import statistics as st
import sys

ARM = sys.argv[2] if len(sys.argv) > 2 else "f2"
SPLIT = re.compile(r"demote helper split: ticket seq=\d+ copy ([\d.]+) ms over [\d.]+ MB \(minflt \+-?\d+\), hash "
                   r"([\d.]+) ms \(helper ([\d.]+) ms\)")
WALL = re.compile(r"demote digests landed off the tick: .*; wall ([\d.]+)ms t0 to publication")
PIN = re.compile(r"\[prefix-host\] promote: \d+ tokens, [\d.]+MB in ([\d.]+)ms")
PUB = re.compile(r"promote published off the tick: .*timeline from t0: submitted [+-][\d.]+ms \(tick (\d+), its top "
                 r"[+-][\d.]+ms\);.* published [+-][\d.]+ms \(tick (\d+), its top")
PH = re.compile(r"span receipt: fill ([\d.]+) ms, copies ([\d.]+) ms, digests ([\d.]+) ms")
SET = re.compile(r"the set holds (\d+) bytes charged")
MODES = {"demote": "demote", "chain": "promote-long", "promote": "promote"}


def med(xs):
    return st.median(xs) if xs else float("nan")


def rd(p):
    try:
        return open(p, errors="replace").read().strip()
    except OSError:
        return None


def ab(root, cell):
    mode = MODES[cell]
    complete, notes, pools = True, [], {}
    for order in ("o1", "o2"):
        for arm in ("base", ARM):
            ds = sorted(glob.glob(os.path.join(root, cell, "ab", order, f"b*-{arm}")))
            if len(ds) != 5:
                complete = False
                notes.append(f"{cell} {order} {arm} boots={len(ds)}")
            p = {k: [] for k in ("stall", "e2e", "chain", "helper", "copy", "wall", "pin", "ph", "set")}
            p.update(steady=0, late=0, demotes=0)
            for d in ds:
                rec = os.path.join(d, mode, "receipt.json")
                if not os.path.exists(rec):
                    complete = False
                    notes.append(f"{cell}/{order}/{os.path.basename(d)} no receipt")
                    continue
                j = json.load(open(rec))
                if j.get("summary", {}).get("errors"):
                    complete = False
                    notes.append(f"{cell}/{order}/{os.path.basename(d)} errors")
                for r in j["runs"]:
                    if r.get("arm") != mode:
                        continue
                    if "stall_ms" in r:
                        p["stall"].append(r["stall_ms"])
                    i = r.get("intruder") or {}
                    if "wall_ms" in i:
                        p["e2e"].append(i["wall_ms"])
                    if "chain_wall_ms" in i:
                        p["chain"].append(i["chain_wall_ms"])
                text = open(os.path.join(d, "server.log"), errors="replace").read()
                sp = list(SPLIT.finditer(text))[1:]
                p["helper"] += [float(m.group(3)) for m in sp]
                p["copy"] += [float(m.group(1)) for m in sp]
                p["wall"] += [float(m.group(1)) for m in list(WALL.finditer(text))[1:]]
                p["pin"] += [float(m.group(1)) for m in list(PIN.finditer(text))[1:]]
                pubs = list(PUB.finditer(text))[1:]
                p["steady"] += len(pubs)
                p["late"] += sum(1 for m in pubs if int(m.group(2)) > int(m.group(1)) + 1)
                p["ph"] += [tuple(float(x) for x in m.groups()) for m in list(PH.finditer(text))[1:]]
                sets = SET.findall(text)
                if sets:
                    p["set"].append(int(sets[-1]))
                p["demotes"] += text.count("[prefix-host] demote:")
            pools[(order, "base" if arm == "base" else "f2")] = p
    passes = (rd(os.path.join(root, cell, "ab", "replays.log")) or "").count("STALL REPLAY: PASS")
    if passes != 20:
        complete = False
        notes.append(f"{cell} replays PASS={passes} of 20")
    return complete, notes, pools


def hump(root):
    by = {}
    for f in sorted(glob.glob(os.path.join(root, "hump", "b*-x*", "demote", "receipt.json"))):
        arm = f.split(os.sep)[-3].split("-")[1]
        rs = [r for r in json.load(open(f))["runs"] if r.get("arm") == "demote" and r.get("itl_ms")]
        itl = [st.median(r["itl_ms"][:22]) for r in rs]
        if len(itl) < 16:
            return None
        by.setdefault(arm, []).append(max(itl[3:16]) - st.median(itl[:3]))
    return by


def main():
    root = sys.argv[1]
    exits = {os.path.basename(p)[:-5]: rd(p) for p in sorted(glob.glob(os.path.join(root, "gates", "*.exit")))}
    a = len(exits) == 11 and all(v == "0" for v in exits.values())
    print(f"F2 (a) gates {exits}")
    cells = {c: ab(root, c) for c in MODES}
    complete = all(v[0] for v in cells.values())
    notes = [n for v in cells.values() for n in v[1]]
    b_ok, c_ok, d_ok, e_ok, f_ok = [], [], [], [], []
    for order in ("o1", "o2"):
        for cell, (_, _, pools) in cells.items():
            B, F = pools[(order, "base")], pools[(order, "f2")]
            e_ok.append(med(F["stall"]) <= med(B["stall"]) + 1.0)
            f_ok.append(F["demotes"] == B["demotes"])
            line = (f"F2 READING cell={cell} order={order} stall base={med(B['stall']):.2f} f2={med(F['stall']):.2f} "
                    f"| demotes base={B['demotes']} f2={F['demotes']}")
            if cell == "demote":
                d_ok.append(med(F["wall"]) <= med(B["wall"]) + 1.0 and med(F["helper"]) <= med(B["helper"]) + 1.0)
                line += (f" | helper base={med(B['helper']):.1f} f2={med(F['helper']):.1f} ms (copy base="
                         f"{med(B['copy']):.2f} f2={med(F['copy']):.2f}) | wall base={med(B['wall']):.1f} "
                         f"f2={med(F['wall']):.1f} ms")
            if cell == "chain":
                e_ok.append(med(F["chain"]) <= med(B["chain"]) + 1.0)
                line += f" | chain base={med(B['chain']):.1f} f2={med(F['chain']):.1f} ms"
            if cell == "promote":
                fill = med([x[0] for x in F["ph"]])
                b_ok.append(fill <= 0.5)
                c_ok.append(med(F["e2e"]) <= med(B["e2e"]) - 5.0 and med(F["pin"]) <= med(B["pin"]) + 1.0)
                phase = lambda P, k: med([x[k] for x in P["ph"]])
                line += (f" | late base={B['late']}/{B['steady']} f2={F['late']}/{F['steady']} | pin base="
                         f"{med(B['pin']):.2f} f2={med(F['pin']):.2f} | e2e base={med(B['e2e']):.1f} f2="
                         f"{med(F['e2e']):.1f} | phases base fill {phase(B, 0):.2f} copies {phase(B, 1):.2f} digests "
                         f"{phase(B, 2):.2f}; f2 fill {phase(F, 0):.2f} copies {phase(F, 1):.2f} digests "
                         f"{phase(F, 2):.2f} ms")
            line += f" | set charged base={B['set'][-1:] or '-'} f2={F['set'][-1:] or '-'}"
            print(line)
    P = cells["promote"][2]
    late_f = sum(P[(o, "f2")]["late"] for o in ("o1", "o2"))
    steady_f = sum(P[(o, "f2")]["steady"] for o in ("o1", "o2"))
    late_b = sum(P[(o, "base")]["late"] for o in ("o1", "o2"))
    steady_b = sum(P[(o, "base")]["steady"] for o in ("o1", "o2"))
    b_ok.append(steady_f > 0 and late_f <= 0.05 * steady_f)
    print(f"F2 READING late pooled base={late_b}/{steady_b} f2={late_f}/{steady_f}")
    hb = hump(root)
    if hb is None or set(hb) != {"xbase", f"x{ARM}"}:
        complete = False
        notes.append("hump incomplete")
    else:
        e_ok.append(med(hb[f"x{ARM}"]) <= med(hb["xbase"]) + 0.15)
        print(f"F2 READING hump base={med(hb['xbase']):+.3f} f2={med(hb[f'x{ARM}']):+.3f} ms")
    if not complete:
        print(f"F2 INCOMPLETE ({'; '.join(notes)}) -> (b) to (f) read nothing; the cells repeat whole once")
    clauses = (("b", b_ok), ("c", c_ok), ("d", d_ok), ("e", e_ok), ("f", f_ok))
    res = {k: complete and all(v) for k, v in clauses}
    for k, v in clauses:
        print(f"F2 ({k}) {'PASS' if res[k] else 'FAIL'} {v}")
    failed = [k for k, _ in clauses if not res[k]]
    if not a:
        v = "REFUTED ((a) failed): revert in one commit, red receipts banked"
    elif not complete:
        v = "INCOMPLETE: repeat the cells whole once"
    elif failed:
        v = f"REVERT ((a) passed; failed {', '.join(failed)}): recorded as read, reverted in one commit"
    else:
        v = "ADOPT (F2 is the naked program)"
    print(f"F2 VERDICT -> {v}")


if __name__ == "__main__":
    main()
