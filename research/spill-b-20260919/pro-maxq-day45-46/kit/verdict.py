#!/usr/bin/env python3
"""One VERDICT line for the admit-w-release Max-Q cell (PREREG.md), from the registered readers' clause lines.

usage: verdict.py [--card LABEL] [--day45 DIR] [--day46 DIR] [--root ROOT] [--out OUTDIR] [--rerun-readers]

DIR is a half's receipt root (the R of chain-b.sh or chain.sh: boots/, read.log). With --root, a half that is not
given and absent from OUTDIR (not attempted there) is taken from the newest ROOT/out-*/day45 or day46 whose boots all
finished, so two parts run in separate gaps combine into one verdict; a half attempted in OUTDIR is always read from
OUTDIR. --rerun-readers runs the registered readers (day45-read.py, day46-read.py)
on DIR first and uses their stdout instead of read.log. The rule (PREREG.md section 3):

- DAY45 2.2 PASS: these 12 clause lines all read PASS: W4 and W1 on O1-off, O1-on, O2-on, O2-off; W2 on O1-on and
  O2-on; W3 for O1 and O2 (DAY45.md 2.2: "Every clause PASS on the target card under addendum B").
- DAY46 2.2 PASS: these 24 clause lines all read PASS: P2, P1 and P3 on the four enforcing boots; P5-PROBE on all six
  boots; P5 on O1-enforce-wrel and O2-enforce-wrel; P4 for each order and enforcing arm (DAY46.md 2.2: "P1 to P5
  PASS on all four enforcing boots, both orders"; P5's probe half applies on every arm, DAY46.md 1.3).
- A clause line of a boot that did not finish (no <boot>.exit or no finished.txt; a boot cut by TIMEOUT.txt or
  THERMAL-STOP.txt has neither) does not count; a missing required line makes the half INCOMPLETE, never PASS.
- VERDICT PASS when both halves PASS; FAIL when a finished boot's required line reads FAIL; INCOMPLETE otherwise.
The value reading (DAY46 wave2 ok200 on enforce against enforce-wrel) and the shared-box labels are printed as
READING lines before the VERDICT line; they carry no bound.
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--card", default="pro6000maxq")
ap.add_argument("--day45")
ap.add_argument("--day46")
ap.add_argument("--root")
ap.add_argument("--out")
ap.add_argument("--rerun-readers", action="store_true")
a = ap.parse_args()

D45_BOOTS = ["O1-off", "O1-on", "O2-on", "O2-off"]
D46_BOOTS = ["O1-shadow", "O1-enforce", "O1-enforce-wrel", "O2-enforce-wrel", "O2-enforce", "O2-shadow"]
D46_ENF = ["O1-enforce", "O1-enforce-wrel", "O2-enforce-wrel", "O2-enforce"]


def req45():
    r = [("W4", ("boot", b)) for b in D45_BOOTS] + [("W1", ("boot", b)) for b in D45_BOOTS]
    r += [("W2", ("boot", b)) for b in ("O1-on", "O2-on")] + [("W3", ("order", o)) for o in ("O1", "O2")]
    return r


def req46():
    r = [(c, ("boot", b)) for c in ("P2", "P1", "P3") for b in D46_ENF]
    r += [("P5-PROBE", ("boot", b)) for b in D46_BOOTS] + [("P5", ("boot", b)) for b in ("O1-enforce-wrel", "O2-enforce-wrel")]
    r += [("P4", ("order-arm", f"{o}:{arm}")) for o in ("O1", "O2") for arm in ("enforce", "enforce-wrel")]
    return r


def boot_done(d, b):
    return os.path.exists(os.path.join(d, "boots", f"{b}.exit")) and os.path.exists(os.path.join(d, "boots", b, "finished.txt"))


def half_complete(d, boots):
    return bool(d) and os.path.isdir(d) and all(boot_done(d, b) for b in boots)


def newest_complete(root, half, boots):
    cands = sorted(glob.glob(os.path.join(root, "out-*", half)), reverse=True)
    for c in cands:
        if half_complete(c, boots) and not any(os.path.exists(os.path.join(os.path.dirname(c), f))
                                               for f in ("TIMEOUT.txt", "THERMAL-STOP.txt")):
            return c
    return None


def reader_lines(d, reader):
    if a.rerun_readers:
        p = subprocess.run([sys.executable, "-I", reader, a.card, d], capture_output=True, text=True)
        return p.stdout.splitlines()
    try:
        return open(os.path.join(d, "read.log"), errors="replace").read().splitlines()
    except OSError:
        return []


LINE = re.compile(r"^(DAY4[56]) (\S+) card=\S+ (?:boot=(\S+)|order=(\S+?)(?: arm=(\S+))?) .*-> (PASS|FAIL)\s*$")


def judge(name, d, lines, required, boots):
    if not d or not os.path.isdir(d):
        return "INCOMPLETE", f"{name}=NOT-RUN", []
    stopped = [f for f in ("TIMEOUT.txt", "THERMAL-STOP.txt") if os.path.exists(os.path.join(os.path.dirname(d), f))]
    got = {}
    for l in lines:
        m = LINE.match(l)
        if not m:
            continue
        clause = m.group(2)
        if m.group(3):
            key = ("boot", m.group(3))
        elif m.group(5):
            key = ("order-arm", f"{m.group(4)}:{m.group(5)}")
        else:
            key = ("order", m.group(4))
        got[(clause, key)] = m.group(6)

    def finished(key):
        kind, v = key
        if kind == "boot":
            return boot_done(d, v)
        order = v.split(":")[0]
        if name == "day45":
            return boot_done(d, f"{order}-off") and boot_done(d, f"{order}-on")
        arm = v.split(":")[1]
        return boot_done(d, f"{order}-shadow") and boot_done(d, f"{order}-{arm}")

    passed, failed, missing = 0, [], []
    for r in required:
        clause, key = r
        if not finished(key):
            missing.append(f"{clause}@{key[1]}")
            continue
        v = got.get(r)
        if v == "PASS":
            passed += 1
        elif v == "FAIL":
            failed.append(f"{clause}@{key[1]}")
        else:
            missing.append(f"{clause}@{key[1]}")
    state = "FAIL" if failed else ("PASS" if passed == len(required) else "INCOMPLETE")
    detail = f"{name}={state}({passed}/{len(required)})"
    notes = []
    if failed:
        notes.append(f"{name}_failed={failed[:6]}")
    if missing:
        notes.append(f"{name}_missing={missing[:6]}")
    if stopped:
        notes.append(f"{name}_stopped_by={stopped}")
    return state, detail, notes


out_dir = a.out
d45 = a.day45 or (os.path.join(out_dir, "day45") if out_dir and os.path.isdir(os.path.join(out_dir, "day45")) else None)
d46 = a.day46 or (os.path.join(out_dir, "day46") if out_dir and os.path.isdir(os.path.join(out_dir, "day46")) else None)
src = {}
if a.root:
    # Only a half this out dir did not attempt comes from another out dir; a half attempted here is read as it is,
    # cut or not, so an older complete copy never masks this run.
    if d45 is None:
        alt = newest_complete(a.root, "day45", D45_BOOTS)
        if alt:
            src["day45"], d45 = alt, alt
    if d46 is None:
        alt = newest_complete(a.root, "day46", D46_BOOTS)
        if alt:
            src["day46"], d46 = alt, alt

l45 = reader_lines(d45, os.path.join(HERE, "d45", "research", "spill-b-20260919", "day45-read.py")) if d45 else []
l46 = reader_lines(d46, os.path.join(HERE, "d46", "research", "spill-b-20260919", "day46-read.py")) if d46 else []
s45, t45, n45 = judge("day45", d45, l45, req45(), D45_BOOTS)
s46, t46, n46 = judge("day46", d46, l46, req46(), D46_BOOTS)
readings = []
for k, v in src.items():
    readings.append(f"READING {k} taken from {v} (the newest out dir whose {k} boots all finished)")
for l in l46:
    if l.startswith("DAY46 READING") and "wave=wave2" in l and ("-enforce " in l or "-enforce-wrel " in l):
        readings.append("READING " + l)
for d, half in ((d45, "day45"), (d46, "day46")):
    if not d:
        continue
    gw = 0
    for p in glob.glob(os.path.join(d, "boots", "*", "server.log")):
        gw += sum(1 for l in open(p, errors="replace") if "[gpu-watch]" in l and ("fault" in l.lower() or "FATAL" in l))
    try:
        card_csv = open(os.path.join(d, "card.csv")).read().strip().splitlines()[-1]
    except (OSError, IndexError):
        card_csv = None
    try:
        run_card = open(os.path.join(os.path.dirname(d), "card.csv")).read().strip().splitlines()[-1]
    except (OSError, IndexError):
        run_card = None
    readings.append(f"READING {half} dir={d} card_csv=[{card_csv}] run_card=[{run_card}] gpu_watch_fault_lines={gw}")
if out_dir:
    tp = os.path.join(out_dir, "telemetry.csv")
    if os.path.exists(tp):
        temps, pw, sm, fans = [], [], [], []
        for l in open(tp, errors="replace"):
            f = [x.strip() for x in l.split(",")]
            try:
                temps.append(float(f[2])); fans.append(float(f[3])); pw.append(float(f[4])); sm.append(float(f[7]))
            except (IndexError, ValueError):
                pass
        if temps:
            sm_sorted = sorted(sm)
            readings.append(f"READING telemetry samples={len(temps)} (250 ms, card only) temp_max={max(temps):.0f}C "
                            f"fan_max={max(fans):.0f}% power_max={max(pw):.1f}W sm_clock_p50={sm_sorted[len(sm_sorted) // 2]:.0f}MHz")
overall = "FAIL" if "FAIL" in (s45, s46) else ("PASS" if (s45, s46) == ("PASS", "PASS") else "INCOMPLETE")
for r in readings + [f"NOTE {n}" for n in n45 + n46]:
    print(r)
verdict = f"VERDICT admit-w-release-default card={a.card} {t45} {t46} -> {overall}"
print(verdict)
if out_dir:
    with open(os.path.join(out_dir, "VERDICT.txt"), "w") as f:
        f.write(verdict + "\n")
    with open(os.path.join(out_dir, "READINGS.txt"), "w") as f:
        f.write("\n".join(readings + [f"NOTE {n}" for n in n45 + n46]) + "\n")
