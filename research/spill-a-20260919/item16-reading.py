#!/usr/bin/env python3
"""WP-A day 45 (DAY45.md section 1, OWED item 16) reader, written before the cell runs: the regime check and the placing
rule over DAY38 section 20's cell replicated in the G4 hold's thermal regime.

usage: item16-reading.py ROOT
Input: ROOT/hump/reading-hump.log (day38-hump-reading.py's lines), ROOT/hump/bNN-x<arm>/BOOT.txt (the start temperature
line and the local start and end stamps), ROOT/card-250ms.csv (nvidia-smi's 250 ms telemetry, local stamps).
Regime: every hump boot starts at >= 85 C and its SM clock median over its own window (start to end stamp) is at most
2000 MHz (the G4 hold's hump cell: 87 to 88 C, 1830 to 1995 MHz). Validity: the control xgpp's median HUMP > 0.15 ms,
and every arm read over two boots. Placing, with r = median HUMP of xbase and d = median HUMP of xg4 minus r:
  d > 0.15                 -> the design (G4 humps beyond the base arm in this regime)
  else xg4's HUMP > 0.15   -> the regime (the base arm shares G4's rise within the bound)
  else                     -> not reproduced (G4 flat in the regime that failed it)
G''' (xg3) is placed by the same rule, a reading.
"""
import csv
import datetime as dt
import glob
import os
import re
import statistics
import sys

HUMP = re.compile(r"HUMP arm=(\S+) boots=(\d+) median-hump=([+-][\d.]+) humps=(\S+)")
START_T = re.compile(r"start temperature\.gpu,clocks\.sm,power\.draw: (\d+), (\d+) MHz")
STAMP = re.compile(r"(start|end)_local=(\S+ \S+)")


def parse_ts(s):
    s = s.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None


def main():
    root = sys.argv[1]
    hump = {}
    for ln in open(os.path.join(root, "hump", "reading-hump.log"), errors="replace"):
        m = HUMP.search(ln)
        if m:
            hump[m.group(1)] = (int(m.group(2)), float(m.group(3)))
    tele = []
    with open(os.path.join(root, "card-250ms.csv"), errors="replace") as f:
        for row in csv.reader(f):
            if len(row) < 4 or row[0].startswith("timestamp"):
                continue
            t = parse_ts(row[0])
            try:
                temp = int(row[1].strip())
                sm = int(row[3].strip().split()[0])
            except (ValueError, IndexError):
                continue
            if t:
                tele.append((t, temp, sm))
    regime_ok, notes = True, []
    for bd in sorted(glob.glob(os.path.join(root, "hump", "b*-x*"))):
        txt = open(os.path.join(bd, "BOOT.txt"), errors="replace").read()
        m = START_T.search(txt)
        stamps = dict(STAMP.findall(txt))
        start, end = parse_ts(stamps.get("start", "")), parse_ts(stamps.get("end", ""))
        win = [s for (t, _, s) in tele if start and end and start <= t <= end]
        temps = [c for (t, c, _) in tele if start and end and start <= t <= end]
        t0 = int(m.group(1)) if m else None
        smed = statistics.median(win) if win else None
        ok = t0 is not None and t0 >= 85 and smed is not None and smed <= 2000
        regime_ok &= ok
        print(f"ITEM16 BOOT {os.path.basename(bd)} start_temp={t0} window_temp={min(temps) if temps else None}.."
              f"{max(temps) if temps else None} sm_median={smed} samples={len(win)} -> {'HOT' if ok else 'NOT HOT'}")
    for arm in ("xbase", "xg4", "xg3", "xgpp"):
        if arm not in hump or hump[arm][0] != 2:
            notes.append(f"{arm} not read over two boots")
    print(f"ITEM16 HUMPS {hump}")
    if notes:
        print(f"ITEM16 INCOMPLETE ({'; '.join(notes)}) -> nothing is placed; the cell repeats whole once")
        sys.exit(2)
    if not regime_ok:
        print("ITEM16 REGIME NOT REPRODUCED -> nothing is placed; the cell repeats once with the warm-up doubled")
        sys.exit(3)
    if hump["xgpp"][1] <= 0.15:
        print(f"ITEM16 CONTROL INVALID (xgpp {hump['xgpp'][1]:+.3f} <= 0.15) -> nothing is placed; the cell repeats once")
        sys.exit(4)
    r = hump["xbase"][1]

    def place(arm):
        d = hump[arm][1] - r
        if d > 0.15:
            return d, "THE DESIGN"
        if hump[arm][1] > 0.15:
            return d, "THE REGIME"
        return d, "NOT REPRODUCED"

    for arm in ("xg3", "xg4"):
        d, where = place(arm)
        print(f"ITEM16 PLACE arm={arm} hump={hump[arm][1]:+.3f} base={r:+.3f} minus-base={d:+.3f} -> {where}")
    d, where = place("xg4")
    print(f"ITEM16 -> G4's 5090 (f) FAIL (DAY38 section 19) placed on {where}")


if __name__ == "__main__":
    main()
