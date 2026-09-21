#!/usr/bin/env python3
"""Day-16 WC pair analysis (research/spill-c-20260919/WC-DESTINATIONS.md): demote and promote wall times
OFF versus ON for the same two entries from the mirrored wc-pair cell, N=5 per arm per order, both orders,
one lock hold. Prints per-order and pooled medians with N, the promote with and without its inline demote,
and the telemetry regime from the collector's 250 ms sampler. Integrity and arithmetic only: one card, one
window, executed-not-qualified; the first cell of the decide-by review, not a verdict.
usage: wc-pair.py [cell-dir]   (default: pro-single-day16/wc-pair next to this file)
"""
import csv
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "pro-single-day16" / "wc-pair"
EV = ROOT / "ev"
DEMOTE = re.compile(r"\[prefix-host\] demote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
PROMOTE = re.compile(r"\[prefix-host\] promote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
ARMS = ["o1-off", "o1-on", "o2-on", "o2-off"]


def parse(label):
    log = (EV / f"{label}-server.log").read_text(errors="replace").splitlines()
    demotes, promotes, receipts = [], [], {"D2H": 0, "H2D": 0}
    last_demote = None
    for l in log:
        if "contracts door D2H receipt:" in l:
            receipts["D2H"] += 1
        if "contracts door H2D receipt:" in l:
            receipts["H2D"] += 1
        m = DEMOTE.search(l)
        if m:
            last_demote = float(m.group(3))
            demotes.append((int(m.group(1)), float(m.group(2)), last_demote))
            continue
        m = PROMOTE.search(l)
        if m:
            ms = float(m.group(3))
            inline = last_demote
            promotes.append((int(m.group(1)), float(m.group(2)), ms, inline, ms - inline if inline is not None else None))
            last_demote = None
    return demotes, promotes, receipts


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt(xs):
    return f"median {med(xs):.1f} ms (N={len(xs)}, min {min(xs):.1f}, max {max(xs):.1f})" if xs else "none"


def regime():
    sampler = ROOT / "command.sampler.log"
    if not sampler.exists():
        return "telemetry: sampler log absent"
    rows = list(csv.DictReader(sampler.open(errors="replace"), skipinitialspace=True))
    def col(name, conv=float):
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            v = v.split()[0]
            try:
                out.append(conv(v))
            except ValueError:
                pass
        return out
    temps = col("temperature.gpu")
    power = col("power.draw [W]")
    sm = col("clocks.current.sm [MHz]")
    limit = {r.get("power.limit [W]", "").strip() for r in rows}
    return (f"telemetry: {len(rows)} samples at 250 ms; temperature {min(temps):.0f}..{max(temps):.0f} C; "
            f"power draw max {max(power):.0f} W (limit {', '.join(sorted(limit))}); SM clock {min(sm):.0f}..{max(sm):.0f} MHz")


def main():
    print(f"WC pair cell: {ROOT}")
    per = {}
    for label in ARMS:
        d, p, r = parse(label)
        per[label] = (d, p, r)
        dm = [x[2] for x in d[:5]]
        pm = [x[2] for x in p[:5]]
        pe = [x[4] for x in p[:5] if x[4] is not None]
        print(f"\n{label}: {len(d)} demotes, {len(p)} promotes, receipts D2H={r['D2H']} H2D={r['H2D']}")
        print(f"  demote  (r2..r6): {fmt(dm)}")
        print(f"  promote (r3..r7): {fmt(pm)}")
        print(f"  promote minus inline demote: {fmt(pe)}")
        print(f"  entries: " + ", ".join(sorted({f'{t} tok/{mb} MB' for t, mb, *_ in d})))
    for arm in ("off", "on"):
        labels = [l for l in ARMS if l.endswith(f"-{arm}")]
        dm = [x[2] for l in labels for x in per[l][0][:5]]
        pm = [x[2] for l in labels for x in per[l][1][:5]]
        pe = [x[4] for l in labels for x in per[l][1][:5] if x[4] is not None]
        print(f"\npooled {arm.upper()} ({' + '.join(labels)}):")
        print(f"  demote:  {fmt(dm)}")
        print(f"  promote: {fmt(pm)}")
        print(f"  promote minus inline demote: {fmt(pe)}")
    print()
    print(regime())
    marks = EV / "marks.tsv"
    if marks.exists():
        rows = [l.split("\t") for l in marks.read_text().splitlines() if l.strip()]
        t0 = datetime.fromisoformat(rows[0][0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(rows[-1][0].replace("Z", "+00:00"))
        print(f"window: {rows[0][0]} .. {rows[-1][0]} ({(t1 - t0).total_seconds():.0f} s, one lock hold)")
    checks = []
    for label in ARMS:
        d, p, r = per[label]
        on = label.endswith("-on")
        checks.append((f"{label}: 6 demotes and 5 promotes", len(d) == 6 and len(p) == 5))
        checks.append((f"{label}: receipts {'present' if on else 'absent'} as the arm requires", (r["D2H"] == 6 and r["H2D"] == 5) if on else (r["D2H"] == 0 and r["H2D"] == 0)))
        checks.append((f"{label}: every promote had an inline demote in its window", all(x[3] is not None for x in p)))
    fails = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'ok' if ok else 'FAIL'}: {n}")
    print(f"WC PAIR REPLAY: {'PASS' if not fails else 'FAIL ' + str(len(fails))} ({len(checks)} checks)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
