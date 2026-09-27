#!/usr/bin/env python3
"""Day-31 reading of the RTX 5090 pair cell (day31-wc-cell.sh under the collector), fixed before the run.

Reads the four server logs with day 16's regexes (the `[prefix-host] demote: N tokens, X MB in Y ms` and
`promote: ... in Y ms` lines), pools the first five demotes (r2..r6) and five promotes (r3..r7) of each boot per
arm (N=10 per arm over both orders), and prints beside them what the day-16 replay does not read: the door's
own `published off the tick ... Xms from submission to completion` figures on the ON arm, the `request parked`
and `restore submitted off the tick` counts per boot, the PINNED-DEFAULT line the transfer gate printed on this
card, and the regime from the collector's 250 ms CSV and the cell's own 1 s CSV. Pooled medians with N and
IQR; `on_minus_off` with the two IQRs in quadrature as `unc`, `isolated` when |value| > unc. Integrity and
arithmetic only: one card, one window, executed-not-qualified; nothing here decides the door.
usage: day31-wc-reading.py <collector-cell-dir>   (the dir holding CELL.jsonl; ev/ under the base name beside it)
"""
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
BASE = ROOT.parent / re.sub(r"-retry\d+$", "", ROOT.name)
EV = BASE / "ev"
DEMOTE = re.compile(r"\[prefix-host\] demote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
PROMOTE = re.compile(r"\[prefix-host\] promote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
DEMOTE_PUB = re.compile(r"demote published off the tick: ticket seq=\d+ complete after (\d+) poll\(s\), ([\d.]+)ms from submission to completion")
PROMOTE_PUB = re.compile(r"promote published off the tick: ticket complete after (\d+) poll\(s\), ([\d.]+)ms from submission to completion")
ARMS = ["o1-off", "o1-on", "o2-on", "o2-off"]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr(xs):
    if len(xs) < 4:
        return float("nan")
    q = statistics.quantiles(xs, n=4)
    return q[2] - q[0]


def fmt(xs):
    return f"median {med(xs):.1f} (N={len(xs)}, min {min(xs):.1f}, max {max(xs):.1f}, IQR {iqr(xs):.1f})" if xs else "none"


def parse(label):
    lines = (EV / f"{label}-server.log").read_text(errors="replace").splitlines()
    d, p, dpub, ppub = [], [], [], []
    last_demote = None
    counts = {"parked": 0, "restore_submitted": 0, "refused": 0, "disabled": 0, "dropped": 0, "d2h": 0, "h2d": 0, "door_on_line": 0}
    for l in lines:
        counts["parked"] += l.count("request parked")
        counts["restore_submitted"] += "restore submitted off the tick" in l
        counts["refused"] += "refused" in l
        counts["disabled"] += "DISABLED" in l
        counts["dropped"] += "dropped" in l
        counts["d2h"] += "contracts door D2H receipt:" in l
        counts["h2d"] += "contracts door H2D receipt:" in l
        counts["door_on_line"] += "[prefix-host] contracts door ON (MEMRA_KV_HOST_CONTRACTS=1)" in l
        m = DEMOTE_PUB.search(l)
        if m:
            dpub.append(float(m.group(2)))
        m = PROMOTE_PUB.search(l)
        if m:
            ppub.append(float(m.group(2)))
        m = DEMOTE.search(l)
        if m:
            last_demote = float(m.group(3))
            d.append((int(m.group(1)), float(m.group(2)), last_demote))
            continue
        m = PROMOTE.search(l)
        if m:
            ms = float(m.group(3))
            p.append((int(m.group(1)), float(m.group(2)), ms, last_demote, ms - last_demote if last_demote is not None else None))
            last_demote = None
    return d, p, dpub, ppub, counts


def col(rows, name):
    out = []
    for r in rows:
        v = r.get(name)
        if v is None:
            continue
        try:
            out.append(float(v.split()[0]))
        except (ValueError, IndexError):
            pass
    return out


def regime(path, label):
    if not path.exists():
        return f"{label}: {path.name} absent"
    rows = list(csv.DictReader(path.open(errors="replace"), skipinitialspace=True))
    t = col(rows, "temperature.gpu")
    w = col(rows, "power.draw [W]")
    m = col(rows, "memory.used [MiB]")
    s = col(rows, "clocks.current.sm [MHz]")
    if not t:
        return f"{label}: {path.name} has no parseable rows ({len(rows)} rows)"
    return (f"{label}: {len(rows)} samples; temperature {min(t):.0f}..{max(t):.0f} C; power {min(w):.1f}..{max(w):.1f} W; "
            f"memory.used {min(m):.0f}..{max(m):.0f} MiB; SM clock {min(s):.0f}..{max(s):.0f} MHz")


def main():
    print(f"DAY31 PAIR reading: evidence {EV}, collector capture {ROOT.name}")
    for name in ("CELL.txt", "binary.sha256", "LOCK.json"):
        f = EV / name
        if f.exists():
            print(f"  {name}: " + " | ".join(f.read_text().strip().splitlines()))
    pinned = None
    roundtrips = []
    pl = EV / "pinned-default.log"
    if pl.exists():
        for l in pl.read_text(errors="replace").splitlines():
            if l.startswith("PINNED-DEFAULT device="):
                pinned = l.strip()
            elif l.startswith("PINNED-DEFAULT roundtrip"):
                roundtrips.append(l.strip())
        print(f"  pinned arm on this card: {pinned or 'no PINNED-DEFAULT device= line (' + pl.read_text(errors='replace').strip().splitlines()[0] + ')'}")
        for l in roundtrips:
            print(f"  {l}")
    per = {}
    checks = []
    for label in ARMS:
        d, p, dpub, ppub, c = parse(label)
        per[label] = (d, p, dpub, ppub, c)
        on = label.endswith("-on")
        print(f"\n{label}: {len(d)} demotes, {len(p)} promotes, receipts D2H={c['d2h']} H2D={c['h2d']}, parked={c['parked']}, "
              f"restore_submitted={c['restore_submitted']}, refused={c['refused']}, disabled={c['disabled']}, dropped={c['dropped']}, door_on_line={c['door_on_line']}")
        print(f"  raw demote in-ms (r2..r7):  {[x[2] for x in d]}")
        print(f"  raw promote in-ms (r3..r7): {[x[2] for x in p]}  inline demote ms: {[x[3] for x in p]}")
        print(f"  demote  (r2..r6): {fmt([x[2] for x in d[:5]])}")
        print(f"  promote (r3..r7): {fmt([x[2] for x in p[:5]])}")
        print(f"  promote minus inline demote: {fmt([x[4] for x in p[:5] if x[4] is not None])}")
        if on:
            print(f"  door demote submission-to-completion: {fmt(dpub)}  raw {dpub}")
            print(f"  door promote submission-to-completion: {fmt(ppub)}  raw {ppub}")
        print("  entries: " + ", ".join(sorted({f"{t} tok/{mb} MB" for t, mb, *_ in d})))
        checks.append((f"{label}: 6 demotes and 5 promotes", len(d) == 6 and len(p) == 5))
        checks.append((f"{label}: receipts {'present' if on else 'absent'} as the arm requires",
                       (c["d2h"] == 6 and c["h2d"] == 5) if on else (c["d2h"] == 0 and c["h2d"] == 0)))
        checks.append((f"{label}: the door boot line {'present' if on else 'absent'}", (c["door_on_line"] == 1) == on))
        checks.append((f"{label}: every promote had an inline demote in its window", all(x[3] is not None for x in p)))
        checks.append((f"{label}: zero refused, DISABLED or dropped lines", c["refused"] == 0 and c["disabled"] == 0 and c["dropped"] == 0))
        if on:
            checks.append((f"{label}: 6 demote and 5 promote completion lines", len(dpub) == 6 and len(ppub) == 5))
    pooled = {}
    for arm in ("off", "on"):
        labels = [l for l in ARMS if l.endswith(f"-{arm}")]
        dm = [x[2] for l in labels for x in per[l][0][:5]]
        pm = [x[2] for l in labels for x in per[l][1][:5]]
        pe = [x[4] for l in labels for x in per[l][1][:5] if x[4] is not None]
        pooled[arm] = (dm, pm, pe)
        print(f"\npooled {arm.upper()} ({' + '.join(labels)}):")
        print(f"  demote:  {fmt(dm)}")
        print(f"  promote: {fmt(pm)}")
        print(f"  promote minus inline demote: {fmt(pe)}")
        if arm == "on":
            dpub = [x for l in labels for x in per[l][2]]
            ppub = [x for l in labels for x in per[l][3]]
            print(f"  door demote submission-to-completion: {fmt(dpub)}")
            print(f"  door promote submission-to-completion: {fmt(ppub)}")
    print()
    print(regime(ROOT / "command.gpu.csv", "collector 250 ms"))
    print(regime(EV / "card.during.csv", "cell 1 s"))
    for snap in ("before", "after"):
        f = EV / f"card.{snap}.csv"
        if f.exists():
            print(f"card.{snap}: " + " | ".join(f.read_text().strip().splitlines()))
    checks.append(("pinned arm on this card printed as write-combined", pinned is not None and "kind=write-combined" in pinned))
    fails = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'ok' if ok else 'FAIL'}: {n}")
    verdict = []
    for name, idx in (("demote", 0), ("promote", 1), ("promote_minus_inline", 2)):
        off, on = pooled["off"][idx], pooled["on"][idx]
        if off and on:
            diff = med(on) - med(off)
            unc = (iqr(on) ** 2 + iqr(off) ** 2) ** 0.5
            tag = "isolated" if abs(diff) > unc else "under_resolution"
            verdict.append(f"{name} off {med(off):.1f} (N={len(off)}) on {med(on):.1f} (N={len(on)}) on_minus_off {diff:+.1f} unc {unc:.1f} {tag}")
        else:
            verdict.append(f"{name} unread")
    parked = [per[l][4]["parked"] for l in ARMS]
    print(f"DAY31 PAIR VERDICT: {'; '.join(verdict)}; parked per boot {parked}; pinned={'write-combined' if pinned and 'kind=write-combined' in pinned else 'unverified'}; admissible={not fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
