#!/usr/bin/env python3
"""Day 72 reader for cell `gap15` (research/spill-c-20260919/DAY72.md section 1, registered before the cell's scripts).

Part A: DAY64 section 5's admissibility clause over the four timed arms, then `day60-gap.py` unchanged (its lines are
printed as it prints them). Part B: each profiled run's window located from its own log (the stamped `STEADY-STATE
window` line is the window's end, the printed seconds its length), mapped onto the trace through the SQLite export's
session start; over it, per window token, B1 `gpu_busy` and `gpu_idle`, B2 `h2d` count, bytes, busy and `h2d_exposed`,
B3 `kernel_sum` and the five kernel names that differ most between the arms. Each profiled run's window rows (every
kernel and copy that overlaps the window) are written to `ev/<label>.window.tsv`. Then the registered verdict line.

usage: day72-read.py <cell-dir> [--rig NAME]
"""
import importlib.util
import re
import sqlite3
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("day40_attrib", HERE / "day40-attrib.py")
d40 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d40)

N = 32
ADMISSIBLE_IQR = 0.005
ARMS = ("ref", "refc", "on", "onc")
WINDOW = re.compile(r"^(\d\d:\d\d:\d\d\.\d{3})\tMoE cache STEADY-STATE window: (\d+) decode steps in ([0-9.]+)s", re.M)
FILL = re.compile(r"\[experts-via-tier\] fill complete before decode in ([0-9.]+) ms")


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def iqr(values):
    if len(values) < 4:
        return float("nan")
    q = statistics.quantiles(values, n=4)
    return q[2] - q[0]


def union(intervals):
    """Total length of the union of (start, end) intervals and the merged list."""
    merged = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return sum(b - a for a, b in merged), merged


def overlap(merged_a, merged_b):
    """Length of the intersection of two merged interval lists."""
    i = j = 0
    total = 0
    while i < len(merged_a) and j < len(merged_b):
        a0, a1 = merged_a[i]
        b0, b1 = merged_b[j]
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            total += hi - lo
        if a1 < b1:
            i += 1
        else:
            j += 1
    return total


def clip(rows, lo, hi):
    return [(max(a, lo), min(b, hi)) for a, b in rows if b > lo and a < hi]


def part_a(cell, rig):
    ev = cell / "ev"
    values = defaultdict(lambda: defaultdict(list))
    for log in sorted(ev.glob("o[12]-*-r[1-5].log")):
        arm = log.stem.split("-")[1]
        r = d40.parse_run(log)
        for key in ("gen_s", "window_s"):
            if key in r:
                values[arm][key].append(r[key])
    failing = []
    for a in ARMS:
        for key in ("gen_s", "window_s"):
            spread = iqr(values[a][key])
            if not spread <= ADMISSIBLE_IQR:
                failing.append(f"{a}:{key}={spread:.4f}")
    admissible = not failing
    worst = {key: max(iqr(values[a][key]) for a in ARMS) for key in ("gen_s", "window_s")}
    print(f"DAY72 ADMISSIBILITY rig={rig} ceiling={ADMISSIBLE_IQR} max_iqr_gen={worst['gen_s']:.4f}"
          f" max_iqr_window={worst['window_s']:.4f} failing={failing}"
          f" -> {'admissible' if admissible else 'inadmissible'}")
    out = subprocess.run([sys.executable, str(HERE / "day60-gap.py"), str(cell), "--rig", rig],
                         capture_output=True, text=True)
    sys.stdout.write(out.stdout)
    if out.stderr:
        sys.stdout.write("day60-gap.py stderr: " + out.stderr)
    gap = re.search(r"DAY60 GAP rig=\S+ integrity=(\w+) window: wall_gap=([+-][0-9.]+) cpu_gap=\S+ top=\S+ (\w+);",
                    out.stdout)
    if not gap or gap.group(1) != "ok":
        return admissible, "void", None
    return admissible, gap.group(3), float(gap.group(2))


def session_start_ns(db):
    return db.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]


def strings(db):
    return dict(db.execute("select id, value from StringIds"))


def profiled(ev, label):
    """Part B for one profiled run: its window's per-token figures, or a reason it is not read."""
    log = ev / f"{label}.log"
    db_path = ev / f"{label}.sqlite"
    if not log.exists() or not db_path.exists():
        return None, "log or sqlite missing"
    r = d40.parse_run(log)
    exit_code = int((ev / f"{label}.exit").read_text().strip())
    if exit_code != 0 or not r.get("match", "").endswith("MATCH") or r.get("window_n") != N:
        return None, f"exit={exit_code} match={r.get('match')} window_n={r.get('window_n')}"
    text = log.read_text(errors="replace")
    if "-on-" in label:
        fill = FILL.search(text)
        if not fill or r.get("physical_reads") != 0:
            return None, f"fill={bool(fill)} physical_reads={r.get('physical_reads')}"
    m = WINDOW.search(text)
    db = sqlite3.connect(str(db_path))
    try:
        start_ns = session_start_ns(db)
        names = strings(db)
        kernels = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
        copies = db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    except sqlite3.Error as e:
        return None, f"sqlite: {e}"
    session = datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc)
    end_clock = datetime.strptime(m.group(1), "%H:%M:%S.%f").time()
    end_utc = datetime.combine(session.date(), end_clock, tzinfo=timezone.utc)
    if end_utc < session - timedelta(hours=12):
        end_utc += timedelta(days=1)
    hi = int(end_utc.timestamp() * 1e9) - start_ns
    lo = hi - int(float(m.group(3)) * 1e9)
    wk = [(a, b, names.get(n, str(n))) for a, b, n in kernels if b > lo and a < hi]
    wc = [(a, b, nbytes, kind) for a, b, nbytes, kind in copies if b > lo and a < hi]
    if not wk:
        return None, "no kernel rows inside the window"
    with open(ev / f"{label}.window.tsv", "w") as f:
        f.write(f"# window lo_ns={lo} hi_ns={hi} (trace ns since session start)\n")
        for a, b, name in wk:
            f.write(f"K\t{a}\t{b}\t{name}\n")
        for a, b, nbytes, kind in wc:
            f.write(f"M\t{a}\t{b}\t{nbytes}\t{kind}\n")
    busy, kmerged = union(clip([(a, b) for a, b, _ in wk], lo, hi))
    h2d = [(a, b, n) for a, b, n, kind in wc if kind == 1]  # CUPTI copyKind 1: host to device
    h2d_busy, hmerged = union(clip([(a, b) for a, b, _ in h2d], lo, hi))
    per_name = defaultdict(int)
    for a, b, name in wk:
        per_name[name] += min(b, hi) - max(a, lo)
    ms = 1e6 * N
    return {
        "window_ms": (hi - lo) / 1e6,
        "gpu_busy": busy / ms,
        "gpu_idle": (hi - lo - busy) / ms,
        "h2d_count": len(h2d) / N,
        "h2d_mb": sum(n for _, _, n in h2d) / 1e6 / N,
        "h2d_busy": h2d_busy / ms,
        "h2d_exposed": (h2d_busy - overlap(hmerged, kmerged)) / ms,
        "kernel_sum": sum(per_name.values()) / ms,
        "per_name": {k: v / ms for k, v in per_name.items()},
        "kernels": len(wk) / N,
    }, None


def main():
    cell = Path(sys.argv[1])
    rig = sys.argv[sys.argv.index("--rig") + 1] if "--rig" in sys.argv else "?"
    ev = cell / "ev"
    admissible, part_a_verdict, window_gap = part_a(cell, rig)
    nsys = (ev / "nsys.txt").read_text().strip() if (ev / "nsys.txt").exists() else "nsys.txt missing"
    print(f"DAY72 PART B rig={rig} {nsys.splitlines()[0] if nsys else '-'}")
    got = defaultdict(list)
    reasons = []
    for label in ("p-ref-r1", "p-on-r1", "p-on-r2", "p-ref-r2"):
        res, why = profiled(ev, label) if not nsys.startswith("nsys=none") else (None, "nsys=none")
        if res is None:
            reasons.append(f"{label}: {why}")
            print(f"DAY72 B {label}: not read ({why})")
            continue
        got[label.split("-")[1]].append(res)
        print(f"DAY72 B {label}: window_ms={res['window_ms']:.1f} per window token (ms): gpu_busy={res['gpu_busy']:.4f}"
              f" gpu_idle={res['gpu_idle']:.4f} kernel_sum={res['kernel_sum']:.4f} kernels={res['kernels']:.1f}"
              f" h2d_count={res['h2d_count']:.1f} h2d_mb={res['h2d_mb']:.2f} h2d_busy={res['h2d_busy']:.4f}"
              f" h2d_exposed={res['h2d_exposed']:.4f}")
    part_b = "not_read"
    if len(got["ref"]) == 2 and len(got["on"]) == 2 and not reasons:
        diff = {k: med([x[k] for x in got["on"]]) - med([x[k] for x in got["ref"]])
                for k in ("gpu_busy", "gpu_idle", "kernel_sum", "h2d_busy", "h2d_exposed", "h2d_count", "h2d_mb")}
        print("DAY72 B door_minus_ref per window token (ms, medians of two): "
              + " ".join(f"{k}={v:+.4f}" for k, v in diff.items()))
        names = set().union(*(x["per_name"] for x in got["ref"] + got["on"]))
        per = {n: med([x["per_name"].get(n, 0.0) for x in got["on"]]) - med([x["per_name"].get(n, 0.0)
                                                                              for x in got["ref"]]) for n in names}
        top = sorted(per.items(), key=lambda kv: -abs(kv[1]))[:5]
        print("DAY72 B3 kernels differing most (door minus ref, ms per window token): "
              + ", ".join(f"{n}={v:+.4f}" for n, v in top))
        if window_gap is None:
            part_b = "unplaced"
            print("DAY72 B note: Part A gave no window_gap; Part B is read against nothing")
        else:
            half = window_gap / 2
            if diff["gpu_idle"] >= half:
                part_b = "gpu_stall"
            elif diff["kernel_sum"] >= half:
                part_b = "kernels_longer"
            else:
                part_b = "unplaced"
            print(f"DAY72 B rule: half of Part A's window_gap={half:+.4f} ms per token")
    integrity = "ok" if part_a_verdict != "void" else "FAIL"
    print(f"DAY72 GAP15 VERDICT rig={rig} integrity={integrity} admissible={'yes' if admissible else 'no'}"
          f" partA={part_a_verdict} partB={part_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
