#!/usr/bin/env python3
"""Day 39: the thermal, power and clock regime of the three-binary stall cell on the target card, whole hold and per
program (day37-regime.py with the day-39 programs, the SM clock added, and the box's UTC offset passed in).

Reads the collector's 250 ms `command.gpu.csv` (nvidia-smi local-time stamps in the BOX's zone, converted to UTC with
the offset the runner recorded on the box as `host_utc_offset=` in `stall/provenance.log`, never the reading host's
zone) and the cell's `ev/marks.tsv` (UTC). A program's window runs from its
first `boot-<prog>-` mark to its last `stopped-<prog>-` mark. Prints samples, temperature and power.draw ranges and
medians, and the card-wide memory range. No figure here is compared to another day or card.
usage: day39-regime.py <collector_dir> <ev_dir> <utc_offset like +0000>
"""
import csv
import datetime as dt
import statistics
import sys


def load_gpu(path, tz):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f, skipinitialspace=True)
        head = next(rd)
        idx = {h.strip(): i for i, h in enumerate(head)}
        for r in rd:
            if not r or len(r) < len(head):
                continue
            t = dt.datetime.strptime(r[idx["timestamp"]].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).astimezone(dt.timezone.utc)
            try:
                temp = float(r[idx["temperature.gpu"]])
                watt = float(r[idx["power.draw [W]"]].replace("W", "").strip())
                mem = float(r[idx["memory.used [MiB]"]].replace("MiB", "").strip())
                sm = float(r[idx["clocks.current.sm [MHz]"]].replace("MHz", "").strip())
            except ValueError:
                continue
            rows.append((t, temp, watt, mem, r[idx["power.limit [W]"]].strip(), sm))
    return rows


def load_marks(path):
    out = []
    with open(path) as f:
        for ln in f:
            ts, name = ln.rstrip("\n").split("\t")
            # the mark stamp carries 3 (the box's date) or 9 (the local rig's) fraction digits: pad or cut to 6
            base, frac = ts.rstrip("Z").split(".")
            out.append((dt.datetime.strptime(base + "." + (frac + "000000")[:6], "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=dt.timezone.utc), name))
    return out


def show(label, rows):
    if not rows:
        print(f"DAY39 REGIME {label}: 0 samples")
        return
    temps = [r[1] for r in rows]
    watts = [r[2] for r in rows]
    mems = [r[3] for r in rows]
    sms = [r[5] for r in rows]
    print(f"DAY39 REGIME {label}: samples={len(rows)} temp_c={min(temps):.0f}..{max(temps):.0f} (median {statistics.median(temps):.0f}) "
          f"power_w={min(watts):.2f}..{max(watts):.2f} (median {statistics.median(watts):.2f}) "
          f"clocks_sm_mhz={min(sms):.0f}..{max(sms):.0f} (median {statistics.median(sms):.0f}) "
          f"mem_used_mib={min(mems):.0f}..{max(mems):.0f} power_limit={sorted({r[4] for r in rows})}")


def main():
    off = sys.argv[3]
    sign = -1 if off[0] == "-" else 1
    tz = dt.timezone(sign * dt.timedelta(hours=int(off[1:3]), minutes=int(off[3:5])))
    gpu = load_gpu(f"{sys.argv[1]}/command.gpu.csv", tz)
    marks = load_marks(f"{sys.argv[2]}/marks.tsv")
    show("hold", gpu)
    first, last = marks[0][0], marks[-1][0]
    show(f"marks {first.isoformat()} to {last.isoformat()}", [r for r in gpu if first <= r[0] <= last])
    for prog in ("p1-b0", "p2-b1", "p3-b2", "p4-b2", "p5-b1", "p6-b0"):
        b = [t for t, n in marks if n.startswith(f"boot-{prog}-")]
        s = [t for t, n in marks if n.startswith(f"stopped-{prog}-")]
        if not b or not s:
            print(f"DAY39 REGIME {prog}: no marks")
            continue
        lo, hi = min(b), max(s)
        show(f"{prog} {lo.strftime('%H:%M:%S')}Z to {hi.strftime('%H:%M:%S')}Z", [r for r in gpu if lo <= r[0] <= hi])
    return 0


if __name__ == "__main__":
    sys.exit(main())
