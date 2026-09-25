#!/usr/bin/env python3
"""Day 55 reader (research/spill-c-20260919/DAY55.md section 1, registered before the cell ran): the card guard, the
memory admission's lines in every prime boot and the co-tenant trace of the 1 s card CSV, then the C8 verdict. The
stall readings themselves are day35-stall-reading.py's, run unchanged on the same ev/ (pass1, pass2).

usage: day55-guard.py <ev-dir>
"""
import csv
import datetime
import json
import sys
from pathlib import Path

COTENANT_MIB = 14000  # the 9B server's own clean footprint read 7,641 to 9,753 MiB on day 35
BOOTS = [("warmup", "prime"), ("pass1", "prime"), ("pass1", "off"), ("pass1", "on"),
         ("pass2", "on"), ("pass2", "off"), ("pass2", "prime")]


def ts(s):
    """A marks.tsv stamp (UTC)."""
    t = datetime.datetime.strptime(s.strip()[:23], "%Y-%m-%dT%H:%M:%S.%f")
    return t.replace(tzinfo=datetime.timezone.utc)


def main():
    ev = Path(sys.argv[1])
    marks = {}
    for row in (ev / "marks.tsv").read_text().splitlines():
        stamp, label = row.split("\t")
        marks[label] = ts(stamp)
    samples = []
    with open(ev / "card.during.csv") as f:
        for r in csv.reader(f):
            if not r or r[0].startswith("timestamp"):
                continue
            try:
                # nvidia-smi stamps local time; astimezone() reads a naive time as the host's zone.
                t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").astimezone(
                    datetime.timezone.utc)
                used = int(r[8].split()[0])
            except (ValueError, IndexError):
                continue
            samples.append((t, used))
    fails = []
    for p, kind in BOOTS:
        d = ev / p / kind
        boot = (d / "BOOT.txt").read_text() if (d / "BOOT.txt").exists() else ""
        guard = boot.split("guard=")[1].split()[0] if "guard=" in boot else "missing"
        log = (d / "server.log").read_text(errors="replace") if (d / "server.log").exists() else ""
        defers = log.count("[admit-oom] VRAM defer")
        rejects = log.count("[admit-oom] capacity reject")
        t0, t1 = marks.get(f"boot-{p}-{kind}"), marks.get(f"stopped-{p}-{kind}")
        window = [u for t, u in samples if t0 and t1 and t0 <= t <= t1]
        peak = max(window) if window else -1
        errors = None
        if kind == "prime" and (d / "prime" / "receipt.json").exists():
            r = json.loads((d / "prime" / "receipt.json").read_text())
            # the harness records a refused intruder in the summary's `errors` (and the run's `intruder.error`)
            errors = len(r["summary"].get("errors") or [])
        print(f"DAY55 BOOT {p}/{kind} guard={guard} admit_defers={defers} admit_rejects={rejects} "
              f"window_peak_memory_used_mib={peak} cotenant={'yes' if peak > COTENANT_MIB else 'no'}"
              + ("" if errors is None else f" prime_run_errors={errors}"))
        if guard != "clean":
            fails.append(f"{p}/{kind} guard={guard}")
        if peak > COTENANT_MIB:
            fails.append(f"{p}/{kind} co-tenant peak {peak} MiB")
        if kind == "prime" and p != "warmup":
            if defers or rejects:
                fails.append(f"{p}/prime admission defers={defers} rejects={rejects}")
            if errors != 0:
                fails.append(f"{p}/prime run errors={errors}")
    verdict = "prime_always_admitted" if not fails else "prime_not_admitted"
    print(f"DAY55 VERDICT -> {verdict}" + ("" if not fails else " failed=" + "; ".join(fails)))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
