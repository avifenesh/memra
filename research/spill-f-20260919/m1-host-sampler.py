#!/usr/bin/env python3
"""M1 host/storage sampler and validator (M1-PREREG.md section B telemetry; OWED 10).

  sample   --devices nvme0n1,dm-0 --out FILE [--interval-ms 250] [--pid N] [--duration-s S]
  validate FILE --devices nvme0n1,dm-0 [--max-gap-ms 500]

`sample` writes JSONL until SIGTERM/SIGINT or --duration-s: one header line, then one line per
tick on absolute monotonic deadlines (no drift) with /proc/diskstats rows for the named devices,
/proc/meminfo (MemAvailable, MemFree, Cached, Dirty, Writeback, Mlocked), aggregate /proc/stat
CPU jiffies, /proc/<pid>/io of the target when alive, and NVMe hwmon temperatures (milli-degC)
of the controllers behind the named namespaces. It reads only procfs/sysfs. A device missing
from a tick is recorded as missing, never filled in.

`validate` exits 0 only if the header is present, sequence numbers are contiguous, monotonic
time strictly increases, no gap exceeds --max-gap-ms, and every named device appears in every
tick. It prints every violation.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

SCHEMA = "m1-host-sampler-v1"
MEMINFO_KEYS = ("MemAvailable", "MemFree", "Cached", "Dirty", "Writeback", "Mlocked")
DISKSTAT_FIELDS = ("reads", "reads_merged", "read_sectors", "read_ms", "writes", "writes_merged",
                   "write_sectors", "write_ms", "in_flight", "io_ms", "weighted_io_ms")


def diskstats(names):
    rows = {}
    try:
        text = Path("/proc/diskstats").read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 14 and f[2] in names:
            rows[f[2]] = dict(zip(DISKSTAT_FIELDS, map(int, f[3:14])))
    return rows


def meminfo():
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            if key in MEMINFO_KEYS:
                out[key + "_kB"] = int(rest.split()[0])
    except OSError:
        pass
    return out


def cpu_jiffies():
    try:
        first = Path("/proc/stat").read_text().splitlines()[0].split()
    except (OSError, IndexError):
        return None
    return dict(zip(("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal"),
                    map(int, first[1:9])))


def proc_io(pid):
    if pid is None:
        return None
    try:
        return {k: int(v) for k, v in (l.split(": ") for l in
                                       Path(f"/proc/{pid}/io").read_text().splitlines())}
    except (OSError, ValueError):
        return None


def hwmon_inputs(names):
    """temp*_input files of the NVMe controllers behind the named namespaces."""
    inputs, limits = {}, {}
    for name in names:
        m = re.fullmatch(r"(nvme\d+)(?:c\d+)?n\d+", name)
        if not m:
            continue
        for mon in sorted(Path(f"/sys/class/nvme/{m[1]}").glob("hwmon*")) + \
                sorted(Path(f"/sys/class/nvme/{m[1]}/device").glob("hwmon/hwmon*")):
            for temp in sorted(mon.glob("temp*_input")):
                key = f"{m[1]}:{temp.name[:-6]}"
                inputs[key] = temp
                for lim in ("max", "crit"):
                    try:
                        limits[f"{key}_{lim}_mC"] = int(Path(str(temp)[:-6] + "_" + lim).read_text())
                    except (OSError, ValueError):
                        pass
    return inputs, limits


def read_temps(inputs):
    out = {}
    for key, path in inputs.items():
        try:
            out[key] = int(path.read_text())
        except (OSError, ValueError):
            out[key] = None
    return out


def sample(args):
    names = [d for d in args.devices.split(",") if d]
    stop = {"flag": False}

    def handler(signum, frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    inputs, limits = hwmon_inputs(names)
    interval_ns = args.interval_ms * 1_000_000
    with open(args.out, "x", buffering=1) as out:
        out.write(json.dumps({"kind": "header", "schema": SCHEMA, "devices": names,
                              "interval_ms": args.interval_ms, "pid": args.pid,
                              "hwmon_sensors": sorted(inputs), "hwmon_limits_mC": limits,
                              "start_unix_ns": time.time_ns()}) + "\n")
        start = time.monotonic_ns()
        seq = 0
        end = start + int(args.duration_s * 1e9) if args.duration_s else None
        while not stop["flag"]:
            deadline = start + seq * interval_ns
            now = time.monotonic_ns()
            if deadline > now:
                time.sleep((deadline - now) / 1e9)
                if stop["flag"]:
                    break
            t = time.monotonic_ns()
            if end is not None and t > end:
                break
            disks = diskstats(names)
            out.write(json.dumps({"kind": "tick", "seq": seq, "mono_ns": t, "late_ns": max(0, t - deadline),
                                  "unix_ns": time.time_ns(), "diskstats": disks,
                                  "missing_devices": [n for n in names if n not in disks],
                                  "meminfo": meminfo(), "cpu": cpu_jiffies(), "proc_io": proc_io(args.pid),
                                  "nvme_temp_mC": read_temps(inputs)}) + "\n")
            seq += 1
    return 0


def validate(path, names, max_gap_ms):
    problems = []
    lines = Path(path).read_text().splitlines()
    rows = [json.loads(l) for l in lines if l.strip()]
    if not rows or rows[0].get("kind") != "header" or rows[0].get("schema") != SCHEMA:
        return ["no m1-host-sampler-v1 header"]
    ticks = [r for r in rows[1:] if r.get("kind") == "tick"]
    if not ticks:
        problems.append("no ticks")
    prev = None
    for t in ticks:
        expected = 0 if prev is None else prev["seq"] + 1
        if t["seq"] != expected:
            problems.append(f"sequence gap: seq {t['seq']} follows {expected - 1}")
        missing = [n for n in names if n not in t["diskstats"]]
        if missing:
            problems.append(f"tick {t['seq']}: missing devices {missing}")
        if prev is not None:
            dt = t["mono_ns"] - prev["mono_ns"]
            if dt <= 0:
                problems.append(f"tick {t['seq']}: monotonic time did not advance")
            elif dt > max_gap_ms * 1_000_000:
                problems.append(f"tick {t['seq']}: gap {dt / 1e6:.1f} ms > {max_gap_ms} ms")
        prev = t
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--devices", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--interval-ms", type=int, default=250)
    s.add_argument("--pid", type=int)
    s.add_argument("--duration-s", type=float)
    v = sub.add_parser("validate")
    v.add_argument("file")
    v.add_argument("--devices", required=True)
    v.add_argument("--max-gap-ms", type=float, default=500)
    args = ap.parse_args(argv)
    if args.cmd == "sample":
        return sample(args)
    problems = validate(args.file, [d for d in args.devices.split(",") if d], args.max_gap_ms)
    ticks = sum(1 for l in Path(args.file).read_text().splitlines() if '"kind": "tick"' in l)
    print(f"M1-HOST-SAMPLER validate={'PASS' if not problems else 'FAIL'} ticks={ticks} problems={len(problems)}")
    for p in problems[:50]:
        print(f"  {p}")
    return 0 if not problems else 3


if __name__ == "__main__":
    sys.exit(main())
