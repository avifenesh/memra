#!/usr/bin/env python3
"""Per-visit GPU telemetry (M1-PREREG.md, telemetry amendment): recording only, never a gate.

`GpuSampler(path).start()` runs a 250 ms `nvidia-smi` sampler into `path` (CSV, no units);
`.stop()` ends it and returns `summarize(path)`. `summarize_collector(csv, t0_unix, t1_unix)`
gives the same summary, without reason bits, from a collector `command.gpu.csv` slice.
"""
import csv
import datetime
import os
from pathlib import Path
import signal
import statistics
import subprocess

FIELDS = ["timestamp", "clocks.sm", "clocks.max.sm", "power.draw", "enforced.power.limit",
          "temperature.gpu", "clocks_event_reasons.active"]
# nvidia-smi clocks_event_reasons bits.
REASONS = {0x1: "gpu_idle", 0x2: "applications_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
           0x10: "sync_boost", 0x20: "sw_thermal", 0x40: "hw_thermal", 0x80: "hw_power_brake",
           0x100: "display_clock"}


def _num(s):
    s = s.strip().split(" ")[0]
    try:
        return float(s)
    except ValueError:
        return None


class GpuSampler:
    def __init__(self, path, device="0"):
        self.path = Path(path)
        self.device = device
        self.proc = None

    def start(self):
        out = self.path.open("x")
        err = Path(str(self.path) + ".err").open("x")
        self.proc = subprocess.Popen(["nvidia-smi", "-i", self.device, "--query-gpu=" + ",".join(FIELDS),
                                      "--format=csv,noheader,nounits", "-lms", "250"],
                                     stdout=out, stderr=err, start_new_session=True)
        out.close()
        err.close()
        return self

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        return summarize(self.path)


def _summary(sm, smax, power, temp, reasons, n):
    out = {"samples": n}
    if sm:
        out.update(sm_clock_mhz_median=statistics.median(sm), sm_clock_mhz_min=min(sm))
    if smax:
        out["sm_clock_max_mhz"] = max(smax)
    if power:
        out.update(power_w_median=round(statistics.median(power), 2), power_w_max=max(power))
    if temp:
        out["temperature_c_max"] = max(temp)
    if reasons is not None and n:
        out["reason_share"] = {name: round(sum(1 for r in reasons if r & bit) / n, 3)
                               for bit, name in REASONS.items() if any(r & bit for r in reasons)}
    return out


def summarize(path):
    rows = [r for r in csv.reader(Path(path).read_text(errors="replace").splitlines()) if len(r) == len(FIELDS)]
    sm = [v for v in (_num(r[1]) for r in rows) if v is not None]
    smax = [v for v in (_num(r[2]) for r in rows) if v is not None]
    power = [v for v in (_num(r[3]) for r in rows) if v is not None]
    temp = [v for v in (_num(r[5]) for r in rows) if v is not None]
    reasons = []
    for r in rows:
        try:
            reasons.append(int(r[6].strip(), 16))
        except ValueError:
            pass
    return _summary(sm, smax, power, temp, reasons, len(rows))


def summarize_collector(path, t0_unix, t1_unix):
    """Slice a collector `command.gpu.csv` (local-time `YYYY/MM/DD HH:MM:SS.mmm` stamps)."""
    lines = Path(path).read_text(errors="replace").splitlines()
    head = [h.strip() for h in lines[0].split(",")]
    idx = {name: i for i, name in enumerate(head)}
    sm, power, temp, n = [], [], [], 0
    for line in lines[1:]:
        r = [c.strip() for c in line.split(",")]
        if len(r) != len(head):
            continue
        try:
            t = datetime.datetime.strptime(r[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        if not t0_unix <= t <= t1_unix:
            continue
        n += 1
        for dst, key in ((sm, "clocks.current.sm [MHz]"), (power, "power.draw [W]"), (temp, "temperature.gpu")):
            v = _num(r[idx[key]]) if key in idx else None
            if v is not None:
                dst.append(v)
    return _summary(sm, [], power, temp, None, n)
