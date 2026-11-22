#!/usr/bin/env python3
"""Telemetry and identity checks for a gpu-route batch directory.

judge.py BATCH_DIR [--timed-from HH:MM:SS.f --timed-to HH:MM:SS.f]
  - telemetry: smi.csv (250 ms). A loaded sample (power over 60 W) at P2 or with memory at 11001 MHz
    voids the batch's timed takes; samples with SW thermal slowdown (0x20) or SW power cap (0x4)
    are counted.
"""
import csv, json, sys

def telemetry(path, lo=None, hi=None):
    rows = list(csv.reader(open(path)))[1:]
    out = {"samples": 0, "loaded": 0, "loaded_p2_or_11001": 0, "loaded_sw_thermal_0x20": 0,
           "loaded_sw_power_cap_0x4": 0, "loaded_sm_mhz_min": None, "loaded_sm_mhz_max": None,
           "loaded_temp_c_range": None}
    temps, sms = [], []
    for r in rows:
        if len(r) < 8:
            continue
        t = r[0].strip().split(" ")[-1]
        if lo and (t < lo or t > hi):
            continue
        out["samples"] += 1
        power = float(r[4].split()[0])
        if power <= 60:
            continue
        out["loaded"] += 1
        reasons = int(r[6].strip(), 16)
        if r[1].strip() == "P2" or r[3].strip().startswith("11001"):
            out["loaded_p2_or_11001"] += 1
        if reasons & 0x20:
            out["loaded_sw_thermal_0x20"] += 1
        if reasons & 0x4:
            out["loaded_sw_power_cap_0x4"] += 1
        sms.append(int(r[2].split()[0]))
        temps.append(int(r[7].strip()))
    if sms:
        out["loaded_sm_mhz_min"], out["loaded_sm_mhz_max"] = min(sms), max(sms)
        out["loaded_temp_c_range"] = [min(temps), max(temps)]
    out["full_clocks"] = out["loaded"] > 0 and out["loaded_p2_or_11001"] == 0
    return out

if __name__ == "__main__":
    d = sys.argv[1]
    lo = hi = None
    if "--timed-from" in sys.argv:
        lo = sys.argv[sys.argv.index("--timed-from") + 1]
        hi = sys.argv[sys.argv.index("--timed-to") + 1]
    print(json.dumps(telemetry(f"{d}/smi.csv", lo, hi), indent=1))
