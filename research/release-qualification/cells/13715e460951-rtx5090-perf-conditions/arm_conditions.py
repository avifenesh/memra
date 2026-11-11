"""Per-arm card conditions for a timed perf cell: joins a 1 Hz nvidia-smi sample log with the perf run logs'
end times (file mtimes). usage: arm_conditions.py THROTTLE_CSV PERF_DIR START_ISO OUT_TSV"""
import csv
import datetime as dt
import sys
from pathlib import Path

throttle, perf, start, out = sys.argv[1:5]
rows = list(csv.reader(open(throttle), skipinitialspace=True))
head = [h.split(" [")[0] for h in rows[0]]
samples = []
for r in rows[1:]:
    if len(r) != len(head):
        continue
    d = dict(zip(head, r))
    d["t"] = dt.datetime.strptime(d["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
    samples.append(d)
flags = [h for h in head if h.startswith("clocks_event_reasons.") and "thermal" in h or h.endswith(".hw_slowdown")]
counter = "clocks_event_reasons_counters.sw_thermal_slowdown"
logs = sorted(p for p in Path(perf).glob("*.log"))
edge = dt.datetime.fromisoformat(start).replace(tzinfo=None)
lines = ["run\tended\tsamples\tthermal_or_hw_slowdown_active\tsw_thermal_counter_delta_us\ttemp_c_min\ttemp_c_max\tpower_w_max"]
for log in logs:
    end = dt.datetime.fromtimestamp(log.stat().st_mtime)
    window = [s for s in samples if edge <= s["t"] <= end]
    active = sum(1 for s in window if any(s[f] != "Not Active" for f in flags))
    temps = [int(s["temperature.gpu"]) for s in window]
    power = [float(s["power.draw"].split()[0]) for s in window]
    delta = "n/a"
    if counter in head and window:
        delta = str(int(window[-1][counter].split()[0]) - int(window[0][counter].split()[0]))
    lines.append(f"{log.name}\t{end.strftime('%H:%M:%S')}\t{len(window)}\t{active}\t{delta}\t"
                 f"{min(temps) if temps else ''}\t{max(temps) if temps else ''}\t{max(power) if power else ''}")
    edge = end
Path(out).write_text("\n".join(lines) + "\n")
print("\n".join(lines))
