#!/usr/bin/env python3
"""Validate the ABBAx3 log and correlate prefill intervals with GPU metadata."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics


def number(line, name, cast=float):
    found = re.search(r"\b" + re.escape(name) + r"=([^\s]+)", line)
    if not found:
        raise ValueError(f"missing {name}")
    return cast(found.group(1))


def spread(values):
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "max": max(values), "mean": statistics.mean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}


def audit(prefix):
    model = Path(str(prefix) + "-model.log").read_text()
    controller = Path(str(prefix) + "-controller.log").read_text()
    if not re.search(r"^EXIT .* status=0$", controller, re.M):
        raise ValueError("controller has no successful terminal receipt")
    if "PASS matrix/EP chunk identity; balanced comparison complete, no serving admission" not in model:
        raise ValueError("missing balanced completion")
    if model.count("\nEXACT ordinal=") != 12:
        raise ValueError("not all twelve exactness checks completed")
    lines = [line for line in model.splitlines() if line.startswith("MEASURE ")]
    if len(lines) != 12:
        raise ValueError("expected twelve measurements")
    rows = []
    for index, line in enumerate(lines):
        row = {name: number(line, name, int) for name in
               ("ordinal", "width", "prompt", "start_unix_ms", "end_unix_ms", "ep_calls", "device_route_calls")}
        row["seconds"] = number(line, "prefill_seconds")
        row["signature"] = number(line, "signature", str)
        if row["ordinal"] != index or row["width"] != [32, 512, 512, 32][index % 4]:
            raise ValueError("ABBA order mismatch")
        if row["prompt"] != 9900 or row["ep_calls"] <= 0 or row["device_route_calls"] <= 0:
            raise ValueError("input/engagement mismatch")
        if not math.isfinite(row["seconds"]) or row["seconds"] <= 0 or row["end_unix_ms"] <= row["start_unix_ms"]:
            raise ValueError("invalid measured interval")
        if rows and row["start_unix_ms"] < rows[-1]["end_unix_ms"]:
            raise ValueError("overlapping measurement intervals")
        rows.append(row)
    if len({row["signature"] for row in rows}) != 1:
        raise ValueError("request-state/output signature changed")
    processes = Path(str(prefix) + "-processes.log").read_text()
    if "REFUSED" in processes:
        raise ValueError("process interference refusal")
    pids = set(re.findall(r"^(\d+),", processes, re.M))
    if len(pids) != 1:
        raise ValueError("expected one GPU process")
    telemetry = []
    with Path(str(prefix) + "-gpu.csv").open() as stream:
        for raw in csv.DictReader(stream, skipinitialspace=True):
            sample = {"ms": datetime.strptime(raw["timestamp"], "%Y/%m/%d %H:%M:%S.%f")
                      .replace(tzinfo=timezone.utc).timestamp() * 1000,
                      "gpu": int(raw["index"])}
            for key, label in [("sm_mhz", "clocks.current.sm [MHz]"),
                               ("memory_mhz", "clocks.current.memory [MHz]"),
                               ("power_w", "power.draw [W]"), ("limit_w", "power.limit [W]"),
                               ("temperature_c", "temperature.gpu")]:
                sample[key] = float(raw[label].split()[0])
            telemetry.append(sample)
    for row in rows:
        row["gpu_metadata"] = {}
        for gpu in (0, 1):
            samples = [s for s in telemetry if s["gpu"] == gpu and row["start_unix_ms"] <= s["ms"] <= row["end_unix_ms"]]
            if not samples:
                raise ValueError("missing phase-correlated telemetry")
            times = [row["start_unix_ms"]] + [s["ms"] for s in samples] + [row["end_unix_ms"]]
            row["gpu_metadata"][gpu] = {"samples": len(samples),
                "max_observation_gap_ms": max(b - a for a, b in zip(times, times[1:])),
                **{key: spread([s[key] for s in samples]) for key in
                   ("sm_mhz", "memory_mhz", "power_w", "limit_w", "temperature_c")}}
    arms = {width: spread([row["seconds"] for row in rows if row["width"] == width]) for width in (32, 512)}
    pairs = []
    for a, b in zip(rows[::2], rows[1::2]):
        baseline, candidate = (a, b) if a["width"] == 32 else (b, a)
        pairs.append(baseline["seconds"] / candidate["seconds"])
    return {"status": "BALANCED_ENGINE_PREFILL_NOT_SERVING_ADMISSION", "gpu_pid": next(iter(pids)),
            "seconds": arms, "median_input_tps": {w: 9900 / arms[w]["median"] for w in arms},
            "median_time_ratio_32_over_512": arms[32]["median"] / arms[512]["median"],
            "paired_time_ratios": spread(pairs), "rows": rows,
            "note": "Power and clocks are receipt metadata, not proof of a power-limited critical path."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt_prefix", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.receipt_prefix), indent=2, allow_nan=False))
