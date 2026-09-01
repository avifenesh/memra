#!/usr/bin/env python3
"""Sample both GPUs every 500 ms without adding a Python/NVML dependency."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time


STOP = False


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def number(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def sample() -> dict[str, object]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,temperature.gpu,power.draw,memory.used,clocks.sm,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gpus: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        gpus.append(
            {
                "index": int(fields[0]),
                "temperature_C": number(fields[1]),
                "power_W": number(fields[2]),
                "memory_MiB": number(fields[3]),
                "clock_MHz": number(fields[4]),
                "utilization_pct": number(fields[5]),
            }
        )
    return {"unix_ms": int(time.time() * 1000), "gpus": gpus}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: sample-nvml.py <output.jsonl>", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with open(sys.argv[1], "x", encoding="utf-8") as output:
        while not STOP:
            print(json.dumps(sample(), separators=(",", ":")), file=output, flush=True)
            deadline = time.monotonic() + 0.5
            while not STOP and time.monotonic() < deadline:
                time.sleep(min(0.05, deadline - time.monotonic()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
