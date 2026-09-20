#!/usr/bin/env python3
"""Collector command: normalize only the exact native host-capacity refusal to exit 2.

Keeps the entire native stdout/stderr and its exit status. No GPU work may invoke
this script outside tier-battery.py. An unrelated failure never becomes REFUSED.
"""
import subprocess
import sys

REASON = "experts-via-tier host bank budget cannot hold one expert record"


def verdict(code, output):
    lines = output.splitlines()
    if code == 1 and f'Error: "{REASON}"' in lines and not any("[expert-host-slru]" in line for line in lines):
        return 2, "REFUSED: " + REASON
    return 1, "FAIL: expected native host-capacity refusal was not observed"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: pressure-refusal.py <native command...>")
    native = subprocess.run(sys.argv[1:], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(native.stdout, end="", flush=True)
    print(f"native_exit_code={native.returncode}", flush=True)
    code, message = verdict(native.returncode, native.stdout)
    print(message, flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
