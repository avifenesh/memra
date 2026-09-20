#!/usr/bin/env python3
"""Collector red arm: the native host-capacity refusal must arrive as the refusal token.

Since day ten the gate binaries print the final stderr line `REFUSED: <reason>` and
exit 2 themselves for a typed expert bank budget refusal. This wrapper keeps the entire
native stdout/stderr and exit status, passes exactly that native token through, and
turns anything else (including the pre-day-ten `Error: "..."` exit 1 shape this script
used to normalize) into a FAIL. No GPU work may invoke it outside tier-battery.py.
"""
import subprocess
import sys

REASON = "experts-via-tier host bank budget cannot hold one expert record"
TOKEN = "REFUSED: " + REASON


def verdict(code, output):
    lines = output.splitlines()
    native = bool(lines) and lines[-1].startswith(TOKEN)
    if code == 2 and native and not any("[expert-host-slru]" in line for line in lines):
        return 2, lines[-1]
    return 1, "FAIL: native host-capacity refusal token was not observed"


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
