#!/usr/bin/env python3
"""Process helpers for run.sh (Max-Q kit), standard library only.

  reap --run-id ID [--spare PID,PID]
      SIGTERM, then SIGKILL after 5 s, every process of this user whose environment carries
      MAXQ_RUN_ID=ID (run.sh exports it, so the collector, driver, servers and every sampler,
      including the ones started in their own session, inherit it), except the spared PIDs and
      this process.
  thermal-watch --csv FILE --run-id ID --spare PID,PID --out FILE
      Tails run.sh's 250 ms card CSV. The owner's Max-Q thermal rule: stop only when the card is
      hot AND its fan is near full, three samples in a row (temperature >= 92 C and fan >= 95 %).
      On a stop it writes FILE and reaps the run (the spared PIDs stay).
"""
import argparse
import os
from pathlib import Path
import signal
import sys
import time

HOT_C = 92.0
FAN_PCT = 95.0
IN_A_ROW = 3


def tagged(run_id, spare):
    want = b"MAXQ_RUN_ID=" + run_id.encode()
    me = os.getpid()
    found = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        pid = int(p.name)
        if pid == me or pid in spare:
            continue
        try:
            env = (p / "environ").read_bytes()
            state = (p / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if state != "Z" and want in env.split(b"\0"):
            found.append(pid)
    return found


def reap(run_id, spare):
    victims = tagged(run_id, spare)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and tagged(run_id, spare):
        time.sleep(0.2)
    left = tagged(run_id, spare)
    for pid in left:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(f"kit-procs reap: run {run_id}: SIGTERM {len(victims)}, SIGKILL {len(left)}", flush=True)


def number(text):
    try:
        return float(text.strip().split(" ")[0])
    except ValueError:
        return None


def thermal_watch(csv, run_id, spare, out):
    path = Path(csv)
    while not path.exists():
        time.sleep(1)
    with path.open(errors="replace") as f:
        header = None
        streak = 0
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            cells = [c.strip() for c in line.rstrip("\n").split(",")]
            if header is None:
                header = cells
                temp = next(i for i, h in enumerate(header) if h.startswith("temperature.gpu"))
                fan = next(i for i, h in enumerate(header) if h.startswith("fan.speed"))
                continue
            if len(cells) != len(header):
                continue
            t, s = number(cells[temp]), number(cells[fan])
            streak = streak + 1 if t is not None and s is not None and t >= HOT_C and s >= FAN_PCT else 0
            if streak >= IN_A_ROW:
                Path(out).write_text(f"THERMAL STOP at {cells[0]}: {IN_A_ROW} samples in a row at "
                                     f"temperature >= {HOT_C} C and fan >= {FAN_PCT} % (last {t} C, {s} %)\n")
                print(f"kit-procs thermal-watch: stop ({t} C, fan {s} %)", flush=True)
                reap(run_id, spare)
                return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reap")
    r.add_argument("--run-id", required=True)
    r.add_argument("--spare", default="")
    t = sub.add_parser("thermal-watch")
    for name in ("--csv", "--run-id", "--out"):
        t.add_argument(name, required=True)
    t.add_argument("--spare", default="")
    a = ap.parse_args()
    spare = {int(x) for x in a.spare.split(",") if x.strip()}
    if a.cmd == "reap":
        reap(a.run_id, spare)
        return 0
    return thermal_watch(a.csv, a.run_id, spare, a.out)


if __name__ == "__main__":
    sys.exit(main())
