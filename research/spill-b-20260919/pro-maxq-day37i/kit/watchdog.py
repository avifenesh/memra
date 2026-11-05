#!/usr/bin/env python3
"""Window deadline and Max-Q thermal guard for one kit run; also the kill step of the runner's cleanup.

Stops the run (every process whose environment carries the run's marker, KIT_RUN_ID=<id>, whatever session or
process group it moved to) when:
  - the window deadline passes (WINDOW-DEADLINE), or
  - the card reads >= 92 C with fan >= 95 % on 3 consecutive 10 s samples, or >= 95 C on any sample (THERMAL-STOP,
    the Max-Q guard rule of this box).
The reason goes to --stopfile and --log. SIGTERM first, SIGKILL after 20 s. Only this user's processes are read.

usage: watchdog.py --csv <telemetry csv> --deadline <epoch s> --marker KIT_RUN_ID=<id> --log <chain.log> --stopfile <f>
       watchdog.py --kill-only --marker KIT_RUN_ID=<id> --log <chain.log>
"""
import argparse, csv, os, signal, sys, time


def marked(marker):
    want = marker.encode() + b"\0"
    me = os.getpid()
    pids = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            if os.stat(f"/proc/{d}").st_uid != os.getuid():
                continue
            env = open(f"/proc/{d}/environ", "rb").read()
        except OSError:
            continue
        if (b"\0" + env).find(b"\0" + want) >= 0:
            pids.append(int(d))
    return pids


def kill_marked(marker, log):
    pids = marked(marker)
    if not pids:
        return 0
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    end = time.time() + 20
    while time.time() < end and marked(marker):
        time.sleep(0.5)
    left = marked(marker)
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(log, "a") as f:
        f.write(f"{stamp} watchdog: stopped {len(pids)} marked processes (SIGTERM), {len(left)} needed SIGKILL\n")
    return len(pids)


def num(s):
    s = (s or "").strip().split(" ")[0]
    try:
        return float(s)
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--marker", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--stopfile")
    ap.add_argument("--kill-only", action="store_true")
    a = ap.parse_args()
    if a.kill_only:
        kill_marked(a.marker, a.log)
        return 0
    hot_ticks = 0
    next_tick = time.time() + 10
    pos = 0
    header = None
    last = None
    reason = None
    while reason is None:
        time.sleep(2)
        now = time.time()
        if now >= a.deadline:
            reason = f"WINDOW-DEADLINE at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(a.deadline))}"
            break
        try:
            with open(a.csv) as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
        except OSError:
            continue
        for line in chunk.splitlines():
            row = next(csv.reader([line], skipinitialspace=True), [])
            if header is None:
                header = row
                continue
            if len(row) != len(header):
                continue
            rec = dict(zip(header, row))
            temp = num(next((v for k, v in rec.items() if k.split(" ")[0] == "temperature.gpu"), None))
            fan = num(next((v for k, v in rec.items() if k.split(" ")[0] == "fan.speed"), None))
            last = (temp, fan)
            if temp is not None and temp >= 95:
                reason = f"THERMAL-STOP temperature {temp:.0f} C >= 95 (fan {fan})"
                break
        if reason is None and now >= next_tick:
            next_tick += 10
            if last and last[0] is not None and last[1] is not None and last[0] >= 92 and last[1] >= 95:
                hot_ticks += 1
            else:
                hot_ticks = 0
            if hot_ticks >= 3:
                reason = f"THERMAL-STOP temperature {last[0]:.0f} C with fan {last[1]:.0f} % on 3 consecutive 10 s samples"
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(a.stopfile, "a") as f:
        f.write(f"{stamp} {reason}\n")
    with open(a.log, "a") as f:
        f.write(f"{stamp} watchdog: {reason}; stopping the run\n")
    kill_marked(a.marker, a.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
