#!/usr/bin/env python3
"""Telemetry, thermal guard and deadline for the admit-w-release Max-Q kit (stdlib only).

usage: guard.py --card N --pgid PGID --out DIR --deadline-s S [--grace-s 60]

- Streams `nvidia-smi -i N ... -lms 250` (that card only) into DIR/telemetry.csv, restarting the stream if it dies.
- Every 10 s: appends that card's compute apps to DIR/compute-apps-card.csv, and applies the owner's Max-Q rule to the
  newest sample: stop at temperature >= 92 C with fan >= 95 % on 3 consecutive 10 s checks, or at >= 95 C.
- At the deadline or on a thermal stop: writes DIR/TIMEOUT.txt or DIR/THERMAL-STOP.txt, sends SIGTERM to the cell's
  process group PGID (the group run.sh started with setsid, nothing else), then SIGKILL after --grace-s.
- Exits when the group is gone or on SIGTERM.
"""
import argparse, os, signal, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("--pgid", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--deadline-s", type=float, required=True)
ap.add_argument("--grace-s", type=float, default=60.0)
ap.add_argument("--hot-c", type=float, default=92.0)
ap.add_argument("--fan-pct", type=float, default=95.0)
ap.add_argument("--hard-c", type=float, default=95.0)
ap.add_argument("--checks", type=int, default=3)
a = ap.parse_args()

FIELDS = ("timestamp,index,temperature.gpu,fan.speed,power.draw,power.limit,enforced.power.limit,clocks.sm,clocks.mem,"
          "utilization.gpu,memory.used,clocks_event_reasons.active")
tele_path = os.path.join(a.out, "telemetry.csv")
apps_path = os.path.join(a.out, "compute-apps-card.csv")
events_path = os.path.join(a.out, "guard.log")
latest = {"line": None}
stop = threading.Event()
start = time.time()


def event(msg):
    with open(events_path, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}\n")


def group_alive():
    try:
        os.killpg(a.pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def end_group(reason_file, text):
    with open(os.path.join(a.out, reason_file), "w") as f:
        f.write(text + "\n")
    event(f"{reason_file}: {text}; SIGTERM to process group {a.pgid}")
    try:
        os.killpg(a.pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    t0 = time.time()
    while time.time() - t0 < a.grace_s and group_alive():
        time.sleep(1)
    if group_alive():
        event(f"group {a.pgid} still alive after {a.grace_s:.0f} s; SIGKILL")
        try:
            os.killpg(a.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def stream():
    with open(tele_path, "a") as f:
        f.write(FIELDS + "\n")
    while not stop.is_set():
        p = subprocess.Popen(["nvidia-smi", "-i", a.card, f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                              "-lms", "250"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        with open(tele_path, "a") as f:
            for line in p.stdout:
                f.write(line)
                f.flush()
                latest["line"] = line.strip()
                if stop.is_set():
                    break
        p.terminate()
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
        if not stop.is_set():
            event(f"telemetry stream ended rc={p.returncode}; restarting")
            time.sleep(1)


def on_term(*_):
    stop.set()


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)
threading.Thread(target=stream, daemon=True).start()
event(f"guard on: card={a.card} pgid={a.pgid} deadline_s={a.deadline_s:.0f} rule=(>= {a.hot_c} C and fan >= {a.fan_pct} %"
      f" on {a.checks} consecutive 10 s checks) or >= {a.hard_c} C")
streak = 0
while not stop.is_set():
    stop.wait(10)
    if stop.is_set():
        break
    if not group_alive():
        event("cell group gone; guard exits")
        break
    try:
        apps = subprocess.run(["nvidia-smi", "-i", a.card, "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout
        with open(apps_path, "a") as f:
            stamp = int(time.time() * 1000)
            for l in apps.splitlines() or ["(none)"]:
                f.write(f"{stamp}, {l}\n")
    except (subprocess.TimeoutExpired, OSError) as e:
        event(f"compute-apps query failed: {e!r}")
    line = latest["line"]
    if line:
        parts = [x.strip() for x in line.split(",")]
        try:
            temp = float(parts[2])
        except (IndexError, ValueError):
            temp = None
        try:
            fan = float(parts[3])
        except (IndexError, ValueError):
            fan = None
        if temp is not None and temp >= a.hard_c:
            end_group("THERMAL-STOP.txt", f"temperature {temp} C >= {a.hard_c} C on card {a.card} (sample: {line})")
            break
        if temp is not None and fan is not None and temp >= a.hot_c and fan >= a.fan_pct:
            streak += 1
            event(f"hot with fans near full: {temp} C fan {fan} % (check {streak} of {a.checks})")
            if streak >= a.checks:
                end_group("THERMAL-STOP.txt", f"{temp} C with fan {fan} % on {a.checks} consecutive 10 s checks on card "
                          f"{a.card} (sample: {line})")
                break
        else:
            streak = 0
    if time.time() - start >= a.deadline_s:
        end_group("TIMEOUT.txt", f"total timeout {a.deadline_s:.0f} s reached")
        break
stop.set()
sys.exit(0)
