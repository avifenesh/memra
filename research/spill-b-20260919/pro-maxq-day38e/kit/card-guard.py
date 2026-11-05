#!/usr/bin/env python3
"""250 ms telemetry of ONE card, plus the owner's Max-Q thermal guard.

usage: card-guard.py --card N --out telemetry.csv --stop-file THERMAL-STOP.txt --pid-file boot.pid

Samples `nvidia-smi -i N ... -lms 250` (that card only) and writes each line prefixed with epoch ms, the same
stamp the server log carries. The guard (owner rule for the <box> Max-Q cards): a high temperature alone is
normal; stop at >= 92 C with fan >= 95 % on 3 consecutive samples 10 s apart, or at >= 95 C at once. On a trip
it writes the stop file and sends SIGTERM to the pid in --pid-file (the boot's `timeout`, which forwards the
signal to the boot's process group), then keeps recording until it is terminated itself.
"""
import argparse, os, signal, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--stop-file", required=True)
ap.add_argument("--pid-file", required=True)
ap.add_argument("--interval-ms", type=int, default=250)
ap.add_argument("--hot-c", type=float, default=92.0)
ap.add_argument("--fan-pct", type=float, default=95.0)
ap.add_argument("--hard-c", type=float, default=95.0)
ap.add_argument("--every-s", type=float, default=10.0)
ap.add_argument("--consecutive", type=int, default=3)
a = ap.parse_args()

FIELDS = ("timestamp,index,temperature.gpu,fan.speed,power.draw,power.limit,enforced.power.limit,clocks.sm,"
          "clocks.mem,utilization.gpu,memory.used,pstate,clocks_event_reasons.active")
proc = None
stopping = False


def on_term(_sig, _frm):
    global stopping
    stopping = True
    if proc and proc.poll() is None:
        proc.terminate()


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)


def num(s):
    try:
        return float(s.strip())
    except ValueError:
        return None


def trip(reason, recent):
    with open(a.stop_file, "a") as f:
        f.write(f"THERMAL-STOP {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} card={a.card} {reason}\n")
        for line in recent:
            f.write(f"  {line}\n")
    try:
        pid = int(open(a.pid_file).read().strip())
        os.kill(pid, signal.SIGTERM)
        with open(a.stop_file, "a") as f:
            f.write(f"  sent SIGTERM to boot pid {pid}\n")
    except (OSError, ValueError) as e:
        with open(a.stop_file, "a") as f:
            f.write(f"  no boot to stop ({e!r})\n")


hot_run, last_eval, tripped, recent, restarts = 0, 0.0, False, [], 0
with open(a.out, "a", buffering=1) as out:
    out.write(f"epoch_ms,{FIELDS}\n")
    while not stopping and restarts < 20:
        proc = subprocess.Popen(["nvidia-smi", "-i", a.card, f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits",
                                 f"-lms", str(a.interval_ms)], stdout=subprocess.PIPE, text=True)
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            out.write(f"{int(time.time() * 1000)},{line}\n")
            recent = (recent + [line])[-8:]
            parts = line.split(",")
            if tripped or len(parts) < 4:
                continue
            temp, fan = num(parts[2]), num(parts[3])
            now = time.monotonic()
            if temp is not None and temp >= a.hard_c:
                tripped = True
                trip(f"temperature {temp} C >= {a.hard_c} C", recent)
                continue
            if now - last_eval >= a.every_s:
                last_eval = now
                hot = temp is not None and fan is not None and temp >= a.hot_c and fan >= a.fan_pct
                hot_run = hot_run + 1 if hot else 0
                if hot_run >= a.consecutive:
                    tripped = True
                    trip(f"temperature >= {a.hot_c} C with fan >= {a.fan_pct} % on {hot_run} samples {a.every_s:.0f} s apart",
                         recent)
        proc.wait()
        if not stopping:
            restarts += 1
            out.write(f"{int(time.time() * 1000)},# nvidia-smi exited rc={proc.returncode}; restart {restarts}\n")
            time.sleep(1)
sys.exit(0)
