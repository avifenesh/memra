#!/usr/bin/env python3
"""Card telemetry, the Max-Q thermal guard and the card-pin check for the batch-oom-recover kit.

usage: guard.py watch --kit KIT --out OUT --card IDX --uuid GPU-...
       guard.py reap --kit KIT [--grace-s 20]

watch: runs `nvidia-smi -i IDX ... -lms 250` (the pinned card only) into OUT/telemetry.csv. Thermal rule
(the owner's Max-Q rule): stop at >= 95 C, or at >= 92 C with the fan >= 95 % on 3 consecutive samples.
Every 2 s it lists the box's compute apps (read-only) and records every process of this kit (its executable
is under KIT/bins/) with the card it holds, in OUT/pins.csv; a kit process on any other card is a pin
violation. Either stop writes OUT/THERMAL-STOP.txt or OUT/PIN-VIOLATION.txt, then OUT/STOP and
OUT/serve/STOP (day49d-run.sh stops between boots), and sends SIGTERM to this kit's servers only.
reap: SIGTERM to this kit's servers (executable under KIT/bins/ AND cwd == KIT), SIGKILL after the grace.
Never a signal to a process this kit did not start.
"""
import argparse
import os
import signal
import subprocess
import sys
import threading
import time

FIELDS = ("timestamp,index,temperature.gpu,fan.speed,power.draw,power.limit,clocks.sm,clocks.mem,"
          "utilization.gpu,utilization.memory,memory.used,clocks_event_reasons.active,pstate")
HOT_C, HOT_FAN, HOT_N, CRIT_C = 92.0, 95.0, 3, 95.0


def ours(kit, pid):
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return False
    return exe.startswith(os.path.join(kit, "bins") + "/") and cwd == kit


def kit_pids(kit):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit() and int(d) != os.getpid() and ours(kit, int(d)):
            out.append(int(d))
    return out


def reap(kit, grace_s):
    pids = kit_pids(kit)
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    end = time.time() + grace_s
    while time.time() < end and any(ours(kit, p) for p in pids):
        time.sleep(0.5)
    left = [p for p in pids if ours(kit, p)]
    for p in left:
        try:
            os.kill(p, signal.SIGKILL)
        except OSError:
            pass
    return pids, left


def num(s):
    try:
        return float(s.strip().split()[0])
    except (ValueError, IndexError):
        return None


def stop(kit, out, name, text):
    with open(os.path.join(out, name), "a") as f:
        f.write(text.rstrip() + "\n")
    for p in (os.path.join(out, "STOP"), os.path.join(out, "serve", "STOP")):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a") as f:
            f.write(f"{name}: {text.splitlines()[0]}\n")
    pids, left = reap(kit, 20)
    with open(os.path.join(out, name), "a") as f:
        f.write(f"sigterm={pids} sigkill={left}\n")


def pins(kit, out, uuid, done):
    path = os.path.join(out, "pins.csv")
    with open(path, "a") as f:
        f.write("t_unix,pid,gpu_uuid,used_mib,on_card\n")
    flagged = False
    while not done.is_set():
        # After any stop (thermal, pin, or the runner's deadline), no kit server may keep or take the card.
        if os.path.exists(os.path.join(out, "STOP")):
            pids, left = reap(kit, 10)
            if pids:
                with open(os.path.join(out, "STOP"), "a") as f:
                    f.write(f"reaped after stop: sigterm={pids} sigkill={left}\n")
        try:
            r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
            rows = [l.split(",") for l in r.stdout.splitlines() if l.strip()]
        except (OSError, subprocess.TimeoutExpired):
            rows = []
        now = time.time()
        for row in rows:
            if len(row) < 3 or not row[0].strip().isdigit():
                continue
            pid, g, used = int(row[0]), row[1].strip(), row[2].strip()
            if not ours(kit, pid):
                continue
            on = g == uuid
            with open(path, "a") as f:
                f.write(f"{now:.3f},{pid},{g},{used},{int(on)}\n")
            if not on and not flagged:
                flagged = True
                stop(kit, out, "PIN-VIOLATION.txt", f"kit pid {pid} holds {g}, not the pinned card {uuid}")
        done.wait(2.0)


def watch(a):
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())
    t = threading.Thread(target=pins, args=(a.kit, a.out, a.uuid, done), daemon=True)
    t.start()
    tel = open(os.path.join(a.out, "telemetry.csv"), "a")
    tel.write(FIELDS + "\n")
    tel.flush()
    p = subprocess.Popen(["nvidia-smi", "-i", str(a.card), f"--query-gpu={FIELDS}", "--format=csv,noheader",
                          "-lms", "250"], stdout=subprocess.PIPE, text=True)
    streak, fired = 0, False
    try:
        while not done.is_set():
            line = p.stdout.readline()
            if not line:
                break
            tel.write(line)
            tel.flush()
            f = line.split(",")
            if len(f) < 4 or fired:
                continue
            temp, fan = num(f[2]), num(f[3])
            if temp is None:
                continue
            hot = fan is not None and temp >= HOT_C and fan >= HOT_FAN
            streak = streak + 1 if hot else 0
            if temp >= CRIT_C or streak >= HOT_N:
                fired = True
                stop(a.kit, a.out, "THERMAL-STOP.txt",
                     f"card {a.card} temp={temp} C fan={fan} % streak={streak}: the Max-Q thermal rule "
                     f"(>= {CRIT_C} C, or >= {HOT_C} C with fan >= {HOT_FAN} % x{HOT_N})\n{line}")
    finally:
        done.set()
        p.terminate()
        try:
            p.wait(5)
        except subprocess.TimeoutExpired:
            p.kill()
        tel.close()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--kit", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--card", required=True)
    w.add_argument("--uuid", required=True)
    r = sub.add_parser("reap")
    r.add_argument("--kit", required=True)
    r.add_argument("--grace-s", type=float, default=20)
    a = ap.parse_args()
    a.kit = os.path.realpath(a.kit)
    if a.cmd == "watch":
        watch(a)
    else:
        pids, left = reap(a.kit, a.grace_s)
        print(f"reap: sigterm={pids} sigkill={left}")


if __name__ == "__main__":
    sys.exit(main())
