#!/usr/bin/env python3
"""server_boot.py OUT.jsonl N GAP_MS BIN_DIR[,BIN_DIR...]: memra#1316's boot-only loop of the real
server on a GB10 host, under the caller's lease and with the serving's environment already set
(serve_env.sh). N boots, GAP_MS apart, the binaries taken in turn (arms). Each boot starts
`BIN_DIR/memra-server` and reads its output until the worker logs `Engine ready` (the boot is
past `Engine::new`: the process is killed there, before it reads the weights) or the process
ends by itself (a FATAL). One row per boot: the arm, `ready` or `fatal`, the milliseconds from
the start to that line, the FATAL and `[ctx-retry]` lines, and MemFree, MemAvailable, Cached and
the Normal zone's free blocks of order 9 and up read right before the start. Standard library
only; starts and stops only its own server."""
import json
import os
import signal
import subprocess
import sys
import time

out, n, gap_ms, bins = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
# Optional: ROUND tags every row of this call; ARM_MB and ARM_WAIT_S make the first boot wait (at
# most ARM_WAIT_S seconds) until MemFree is under ARM_MB, so a round starts at the moment the
# readers have used up what an exiting process freed. With ARM_SKIP=1 a wait that times out
# writes one `skipped` row and exits 3 without booting.
ROUND = os.environ.get("ROUND")
ARM_MB = int(os.environ.get("ARM_MB", "0"))
ARM_WAIT_S = float(os.environ.get("ARM_WAIT_S", "60"))


def mem_free_mb():
    with open("/proc/meminfo") as m:
        for line in m:
            if line.startswith("MemFree:"):
                return int(line.split()[1]) // 1024
    return -1


def host():
    row = {}
    with open("/proc/meminfo") as m:
        for line in m:
            k, v = line.split(":", 1)
            if k in ("MemFree", "MemAvailable", "Cached"):
                row[k + "_mb"] = int(v.split()[0]) // 1024
    with open("/proc/buddyinfo") as b:
        for line in b:
            p = line.split()
            if p[3] == "Normal":
                row["hi"] = [int(x) for x in p[13:]]
    return row


counts = {}
armed = None
if ARM_MB:
    t_arm = time.monotonic()
    armed = "timeout"
    while time.monotonic() - t_arm < ARM_WAIT_S:
        if mem_free_mb() < ARM_MB:
            armed = "threshold"
            break
        time.sleep(0.02)
    armed = {"mb": ARM_MB, "fired": armed, "waited_s": round(time.monotonic() - t_arm, 2)}
    if armed["fired"] == "timeout" and os.environ.get("ARM_SKIP") == "1":
        # MemFree did not come back under the threshold: the kernel is holding more memory free
        # than the state this round is for. No boots; the caller starts a new episode.
        with open(out, "a") as f:
            f.write(json.dumps({"round": int(ROUND or 0), "skipped": "MemFree stayed over the threshold",
                                "t": round(time.time(), 3), "host": host(), "armed": armed}) + "\n")
        sys.exit(3)
with open(out, "a") as f:
    for i in range(n):
        bin_dir = bins[i % len(bins)]
        arm = os.path.basename(bin_dir.rstrip("/"))
        row = {"boot": i, "arm": arm, "t": round(time.time(), 3), "host": host()}
        if ROUND is not None:
            row["round"] = int(ROUND)
        if armed and i == 0:
            row["armed"] = armed
        t0 = time.monotonic()
        p = subprocess.Popen([os.path.join(bin_dir, "memra-server")], stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        outcome, keep = "fatal", []
        for raw in p.stdout:
            line = raw.decode(errors="replace").rstrip()
            if "[ctx-retry]" in line or "FATAL" in line:
                keep.append(f"{(time.monotonic() - t0) * 1e3:.0f}ms {line}")
            if "Engine ready" in line:
                outcome = "ready"
                break
            if "FATAL" in line:
                break
        row["ms"] = round((time.monotonic() - t0) * 1e3, 1)
        # The server runs in its own process group, so the kill also ends the kernel-log tail and
        # the nvidia-smi canary it started (a failed boot may leave them behind as well).
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.stdout.close()
        row["exit"] = p.wait()
        row["outcome"] = outcome
        row["lines"] = keep
        row["first_try_failed"] = any("[ctx-retry]" in x for x in keep)
        f.write(json.dumps(row) + "\n")
        f.flush()
        key = (arm, outcome, row["first_try_failed"])
        counts[key] = counts.get(key, 0) + 1
        if outcome != "ready" or row["first_try_failed"]:
            print(f"{'round ' + ROUND + ' ' if ROUND else ''}boot {i} arm {arm}: {outcome} {keep[-1] if keep else ''}"[:400], flush=True)
        time.sleep(gap_ms / 1e3)
if ROUND is None:
    for (arm, outcome, ff), c in sorted(counts.items()):
        print(f"arm {arm}: {outcome}{' after a failed first try' if ff else ''}: {c}", flush=True)
