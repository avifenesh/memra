#!/usr/bin/env python3
"""mem_holder.py TARGET_MB CAP_S: memra#1221's truly-short condition on a GB10 unit. Allocates and
touches anonymous memory in 256 MB chunks until /proc/meminfo MemAvailable is at or under
TARGET_MB, holds it, and exits on SIGTERM or after CAP_S seconds whatever happens (so a hung cell
cannot leave the unit short). It sets its own oom_score_adj to 1000 first, so if the kernel must
kill something it takes this process. Prints MemAvailable and the bytes held once a second.
Touches nothing but its own memory; no system setting is changed."""
import mmap
import signal
import sys
import time

target_kb = int(sys.argv[1]) * 1024
cap_s = float(sys.argv[2])
with open("/proc/self/oom_score_adj", "w") as f:
    f.write("1000")
stop = False


def on_term(n, f):
    global stop
    stop = True


signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)


def avail_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


CHUNK = 256 << 20
held = []
t0 = time.time()
last = 0.0
page = b"\1" * 4096
while not stop and time.time() - t0 < cap_s:
    a = avail_kb()
    if a > target_kb + CHUNK // 1024:
        m = mmap.mmap(-1, CHUNK)
        for off in range(0, CHUNK, 4096):
            m[off:off + 4096] = page
        held.append(m)
    elif a < target_kb - CHUNK // 1024 and held:
        held.pop().close()  # the server grew: give back a chunk rather than push the unit lower
    if time.time() - last >= 1.0:
        last = time.time()
        print(f"{last:.3f} holder MemAvailable {a // 1024} MB held {len(held) * CHUNK >> 20} MB", flush=True)
    if a <= target_kb + CHUNK // 1024:
        time.sleep(0.05)
for m in held:
    m.close()
print(f"{time.time():.3f} holder released ({'signal' if stop else 'cap'})", flush=True)
