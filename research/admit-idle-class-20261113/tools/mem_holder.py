#!/usr/bin/env python3
"""mem_holder.py TARGET_MB CAP_S [static]: memra#1221's truly-short condition on a GB10 unit. Allocates and
touches anonymous memory in 256 MB chunks until /proc/meminfo MemAvailable is at or under
TARGET_MB, holds it, and exits on SIGTERM or after CAP_S seconds whatever happens (so a hung cell
cannot leave the unit short). It sets its own oom_score_adj to 1000 first, so if the kernel must
kill something it takes this process. Prints MemAvailable and the bytes held once a second.
Touches nothing but its own memory; no system setting is changed.

By default it regulates: when MemAvailable falls a chunk under the target (the server grew) it
gives a chunk back, so the unit stays at the target whatever the server allocates. With `static`
(memra#1304 review) it stops for good once it first reaches the target: what it holds is then a
fixed amount, as another process's memory is, and the server's own growth comes out of the free
memory that is left."""
import mmap
import signal
import sys
import time

target_kb = int(sys.argv[1]) * 1024
cap_s = float(sys.argv[2])
static = len(sys.argv) > 3 and sys.argv[3] == "static"
frozen = False
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
    if frozen:
        pass
    elif a > target_kb + CHUNK // 1024:
        m = mmap.mmap(-1, CHUNK)
        for off in range(0, CHUNK, 4096):
            m[off:off + 4096] = page
        held.append(m)
    elif static:
        frozen = True  # at the target: hold exactly this from here on
    elif a < target_kb - CHUNK // 1024 and held:
        held.pop().close()  # the server grew: give back a chunk rather than push the unit lower
    if time.time() - last >= 1.0:
        last = time.time()
        print(f"{last:.3f} holder MemAvailable {a // 1024} MB held {len(held) * CHUNK >> 20} MB", flush=True)
    if frozen or a <= target_kb + CHUNK // 1024:
        time.sleep(0.05)
for m in held:
    m.close()
print(f"{time.time():.3f} holder released ({'signal' if stop else 'cap'})", flush=True)
