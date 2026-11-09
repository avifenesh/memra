#!/usr/bin/env python3
"""After the suggested tgkill nudge, does one group SIGCONT resume everything cleanly?

Leader: main-spawn-nocldstop (main thread parked in vfork, worker counting, SA_NOCLDSTOP).
Group SIGSTOP, tgkill SIGSTOP to the worker, then group SIGCONT; check the worker counts again
and no SIGSTOP is left pending anywhere; then release the FIFO and check the leader's main
thread returns from posix_spawn and the process exits 0.

usage: resume_after_tgkill.py LEADER_BINARY WORKDIR
"""
import ctypes
import os
import signal
import subprocess
import sys
import time

LIBC = ctypes.CDLL(None, use_errno=True)
BIT = 1 << (signal.SIGSTOP - 1)


def states(pid):
    out = {}
    for tid in os.listdir(f"/proc/{pid}/task"):
        s = open(f"/proc/{pid}/task/{tid}/stat").read()
        out[int(tid)] = s[s.rindex(")") + 1:].split()[0]
    return out


def pend(pid, tid, key):
    for line in open(f"/proc/{pid}/task/{tid}/status"):
        if line.startswith(key):
            return int(line.split()[1], 16)


def ctr(path):
    b = open(path, "rb").read(8)
    return int.from_bytes(b, "little") if len(b) == 8 else 0


leader, work = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
fifo, cpath = os.path.join(work, "fifo"), os.path.join(work, "ctr")
for p in (fifo, cpath):
    if os.path.exists(p):
        os.unlink(p)
os.mkfifo(fifo, 0o600)
proc = subprocess.Popen([leader, "main-spawn-nocldstop", fifo, cpath], process_group=0)
pid = proc.pid
t0 = time.monotonic()
while states(pid).get(pid) != "D" or len(states(pid)) != 2:
    if time.monotonic() - t0 > 5:
        sys.exit("setup timed out")
    time.sleep(0.002)
time.sleep(0.05)
worker = [t for t in states(pid) if t != pid][0]
os.killpg(pid, signal.SIGSTOP)
time.sleep(0.2)
print("after group STOP, 200 ms:", states(pid), "counter moving:", ctr(cpath) != (time.sleep(0.05) or ctr(cpath)))
LIBC.tgkill(pid, worker, signal.SIGSTOP)
time.sleep(0.2)
c1 = ctr(cpath)
time.sleep(0.2)
print("after tgkill:", states(pid), "counter frozen:", ctr(cpath) == c1)
os.killpg(pid, signal.SIGCONT)
time.sleep(0.2)
c2 = ctr(cpath)
time.sleep(0.2)
print("after group CONT:", states(pid), "counter moving:", ctr(cpath) > c2,
      "SIGSTOP pending shared/main/worker:",
      bool(pend(pid, pid, "ShdPnd:") & BIT), bool(pend(pid, pid, "SigPnd:") & BIT),
      bool(pend(pid, worker, "SigPnd:") & BIT))
fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
t1 = time.monotonic()
while states(pid).get(pid) == "D" and time.monotonic() - t1 < 5:
    time.sleep(0.005)
os.close(fd)
print("after FIFO writer: main thread", states(pid).get(pid), "(left the vfork wait)")
try:
    rc = proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    rc = "still running"
print("leader exit:", rc)
if rc == "still running":
    os.killpg(pid, signal.SIGKILL)
    proc.wait()
