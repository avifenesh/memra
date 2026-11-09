#!/usr/bin/env python3
"""Does PR #1130's acknowledgement land before the vfork child has stopped?

The runner acknowledges a single-threaded leader parked in vfork (D, SIGSTOP in ShdPnd)
without looking at the child. The repeated-cycle test, the stress test and the deterministic
test then assert right away (not under a wait) that the unexeced child reads T. This probe
starves the child (leader and child at nice 19 on one CPU with two nice-0 hogs), emulates the
runner's tick (5 ms sleep, then the PR's check) and records the child's state at the moment
of acknowledgement, which is what those assertions read.

usage: starve.py LEADER_BINARY OUT_JSONL CPU TRIALS
"""
import json
import os
import signal
import subprocess
import sys
import time

SIGSTOP_BIT = 1 << (signal.SIGSTOP - 1)
PF_FORKNOEXEC = 0x40


def stat_fields(path):
    try:
        s = open(path).read()
    except OSError:
        return None
    rest = s[s.rindex(")") + 1:].split()
    return rest[0], int(rest[1]), int(rest[6])


def thread_states(pid):
    out = []
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        f = stat_fields(f"/proc/{pid}/task/{tid}/stat")
        if f:
            out.append(f[0])
    return out


def shdpnd(pid):
    for line in open(f"/proc/{pid}/status"):
        if line.startswith("ShdPnd:"):
            return int(line.split()[1], 16)
    return None


def unexeced_children(pid):
    kids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            f = stat_fields(f"/proc/{name}/stat")
            if f and f[1] == pid and f[2] & PF_FORKNOEXEC:
                kids.append(f[0])
    return kids


def pr_ack(pid):
    r = os.waitid(os.P_PID, pid, os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if r is not None and r.si_code == os.CLD_STOPPED:
        return True
    if r is not None and r.si_pid != 0:
        return False
    live, stopped = 0, False
    for s in thread_states(pid):
        if s in "ZX":
            continue
        if s in "Tt":
            live, stopped = live + 1, True
        elif s in "DI":
            live += 1
        else:
            return False
    if live == 0:
        return False
    if stopped:
        return True
    return bool(shdpnd(pid) & SIGSTOP_BIT)


def main():
    leader, out, cpu, trials = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    work = os.path.join(os.path.dirname(out), "work-starve")
    os.makedirs(work, exist_ok=True)
    hogs = [subprocess.Popen(["taskset", "-c", cpu, "sh", "-c", "while :; do :; done"])
            for _ in range(2)]
    rows = []
    try:
        for trial in range(trials):
            fifo = os.path.join(work, "fifo")
            if os.path.exists(fifo):
                os.unlink(fifo)
            os.mkfifo(fifo, 0o600)
            proc = subprocess.Popen(["taskset", "-c", cpu, "nice", "-n", "19", leader,
                                     "single", fifo, os.path.join(work, "ctr")],
                                    process_group=0)
            pid = proc.pid
            t0 = time.monotonic()
            while not (thread_states(pid) == ["D"] and "S" in unexeced_children(pid)):
                if time.monotonic() - t0 > 10:
                    raise SystemExit("setup timed out")
                time.sleep(0.002)
            time.sleep(0.02)
            os.killpg(pid, signal.SIGSTOP)
            ts = time.monotonic()
            ticks = 0
            while True:
                time.sleep(0.005)  # the test config's poll_ms
                ticks += 1
                if pr_ack(pid):
                    break
            row = {
                "trial": trial,
                "ack_ms": round((time.monotonic() - ts) * 1000, 1),
                "ticks": ticks,
                "leader_threads_at_ack": thread_states(pid),
                "unexeced_children_at_ack": unexeced_children(pid),
            }
            time.sleep(0.2)
            row["unexeced_children_after_200ms"] = unexeced_children(pid)
            row["test_assert_would_pass_at_ack"] = (
                all(s == "T" for s in row["leader_threads_at_ack"])
                or "T" in row["unexeced_children_at_ack"])
            rows.append(row)
            os.killpg(pid, signal.SIGKILL)
            proc.wait(timeout=10)
    finally:
        for h in hogs:
            h.kill()
            h.wait()
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    bad = [r for r in rows if not r["test_assert_would_pass_at_ack"]]
    print(f"trials={len(rows)} acknowledged_before_child_stopped={len(bad)}")
    for r in bad[:5]:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
