#!/usr/bin/env python3
"""Runner-side driver for the PR #1130 review probe.

Mirrors the runner: the leader runs in its own process group, the driver sends the group
SIGSTOP, then polls PR #1130's acknowledgement (waitid WSTOPPED|WEXITED|WNOHANG|WNOWAIT, then
the /proc reading of classify_leader_threads / leader_stop_took_effect, ported line by line)
every 25 ms. It also records every leader thread's state, ShdPnd, and a counter the leader's
worker thread bumps every 1 ms (progress there means job code runs).

usage: driver.py LEADER_BINARY OUT_DIR MODE [MODE...]
"""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time

LIBC = ctypes.CDLL(None, use_errno=True)
PF_FORKNOEXEC = 0x40
SIGSTOP_BIT = 1 << (signal.SIGSTOP - 1)


def stat_fields(path):
    try:
        s = open(path).read()
    except OSError:
        return None
    rest = s[s.rindex(")") + 1:].split()
    return rest[0], int(rest[1]), int(rest[6])  # state, ppid, flags


def thread_states(pid):
    out = {}
    try:
        tids = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return out
    for tid in tids:
        f = stat_fields(f"/proc/{pid}/task/{tid}/stat")
        if f:
            out[int(tid)] = f[0]
    return out


def shdpnd(pid):
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("ShdPnd:"):
                return int(line.split()[1], 16)
    except OSError:
        return None
    return None


def unexeced_children(pid):
    kids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        f = stat_fields(f"/proc/{name}/stat")
        if f and f[1] == pid and f[2] & PF_FORKNOEXEC:
            kids.append((int(name), f[0]))
    return kids


def classify(states):
    """classify_leader_threads, darklane.rs:397-417."""
    live, stopped = 0, False
    for s in states:
        if s in "ZX":
            continue
        if s in "Tt":
            live += 1
            stopped = True
        elif s in "DI":
            live += 1
        else:
            return "Running"
    if live == 0:
        return "Running"
    return "Stopping" if stopped else "Blocked"


def pr_ack(pid):
    """child_stop_acknowledged + leader_stop_took_effect, darklane.rs:362-468."""
    r = os.waitid(os.P_PID, pid, os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if r is not None and r.si_code == os.CLD_STOPPED:
        return True, "waitid"
    if r is not None and r.si_pid != 0:
        return False, "exit"
    c = classify(list(thread_states(pid).values()))
    if c == "Running":
        return False, c
    if c == "Stopping":
        return True, c
    p = shdpnd(pid)
    return bool(p is not None and p & SIGSTOP_BIT), c


def counter(path):
    try:
        b = open(path, "rb").read(8)
    except OSError:
        return 0
    return int.from_bytes(b, "little") if len(b) == 8 else 0


def wait_until(what, f, timeout=5.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if f():
            return
        time.sleep(0.002)
    raise SystemExit(f"setup timed out: {what}")


def sample(pid, ctr, t0, arm):
    ack, why = pr_ack(pid)
    p = shdpnd(pid)
    return {
        "arm": arm,
        "t_ms": round((time.monotonic() - t0) * 1000),
        "pr_ack": ack,
        "pr_path": why,
        "threads": {("main" if t == pid else str(t)): s for t, s in sorted(thread_states(pid).items())},
        "shdpnd": None if p is None else f"{p:016x}",
        "sigstop_in_shdpnd": bool(p is not None and p & SIGSTOP_BIT),
        "unexeced_children": unexeced_children(pid),
        "worker_counter": counter(ctr),
    }


def run(leader, out_dir, mode):
    # MODE+tgkill: the leader runs MODE; right after the group SIGSTOP the driver also sends a
    # thread-directed SIGSTOP (tgkill) to every other live leader thread (the suggested fix).
    tgkill_now = mode.endswith("+tgkill")
    leader_mode = mode[: -len("+tgkill")] if tgkill_now else mode
    work = os.path.join(out_dir, f"work-{mode}")
    os.makedirs(work, exist_ok=True)
    fifo = os.path.join(work, "fifo")
    ctr = os.path.join(work, "ctr")
    for p in (fifo, ctr):
        if os.path.exists(p):
            os.unlink(p)
    os.mkfifo(fifo, 0o600)
    proc = subprocess.Popen([leader, leader_mode, fifo, ctr], process_group=0)
    pid = proc.pid
    rows = []
    try:
        def ready():
            th = thread_states(pid)
            kids = unexeced_children(pid)
            if not any(s == "S" for _, s in kids):
                return False
            if mode.startswith("two-block"):
                return len(th) == 2 and all(s == "D" for s in th.values())
            if mode == "thread-spawn":
                return "D" in th.values()
            return th.get(pid) == "D"

        wait_until("leader in vfork wait on the FIFO child", ready)
        t0 = time.monotonic()
        rows.append(dict(sample(pid, ctr, t0, "before-stop"), event="pre"))
        os.killpg(pid, signal.SIGSTOP)
        if tgkill_now:
            for tid, s in thread_states(pid).items():
                if tid != pid and s not in "ZX":
                    rc = LIBC.tgkill(pid, tid, signal.SIGSTOP)
                    rows.append({"arm": "tgkill", "event": "tgkill", "tid": tid, "rc": rc})
        first_ack = None
        last_key = None
        end = t0 + 3.0
        while time.monotonic() < end:
            row = sample(pid, ctr, t0, "group-sigstop")
            key = (row["pr_ack"], row["pr_path"], tuple(row["threads"].values()))
            if row["pr_ack"] and first_ack is None:
                first_ack = row
                rows.append(dict(row, event="first-ack"))
            elif key != last_key:
                rows.append(dict(row, event="change"))
            last_key = key
            time.sleep(0.025)
        rows.append(dict(sample(pid, ctr, t0, "group-sigstop"), event="end-3s",
                         first_ack_ms=None if first_ack is None else first_ack["t_ms"],
                         counter_at_first_ack=None if first_ack is None else first_ack["worker_counter"]))
        if mode.startswith("main-spawn") and not tgkill_now:
            # The suggested fix: a thread-directed SIGSTOP to each other live leader thread.
            for tid, s in thread_states(pid).items():
                if tid != pid and s not in "ZX":
                    rc = LIBC.tgkill(pid, tid, signal.SIGSTOP)
                    rows.append({"arm": "tgkill", "event": "tgkill", "tid": tid, "rc": rc})
            t1 = time.monotonic()
            c1 = None
            while time.monotonic() - t1 < 1.0:
                row = sample(pid, ctr, t0, "tgkill")
                if row["pr_ack"] and c1 is None:
                    c1 = row
                    rows.append(dict(row, event="first-ack"))
                time.sleep(0.025)
            rows.append(dict(sample(pid, ctr, t0, "tgkill"), event="end-tgkill"))
    finally:
        kids = [int(n) for n in os.listdir("/proc") if n.isdigit()
                and (stat_fields(f"/proc/{n}/stat") or ("", 0, 0))[1] == pid]
        os.killpg(pid, signal.SIGCONT)
        try:
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            os.close(fd)
        except OSError:
            pass
        time.sleep(0.05)
        os.killpg(pid, signal.SIGKILL)
        for k in kids:
            try:
                os.kill(k, signal.SIGKILL)
            except OSError:
                pass
        proc.wait()
    with open(os.path.join(out_dir, f"{mode}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


def main():
    leader, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    for mode in sys.argv[3:]:
        rows = run(leader, out_dir, mode)
        print(f"== {mode}")
        for r in rows:
            print(json.dumps(r))


if __name__ == "__main__":
    main()
