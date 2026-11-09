#!/usr/bin/env python3
"""Catch a SIGSTOPped process group whose leader never reports stopped.

Mirrors the memra background runner: the job is `SHELL -c CMD` in its own
process group, the yield is kill(-pgid, SIGSTOP), and the acknowledgement is
waitid(P_PID, leader, WSTOPPED|WEXITED|WNOHANG|WNOWAIT) reporting CLD_STOPPED.

Usage: vfork_stop_probe.py SHELL CMD ATTEMPTS MAX_HITS [HOLD_S]
HOLD_S (default 3) re-reads a missed group after that long, then re-stops it right
after SIGCONT; HOLD_S=0 only counts misses (one snapshot each) to measure the rate.
Prints one JSON line per missed acknowledgement and one summary line.
"""
import ctypes
import json
import os
import random
import signal
import subprocess
import sys
import time

PF_FORKNOEXEC = 0x40
ACK_WAIT_S = 0.5


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as e:
        return f"<{type(e).__name__}: {e.strerror}>"


def stat_fields(path):
    s = read(path)
    if s.startswith("<"):
        return None
    i = s.rfind(")")
    rest = s[i + 2:].split()
    return {
        "comm": s[s.find("(") + 1:i],
        "state": rest[0],
        "ppid": int(rest[1]),
        "pgrp": int(rest[2]),
        "flags": int(rest[6]),
    }


def group_members(pgid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            f = stat_fields(f"/proc/{d}/stat")
            if f and f["pgrp"] == pgid:
                out.append(int(d))
    return sorted(out)


libc = ctypes.CDLL(None, use_errno=True)
SYS_KCMP = {"x86_64": 312, "aarch64": 272}[os.uname().machine]
KCMP_VM = 1


def kcmp_vm(a, b):
    r = libc.syscall(SYS_KCMP, a, b, KCMP_VM, 0, 0)
    if r < 0:
        return f"errno {ctypes.get_errno()}"
    return {0: "same mm", 1: "different (lt)", 2: "different (gt)"}.get(r, r)


def snapshot(pgid):
    snap = {}
    for pid in group_members(pgid):
        f = stat_fields(f"/proc/{pid}/stat") or {}
        status = {}
        for line in read(f"/proc/{pid}/status").splitlines():
            k, _, v = line.partition(":")
            if k in ("State", "Threads", "SigPnd", "ShdPnd", "SigBlk"):
                status[k] = v.strip()
        tasks = {}
        try:
            tids = sorted(os.listdir(f"/proc/{pid}/task"), key=int)
        except OSError:
            tids = []
        for t in tids:
            tf = stat_fields(f"/proc/{pid}/task/{t}/stat") or {}
            tasks[t] = {
                "state": tf.get("state"),
                "wchan": read(f"/proc/{pid}/task/{t}/wchan"),
            }
        try:
            exe = os.readlink(f"/proc/{pid}/exe")
        except OSError as e:
            exe = f"<{e.strerror}>"
        snap[pid] = {
            "comm": f.get("comm"),
            "state": f.get("state"),
            "ppid": f.get("ppid"),
            "pgrp": f.get("pgrp"),
            "flags": hex(f.get("flags", 0)),
            "PF_FORKNOEXEC": bool(f.get("flags", 0) & PF_FORKNOEXEC),
            "exe": exe,
            "cmdline": read(f"/proc/{pid}/cmdline").replace("\0", " ").strip(),
            "wchan": read(f"/proc/{pid}/wchan"),
            "syscall": read(f"/proc/{pid}/syscall").strip(),
            "status": status,
            "tasks": tasks,
            "kcmp_vm_vs_leader": None if pid == pgid else kcmp_vm(pgid, pid),
        }
    return snap


def leader_reports_stopped(pid):
    r = os.waitid(os.P_PID, pid, os.WSTOPPED | os.WEXITED | os.WNOHANG | os.WNOWAIT)
    if r is None:
        return "nothing"
    return {os.CLD_STOPPED: "CLD_STOPPED", os.CLD_EXITED: "CLD_EXITED",
            os.CLD_KILLED: "CLD_KILLED"}.get(r.si_code, str(r.si_code))


def main():
    shell, cmd, attempts, max_hits = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    hold_s = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
    child = subprocess.Popen([shell, "-c", cmd], process_group=0)
    pid = child.pid
    time.sleep(0.05)
    hits = 0
    acked = 0
    ack_ms = []
    t_start = time.monotonic()
    for n in range(attempts):
        time.sleep(random.uniform(0.001, 0.015))
        t0 = time.monotonic()
        os.killpg(pid, signal.SIGSTOP)
        got = None
        while time.monotonic() - t0 < ACK_WAIT_S:
            if leader_reports_stopped(pid) == "CLD_STOPPED":
                got = time.monotonic() - t0
                break
            time.sleep(0.0005)
        if got is not None:
            acked += 1
            ack_ms.append(got * 1000)
            os.killpg(pid, signal.SIGCONT)
            continue
        hits += 1
        snap_a = snapshot(pid)
        if hold_s == 0:
            print(json.dumps({"event": "no_stop_ack", "shell": shell, "cmd": cmd,
                              "attempt": n, "leader": pid, "waitid_at_0.5s": "nothing",
                              "group_at_0.5s": snap_a}), flush=True)
            os.killpg(pid, signal.SIGCONT)
            if hits >= max_hits:
                break
            continue
        time.sleep(hold_s)
        snap_b = snapshot(pid)
        print(json.dumps({
            "event": "no_stop_ack",
            "shell": shell, "cmd": cmd, "attempt": n,
            "leader": pid,
            "waitid_at_0.5s": "nothing",
            "group_at_0.5s": snap_a,
            f"waitid_at_{0.5 + hold_s}s": leader_reports_stopped(pid),
            f"group_at_{0.5 + hold_s}s": snap_b,
        }), flush=True)
        os.killpg(pid, signal.SIGCONT)
        t1 = time.monotonic()
        os.killpg(pid, signal.SIGSTOP)
        rec = None
        while time.monotonic() - t1 < ACK_WAIT_S:
            if leader_reports_stopped(pid) == "CLD_STOPPED":
                rec = (time.monotonic() - t1) * 1000
                break
            time.sleep(0.0005)
        print(json.dumps({"event": "after_SIGCONT_restop", "attempt": n,
                          "acked_ms": rec, "group": snapshot(pid) if rec is None else None}),
              flush=True)
        os.killpg(pid, signal.SIGCONT)
        if hits >= max_hits:
            break
    os.killpg(pid, signal.SIGKILL)
    child.wait()
    ack_ms.sort()
    print(json.dumps({
        "event": "summary", "shell": shell, "cmd": cmd,
        "attempts": n + 1, "acked": acked, "no_ack": hits,
        "elapsed_s": round(time.monotonic() - t_start, 1),
        "ack_ms_p50": round(ack_ms[len(ack_ms) // 2], 2) if ack_ms else None,
        "ack_ms_max": round(ack_ms[-1], 2) if ack_ms else None,
        "kernel": f"{os.uname().sysname} {os.uname().release} {os.uname().machine}",
    }), flush=True)


if __name__ == "__main__":
    main()
