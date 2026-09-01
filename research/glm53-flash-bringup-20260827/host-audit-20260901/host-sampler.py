#!/usr/bin/env python3
"""HOST-MICRO SAMPLER — per-thread scheduling receipts for a running memra-server.

Engine-wide (lane/glm5-host-audit, 2026-09-01). Reads ONLY procfs: no ptrace, no
signals, no requests, no perf. Safe to run against a live serving boot; safe to run
as a co-tenant. Its own cost is one procfs read set per thread per tick.

WHY procfs AND NOT perf: Box B is an unprivileged container with
`kernel.perf_event_paranoid = 4` and no kernel-matched linux-tools, so `perf stat`,
`perf trace` and every PMU counter (cache-misses, LLC-misses, cpu-migrations) are
UNAVAILABLE there. cpu-migrations are recovered here by SAMPLING the scheduler's
last-CPU field (/proc/<tid>/stat field 39) instead of counting them exactly: a
sampled migration count is a LOWER BOUND on real migrations and is labelled as such
in every row. voluntary/nonvoluntary context switches ARE exact (they are counters,
not samples) — but they must be read PER THREAD: /proc/<pid>/status reports the MAIN
thread only, which reads as a near-idle process no matter what the worker is doing.

Usage:
  host-sampler.py --pid <pid> [--secs 60] [--interval 0.05]
                  [--comm memra-gpu-worke,memra-server,cuda-EvtHandlr]
                  [--tokens-before N --tokens-after N] [--json out.json]

CCX/CCD MAP is read from sysfs (index3 shared_cpu_list), never assumed: on the EPYC
9654 in Box B each CCX is 8 cores + their 8 SMT siblings sharing one 32 MB L3, and a
cross-CCX migration throws that L3 away. An SMT-sibling hop inside one core is
counted separately because it costs nothing in L3 terms.
"""

import argparse
import json
import os
import signal
import sys
import time
from collections import defaultdict

STAT_PROCESSOR_FIELD = 39  # 1-based, per proc(5): "processor" = last CPU executed on


def read(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return None


def l3_map():
    """cpu -> L3 domain id, read from sysfs. Returns ({}, note) if sysfs is blind."""
    m, dom, order = {}, {}, 0
    for cpu in sorted(
        int(d[3:])
        for d in os.listdir("/sys/devices/system/cpu")
        if d.startswith("cpu") and d[3:].isdigit()
    ):
        shared = read(f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list")
        if shared is None:
            continue
        key = shared.strip()
        if key not in dom:
            dom[key] = order
            order += 1
        m[cpu] = dom[key]
    return m, {v: k for k, v in dom.items()}


def core_map():
    """cpu -> physical core id (thread_siblings_list), for SMT-hop accounting."""
    m, dom, order = {}, {}, 0
    for cpu in sorted(
        int(d[3:])
        for d in os.listdir("/sys/devices/system/cpu")
        if d.startswith("cpu") and d[3:].isdigit()
    ):
        sib = read(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list")
        if sib is None:
            continue
        key = sib.strip()
        if key not in dom:
            dom[key] = order
            order += 1
        m[cpu] = dom[key]
    return m


def thread_snapshot(pid, tid):
    st = read(f"/proc/{pid}/task/{tid}/stat")
    status = read(f"/proc/{pid}/task/{tid}/status")
    if st is None or status is None:
        return None
    # comm can contain spaces and parens: split on the LAST ')' per proc(5).
    close = st.rfind(")")
    comm = st[st.find("(") + 1 : close]
    rest = st[close + 2 :].split()
    # rest[0] is field 3 (state), so field N is rest[N-3].
    cpu = int(rest[STAT_PROCESSOR_FIELD - 3])
    utime, stime = int(rest[13 - 3]), int(rest[14 - 3])
    vol = nonvol = None
    for line in status.splitlines():
        if line.startswith("voluntary_ctxt_switches:"):
            vol = int(line.split()[1])
        elif line.startswith("nonvoluntary_ctxt_switches:"):
            nonvol = int(line.split()[1])
    return comm, cpu, utime, stime, vol, nonvol


def rollup(pid):
    """THP receipt: AnonHugePages vs Rss from smaps_rollup (one read, cheap)."""
    out = {}
    txt = read(f"/proc/{pid}/smaps_rollup")
    if txt is None:
        return out
    for line in txt.splitlines():
        for key in ("Rss:", "AnonHugePages:", "Anonymous:", "ShmemPmdMapped:", "FilePmdMapped:"):
            if line.startswith(key):
                out[key.rstrip(":")] = int(line.split()[1])  # kB
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--secs", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=0.05)
    ap.add_argument(
        "--comm",
        default="memra-gpu-worke,memra-server,memra-gpu-watch,cuda-EvtHandlr",
        help="comma list of thread comm prefixes to track (comm is capped at 15 chars)",
    )
    ap.add_argument("--tokens", type=int, default=0, help="completion tokens generated during the window, for per-1k rates")
    ap.add_argument("--label", default="")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    prefixes = tuple(p for p in args.comm.split(",") if p)
    l3, l3names = l3_map()
    cores = core_map()

    first, last, cpuhist = {}, {}, defaultdict(lambda: defaultdict(int))
    seq = defaultdict(list)  # tid -> observed cpu sequence (compressed: only changes)
    names, ticks = {}, 0
    roll_before = rollup(args.pid)

    # SIGTERM must STOP the loop, not kill the process. The caller bounds this sampler by the
    # timed request it runs alongside and then terminates it — and the receipt is only written
    # after the loop, so the default SIGTERM disposition threw away every scheduling receipt in
    # the first arm sweep while the tok/s rows came back perfectly. A collector whose output
    # exists only on the clean-exit path has no output.
    stop = {"now": False}

    def _stop(_signum, _frame):
        stop["now"] = True

    for _sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(_sig, _stop)
        except (ValueError, OSError):
            pass  # not the main thread, or a platform without it: fall back to --secs

    t_end = time.monotonic() + args.secs
    t0 = time.monotonic()
    while time.monotonic() < t_end and not stop["now"]:
        try:
            tids = os.listdir(f"/proc/{args.pid}/task")
        except OSError:
            print(f"pid {args.pid} gone after {ticks} ticks", file=sys.stderr)
            break
        for tid in tids:
            snap = thread_snapshot(args.pid, tid)
            if snap is None:
                continue
            comm, cpu, utime, stime, vol, nonvol = snap
            if not comm.startswith(prefixes):
                continue
            names[tid] = comm
            if tid not in first:
                first[tid] = (utime, stime, vol, nonvol)
            last[tid] = (utime, stime, vol, nonvol)
            cpuhist[tid][cpu] += 1
            if not seq[tid] or seq[tid][-1] != cpu:
                seq[tid].append(cpu)
        ticks += 1
        time.sleep(args.interval)
    wall = time.monotonic() - t0
    roll_after = rollup(args.pid)

    hz = os.sysconf("SC_CLK_TCK")
    rows = []
    for tid, comm in sorted(names.items(), key=lambda kv: (kv[1], int(kv[0]))):
        u0, s0, v0, n0 = first[tid]
        u1, s1, v1, n1 = last[tid]
        hist = cpuhist[tid]
        cpu_seq = seq[tid]
        ccx_seq = [l3.get(c, -1) for c in cpu_seq]
        core_seq = [cores.get(c, -1) for c in cpu_seq]
        ccx_cross = sum(1 for a, b in zip(ccx_seq, ccx_seq[1:]) if a != b)
        core_hops = sum(1 for a, b in zip(core_seq, core_seq[1:]) if a != b)
        smt_hops = sum(
            1
            for a, b in zip(cpu_seq, cpu_seq[1:])
            if a != b and cores.get(a, -1) == cores.get(b, -2)
        )
        busy = ((u1 - u0) + (s1 - s0)) / hz
        row = {
            "tid": int(tid),
            "comm": comm,
            "samples": sum(hist.values()),
            "wall_s": round(wall, 3),
            "cpu_busy_s": round(busy, 3),
            "cpu_busy_pct": round(100.0 * busy / wall, 1) if wall else None,
            "utime_s": round((u1 - u0) / hz, 3),
            "stime_s": round((s1 - s0) / hz, 3),
            "vol_ctxt": v1 - v0,
            "nonvol_ctxt": n1 - n0,
            "vol_per_s": round((v1 - v0) / wall, 2) if wall else None,
            "nonvol_per_s": round((n1 - n0) / wall, 2) if wall else None,
            "distinct_cpus": len(hist),
            "distinct_ccx": len({l3.get(c, -1) for c in hist}),
            "sampled_cpu_changes": max(0, len(cpu_seq) - 1),
            "sampled_ccx_crossings": ccx_cross,
            "sampled_core_hops": core_hops,
            "sampled_smt_sibling_hops": smt_hops,
            "top_cpus": sorted(hist.items(), key=lambda kv: -kv[1])[:6],
        }
        if args.tokens:
            row["vol_per_1k_tok"] = round(1000.0 * (v1 - v0) / args.tokens, 1)
            row["nonvol_per_1k_tok"] = round(1000.0 * (n1 - n0) / args.tokens, 1)
            row["cpu_busy_ms_per_tok"] = round(1000.0 * busy / args.tokens, 3)
        rows.append(row)

    out = {
        "label": args.label,
        "pid": args.pid,
        "wall_s": round(wall, 3),
        "ticks": ticks,
        "interval_s": args.interval,
        "tokens": args.tokens,
        "l3_domains": {str(k): v for k, v in l3names.items()},
        "smaps_rollup_kb_before": roll_before,
        "smaps_rollup_kb_after": roll_after,
        "threads": rows,
        "caveats": [
            "cpu-migration counts are SAMPLED (interval above), so they are a LOWER BOUND; "
            "perf's exact cpu-migrations counter is unavailable on this host "
            "(perf_event_paranoid=4, unprivileged container, no kernel-matched linux-tools).",
            "voluntary/nonvoluntary ctxt switches are EXACT per-thread counters read from "
            "/proc/<pid>/task/<tid>/status. /proc/<pid>/status is the MAIN thread only and "
            "must never be used for this.",
            "cache-misses / LLC-misses are NOT measurable on this host: no PMU access.",
        ],
    }
    text = json.dumps(out, indent=1)
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
