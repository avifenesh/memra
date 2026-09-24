#!/usr/bin/env python3
"""WP-A day 36 reader (DAY36.md section 1: the restore price cell of DAY33 section 6), written before the cell runs.

Input: an evidence dir with server.log and replays.log (the restore arm's). Complete: `STALL REPLAY: PASS` once, and every
landed restore's line carrying the owner-stream term. Readings: over every `restore submitted off the tick` line's
`recurrent copy H.HHms host` and every `restore landed off the tick` line's `recurrent copy G.GGms owner stream`: median,
min, max, N; the first restore of the boot alone; the count of `pending` and `n/a`.
Default mode (the 5090, not the rule's card): `DAY36 PRICE READING (5090, not the rule's card)`, no verdict.
--target (the rule's card, DAY33 section 6 verbatim): the owner-stream GPU median under 0.5 ms and the host median under
0.5 ms per restore -> `DAY36 PRICE VERDICT .. -> CLOSES`, otherwise `-> DESIGN OWED`.
usage: day36-reading.py [--target] EV_DIR
"""
import os
import re
import statistics
import sys

HOST = re.compile(r"\[prefix-cache\] restore submitted off the tick: .*; recurrent copy ([\d.]+)ms host")
GPU = re.compile(r"\[prefix-cache\] restore landed off the tick: .*; recurrent copy ([\d.]+)ms owner stream")


def stat(name, xs):
    if not xs:
        return f"{name} N=0"
    return (f"{name} N={len(xs)} median={statistics.median(xs):.3f} min={min(xs):.3f} max={max(xs):.3f} "
            f"first={xs[0]:.3f}")


def main():
    target = sys.argv[1] == "--target"
    ev = sys.argv[2] if target else sys.argv[1]
    host, gpu, pending, na, submitted, landed = [], [], 0, 0, 0, 0
    for ln in open(os.path.join(ev, "server.log"), errors="replace"):
        if "restore submitted off the tick" in ln:
            submitted += 1
            m = HOST.search(ln)
            if m:
                host.append(float(m.group(1)))
        if "restore landed off the tick" in ln:
            landed += 1
            m = GPU.search(ln)
            if m:
                gpu.append(float(m.group(1)))
            elif "recurrent copy owner stream pending" in ln:
                pending += 1
            elif "recurrent copy n/a owner stream" in ln:
                na += 1
    rp = os.path.join(ev, "replays.log")
    replay = os.path.exists(rp) and "STALL REPLAY: PASS" in open(rp, errors="replace").read()
    print(f"DAY36 READING submitted={submitted} landed={landed} pending={pending} n/a={na} replay_pass={replay}")
    print(f"DAY36 READING {stat('recurrent-copy-host-ms', host)}")
    print(f"DAY36 READING {stat('recurrent-copy-owner-stream-ms', gpu)}")
    complete = replay and landed > 0 and len(gpu) == landed and len(host) == submitted
    if not complete:
        print("DAY36 PRICE INCOMPLETE (a replay failed, or a landed restore lacks its owner-stream term) -> nothing is read")
        sys.exit(2)
    hm, gm = statistics.median(host), statistics.median(gpu)
    if not target:
        print(f"DAY36 PRICE READING (5090, not the rule's card) host median={hm:.3f} owner-stream median={gm:.3f} "
              f"(the rule's bound 0.5 each, read on the target card only)")
        return
    ok = gm < 0.5 and hm < 0.5
    print(f"DAY36 PRICE VERDICT (target card) owner-stream median={gm:.3f} host median={hm:.3f} rule each < 0.5 ms per "
          f"restore -> {'CLOSES' if ok else 'DESIGN OWED'}")


if __name__ == "__main__":
    main()
