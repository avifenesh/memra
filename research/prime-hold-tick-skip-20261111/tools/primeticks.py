#!/usr/bin/env python3
"""When a plain request is primed, from a server log run with MEMRA_TICK_TRACE=1.

usage: primeticks.py [--whole] SERVER_LOG [SERVER_LOG ...]   (--whole: one cell per log, for the
       normal census, whose prompts are all longer than 40 tokens)

Splits each log into cells at the long request's admission (a `[spec-k]` line whose prompt has
more than 40 tokens; the normal cell has none, so the whole log is one cell). For every
admission to plain decode (`[spec-k] ... K=0`) it counts the ticks from the admission to the
tick that primes it, assuming first-in first-out priming (a single prefill call primes the
oldest pending request, a batch call primes all pending ones). A `[tick]` line is printed at the
end of its tick, so 1 means "primed in its admission tick". On the fix, also the in-tick hold
waits (`hold_waits`, `hold_wait_ms`).
"""
import collections
import re
import sys

SPECK = re.compile(r"\[spec-k\] model=\"q\" .* K=(\d+) source=\S+ prompt=(\d+)")


WHOLE = "--whole" in sys.argv


def cells(lines):
    if WHOLE:
        return [lines]
    cur = []
    out = [cur]
    for line in lines:
        m = SPECK.search(line)
        if m and int(m.group(2)) > 40:
            cur = []
            out.append(cur)
            continue
        cur.append(line)
    return [c for c in out if any(l.startswith("[tick]") for l in c)]


for path in [a for a in sys.argv[1:] if a != "--whole"]:
    lines = open(path, errors="replace").read().splitlines()
    for n, cell in enumerate(cells(lines)):
        waits, pend = collections.Counter(), []
        ticks = hold_waits = 0
        hold_ms = []
        for line in cell:
            m = SPECK.search(line)
            if m:
                if m.group(1) == "0":
                    pend.append(0)
                continue
            if not line.startswith("[tick]"):
                continue
            ticks += 1
            single = int(re.search(r"prefill_single_calls=(\d+)", line).group(1))
            batch = int(re.search(r"prefill_batch_calls=(\d+)", line).group(1))
            hw = re.search(r"hold_waits=(\d+) hold_wait_ms=([0-9.]+)", line)
            if hw:
                hold_waits += int(hw.group(1))
                if int(hw.group(1)):
                    hold_ms.append(float(hw.group(2)))
            pend = [p + 1 for p in pend]
            k = single if batch == 0 else len(pend)
            for _ in range(min(k, len(pend))):
                waits[pend.pop(0)] += 1
        primed = sum(waits.values())
        hold = (f", hold waits {hold_waits} (max {max(hold_ms):.2f} ms, total {sum(hold_ms):.1f} ms)"
                if hold_ms else (", hold waits 0" if hold_waits == 0 and "hold_waits=" in "".join(cell) else ""))
        print(f"{path} cell {n}: {ticks} ticks, plain admissions primed {primed}, ticks to prime "
              f"{dict(sorted(waits.items()))}{hold}")
