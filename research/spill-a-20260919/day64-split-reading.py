#!/usr/bin/env python3
"""WP-A day 64 (DAY64.md section 4 step 1) reader: the span receipt's phases, and the design they select, written before
its cell runs.

usage: day64-split-reading.py ROOT
Input: ROOT/promote/ab/o{1,2}/bNN-promote/server.log (the promote cell, 20 boots; the tip with step 1's lines).
Complete: 20 boots; every `contracts door H2D receipt` line with spans carries a `(span receipt: fill X ms, copies Y
ms, digests Z ms)` term (steady: the second and later of each boot).
The rule (section 4): the digests term at least half of the (copies + digests) in the median selects D1, the
overlapped span digests; otherwise the larger of the fill and the lanes' share is named with its numbers.
Also printed: the late count (day64-reading.py's) on this tree, as a reading.
"""
import glob
import os
import re
import statistics as st
import subprocess
import sys

T = re.compile(r"contracts door H2D receipt: .*span receipt: fill ([\d.]+) ms, copies ([\d.]+) ms, digests ([\d.]+) ms")
R = re.compile(r"contracts door H2D receipt: .*f32 spans landed under the ticket")


def main():
    root = sys.argv[1]
    ds = sorted(glob.glob(os.path.join(root, "promote", "ab", "o*", "b*-promote")))
    fill, copies, digests, missing = [], [], [], 0
    for d in ds:
        lines = open(os.path.join(d, "server.log"), errors="replace").read().splitlines()
        rs = [ln for ln in lines if R.search(ln)][1:]
        for ln in rs:
            m = T.search(ln)
            if not m:
                missing += 1
                continue
            fill.append(float(m.group(1)))
            copies.append(float(m.group(2)))
            digests.append(float(m.group(3)))
    here = os.path.dirname(os.path.abspath(__file__))
    late = subprocess.run([sys.executable, os.path.join(here, "day64-reading.py"), root], capture_output=True,
                          text=True).stdout.strip().splitlines()
    print("\n".join(late))
    if len(ds) != 20 or missing or not digests:
        print(f"DAY64B INCOMPLETE (boots={len(ds)}, receipt lines without timing={missing}) -> nothing selected")
        return
    f, c, g = st.median(fill), st.median(copies), st.median(digests)
    print(f"DAY64B SPLIT N={len(digests)} fill={f:.2f} copies={c:.2f} digests={g:.2f} ms (digests share of copies + "
          f"digests {g / (c + g):.2f})")
    print(f"DAY64B SELECT -> {'D1 (the overlapped span digests)' if g >= 0.5 * (c + g) else 'NOT D1: ' + ('the fill' if f >= c else 'the copies') + ' named with its numbers'}")


if __name__ == "__main__":
    main()
