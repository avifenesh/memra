#!/usr/bin/env python3
"""timeline.py SERVER_LOG[.gz] [ANCHOR_REGEX] [BEFORE_S] [AFTER_S]: memra#1266. Print the memory
timeline around an anchor line (default: the first step OOM; else the first admission of the
3,905-token prompt): every [p1266] line, admission, admit-trim, prefix-cache, spec warn/suspend,
prime, tick-spec, admit-oom and engine-error line in the window, timestamps relative to the anchor."""
import gzip
import re
import sys

path = sys.argv[1]
anchor_re = re.compile(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
before = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
after = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
op = gzip.open if path.endswith(".gz") else open
lines = []
with op(path, "rt", errors="replace") as f:
    for raw in f:
        parts = raw.rstrip("\n").split(" ", 1)
        try:
            t = float(parts[0])
        except ValueError:
            continue
        lines.append((t, parts[1] if len(parts) > 1 else ""))
keep = re.compile(r"\[p1266\]|\[admission\] request cost|\[admit-trim\]|\[prefix-cache\]|\[spec\]|"
                  r"\[prime-|\[tick-spec\]|\[admit-oom\]|\[engine-error\]|\[spec-pool\]|\[device-trim\]|\[f1266\]")
if anchor_re is None:
    cands = [t for t, s in lines if "step OOM" in s or "[p1266] step-oom" in s]
    if not cands:
        cands = [t for t, s in lines if "prompt=3905" in s and "p1266] admit-gate" in s]
    if not cands:
        cands = [t for t, s in lines if "prompt=3905" in s]
else:
    cands = [t for t, s in lines if anchor_re.search(s)]
if not cands:
    print("no anchor")
    sys.exit(1)
t0 = cands[0]
for t, s in lines:
    if t0 - before <= t <= t0 + after and keep.search(s):
        print(f"{t - t0:+8.3f} {s[:420]}")
