#!/usr/bin/env python3
"""Put the group-member round's receipts into the lane: red control, verification, harness.

Redacts the work directory and the host name and strips trailing spaces and final blank lines.
verify/summary.txt goes through the review round's filter (the rc and result line verify.sh
wrote, plus the stress count from each log), which leaves this round's lines unchanged.

Usage: assemble.py FETCHED_DIR SCRATCH_DIR LANE_MEMBERS_DIR FROM=TO...
(each FROM=TO is a redaction: the work directory, the host name)
"""
import os
import re
import sys

fetched, scratch, dst = sys.argv[1:4]
REDACT = [tuple(pair.split("=", 1)) for pair in sys.argv[4:]]
assert REDACT and all(len(pair) == 2 and pair[0] for pair in REDACT)


def clean(text):
    for a, b in REDACT:
        text = text.replace(a, b)
    lines = [l.rstrip() for l in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def put(text, rel):
    out = os.path.join(dst, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(clean(text))


def read(path):
    with open(path, errors="strict") as f:
        return f.read()


for name in ("before.txt", "final.txt", "before-darklane.log", "final-darklane.log"):
    put(read(os.path.join(fetched, "red", name)), f"red/{name}")

v = os.path.join(fetched, "verify")
for name in sorted(os.listdir(v)):
    if name.endswith(".log"):
        put(read(os.path.join(v, name)), f"verify/{name}")

PARKED = re.compile(r"stops acknowledged with the leader parked in vfork: \d+ of 300")
ITER = re.compile(r"^darklane iter (\d+) rc=(\d+) (test result: .*? finished in [0-9.]+s)")
rows = []
for line in read(os.path.join(v, "summary.txt")).splitlines():
    m = ITER.match(line)
    if m:
        log = read(os.path.join(v, f"darklane-{m.group(1)}.log"))
        parked = PARKED.findall(log)
        assert len(parked) == 1, (m.group(1), parked)
        rows.append(f"darklane iter {m.group(1)} rc={m.group(2)} {m.group(3)} {parked[0]}")
    elif line.startswith(("darklane.rs sha256", "build rc=", "full suite ", "fmt rc=", "clippy rc=")):
        rows.append(line)
assert sum(r.startswith("darklane iter") for r in rows) == 20
put("\n".join(rows), "verify/summary.txt")

for name in ("gen-before.py", "red.sh", "verify.sh", "run-all.sh", "assemble.py", "stage_members_probe.py"):
    put(read(os.path.join(scratch, name)), f"harness/{name}")

for root, _, files in os.walk(dst):
    for name in files:
        text = read(os.path.join(root, name))
        for bad in [a for a, _ in REDACT]:
            if bad in text:
                raise SystemExit(f"{os.path.join(root, name)}: {bad}")
        # An absolute /home or /tmp path left over (a redacted `$WORK/tmp/` is fine).
        if re.search(r"(?<![\w.$-])/(home|tmp)/", text):
            raise SystemExit(f"{os.path.join(root, name)}: absolute path")
print("assembled", sum(len(f) for _, _, f in os.walk(dst)), "files")
