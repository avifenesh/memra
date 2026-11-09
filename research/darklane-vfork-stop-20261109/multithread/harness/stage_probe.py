#!/usr/bin/env python3
"""Copy the review's probe (sources, raw JSONL, summaries) into the lane, paths redacted.

Usage: stage_probe.py REVIEW_DIR LANE_MULTITHREAD_DIR FROM=TO...
(each FROM=TO is a redaction: the review's work directory)
"""
import os
import re
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
REDACT = [tuple(pair.split("=", 1)) for pair in sys.argv[3:]]
assert REDACT and all(len(pair) == 2 and pair[0] for pair in REDACT)


def clean(text):
    for a, b in REDACT:
        text = text.replace(a, b)
    lines = [l.rstrip() for l in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def copy(rel_src, rel_dst):
    out = os.path.join(dst, rel_dst)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(os.path.join(src, rel_src)) as f:
        text = f.read()
    with open(out, "w") as f:
        f.write(clean(text))


for name in ("leader.c", "driver.py", "repeat.sh", "resume_after_tgkill.py", "starve.py"):
    copy(f"probe/{name}", f"probe/{name}")
copy("results-repeat-summary.txt", "probe/repeat-summary.txt")
for run in sorted(os.listdir(os.path.join(src, "results-repeat"))):
    rdir = os.path.join(src, "results-repeat", run)
    if not (run.startswith("run-") and os.path.isdir(rdir)):
        continue
    for name in sorted(os.listdir(rdir)):
        if name.endswith(".jsonl"):
            copy(f"results-repeat/{run}/{name}", f"probe/repeat/{run}/{name}")
copy("results-starve/starve.jsonl", "probe/starve.jsonl")
copy("results-starve.txt", "probe/starve-summary.txt")
copy("results-resume.txt", "probe/resume-after-tgkill.txt")
for root, _, files in os.walk(os.path.join(dst, "probe")):
    for name in files:
        text = open(os.path.join(root, name)).read()
        for bad in [a for a, _ in REDACT]:
            assert bad not in text, (name, bad)
        assert not re.search(r"(?<![\w.$-])/(home|tmp)/", text), (name, "absolute path")
print("staged", sum(len(f) for _, _, f in os.walk(os.path.join(dst, "probe"))), "files")
