#!/usr/bin/env python3
"""Copy the re-review's probe (sources, raw JSONL, summaries) into the lane, paths redacted.

Usage: stage_members_probe.py PROBE_DIR LANE_MEMBERS_DIR FROM=TO...
(each FROM=TO is a redaction: the review's work directory)
"""
import os
import re
import sys

src, dst = sys.argv[1], sys.argv[2]
REDACT = [tuple(pair.split("=", 1)) for pair in sys.argv[3:]]
assert REDACT and all(len(pair) == 2 and pair[0] for pair in REDACT)
FILES = ["Cargo.toml", "Cargo.lock", "src/main.rs", "src/ack.rs", "extract.py", "summarize.py",
         "busyleader.c", "results.jsonl", "summary.txt", "busy.jsonl", "busy-summary.txt"]

for rel in FILES:
    text = open(os.path.join(src, rel)).read()
    for a, b in REDACT:
        text = text.replace(a, b)
    lines = [l.rstrip() for l in text.split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    text = "\n".join(lines) + "\n"
    for bad in [a for a, _ in REDACT]:
        assert bad not in text, (rel, bad)
    assert not re.search(r"(?<![\w.$-])/(home|tmp)/", text), (rel, "absolute path")
    out = os.path.join(dst, "probe", rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(text)
print("staged", len(FILES), "files")
