#!/usr/bin/env python3
"""OWED 26 G3: every committed receipt whose `[spill-pread]` totals line shows mmap fallbacks.

Reads files under research/ (logs included, `.ignore` does not apply), groups by the file's
directory, and writes census.json plus the table printed to stdout. Nothing is rescored.
"""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
LINE = re.compile(r"\[spill-pread\] reads=(\d+) bytes=\d+ errors=(\d+) short_reads=(\d+) fallbacks=(\d+)")
rows = {}
for f in sorted((ROOT / "research").rglob("*")):
    if not f.is_file() or f.suffix not in (".log", ".txt", ".jsonl", ".out", ".err", "") or f.stat().st_size > 64 << 20:
        continue
    try:
        text = f.read_text(errors="replace")
    except OSError:
        continue
    for m in LINE.finditer(text):
        rel = f.relative_to(ROOT / "research")
        key = str(rel.parent)
        r = rows.setdefault(key, {"files": 0, "totals_lines": 0, "lines_with_fallbacks": 0, "fallbacks": 0,
                                  "max_per_line": 0, "reads": 0})
        r["totals_lines"] += 1
        fb = int(m[4])
        r["reads"] += int(m[1])
        if fb:
            r["lines_with_fallbacks"] += 1
            r["fallbacks"] += fb
            r["max_per_line"] = max(r["max_per_line"], fb)
    if str(f.relative_to(ROOT / "research").parent) in rows:
        rows[str(f.relative_to(ROOT / "research").parent)]["files"] += 0
hit = {k: v for k, v in rows.items() if v["fallbacks"]}
out = Path(__file__).with_name("census.json")
out.write_text(json.dumps({"directories_with_totals_lines": len(rows), "directories_with_fallbacks": len(hit),
                           "rows": hit}, indent=1) + "\n")
print(f"directories with a totals line: {len(rows)}; with fallbacks: {len(hit)}")
for k, v in sorted(hit.items()):
    print(f"{k} | lines {v['lines_with_fallbacks']}/{v['totals_lines']} | fallbacks {v['fallbacks']} | max {v['max_per_line']}")
