"""Cache residency per arm from serve.log: the resident bytes each insert line reports (after the insert), as a
lens on what the cache keeps when the reclaim runs through the whole run.

  python3 resident.py DIR    (every */serve.log under DIR)
"""
import re, statistics, sys
from pathlib import Path

print("| arm | inserts | resident MB at insert: median / mean / max | inserts into an empty cache (resident = the entry) |")
print("|---|---|---|---|")
for log in sorted(Path(sys.argv[1]).glob("*/serve.log")):
    rows = re.findall(r"\[prefix-cache\] insert \([^)]*\): \d+ tokens, ([0-9.]+)MB \(resident ([0-9.]+)MB / ", log.read_text(errors="replace"))
    if not rows:
        continue
    res = [float(r) for _, r in rows]
    alone = sum(1 for e, r in rows if abs(float(e) - float(r)) < 0.5)
    print(f"| {log.parent.name} | {len(res)} | {statistics.median(res):.0f} / {statistics.mean(res):.0f} / {max(res):.0f} | {alone} |")
