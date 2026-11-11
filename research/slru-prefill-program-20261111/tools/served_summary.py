#!/usr/bin/env python3
"""Summarize served cells: text/tokens sha, server elapsed, client time, peak memory; ITL cells.

usage: served_summary.py <served dir>
"""
import hashlib
import json
import pathlib
import statistics
import sys

root = pathlib.Path(sys.argv[1])
pairs, itl = {}, {}
for d in sorted(p for p in root.iterdir() if p.is_dir()):
    res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    if (d / "response.json").exists():
        body = json.loads((d / "response.json").read_text())
        text = body.get("text") or body["choices"][0]["text"]
        row = {
            "text_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
            "tokens_sha16": hashlib.sha256(json.dumps(body.get("tokens")).encode()).hexdigest()[:16],
            "server_elapsed_s": body.get("elapsed_s"),
            "client_s": res.get("client_s"),
            "boot_s": res.get("boot_s"),
            "peak_mem_mib": res.get("peak_mem_mib"),
        }
        pairs[d.name] = row
    elif "itl_during_b" in res:
        itl[d.name] = res

print("| cell | text sha256 (16) | server elapsed s | client s | boot s | peak memory MiB |")
print("|---|---|---|---|---|---|")
for name, r in pairs.items():
    print(f"| {name} | `{r['text_sha16']}` | {r['server_elapsed_s']:.3f} | {r['client_s']} | {r['boot_s']} | {r['peak_mem_mib']} |")

groups = {}
for name, r in pairs.items():
    key = name.rsplit("-", 1)[0]
    groups.setdefault(key, []).append(r["server_elapsed_s"])
print()
print("| arm | N | server elapsed median s | runs |")
print("|---|---|---|---|")
for key, v in groups.items():
    print(f"| {key} | {len(v)} | {statistics.median(v):.3f} | {', '.join(f'{x:.3f}' for x in v)} |")

# Two-session cells: itl_summary.py (gaps overlapping B; the in-cell window in result.json
# misses the stall, because the gap that spans B ends after B is done).
