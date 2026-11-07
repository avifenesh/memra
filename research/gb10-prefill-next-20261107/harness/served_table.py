#!/usr/bin/env python3
"""served_table.py CAMPAIGN_DIR [CAMPAIGN_DIR ...]: per campaign, the cold TTFT of every cell at
each length (B = main, L = lane), the medians and the change, the answer hashes per length and the
MemAvailable floor per cell (markdown)."""
import json, pathlib, statistics, sys

for cd in map(pathlib.Path, sys.argv[1:]):
    rows = {}
    shas = {}
    floors = []
    for cell in sorted(p for p in cd.iterdir() if p.is_dir()):
        arm = cell.name.split("-", 1)[1]
        for l in (cell / "pf.jsonl").read_text().splitlines():
            if not l.strip():
                continue
            d = json.loads(l)
            n = d["prompt_tokens"]
            rows.setdefault(n, {}).setdefault(arm, []).append(d["ttft_s"])
            shas.setdefault(n, set()).add(d["text_sha256"][:12])
            assert d.get("error") is None and d.get("needles_ok"), (cell, n)
        m = [json.loads(l)["mem_available_kb"] for l in (cell / "mem.jsonl").read_text().splitlines() if l.strip()]
        floors.append(f"{cell.name} {min(m) / 1048576:.2f}")
    print(f"### {cd.name}\n")
    print("| prompt tokens | main | lane | change (medians) | answer sha256 |")
    print("|---:|---|---|---:|---|")
    for n in sorted(rows):
        b, l = rows[n].get("B", []), rows[n].get("L", [])
        mb, ml = statistics.median(b), statistics.median(l)
        print(f"| {n:,} | {', '.join(f'{x:.3f}' for x in b)} s | {', '.join(f'{x:.3f}' for x in l)} s | "
              f"{100 * (ml - mb) / mb:+.2f}% | {', '.join(sorted(shas[n]))} |")
    print(f"\nMemAvailable floor (GiB): {'; '.join(floors)}\n")
