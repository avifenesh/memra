#!/usr/bin/env python3
"""served_peek.py CAMPAIGN_DIR: one line per (cell, target): TTFT, text sha prefix, needles."""
import json, pathlib, sys
cd = pathlib.Path(sys.argv[1])
for cell in sorted(p for p in cd.iterdir() if p.is_dir()):
    f = cell / "pf.jsonl"
    if not f.exists():
        continue
    for l in f.read_text().splitlines():
        if not l.strip():
            continue
        d = json.loads(l)
        print(f"{cell.name:6s} {d.get('prompt_tokens')!s:>7s} ttft={d.get('ttft_s')} sha={str(d.get('text_sha256'))[:12]} needles={d.get('needles_ok')} err={d.get('error')}")
