#!/usr/bin/env python3
"""tab.py GB10_DIR: the cell table of README.md from each cell's run.log, ab.jsonl and trace.txt:
MemAvailable before the request, each request's status, code and Retry-After, tokens, finish, the
text's sha256 prefix, and the admission lines that decided it."""
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
print("| cell | arm | MemAvailable before (MiB) | request | HTTP | code | Retry-After | tokens, finish | text | admission |")
print("|---|---|---|---|---|---|---|---|---|---|")
for cell in sorted(p for p in root.iterdir() if p.is_dir() and (p / "ab.jsonl").exists()):
    run = (cell / "run.log").read_text()
    m = re.search(r"before the request:\n(?:.*\n)*?MemAvailable:\s+(\d+) kB", run)
    avail = f"{int(m.group(1)) // 1024:,}" if m else "-"
    first = run.splitlines()[0]
    arm = re.search(r"arm (\w+)", first).group(1)
    trace = (cell / "trace.txt").read_text() if (cell / "trace.txt").exists() else ""
    lines = []
    for pat, label in [(r"capacity reject: .*?\(available (\d+)MB(?:, post-load capacity (\d+)MB)?", "capacity reject"),
                       (r"idle memory short: .*?capacity \((\d+)MB\).*?available (\d+)MB, required (\d+)MB", "idle memory short"),
                       (r"VRAM defer: (\d+) active", "VRAM defer")]:
        hits = re.findall(pat, trace)
        if hits:
            h = hits[0] if isinstance(hits[0], tuple) else (hits[0],)
            if label == "capacity reject":
                lines.append(f"capacity reject, available {h[0]} MB" + (f", capacity {h[1]} MB" if h[1] else ""))
            elif label == "idle memory short":
                lines.append(f"idle memory short, available {h[1]} MB < required {h[2]} MB, capacity {h[0]} MB")
            else:
                lines.append(f"{len(hits)} VRAM defer lines ({h[0]} active)")
    rows = [json.loads(x) for x in (cell / "ab.jsonl").read_text().splitlines() if x.strip()]
    for i, r in enumerate(rows):
        ref = r.get("refusal") or {}
        code = ((ref.get("body") or {}).get("error") or {}).get("code", "-") if isinstance(ref.get("body"), dict) else "-"
        toks = f"{r['completion_tokens']}, {r['finish']}" if r.get("completion_tokens") else "-"
        text = r["text_sha256"][:8] if r.get("completion_tokens") else "-"
        adm = "; ".join(lines) if i == 0 else ""
        if r["status"] == 200 and not adm and i == 0:
            adm = "admitted"
        print(f"| {cell.name if i == 0 else ''} | {arm if i == 0 else ''} | {avail if i == 0 else ''} | {r['req']} | "
              f"{r['status']} | {code} | {ref.get('retry_after') or '-'} | {toks} | {text} | {adm} |")
