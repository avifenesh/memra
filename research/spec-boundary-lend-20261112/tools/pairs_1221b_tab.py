#!/usr/bin/env python3
"""pairs_1221b_tab.py PAIRS_DIR...: one row per warm == cold pair case (pairs_1221b.py's
verdicts.jsonl in each PAIRS_DIR): whether each later warm turn equals its cold twin, the restores
(tokens of the prompt, each from an `insert (spec-boundary)` entry, with its draft plane), the
answers per turn, the entries' state digests when the case ran with the trace, and from the warm
server log the loan lines at the restore: the last `[prefix-loan] reconcile` before the hit (books,
lent planes whose lender lives, adopted MB) and whether the entry the hit read was borrowed (its lender alive: lent planes, nothing adopted) or
adopted (its lender gone)."""
import gzip
import json
import pathlib
import re
import sys

rec = re.compile(r"\[prefix-loan\] reconcile why=(\S+) books=(\d+) lent_live=(\d+) adopted=([\d.]+)MB")
print("| run | case | later turns == cold | restored (of prompt) with draft plane | answers per turn | "
      "entry digests | at the hit: books / lent live / adopted MB | read |")
print("|---|---|---|---|---|---|---|---|")
for d in map(pathlib.Path, sys.argv[1:]):
    f = d / "verdicts.jsonl"
    if not f.exists():
        print(f"| {d.name} | (no verdicts) |")
        continue
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        later = [t for t in v["turns"] if t != "1"]
        eq = " / ".join("=" if v["turns"][t]["equal"] else "DIFF" for t in later)
        w = v["warm"]
        rest = ", ".join(f"{n:,} of {m:,}" for n, m in w["draft_restore"]) if v["restored_from_spec_entry"] else "NO"
        shas = " / ".join(v["turns"][t]["warm_sha"][:8] for t in sorted(v["turns"]))
        dig = ", ".join(s[3][:12] for s in w["state"] if s[0] == "spec-snapshot") or "-"
        log = d / v["case"] / "warm-server.log"
        gz = d / v["case"] / "warm-server.log.gz"
        text = log.read_text(errors="replace") if log.exists() else (gzip.open(gz, "rt", errors="replace").read() if gz.exists() else "")
        hit = text.find("[prefix-cache] hit")
        before = [m.groups() for m in rec.finditer(text[:hit if hit >= 0 else len(text)])]
        at = f"{before[-1][1]} / {before[-1][2]} / {before[-1][3]}" if before else "-"
        read = ("borrowed" if int(before[-1][2]) > 0 and float(before[-1][3]) == 0 else "adopted") if before else "-"
        print(f"| {d.name} | {v['case']} | {eq} | {rest} | {shas} | {dig} | {at} | {read} |")
