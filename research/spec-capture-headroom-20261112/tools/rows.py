#!/usr/bin/env python3
"""rows.py CAMPAIGN_DIR [LABELS]: markdown rows per cell from specpub_tab.py's cell lines, plus
the g1k draft-graph capture refusals counted in each cell's trace.txt. LABELS maps arm letters to
the names the record uses, e.g. W=V."""
import pathlib
import re
import subprocess
import sys

d = pathlib.Path(sys.argv[1])
labels = dict(kv.split("=") for kv in sys.argv[2:])
out = subprocess.run([sys.executable, str(pathlib.Path(__file__).with_name("specpub_tab.py")), str(d)],
                     capture_output=True, text=True).stdout
pat = re.compile(r"^(\d\d)-([A-Z]): \S+ prompt.*?stall ([-\d.]+) pub ([\d.]+).*?round ([\d.]+); TTFT ([\d.]+) s.*?"
                 r"burst 1 ([\d.]+) ms; trims \[(.*?)\](?:; publisher entry cached (\d+) MB, pool growth (\d+) MB)?")
for line in out.splitlines():
    m = pat.match(line)
    if not m:
        continue
    nn, arm, stall, pub, rnd, ttft, b1, trims, cached, growth = m.groups()
    trims = trims.replace("'", "").replace("released ", "rel. ").replace("kept ", "kept ") or "none"
    tr = d / f"{nn}-{arm}" / "trace.txt"
    refused = sum(1 for l in tr.read_text(errors="replace").splitlines() if "capture failed" in l) if tr.exists() else 0
    print(f"| {nn}-{labels.get(arm, arm)} | {trims} | {cached or ''} | {growth or ''} | {pub} | {stall} | {b1} | "
          f"{rnd} | {ttft} | {'refused' if refused else 'captured'} |")
