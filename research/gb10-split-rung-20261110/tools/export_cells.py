#!/usr/bin/env python3
"""export_cells.py CAMPAIGN_DIR DEST: the small receipts of a served B/T campaign of this lane
(research/gb10-verify-rows-20261109/tools/export_rows.py).

Per cell (NN-B / NN-T): ab.jsonl, guard.log and trace.txt (the server log's `[tick-spec]`, spec K,
draft load, cost calibration and pool lines). Next to them: campaign.log and
rounds-by-request.txt (rounds_by_req.py). Paths are written from the home directory down. The full
server logs, GPU telemetry and memory samples stay in the campaign directory."""
import os
import pathlib
import re
import subprocess
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
tools = pathlib.Path(__file__).resolve().parent
keep = re.compile(r"\[(tick-spec|spec-k|spec-cost|mtp-draft|device-trim|spec-pool)\]")
home = os.path.expanduser("~")


def put(path, text):
    path.write_text(re.sub(r"/home/[A-Za-z0-9_.-]+", "~", text.replace(home, "~")))


dst.mkdir(parents=True, exist_ok=True)
for cell in sorted(p for p in src.iterdir() if p.is_dir() and re.match(r"\d\d-[BT]$", p.name)):
    out = dst / cell.name
    out.mkdir(exist_ok=True)
    for name in ("ab.jsonl", "guard.log"):
        if (cell / name).exists():
            put(out / name, (cell / name).read_text())
    log = cell / "server.log"
    if log.exists():
        lines = [l for l in log.read_text(errors="replace").splitlines() if keep.search(l)]
        put(out / "trace.txt", "\n".join(lines) + "\n")
if (src / "campaign.log").exists():
    put(dst / "campaign.log", (src / "campaign.log").read_text())
r = subprocess.run([sys.executable, str(tools / "rounds_by_req.py"), str(src)], capture_output=True, text=True)
put(dst / "rounds-by-request.txt", r.stdout + r.stderr)
print("exported", dst)
