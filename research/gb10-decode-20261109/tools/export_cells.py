#!/usr/bin/env python3
"""export_cells.py CAMPAIGN_DIR DEST: the small receipts of a served B/T campaign.

Per cell (NN-B / NN-T): ab.jsonl, guard.log and trace.txt, the server log's lines the tables read
(the spec K table and verdicts, the draft load, the cost calibration and every `[tick-spec]` burst).
Next to them: campaign.log, tab.txt (eos_tab.py) and rounds.txt (rounds_tab.py). The full server
logs, GPU telemetry and memory samples stay in the campaign directory.
"""
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
    """Receipts name paths from the home directory down."""
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
for name in ("campaign.log",):
    if (src / name).exists():
        put(dst / name, (src / name).read_text())
for tool, name in (("eos_tab.py", "tab.txt"), ("rounds_tab.py", "rounds.txt")):
    r = subprocess.run([sys.executable, str(tools / tool), str(src)], capture_output=True, text=True)
    put(dst / name, r.stdout + r.stderr)
print("exported", dst)
