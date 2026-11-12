#!/usr/bin/env bash
# memra#1255: regenerate the tables in GATES.md, AGENTIC.md and CELLS.md from the receipts.
# usage: make-tables.sh RESEARCH_DIR   (the harness scripts sit next to this file)
set -eu
R=$1
H=$(cd "$(dirname "$0")" && pwd)
python3 - "$R" "$H" <<'EOF'
import subprocess
import sys
from pathlib import Path

r, h = Path(sys.argv[1]), Path(sys.argv[2])


def run(script, sub):
    d = r / sub
    if not d.exists():
        return f"(no `{sub}/` receipts yet)\n"
    return subprocess.run(["python3", str(h / script), str(d)], capture_output=True, text=True,
                          check=True).stdout


def fill(name, marks):
    p = r / name
    s = p.read_text()
    for mark, body in marks.items():
        start, end = f"<!-- {mark}:START -->", f"<!-- {mark}:END -->"
        a, b = s.index(start) + len(start), s.index(end)
        s = s[:a] + "\n" + body.rstrip() + "\n" + s[b:]
    p.write_text(s)


fill("GATES.md", {"GATES": run("gate-table.py", "gates")})
fill("AGENTIC.md", {"AGENTIC-5090": run("ab-table.py", "agentic-5090"),
                    "AGENTIC-GB10": run("ab-table.py", "agentic-gb10")})
fill("CELLS.md", {"CELLS-5090": run("cell-table.py", "cells-5090"),
                  "CELLS-GB10": run("cell-table.py", "cells-gb10")})
EOF
