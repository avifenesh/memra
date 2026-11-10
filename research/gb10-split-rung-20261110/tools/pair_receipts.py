#!/usr/bin/env python3
"""One receipt per warm == cold pair case (pair_cells_deep.py's output, copied off the unit
without the server logs): the command, the exit code, the case's verdict line and each turn's
prompt, completion and cached tokens and spec rounds. Home paths are written as ~.
usage: pair_receipts.py CELLS_COPY OUT_DIR"""
import json, re, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
HOME = re.compile(r"/home/[A-Za-z0-9_-]+")
for case in sorted(src.glob("pairs-*")):
    lines = ["# cmd"] + (case / "cmd.sh").read_text().splitlines()[2:]
    lines.append(f"# rc {(case / 'rc').read_text().strip()}")
    lines.append("# verdict (pair_cells_deep.py)")
    lines += [l for l in (case / "out" / "verdicts.jsonl").read_text().splitlines() if l.strip()]
    lines.append("# turns (prompt / completion / cached tokens, spec rounds)")
    for f in ("warm-t1", "warm-t2", "warm-t3", "cold-t2", "cold-t3"):
        p = case / "out" / case.name / f"{f}.json"
        if p.exists():
            u = json.loads(p.read_text()).get("usage", {})
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            rounds = (u.get("spec") or {}).get("rounds")
            lines.append(f"{f}: {u.get('prompt_tokens')} / {u.get('completion_tokens')} / {cached}, rounds {rounds}")
    sha = case / "out" / case.name / "binary.sha"
    if sha.exists():
        lines.append("# binary sha256")
        lines += sha.read_text().splitlines()
    (out / f"{case.name}.txt").write_text(HOME.sub("~", "\n".join(lines)) + "\n")
    print(case.name)
