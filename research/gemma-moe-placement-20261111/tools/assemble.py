#!/usr/bin/env python3
"""Copy the lane's receipts into research/gemma-moe-placement-20261111/ of the worktree.

Traced probe logs (MEMRA_MOE_SLRU_PAIRS_TRACE=1) carry one line per MoE layer call (about 63k);
they are committed without those lines, plus trace-summary.txt (mode, runs and lock hold per
cell). Private paths are rewritten to placeholders.
"""
import collections
import pathlib
import re
import shutil
import statistics

HOME = pathlib.Path.home()
SRC = HOME / ".cache/memra-1225"
DST = SRC / "wt/research/gemma-moe-placement-20261111"
SUBS = [
    (str(SRC / "wt"), "<repo>"),
    (str(SRC), "<lane>"),
    ("<models>/", "<models>/"),
    (str(HOME), "<home>"),
]


def scrub(text):
    for a, b in SUBS:
        text = text.replace(a, b)
    text = re.sub(r"leased "leased", text)
    return text


def put(rel, text):
    p = DST / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(scrub(text))


for sub in ("window", "served", "placement-test", "tools"):
    if (DST / sub).exists():
        shutil.rmtree(DST / sub)
summary = []
for stage, sub in (("base", SRC / "step1/window"), ("fix", SRC / "step3/window"),
                   ("final", SRC / "final/window"), ("final-main", SRC / "final/window-main")):
    if not (sub / "cells.txt").exists():
        continue
    put(f"window/{stage}/cells.txt", (sub / "cells.txt").read_text())
    for log in sorted(sub.glob("*/probe.log")):
        lines = log.read_text().splitlines()
        trace = [l for l in lines if l.startswith("[moe-slru-gemma] il=")]
        keep = [l for l in lines if not l.startswith("[moe-slru-gemma] il=")]
        put(f"window/{stage}/{log.parent.name}/probe.log", "\n".join(keep) + "\n")
        if trace:
            modes = collections.Counter(re.search(r"mode=(\w+) runs=(\d+)", l).groups() for l in trace)
            holds = sorted(int(re.search(r"hold_us=(\d+)", l).group(1)) for l in trace)
            q = lambda f: holds[min(len(holds) - 1, int(f * (len(holds) - 1)))]
            summary.append(
                f"{stage}/{log.parent.name}: calls={len(trace)} "
                + " ".join(f"{m}(runs={r})={n}" for (m, r), n in sorted(modes.items()))
                + f" hold_us p50={q(0.5)} p99={q(0.99)} max={holds[-1]}"
            )
put("window/trace-summary.txt", "\n".join(summary) + "\n")
for sub, stage in ((SRC / "step1/served", "served/step1-base"), (SRC / "step3/served", "served/step3"),
                   (SRC / "final/served", "served/final")):
    if not sub.exists():
        continue
    put(f"{stage}/cells.txt", (sub / "cells.txt").read_text())
    for cell in sorted(p for p in sub.iterdir() if p.is_dir()):
        for f in sorted(cell.iterdir()):
            if f.is_file():
                put(f"{stage}/{cell.name}/{f.name}", f.read_text(errors="replace"))
for log in sorted((SRC / "gpu-test").glob("*.log")):
    if log.stem in ("red-main", "green", "green-q8", "red", "green-rebased", "green-q8-rebased"):
        put(f"placement-test/{log.name}", log.read_text())
for tool in ("cells.sh", "serve_cell.py", "col_shas.py", "served_summary.py", "run_test.sh",
             "step1-driver.sh", "step3-driver.sh", "staged-driver.sh", "final-driver.sh", "assemble.py"):
    put(f"tools/{tool}", (SRC / "scripts" / tool).read_text())
print("assembled", DST)
