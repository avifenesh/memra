#!/usr/bin/env python3
"""r2_sections.py: memra#1266 review round 2. Regenerate the README sections that come from the
round-2 runs (sections/*.md) with the lane's table tools: agentic c = 2 and c = 4 on F5, the
need-pad cells (RTX 5090 and GB10), the riders cell, GB10 c = 2, the normal cells (correctness,
no timing), the bench and the gates. A run that is missing renders as "not run"."""
import pathlib
import subprocess
import sys

L = pathlib.Path.home() / ".cache/memra-1266"
S = L / "sections"
T = L / "tools"
REF = str(L / "ref-1221b-head-texts.jsonl")


def run(*args):
    r = subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)
    return (r.stdout or r.stderr).rstrip("\n")


def have(d):
    return d.is_dir() and any(p.is_dir() for p in d.iterdir())


def agentic(camp, *refs):
    d = L / camp
    return run(T / "agentic_table.py", "--no-timing", d, *refs) if have(d) else "(not run)"


out = {}
out["ag35"] = agentic("runs/ag35", REF)
if out["ag35"] == "(not run)":
    out["ag35"] = ("F6, F6 forced cap and M3 at c = 4 run in the RTX 5090's first round-2 block; this section is "
                   "updated when they finish (round 0's main boot at c = 4 answered 29 of 29, below).")
out["pad"] = agentic("runs/pad", REF, L / "runs/ag3/01-M")
out["gpad"] = agentic("gb10-runs/gpad", L / "gb10-runs/gag5/01-M")
out["gag5"] = agentic("gb10-runs/gag5")
d = L / "runs/riders"
out["riders"] = run(T / "riders_table.py", d) if have(d) else "(not run)"
d = L / "gb10-runs/griders2"
out["griders"] = run(T / "riders_table.py", d) if have(d) else "(not run)"
for camp, key in (("runs/nc5", "nc5"), ("gb10-runs/gnc5", "gnc5")):
    d = L / camp
    out[key] = run(T / "normal_table.py", "--no-timing", d) if have(d) else "(not run)"
bench = L / "runs/bench"
if (bench / "summary.txt").exists():
    out["bench_raw"] = (bench / "summary.txt").read_text().rstrip("\n")
else:
    out["bench_raw"] = "(not run)"
gates = L / "runs/gates"
lines = []
for g in sorted(gates.glob("*.log")) if gates.is_dir() else []:
    tail = [x for x in g.read_text().splitlines() if "PASS" in x or "FAIL" in x][-2:]
    lines.append(f"- `{g.stem}`: " + (" / ".join(tail) if tail else "no verdict line"))
out["gates_raw"] = "\n".join(lines) or "(not run)"
for k, v in out.items():
    (S / f"r2-{k}.md").write_text(v + "\n")
print("wrote", ", ".join(f"r2-{k}.md" for k in out))
