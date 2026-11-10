#!/usr/bin/env python3
"""export_cells.py KIND SRC DEST [REQ]: the small receipts of this lane's cells, copied off the unit.

  bt     SRC is a served B/T campaign directory (NN-<ARM> cells): per cell ab.jsonl, guard.log
         and trace.txt (the server log's `[tick-spec]`, `[prefix-cache]`, spec K, draft load and
         cost calibration lines), then campaign.log and table.txt (specpub_tab.py on REQ).
  pairs  SRC holds pairs-*/ cell directories (unit_cell.sh's cmd.sh, rc and out/): one text file
         per case with the command, the exit code, the verdict line (specpub_pairs.py) and each
         turn's prompt, completion and cached tokens and spec counters, the warm boot's
         prefix-cache lines and every boot's published-entry digests.
  nsys   SRC holds nsys*/ cell directories: gap.txt, big.txt, pf.jsonl, guard.log, trace.txt.
  unit   SRC is the unit cells' directory: verdicts.txt, the digest lines of both arms and the
         test logs.
  hitgate  SRC holds hitgate-*/ cell directories: the gate's output (log), rc and its cmd.sh.

Paths are written from the home directory down. Full server logs, profiles, GPU telemetry and
memory samples stay on the machine."""
import json
import os
import pathlib
import re
import subprocess
import sys

kind, src, dst = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
tools = pathlib.Path(__file__).resolve().parent
home = os.path.expanduser("~")
KEEP = re.compile(r"\[(tick-spec|prefix-cache|prefix-cache-split-state|spec-k|spec-cost|mtp-draft|device-trim|spec-pool|spec)\]")


def clean(text):
    return re.sub(r"/home/[A-Za-z0-9_.-]+", "~", text.replace(home, "~"))


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean(text).rstrip("\n") + "\n")  # one newline at the end (git diff --check)


def trace(log):
    return "\n".join(l for l in log.read_text(errors="replace").splitlines() if KEEP.search(l)) + "\n"


if kind == "bt":
    for cell in sorted(p for p in src.iterdir() if p.is_dir() and re.match(r"\d\d-[A-Z]$", p.name)):
        for name in ("ab.jsonl", "guard.log"):
            if (cell / name).exists():
                put(dst / cell.name / name, (cell / name).read_text())
        if (cell / "server.log").exists():
            put(dst / cell.name / "trace.txt", trace(cell / "server.log"))
        elif (cell / "trace.txt").exists():
            put(dst / cell.name / "trace.txt", (cell / "trace.txt").read_text(errors="replace"))
    if (src / "campaign.log").exists():
        put(dst / "campaign.log", (src / "campaign.log").read_text())
    req = sys.argv[4:5]  # the request the table reads (g131k, g260k)
    r = subprocess.run([sys.executable, str(tools / "specpub_tab.py"), str(src), *req], capture_output=True, text=True)
    put(dst / "table.txt", r.stdout + r.stderr)
elif kind == "pairs":
    for case in sorted(src.glob("pairs-*")):
        lines = ["# cmd"] + (case / "cmd.sh").read_text().splitlines()[2:]
        lines.append(f"# rc {(case / 'rc').read_text().strip() if (case / 'rc').exists() else 'missing'}")
        lines.append("# verdict (specpub_pairs.py)")
        v = case / "out" / "verdicts.jsonl"
        lines += [l for l in v.read_text().splitlines() if l.strip()] if v.exists() else ["(no verdict)"]
        ev = case / "out" / case.name
        lines.append("# turns (prompt / completion / cached tokens, spec counters)")
        for f in sorted(ev.glob("*-t?.json")):
            u = json.loads(f.read_text()).get("usage", {})
            cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
            lines.append(f"{f.stem}: {u.get('prompt_tokens')} / {u.get('completion_tokens')} / {cached}, spec {json.dumps(u.get('spec'))}")
        for log in sorted(ev.glob("*-server.log")):
            lines.append(f"# {log.name}: prefix-cache and spec restore lines")
            lines += [l for l in log.read_text(errors="replace").splitlines()
                      if "[prefix-cache" in l or "spec restore" in l or "spec-reuse" in l or "spec-affinity" in l]
        if (ev / "binary.sha").exists():
            lines.append("# binary sha256")
            lines += (ev / "binary.sha").read_text().splitlines()
        put(dst / f"{case.name}.txt", "\n".join(lines) + "\n")
elif kind == "nsys":
    for cell in sorted(src.glob("nsys*")):
        for name in ("gap.txt", "big.txt", "pf.jsonl", "guard.log", "trace.txt"):
            if (cell / name).exists():
                put(dst / cell.name / name, (cell / name).read_text(errors="replace"))
elif kind == "unit":
    for f in sorted(src.glob("*.txt")) + sorted(src.glob("*.log")):
        put(dst / f.name, f.read_text(errors="replace"))
elif kind == "hitgate":
    for case in sorted(src.glob("hitgate-*")):
        lines = ["# cmd"] + (case / "cmd.sh").read_text().splitlines()[2:]
        lines.append(f"# rc {(case / 'rc').read_text().strip() if (case / 'rc').exists() else 'missing'}")
        lines.append("# output")
        lines += (case / "log").read_text(errors="replace").splitlines()
        put(dst / f"{case.name}.txt", "\n".join(lines) + "\n")
else:
    sys.exit(f"unknown kind {kind}")
print("exported", kind, dst)
