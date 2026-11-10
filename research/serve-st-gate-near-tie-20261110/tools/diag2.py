#!/usr/bin/env python3
"""memra#1181, second pass: the CLI program's own top-1/top-2 margin at every one of its 64
steps (so the margin at a flip can be placed in that distribution), and, at the steps where a
server arm left the CLI stream, the logits of the CLI's top ids under the tile prime and under
the tokenwise prime (eager decode in both, teacher-forced on the CLI's ids).

usage: diag2.py BIN_DIR CHECKPOINT_DIR OUT_DIR IDS_JSON STEP...     (run under a GPU lease)
"""
import json
import os
import pathlib
import re
import struct
import subprocess
import sys

BIN, ST, OUT = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
CLI = json.load(open(sys.argv[4]))["cli"]
STEPS = [int(s) for s in sys.argv[5:]]
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "What is the capital of France? Answer in one short sentence."
NGEN = len(CLI)
tape = OUT / "cli.ids"
tape.write_text(" ".join(map(str, CLI)))


def run(name, cmd, env):
    with open(OUT / f"{name}.log", "w") as f:
        rc = subprocess.run(cmd, env={**os.environ, **env}, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        print(f"{name}: exit {rc}", flush=True)
    return (OUT / f"{name}.log").read_text(errors="replace")


rows = []
for g in range(NGEN):
    f32 = OUT / "row.f32"
    if f32.exists():
        f32.unlink()
    run("cli-forced", [str(BIN / "run-gen"), ST, "--prompt", PROMPT],
        {"MEMRA_CHAT": "1", "MEMRA_NGEN": str(NGEN), "MEMRA_FORCE_TOKENS_FILE": str(tape),
         "MEMRA_FORCE_LOGITS_AT": str(g), "MEMRA_FORCE_LOGITS_FILE": str(f32)})
    raw = f32.read_bytes()
    row = struct.unpack(f"<{len(raw) // 4}f", raw)
    top = sorted(range(len(row)), key=lambda i: -row[i])[:8]
    rows.append({"step": g, "cli_id": CLI[g], "top": [[i, round(row[i], 6)] for i in top],
                 "margin": round(row[top[0]] - row[top[1]], 6), "argmax_is_cli_id": top[0] == CLI[g]})
    print(f"step {g}: id {CLI[g]} margin {rows[-1]['margin']:.4f} top2 {top[0]} {top[1]}", flush=True)
(OUT / "cli-margins.json").write_text(json.dumps(rows))
(OUT / "row.f32").unlink()
margins = sorted(r["margin"] for r in rows)
n = len(margins)
print(f"cli margins: n {n}, min {margins[0]:.4f}, p10 {margins[int(0.1 * (n - 1))]:.4f}, "
      f"median {margins[n // 2]:.4f}, max {margins[-1]:.4f}; argmax equals the emitted id at "
      f"{sum(r['argmax_is_cli_id'] for r in rows)} of {n} steps", flush=True)

shifts = []
for g in STEPS:
    watch = [str(i) for i, _ in rows[g]["top"]]
    got = {}
    for name, extra in (("tile", {}), ("tokenwise", {"MEMRA_PRIME_TOKENWISE": "1"})):
        text = run(f"bisect-{name}-{g}", [str(BIN / "verify_shape_bisect"), ST, *map(str, CLI[:g - 1]), "--", str(CLI[g - 1]), *watch],
                   {"MEMRA_PROMPT": PROMPT, "MEMRA_CHAT": "1", **extra})
        line = next((l for l in text.splitlines() if l.startswith("production B=1")), "")
        got[name] = {int(k): float(v) for k, v in re.findall(r"l\[(\d+)\]=(-?[0-9.]+)", line)}
        print(f"step {g} {name} prime: {line[:260]}", flush=True)
    for i in map(int, watch):
        if i in got["tile"] and i in got["tokenwise"]:
            shifts.append({"step": g, "id": i, "tokenwise": got["tokenwise"][i], "tile": got["tile"][i],
                           "shift": round(got["tile"][i] - got["tokenwise"][i], 6)})
    print(f"step {g} shifts (tile minus tokenwise) over the CLI's top 8: "
          + ", ".join(f"{s['id']}:{s['shift']:+.3f}" for s in shifts if s["step"] == g), flush=True)
(OUT / "prime-shifts.json").write_text(json.dumps(shifts))
print("DONE", flush=True)
