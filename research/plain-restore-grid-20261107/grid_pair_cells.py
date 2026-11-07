#!/usr/bin/env python3
"""Warm-vs-cold pair cells for the restore grid law (RTX 5090, Qwen3.8-27B NVFP4+Q5_K + DFlash2).

Each case runs dspark-pair-cell.sh (one warm boot serving three chat turns in order, then one cold
boot per turn 2 and turn 3) with a binary and the warm boot's switches, then judges:
  bytes    turn 2 and turn 3 warm answers byte-equal to their cold twins (the correctness verdict)
  restores the warm boot's DSPARK restores, strict-prefix ones counted (reported, not judged)
  offgrid  warm prefix-cache hits whose restored length is off the 32-token GDN grid on a strict
           prefix (the mechanism the grid law refuses; reported)

usage: grid_pair_cells.py OUT CASE...   with CASE = name=BINARY|WARM_ENV
  WARM_ENV is a space-separated list of VAR=VALUE for the warm boot (empty = the device default).
Writes OUT/<name>/ (the cell's raw files) and OUT/verdicts.jsonl; exit 1 if any case's bytes differ.
"""
import json
import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = pathlib.Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
MODEL = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
DRAFT = "/data/ai-ml/models/q38-gguf/Qwen3.8-27B-DFlash2-memra"
GRID = 32


def content(p):
    m = json.loads(p.read_text())["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")


def judge(ev):
    turns = {}
    for t in (2, 3):
        w, c = content(ev / f"warm-t{t}.json"), content(ev / f"cold-t{t}.json")
        # lengths count the answer's characters, without the reasoning/content NUL separator
        turns[t] = {"equal": w == c and len(w) > 1, "warm_len": len(w) - 1, "cold_len": len(c) - 1}
    t1 = content(ev / "warm-t1.json")
    import hashlib
    turns[1] = {"warm_sha": hashlib.sha256(t1.encode()).hexdigest()[:10], "warm_len": len(t1) - 1}
    log = (ev / "warm-server.log").read_text(errors="replace")
    restores = [tuple(map(int, m.groups())) for m in re.finditer(r"DSPARK restore: (\d+) of (\d+) prompt tokens", log)]
    hits = [tuple(map(int, m.groups())) for m in re.finditer(r"\[prefix-cache\] hit: (\d+) of (\d+) prompt tokens", log)]
    offgrid = [h for h in hits if h[0] < h[1] and h[0] % GRID]
    return {"turns": turns, "bytes": all(v["equal"] for k, v in turns.items() if k != 1),
            "dspark_restores": restores, "strict_prefix_restores": sum(r[0] < r[1] for r in restores),
            "plain_hits": hits, "offgrid_strict_hits": offgrid,
            "grid_refusals": len(re.findall(r"off the prime grid|restore grid law", log))}


def main():
    failed = False
    for spec in sys.argv[2:]:
        name, rest = spec.split("=", 1)
        binary, warm_env = rest.split("|", 1)
        ev = OUT / name
        env = dict(os.environ, EV=str(ev), BIN=binary, MODEL=MODEL, DRAFT=DRAFT,
                   WARM_ENV=warm_env or "MEMRA_COMPAT=openai")
        with open(OUT / f"{name}.log", "w") as log:
            rc = subprocess.run(["bash", str(HERE / "dspark-pair-cell.sh")], env=env,
                                stdout=log, stderr=subprocess.STDOUT).returncode
        verdict = {"case": name, "binary": binary, "warm_env": warm_env or "(device default)", "exit": rc}
        if rc == 0:
            verdict.update(judge(ev))
        else:
            verdict["bytes"] = False
        failed |= not verdict["bytes"]
        print(json.dumps(verdict), flush=True)
        with open(OUT / "verdicts.jsonl", "a") as f:
            f.write(json.dumps(verdict) + "\n")
    sys.exit(1 if failed else 0)


main()
