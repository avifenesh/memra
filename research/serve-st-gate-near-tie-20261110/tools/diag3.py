#!/usr/bin/env python3
"""memra#1181, third pass: the CLI teacher-forced on each server arm's ids. Its EXACTNESS line
counts the steps where the CLI's argmax, given the server's own prefix, is not the server's
token, so it covers the served stream after the parting step.

usage: diag3.py BIN_DIR CHECKPOINT_DIR OUT_DIR IDS_JSON     (run under a GPU lease)
"""
import json
import os
import pathlib
import subprocess
import sys

BIN, ST, OUT = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
IDS = json.load(open(sys.argv[4]))
OUT.mkdir(parents=True, exist_ok=True)
PROMPT = "What is the capital of France? Answer in one short sentence."
for arm in ("cli", "srv", "srv-b1fast", "srv-twprime", "srv-both"):
    ids = IDS[arm]
    tape = OUT / f"{arm}.ids"
    tape.write_text(" ".join(map(str, ids)))
    log = OUT / f"forced-on-{arm}.log"
    with open(log, "w") as f:
        rc = subprocess.run([str(BIN / "run-gen"), ST, "--prompt", PROMPT],
                            env={**os.environ, "MEMRA_CHAT": "1", "MEMRA_NGEN": str(len(ids)), "MEMRA_FORCE_TOKENS_FILE": str(tape)},
                            stdout=f, stderr=subprocess.STDOUT).returncode
    line = next((l for l in log.read_text(errors="replace").splitlines() if l.startswith("teacher-forced EXACTNESS")), "no EXACTNESS line")
    print(f"{arm}: exit {rc}; {line}", flush=True)
print("DONE", flush=True)
