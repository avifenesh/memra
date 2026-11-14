#!/usr/bin/env python3
"""arm_sha.py REV [--deep] [--lib]: the sha256 list of one cell arm's engine files, as build1307.sh
writes it beside the arm's binary (arm.sha): REV's spec.rs, dflash.rs, unified_memory.rs and
dsv4_gpu.rs (and lib.rs with --lib, for the confirming arms) with the [diag1307] lines of
diag1307.py applied (--deep for arm D). Run from a checkout of the repo; nothing in it is changed.

Compare the output with gb10/arms/ARM.sha to check which commit an arm's cells ran on:
  B, Bc: main at the arm's base;  T, D, Tc: the branch's commit the README names for them."""
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

rev = sys.argv[1]
deep = "--deep" in sys.argv[2:]
files = ["spec.rs", "dflash.rs", "unified_memory.rs", "dsv4_gpu.rs"] + (["lib.rs"] if "--lib" in sys.argv[2:] else [])
tools = Path(__file__).resolve().parent
top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
with tempfile.TemporaryDirectory() as tmp:
    src = Path(tmp) / "crates/memra-engine/src"
    src.mkdir(parents=True)
    for f in files:
        blob = subprocess.run(["git", "-C", top, "show", f"{rev}:crates/memra-engine/src/{f}"],
                              capture_output=True, check=True).stdout
        (src / f).write_bytes(blob)
    subprocess.run([sys.executable, str(tools / "diag1307.py"), tmp] + (["--deep"] if deep else []),
                   capture_output=True, check=True)
    for f in files:
        print(f"{hashlib.sha256((src / f).read_bytes()).hexdigest()}  {f}")
