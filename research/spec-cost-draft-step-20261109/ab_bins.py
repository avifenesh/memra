#!/usr/bin/env python3
"""Base against fix binaries at the device default on one RTX PRO 6000 Max-Q (memra#1093).

The cost-model fail-before/pass-after: the base prices every draft length with one K=3 drafting
cost and picks K=3 on cold long prompts; the fix prices each length's own drafting. One boot per
tools/memra-gpu-run lease (card lock and box lock), pinned to the lent cores, in the order given;
before every boot the yield file is checked and the run stops while it exists.

The second check (after the realized-rate fix) runs three arms: `base`, `rung3` (the flip binary
with MEMRA_SPEC_K_COLD_LONG=3, so the model prices K=1..3 as on the base) and `flip` (the flip
binary at its default, cold-long K=2).

usage: ab_bins.py OUT PROMPT_SET ORDER   (ORDER: comma list of base/fix/rung3/flip; PROMPT_SET
long or base)
Environment: GPU_UUID, BOX_LOCK, YIELD_FILE, GPU_RUN, CELL_MODEL, CORES, BIN_BASE, BIN_FIX,
BIN_FLIP, BOOT_OFFSET (first boot number, default 1), DEADLINE (HH:MM; no boot starts after it),
CELL_SCRIPT (the per-boot script beside this file, default cold_long_ab.py; c4_ab.py for the
four-at-once cell).
"""
import os
import pathlib
import subprocess
import sys
import time

OUT = pathlib.Path(sys.argv[1])
PROMPT_SET = sys.argv[2]
ORDER = sys.argv[3].split(",")
HERE = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
bins = {"base": os.environ["BIN_BASE"], "fix": os.environ.get("BIN_FIX", ""),
        "rung3": os.environ.get("BIN_FLIP", ""), "flip": os.environ.get("BIN_FLIP", "")}
deadline = os.environ.get("DEADLINE", "")  # HH:MM, local time: no boot starts after it
yield_file = pathlib.Path(os.environ["YIELD_FILE"])
log = open(OUT / "driver.log", "a")


def say(msg):
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    log.flush()


for i, arm in enumerate(ORDER):
    n = int(os.environ.get("BOOT_OFFSET", "1")) + i
    if deadline and time.strftime("%H:%M") >= deadline:
        say(f"deadline {deadline} reached; stop before boot {n}")
        sys.exit(4)
    if yield_file.exists() or (OUT / "YIELD").exists():
        say(f"yield file present; stop before boot {n}")
        sys.exit(3)
    say(f"boot {n} {arm} start")
    env = dict(os.environ, CELL_PORT="18293", BOOT_START=str(n), TELEMETRY_GPU=os.environ["GPU_UUID"],
               PROMPT_SET="long" if PROMPT_SET == "long" else "")
    cmd = [os.environ["GPU_RUN"], "--gpus", os.environ["GPU_UUID"], "--box-lock", os.environ["BOX_LOCK"],
           "--receipt", str(OUT / f"lease-{n}"), "--wait", "600", "--",
           "taskset", "-c", os.environ["CORES"], "python3",
           str(HERE / os.environ.get("CELL_SCRIPT", "cold_long_ab.py")), str(OUT), bins[arm], arm]
    with open(OUT / "boots.log", "a") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"boot {n} {arm} exit {rc}")
say("BINS_DONE")
