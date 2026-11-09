#!/usr/bin/env python3
"""The cold-long K=2 vs K=3 A/B on one RTX PRO 6000 Blackwell Max-Q of a shared 4-card box.

One boot (one arm) per lease through tools/memra-gpu-run, the card and box locks it takes, in the
order k3 k2 k2 k3 three times; every boot runs cold_long_ab.py pinned to the cores the box's owner
lent (taskset). Before every arm: if the yield file exists, no lease is taken until it is gone
(bounded wait); the card's power limit is left as it is. Runs one prompt set per invocation.

usage: ab_wrx.py OUT BINARY PROMPT_SET   (PROMPT_SET: long, or base for the run-1 set)
Environment: GPU_UUID, BOX_LOCK, YIELD_FILE, GPU_RUN (path to memra-gpu-run), CELL_MODEL, CORES.
"""
import os
import pathlib
import subprocess
import sys
import time

OUT = pathlib.Path(sys.argv[1])
BIN = sys.argv[2]
PROMPT_SET = sys.argv[3]
HERE = pathlib.Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
uuid = os.environ["GPU_UUID"]
yield_file = pathlib.Path(os.environ["YIELD_FILE"])
log = open(OUT / "driver.log", "a")


def say(msg):
    log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    log.flush()


for n, arm in enumerate("k3 k2 k2 k3 k3 k2 k2 k3 k3 k2 k2 k3".split(), 1):
    if (OUT / "YIELD").exists():
        say(f"YIELD file in OUT; stop before boot {n}")
        break
    t0 = time.time()
    while yield_file.exists():
        if time.time() - t0 > 3 * 3600:
            say(f"yield file still present after 3 h; stop before boot {n}")
            sys.exit(2)
        time.sleep(30)
    if time.time() - t0 > 1:
        say(f"waited {time.time() - t0:.0f} s for {yield_file} to clear before boot {n}")
    say(f"boot {n} {arm} start")
    env = dict(os.environ, CELL_PORT="18293", BOOT_START=str(n), TELEMETRY_GPU=uuid,
               PROMPT_SET="long" if PROMPT_SET == "long" else "")
    cmd = [os.environ["GPU_RUN"], "--gpus", uuid, "--box-lock", os.environ["BOX_LOCK"],
           "--receipt", str(OUT / f"lease-{n}"), "--wait", "3600", "--",
           "taskset", "-c", os.environ["CORES"], "python3", str(HERE / "cold_long_ab.py"), str(OUT), BIN, arm]
    with open(OUT / "boots.log", "a") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    say(f"boot {n} {arm} exit {rc}")
subprocess.run(["python3", str(HERE / "cold_long_ab.py"), str(OUT), BIN, "-"],
               env=dict(os.environ, PROMPT_SET="long" if PROMPT_SET == "long" else ""),
               stdout=open(OUT / "summary.log", "w"), stderr=subprocess.STDOUT)
say("AB_DONE")
