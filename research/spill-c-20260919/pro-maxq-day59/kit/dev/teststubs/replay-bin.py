#!/usr/bin/env python3
"""Laptop plumbing-test stub for run-gen-c60 / run-spec-c60: replays the Workstation receipt's log for the label the
cell just marked (the newest ev/marks.tsv's last line), with its exit code. Records argv and the card env it saw.
Never touches a GPU."""
import glob
import os
import sys
import time

R, WS = os.environ["D40_R"], os.environ["MAXQ_TEST_WS"]
marks = max(glob.glob(f"{R}/*/ev/marks.tsv"), key=os.path.getmtime)
cell = marks.split("/")[-3]
label = open(marks).read().splitlines()[-1].split("\t")[1].rsplit(" ", 1)[0]
env = {k: os.environ.get(k) for k in ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "MEMRA_MOE_PREFETCH",
                                      "MEMRA_MOE_RESIDENT", "MEMRA_MOE_SLOTS", "MEMRA_NGEN", "TMPDIR")}
with open(os.environ["MAXQ_STUB_LOG"], "a") as f:
    f.write(f"{os.path.basename(sys.argv[0])} cell={cell} label={label} argv={sys.argv[1:]} env={env}\n")
for line in open(f"{WS}/{cell}/ev/{label}.log", errors="replace"):
    sys.stdout.write(line.split("\t", 1)[1] if "\t" in line else line)
sys.stdout.flush()
time.sleep(0.2)
sys.exit(int(open(f"{WS}/{cell}/ev/{label}.exit").read()))
