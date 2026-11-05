#!/usr/bin/env python3
"""Laptop self-test stand-in for nvidia-smi: four Max-Q cards, no compute apps, CSV rows; -lms loops until killed.
Logs every argv to $FAKE_SMI_LOG so the test can assert that each query carried -i <card>."""
import os, sys, time
argv = sys.argv[1:]
log = os.environ.get("FAKE_SMI_LOG")
if log:
    with open(log, "a") as f:
        f.write(" ".join(argv) + "\n")
if argv and argv[0] in ("topo", "-L", "-q"):
    print("fake"); sys.exit(0)
idx = None
if "-i" in argv:
    idx = argv[argv.index("-i") + 1]
q = next((a for a in argv if a.startswith("--query-")), "")
fmt = next((a for a in argv if a.startswith("--format=")), "--format=csv")
noheader = "noheader" in fmt
nounits = "nounits" in fmt
cards = [str(i) for i in range(4)] if idx is None else [idx]
temp = int(os.environ.get("FAKE_TEMP", "70")); fan = int(os.environ.get("FAKE_FAN", "50"))
vals = {"index": None, "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "memory.used": "3 MiB",
        "memory.total": "97887 MiB", "memory.free": "97000 MiB", "power.limit": "280.00 W", "power.max_limit": "300.00 W",
        "power.default_limit": "300.00 W", "enforced.power.limit": "280.00 W", "temperature.gpu": str(temp),
        "fan.speed": f"{fan} %", "driver_version": "615.71.09", "timestamp": None, "pstate": "P8", "clocks.sm": "180 MHz",
        "clocks.mem": "405 MHz", "power.draw": "24.00 W", "utilization.gpu": "0 %", "clocks_event_reasons.active": "0x0",
        "pcie.link.gen.current": "5", "pcie.link.width.current": "16"}
def row(card):
    out = []
    for k in q.split("=", 1)[1].split(","):
        v = vals.get(k, "N/A")
        if k == "index": v = card
        if k == "timestamp": v = time.strftime("%Y/%m/%d %H:%M:%S.000")
        if nounits and v.split(" ")[-1] in ("MiB", "W", "%", "MHz"): v = v.split(" ")[0]
        out.append(v)
    return ", ".join(out)
if q.startswith("--query-compute-apps"):
    if not noheader: print(q.split("=", 1)[1].replace(",", ", "))
    sys.exit(0)
if q.startswith("--query-gpu"):
    if not noheader: print(q.split("=", 1)[1].replace(",", ", "))
    if "-lms" in argv:
        while True:
            for c in cards: print(row(c), flush=True)
            time.sleep(0.25)
    for c in cards: print(row(c))
    sys.exit(0)
print("fake nvidia-smi: unsupported " + " ".join(argv), file=sys.stderr); sys.exit(6)
