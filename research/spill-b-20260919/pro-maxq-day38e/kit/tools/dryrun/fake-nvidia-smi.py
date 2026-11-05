#!/usr/bin/env python3
"""Fake nvidia-smi for the kit's CPU dry run: one Max-Q card per index, compute apps = stub servers whose cwd is
the test kit. FAKE_SMI_STATE (a JSON file) may override memory.used, temperature.gpu and fan.speed."""
import json, os, sys, time

args = sys.argv[1:]
KIT = os.environ.get("FAKE_SMI_KIT", "")
state = {}
try:
    state = json.load(open(os.environ.get("FAKE_SMI_STATE", "/nonexistent")))
except (OSError, ValueError):
    pass


def opt(name):
    for i, x in enumerate(args):
        if x == name and i + 1 < len(args):
            return args[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def servers():
    out = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            if open(f"/proc/{p}/comm").read().strip() == "memra-server" and os.readlink(f"/proc/{p}/cwd") == KIT:
                out.append(p)
        except OSError:
            pass
    return out


idx = opt("-i") or "0"
fmt = opt("--format") or "csv"
noheader, nounits = "noheader" in fmt, "nounits" in fmt
if "-q" in args:
    print(f"==============NVSMI LOG==============\nGPU {idx}\n    Product Name : NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (fake)")
    sys.exit(0)
q = opt("--query-compute-apps")
if q:
    if not noheader:
        print(", ".join(q.split(",")))
    for p in servers():
        vals = {"pid": p, "process_name": "memra-server", "used_memory": "93000" + ("" if nounits else " MiB"),
                "gpu_bus_id": "00000000:C1:00.0"}
        print(", ".join(vals.get(f, "x") for f in q.split(",")))
    sys.exit(0)
q = opt("--query-gpu")
if not q:
    sys.exit(0)


def value(f):
    used = state.get("memory.used", 93000 if servers() else 3)
    v = {"timestamp": time.strftime("%Y/%m/%d %H:%M:%S.000"), "index": idx,
         "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", "memory.used": used,
         "memory.total": 97887, "uuid": "GPU-fake-0000", "pci.bus_id": "00000000:C1:00.0", "power.limit": "280.00",
         "enforced.power.limit": "280.00", "power.default_limit": "300.00", "power.max_limit": "300.00",
         "driver_version": "615.71.09", "temperature.gpu": state.get("temperature.gpu", 61),
         "fan.speed": state.get("fan.speed", 38), "power.draw": "22.50", "clocks.sm": 180, "clocks.mem": 405,
         "utilization.gpu": 0, "pstate": "P8", "clocks_event_reasons.active": "0x0000000000000400"}.get(f, "N/A")
    unit = {"memory.used": " MiB", "memory.total": " MiB", "power.limit": " W", "power.draw": " W"}.get(f, "")
    return f"{v}{'' if nounits else unit}"


fields = q.split(",")
lms = opt("-lms")
if not noheader:
    print(", ".join(fields))
while True:
    print(", ".join(value(f) for f in fields), flush=True)
    if not lms:
        break
    time.sleep(int(lms) / 1000)
    try:
        state = json.load(open(os.environ.get("FAKE_SMI_STATE", "/nonexistent")))
    except (OSError, ValueError):
        pass
