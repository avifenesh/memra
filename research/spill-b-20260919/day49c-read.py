#!/usr/bin/env python3
"""DAY49 addendum C reader: the i-vmm cell (the gate's arm i under MEMRA_KV_ALLOCATOR=vmm).

usage: day49c-read.py <card> <green receipts> <red receipts>
Each receipts dir is one health-fault-gate run (HFG_ARMS=i): VERDICTS.txt and i-ctrl/, i/, i-red/ server logs.
Green (the fix): the gate's i and i-red verdicts PASS, the door ON in each boot, and one `[kv-vmm] reap (batch-oom)`
line per `retrying` line, printed before it. Red (02dbdfa40): the same gate terms and no reap line.
"""
import os
import re
import sys

card, green, red = sys.argv[1], sys.argv[2], sys.argv[3]
RETRY = re.compile(r"batch OOM: .* retrying the same batched step once")
REAP = "[kv-vmm] reap (batch-oom)"
DOOR_ON = "[kv-vmm] door=ON"


def verdicts(root):
    out = {}
    try:
        with open(os.path.join(root, "VERDICTS.txt")) as f:
            for line in f:
                m = re.match(r"HFG \((i|i-red|i-ctrl)\) ", line)
                if m:
                    out[m.group(1)] = line.strip()
    except OSError:
        pass
    return out


def arm(root, label):
    path = os.path.join(root, label, "server.log")
    try:
        with open(path, errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    door = sum(DOOR_ON in l for l in lines)
    reaps = [i for i, l in enumerate(lines) if REAP in l]
    retries = [i for i, l in enumerate(lines) if RETRY.search(l)]
    # Each retry line has its own reap line after the previous retry and before it.
    paired = len(reaps) == len(retries) and all(
        r < t and (k == 0 or r > retries[k - 1]) for k, (r, t) in enumerate(zip(reaps, retries))
    )
    return dict(door=door, reaps=len(reaps), retries=len(retries), paired=paired,
                reap_text=lines[reaps[0]].strip()[:160] if reaps else "")


def read(root, role):
    v = verdicts(root)
    gate_i = v.get("i", "").endswith("-> PASS")
    gate_red = v.get("i-red", "").endswith("-> PASS")
    ok = gate_i and gate_red
    parts = []
    for label in ("i-ctrl", "i", "i-red"):
        a = arm(root, label)
        if a is None:
            parts.append(f"{label}=[no log]")
            ok = False
            continue
        parts.append(f"{label}=[door_on={a['door']} reap_lines={a['reaps']} retry_lines={a['retries']} paired={a['paired']}]")
        if a["door"] < 1:
            ok = False
        if label == "i-ctrl":
            ok = ok and a["reaps"] == 0 and a["retries"] == 0
        elif role == "green":
            ok = ok and a["retries"] == 1 and a["reaps"] == 1 and a["paired"]
        else:
            ok = ok and a["retries"] == 1 and a["reaps"] == 0
    expect = "one reap per retry, before it" if role == "green" else "no reap line (the defect as it reads)"
    print(f"DAY49C I-VMM card={card} role={role} gate_i={'PASS' if gate_i else 'FAIL'} "
          f"gate_i_red={'PASS' if gate_red else 'FAIL'} {' '.join(parts)} expect={expect} -> "
          f"{'PASS' if ok else 'FAIL'}")
    for label in ("i", "i-red"):
        if label in v:
            print(f"  {role} {v[label]}")
        a = arm(root, label)
        if a and a["reap_text"]:
            print(f"  {role} {label} first reap line: {a['reap_text']}")


read(green, "green")
read(red, "red")
