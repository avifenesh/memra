#!/usr/bin/env python3
"""DAY49 addendum D reader.

usage: day49d-read.py serve <card> <receipt root>     the serving shape (boots/<name>/ from day49d-run.sh)
       day49d-read.py vmm <card> <green> <red>        arm j under MEMRA_KV_ALLOCATOR=vmm (two gate receipt dirs)

serve: a boot counts only if its server log carries a `fired: this batched decode chunk` line naming at least two
sessions. On `on` and `on-plain` every request ends 200 with no error; `off` and `off-plain` are the before reading.
A route whose boots never reach such a chunk reads `not reached`, with the fired counts quoted.
vmm: green passes the gate's j and j-red and prints one `[kv-vmm] reap (batch-oom)` line per `retrying` line, before
it; red passes the same gate terms and prints no reap line.
"""
import collections
import glob
import json
import os
import re
import sys

BATCH = re.compile(r"MEMRA_STEP_OOM_FAULT fired: this batched decode chunk reports a synthetic CUDA OOM \((\d+) session")
FIRED = "MEMRA_STEP_OOM_FAULT fired:"
RETRY = re.compile(r"batch OOM: (\d+) sessions untouched .* retrying the same batched step once")
REAP = "[kv-vmm] reap (batch-oom)"


def lines(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def serve(card, root):
    by_route = collections.defaultdict(list)
    for arm_txt in sorted(glob.glob(os.path.join(root, "boots", "*.arm.txt"))):
        name = os.path.basename(arm_txt)[:-len(".arm.txt")]
        meta = dict(l.split("=", 1) for l in lines(arm_txt) if "=" in l)
        arm = meta.get("arm", "?")
        log = lines(os.path.join(root, "boots", name, "server.log"))
        batch = [int(m.group(1)) for l in log for m in [BATCH.search(l)] if m]
        other = sum(FIRED in l and not BATCH.search(l) for l in log)
        retry = sum(bool(RETRY.search(l)) for l in log)
        ok = sum("batch OOM: retried (ok)" in l for l in log)
        failed = sum("batch OOM: retry failed" in l for l in log)
        rows = [json.loads(l) for l in lines(os.path.join(root, "boots", name, "client.jsonl")) if l.strip()]
        status = collections.Counter(r.get("status") for r in rows)
        errors = sum(1 for r in rows if r.get("error"))
        reached = any(k >= 2 for k in batch)
        route = "plain" if arm.endswith("-plain") else "spec-default"
        head = (f"DAY49D SERVE card={card} boot={name} arm={arm} route={route} batch_fired={batch} other_fired={other} "
                f"retry_lines={retry} retried_ok={ok} retry_failed={failed} rows={len(rows)} "
                f"status={dict(sorted(status.items(), key=str))} error_rows={errors}")
        if not reached:
            print(f"{head} -> NOT REACHED (no batched chunk of >= 2 sessions took the fault)")
        elif arm.startswith("on"):
            good = rows and status.get(200, 0) == len(rows) and errors == 0 and ok == retry == 1
            print(f"{head} -> {'PASS' if good else 'FAIL'}")
        else:
            print(f"{head} -> READING (the before)")
        by_route[route].append(reached)
    for route, seen in sorted(by_route.items()):
        print(f"DAY49D SERVE-ROUTE card={card} route={route} boots={len(seen)} reached={sum(seen)}"
              + ("" if all(seen) else " (a boot that did not reach the chunk is not a reading)"))


def verdicts(root):
    out = {}
    for l in lines(os.path.join(root, "VERDICTS.txt")):
        m = re.match(r"HFG \((j|j-red|j-ctrl)\) ", l)
        if m:
            out[m.group(1)] = l.strip()
    return out


def arm(root, label):
    log = lines(os.path.join(root, label, "server.log"))
    if not log:
        return None
    door = sum("[kv-vmm] door=ON" in l for l in log)
    reaps = [i for i, l in enumerate(log) if REAP in l]
    retries = [i for i, l in enumerate(log) if RETRY.search(l)]
    paired = len(reaps) == len(retries) and all(
        r < t and (k == 0 or r > retries[k - 1]) for k, (r, t) in enumerate(zip(reaps, retries)))
    return dict(door=door, reaps=len(reaps), retries=len(retries), paired=paired,
                reap_text=log[reaps[0]].strip()[:160] if reaps else "")


def vmm(card, green, red):
    for root, role in ((green, "green"), (red, "red")):
        v = verdicts(root)
        gate_j = v.get("j", "").endswith("-> PASS")
        gate_red = v.get("j-red", "").endswith("-> PASS")
        ok = gate_j and gate_red
        parts = []
        for label in ("j-ctrl", "j", "j-red"):
            a = arm(root, label)
            if a is None:
                parts.append(f"{label}=[no log]")
                ok = False
                continue
            parts.append(f"{label}=[door_on={a['door']} reap_lines={a['reaps']} retry_lines={a['retries']} "
                         f"paired={a['paired']}]")
            ok = ok and a["door"] >= 1
            if label == "j-ctrl":
                ok = ok and a["reaps"] == 0 and a["retries"] == 0
            elif role == "green":
                ok = ok and a["retries"] == 1 and a["reaps"] == 1 and a["paired"]
            else:
                ok = ok and a["retries"] == 1 and a["reaps"] == 0
        expect = "one reap per retry, before it" if role == "green" else "no reap line (the defect as it reads)"
        print(f"DAY49D J-VMM card={card} role={role} gate_j={'PASS' if gate_j else 'FAIL'} "
              f"gate_j_red={'PASS' if gate_red else 'FAIL'} {' '.join(parts)} expect={expect} -> "
              f"{'PASS' if ok else 'FAIL'}")
        for label in ("j", "j-red"):
            if label in v:
                print(f"  {role} {v[label]}")


if sys.argv[1] == "serve":
    serve(sys.argv[2], sys.argv[3])
elif sys.argv[1] == "vmm":
    vmm(sys.argv[2], sys.argv[3], sys.argv[4])
else:
    sys.exit(__doc__)
