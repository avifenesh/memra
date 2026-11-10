#!/usr/bin/env python3
"""Assemble the drafting lane's receipts on the run box into one directory for the repo: the gate and measurement
outputs (JSON lines, logs, per-job results, small id and timing arrays; not the tensor dumps), the oracle logs and tape
manifests, the band comparison, and one row per GPU lease. The box's working directory becomes $LANE and card UUIDs are
dropped (a lease row names the card by its index and power cap).

usage: export_receipts.py LANE_DIR OUT_DIR"""
import csv, json, os, re, sys, time

lane, out = sys.argv[1].rstrip("/"), sys.argv[2]
spec = os.path.join(lane, "spec")
UUID = re.compile(rb"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
KEEP_EXT = (".json", ".jsonl", ".log", ".tsv", ".md", ".txt", ".i32", ".csv", ".env")
MAX = 2 << 20


def scrub(b):
    return UUID.sub(b"GPU-<uuid>", b.replace(lane.encode(), b"$LANE"))


def put(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f:
        b = f.read()
    with open(dst, "wb") as f:
        f.write(b if src.endswith((".i32", ".f32")) else scrub(b))


n = 0
for sub in ("out", "gate", "logs"):
    root = os.path.join(spec, sub)
    for d, _, files in os.walk(root):
        for fn in files:
            p = os.path.join(d, fn)
            small = os.path.getsize(p) <= MAX
            if fn == "plain_step_ms.f32" or (fn.endswith(KEEP_EXT) and small):
                put(p, os.path.join(out, sub, os.path.relpath(p, root)))
                n += 1
for tapes in ("tapes", "tapes-cpu"):
    m = os.path.join(spec, tapes, "manifest.json")
    if os.path.exists(m):
        put(m, os.path.join(out, tapes, "manifest.json"))
for raw in ("tapes-raw", "tapes-cpu-raw"):
    root = os.path.join(spec, raw)
    if os.path.isdir(root):
        for case in sorted(os.listdir(root)):
            for fn in ("meta.json", "out_ids.i32", "conf.f32", "gen.i32", "prompt.i32", "tok_in.i32", "pos.i32"):
                p = os.path.join(root, case, fn)
                if os.path.exists(p):
                    put(p, os.path.join(out, raw, case, fn))
                    n += 1
rows = []
leases = os.path.join(lane, "leases")
for name in sorted(os.listdir(leases)):
    if not name.startswith("spec-"):
        continue
    lj = os.path.join(leases, name, "lease.json")
    if not os.path.exists(lj):
        rows.append([name, "", "", "", "", "not granted"])
        continue
    j = json.load(open(lj))
    cards = ",".join(str(d["index"]) for d in j.get("devices", []))
    caps = ""
    cc = os.path.join(leases, name, "cards.csv")
    if os.path.exists(cc):
        idx = {str(d["index"]) for d in j.get("devices", [])}
        for r in csv.reader(open(cc), skipinitialspace=True):
            if r and r[0] in idx:
                caps = f"{r[2]}, {r[3]} C, fan {r[4]}"
    t0, t1 = j.get("started_unix"), j.get("finished_unix")
    rows.append([name, cards, caps, time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t0)) + "Z" if t0 else "",
                 f"{t1 - t0:.0f}" if t0 and t1 else "", f"{j.get('exit_code')}" + (" (timed out)" if j.get("timed_out") else "")])
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, "leases.tsv"), "w") as f:
    f.write("lease\tcard\tcap, temperature, fan at start\tstarted (UTC)\tseconds\texit\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"{n} files, {len(rows)} leases -> {out}")
