#!/usr/bin/env python3
"""WP-B day 45 reader (DAY45.md 1.5): W1 to W4 and the readings from one card's boots.

usage: day45-read.py <card> <card root>
Boot names (fixed by the chains): <O1|O2>-<off|on>.
"""
import json, os, re, sys

card, root = sys.argv[1], sys.argv[2]
out = []


def say(line):
    out.append(line)
    print(line)


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def bdir(n):
    return os.path.join(root, "boots", n)


def have(n):
    return os.path.exists(os.path.join(bdir(n), "client.jsonl"))


def rows(n):
    return {json.loads(l)["tag"]: json.loads(l) for l in read(os.path.join(bdir(n), "client.jsonl")).splitlines() if l.strip()}


def log(n):
    return read(os.path.join(bdir(n), "server.log"))


def metric(n, key):
    try:
        return json.load(open(os.path.join(bdir(n), "metrics-end.json"))).get(key)
    except (OSError, ValueError):
        return None


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")
PRED = re.compile(r"\[admit-predict\] id=(\S+) tenant=\"([^\"]*)\" .*?verdict=(\S+) .*?booked_bytes=(\d+) booked_real=(\d+)")
BOOKED = re.compile(r"\[admit-book\] w-booked id=(\S+) model=\S+ bytes=(\d+)")
RELEASE = re.compile(r"\[admit-book\] w-release id=(\S+) model=\S+ bytes=(\d+)")

names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []
for n in names:
    if not have(n):
        say(f"DAY45 W4 card={card} boot={n} no client rows -> FAIL")
        continue
    t = log(n)
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for x in rows(n).values() if x["status"] == 503)
    ok = oom == 0 and crash == 0 and r503 == 0
    say(f"DAY45 W4 card={card} boot={n} oom_lines={oom} crash_lines={crash} r503={r503} -> {'PASS' if ok else 'FAIL'}")
    preds = PRED.findall(t)
    probe = [p for p in preds if "d45-probe" in p[1]]
    metric_booked = metric(n, "admission_booked_bytes")
    mb = sum(metric_booked.values()) if isinstance(metric_booked, dict) else metric_booked
    ok1 = bool(probe) and probe[-1][3] == "0" and probe[-1][4] == "0" and (mb in (0, None))
    say(f"DAY45 W1 card={card} boot={n} probe_booked={probe[-1][3] if probe else None} "
        f"probe_booked_real={probe[-1][4] if probe else None} metrics_booked={mb} -> {'PASS' if ok1 else 'FAIL'}")
    burst = [p for p in preds if "d45-burst" in p[1] or "d45-warm" in p[1]]
    peak_shadow = max((int(p[3]) for p in burst), default=0)
    peak_real = max((int(p[4]) for p in burst), default=0)
    rejects = sum(1 for p in burst if p[2] == "reject-kv")
    say(f"DAY45 READING card={card} boot={n} predict_lines={len(burst)} peak_booked_shadow={peak_shadow} "
        f"peak_booked_real={peak_real} shadow_reject_kv={rejects}")
    if n.endswith("-on"):
        booked = dict(BOOKED.findall(t))
        rel = RELEASE.findall(t)
        rel_ids = [i for i, _ in rel]
        twice = sorted({i for i in rel_ids if rel_ids.count(i) > 1})
        wrong = [i for i, b in rel if booked.get(i) != b]
        missing = sorted(set(booked) - set(rel_ids))
        ok2 = bool(booked) and not twice and not wrong and not missing
        say(f"DAY45 W2 card={card} boot={n} w_booked={len(booked)} w_release={len(rel)} twice={twice[:4]} "
            f"bytes_mismatch={wrong[:4]} never_released={missing[:4]} -> {'PASS' if ok2 else 'FAIL'}")
for order in ("O1", "O2"):
    a, b = f"{order}-off", f"{order}-on"
    if have(a) and have(b):
        ra, rb = rows(a), rows(b)
        tags = sorted(set(ra) & set(rb))
        differ = [t for t in tags if ra[t]["content_sha256"] != rb[t]["content_sha256"]]
        say(f"DAY45 W3 card={card} order={order} rows={len(tags)} differ={differ[:6]} -> "
            f"{'PASS' if tags and not differ else 'FAIL'}")
with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
