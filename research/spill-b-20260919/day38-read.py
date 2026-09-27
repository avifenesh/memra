#!/usr/bin/env python3
"""WP-B day 38 reader (DAY38.md 1.4 to 1.6, addenda A and B): P1 to P4 and the interaction pair from one card's boots.

usage: day38-read.py <card> <card root>
Boot names (fixed by the chain): main-<O1|O2>-<off|on>, fault-batch, fault-nobatch, fault-nobatch-red, vmm-off, vmm-on.
"""
import glob, json, os, re, statistics, sys

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


def boot_dir(name):
    return os.path.join(root, "boots", name)


def have(name):
    return os.path.exists(os.path.join(boot_dir(name), "client.jsonl"))


def rows(name):
    r = {}
    for l in read(os.path.join(boot_dir(name), "client.jsonl")).splitlines():
        if l.strip():
            x = json.loads(l)
            r[x["tag"]] = x
    return r


def log(name):
    return read(os.path.join(boot_dir(name), "server.log"))


def count(name, pat):
    return len(re.findall(pat, log(name)))


def metric(name, which, key):
    try:
        return json.load(open(os.path.join(boot_dir(name), which))).get(key)
    except (OSError, ValueError):
        return None


FAULT = r"panicked at|CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access|\bFATAL\b"


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def resumed_tags(r):
    return [t for t, x in r.items() if not x["cold"] and x["turn"] >= 2]


# ---- P1 ---------------------------------------------------------------------------------------------------------
p1 = True
for order in ("O1", "O2"):
    off, on = f"main-{order}-off", f"main-{order}-on"
    if not (have(off) and have(on)):
        continue
    ro, rn = rows(off), rows(on)
    tags = sorted(set(ro) | set(rn))
    door_equal = sum(1 for t in tags if t in ro and t in rn and ro[t]["content_sha256"] == rn[t]["content_sha256"])
    door_differ = [t for t in tags if t in ro and t in rn and ro[t]["content_sha256"] != rn[t]["content_sha256"]]
    cold_differ = []
    for name, r in ((off, ro), (on, rn)):
        for t in resumed_tags(r):
            twin = r.get(t + "-cold")
            if twin is None or twin["content_sha256"] != r[t]["content_sha256"]:
                cold_differ.append(f"{name}:{t}")
    non200 = [f"{n}:{t}" for n, r in ((off, ro), (on, rn)) for t, x in r.items() if x["status"] != 200]
    x_resumed = sum(1 for t, x in rn.items() if x["shape"] == "X" and not x["cold"] and x["turn"] >= 2)
    a_resumed = {n: sum(1 for t, x in r.items() if x["shape"] == "A" and not x["cold"] and x["turn"] == 2)
                 for n, r in ((off, ro), (on, rn))}
    grow = count(on, r"\[kv-reuse\] park-compact grow: ")
    parks_on = count(on, r"\[kv-reuse\] park-compact: ")
    failed = count(on, r"park-compact (?:grow )?failed")
    rewound = {n: count(n, r"plain-affinity: rewound to") for n in (off, on)}
    hits = {}
    for n in (off, on):
        a, b = metric(n, "metrics-ready.json", "continuation_pool_hits"), metric(n, "metrics-end.json", "continuation_pool_hits")
        hits[n] = None if a is None or b is None else b - a
    faults = count(off, FAULT) + count(on, FAULT)
    ok = (not door_differ and not cold_differ and not non200 and grow >= x_resumed and failed == 0
          and all(rewound[n] >= a_resumed[n] for n in (off, on)) and faults == 0 and parks_on > 0)
    p1 &= ok
    say(f"DAY38 P1 card={card} order={order} rows={len(tags)} door_equal={door_equal} door_differ={door_differ[:6]} "
        f"cold_differ={cold_differ[:6]} non200={non200[:6]} x_resumed_on={x_resumed} park_compact_grow={grow} "
        f"park_compact_on={parks_on} failed={failed} affinity_rewound={rewound} a_resumed={a_resumed} "
        f"pool_hits={hits} faults={faults} -> {'PASS' if ok else 'FAIL'}")

# ---- P2 ---------------------------------------------------------------------------------------------------------
for name in ("fault-batch", "fault-nobatch", "fault-nobatch-red"):
    if not have(name):
        continue
    r = rows(name)
    fired = count(name, r"MEMRA_STEP_OOM_FAULT fired")
    ok200 = sum(1 for x in r.values() if x["status"] == 200)
    errored = sorted(t for t, x in r.items() if x["status"] != 200)
    parks = count(name, r"\[kv-reuse\] park-compact: ")
    # The conversation whose turn took the forged OOM: its next turn must run cold (equal to its cold twin, and not
    # a pool resume).
    nxt = []
    for t in errored:
        m = re.match(r"(X-\d+-r\d+)-t(\d)$", t)
        if m:
            nt = f"{m.group(1)}-t{int(m.group(2)) + 1}"
            if nt in r:
                cold = r.get(nt + "-cold")
                resumed = (r[nt]["metrics_after"] or {}).get("continuation_pool_hits", 0) - \
                    (r[nt]["metrics_before"] or {}).get("continuation_pool_hits", 0)
                nxt.append(dict(tag=nt, equal_cold=bool(cold) and cold["content_sha256"] == r[nt]["content_sha256"],
                                pool_resumed=resumed))
    faults = count(name, FAULT)
    green = fired == 1 and len(errored) == 1 and parks == ok200 and all(n["pool_resumed"] == 0 and n["equal_cold"] for n in nxt) and faults == 0
    say(f"DAY38 P2 card={card} boot={name} fired={fired} errored_rows={errored} ok200={ok200} park_compact={parks} "
        f"next_turn={nxt} faults={faults} -> {'PASS' if green else 'FAIL'}"
        + (" (red arm: the expected reading is FAIL with park_compact=ok200+1 and a resumed next turn)" if name.endswith("red") else ""))

# ---- P3 ---------------------------------------------------------------------------------------------------------
cost = {}
for name in glob.glob(os.path.join(root, "boots", "main-*-on")):
    for m in re.finditer(r"\[kv-reuse\] park-compact: (\d+) of (\d+) rows retained in ([\d.]+)ms", read(os.path.join(name, "server.log"))):
        rows_, ms = int(m.group(1)), float(m.group(3))
        bucket = min((6144, 30720, 122880), key=lambda L: abs(L - rows_))
        cost.setdefault(bucket, []).append(ms)
for L in sorted(cost):
    v = cost[L]
    say(f"DAY38 P3 card={card} fed~{L} N={len(v)} p50_ms={pct(v, .5):.2f} p95_ms={pct(v, .95):.2f} max_ms={max(v):.2f}")

# ---- P4 ---------------------------------------------------------------------------------------------------------
for order in ("O1", "O2"):
    for arm in ("off", "on"):
        n = f"main-{order}-{arm}"
        if not have(n):
            continue
        r = rows(n)
        e2e = [x["done_ms"] - x["submit_ms"] for x in r.values() if not x["cold"] and x["turn"] >= 2 and x["status"] == 200]
        say(f"DAY38 P4 card={card} boot={n} idle driver_free={metric(n, 'metrics-end.json', 'cuda_driver_free_bytes')} "
            f"pool_cached={metric(n, 'metrics-end.json', 'cuda_pool_cached_bytes')} "
            f"pool_reserved={metric(n, 'metrics-end.json', 'cuda_pool_reserved_bytes')} "
            f"continuation_pool_entries={metric(n, 'metrics-end.json', 'continuation_pool_entries')} "
            f"resumed_e2e_ms N={len(e2e)} p50={pct(e2e, .5):.1f} p95={pct(e2e, .95):.1f}")

# ---- interaction pair -------------------------------------------------------------------------------------------
if have("vmm-off") and have("vmm-on"):
    ro, rn = rows("vmm-off"), rows("vmm-on")
    tags = sorted(set(ro) & set(rn))
    differ = [t for t in tags if ro[t]["content_sha256"] != rn[t]["content_sha256"]]
    say(f"DAY38 VMM-PAIR card={card} rows={len(tags)} differ={differ[:6]} "
        f"vmm_off idle vmm_mapped={metric('vmm-off', 'metrics-end.json', 'kv_vmm_mapped_bytes')} pool_cached={metric('vmm-off', 'metrics-end.json', 'cuda_pool_cached_bytes')} "
        f"trims={count('vmm-off', r'\[kv-vmm\] trim id=')}; vmm_on idle vmm_mapped={metric('vmm-on', 'metrics-end.json', 'kv_vmm_mapped_bytes')} "
        f"pool_cached={metric('vmm-on', 'metrics-end.json', 'cuda_pool_cached_bytes')} park_compact={count('vmm-on', r'\[kv-reuse\] park-compact: ')}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
