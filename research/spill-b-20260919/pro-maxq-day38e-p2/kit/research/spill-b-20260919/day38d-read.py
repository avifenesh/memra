#!/usr/bin/env python3
"""WP-B day 38 addendum D reader (DAY38.md 1.11): P1' and P2' from one card's boots, P3, P4 and the interaction pair
as 1.4 and 1.5 read them. The registered day38-read.py stays the reader of the registered half.

usage: day38d-read.py <card> <card root>
Boot names: main-<O1|O2>-<off|on>, fault-batch, fault-nobatch, fault-nobatch-red, vmm-off, vmm-on.
"""
import glob, json, os, re, sys

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
    r = {}
    for l in read(os.path.join(bdir(n), "client.jsonl")).splitlines():
        if l.strip():
            x = json.loads(l)
            r[x["tag"]] = x
    return r


def log(n):
    return read(os.path.join(bdir(n), "server.log"))


def count(n, pat):
    return len(re.findall(pat, log(n)))


def cached(x):
    return (((x.get("usage") or {}).get("prompt_tokens_details")) or {}).get("cached_tokens") or 0


def hits(x):
    return ((x.get("metrics_after") or {}).get("continuation_pool_hits") or 0) - \
        ((x.get("metrics_before") or {}).get("continuation_pool_hits") or 0)


def metric(n, which, key):
    try:
        return json.load(open(os.path.join(bdir(n), which))).get(key)
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


# ---- P1' --------------------------------------------------------------------------------------------------------
for order in ("O1", "O2"):
    off, on = f"main-{order}-off", f"main-{order}-on"
    if not (have(off) and have(on)):
        continue
    ro, rn = rows(off), rows(on)
    xs = sorted(t for t, x in rn.items() if x["shape"] == "Xp" and not x["cold"] and x["turn"] >= 2)
    both = [t for t in xs if t in ro and cached(ro[t]) > 0 and cached(rn[t]) > 0]
    prompt_mismatch = [t for t in xs if t in ro and ro[t].get("prompt_sha256") != rn[t].get("prompt_sha256")]
    x_differ = [t for t in both if ro[t]["content_sha256"] != rn[t]["content_sha256"]]
    a2 = sorted(t for t, x in rn.items() if x["shape"] == "A" and not x["cold"] and x["turn"] == 2)
    a_differ = [t for t in a2 if t in ro and (ro[t]["content_sha256"] != rn[t]["content_sha256"]
                                            or rn.get(t + "-cold", {}).get("content_sha256") != rn[t]["content_sha256"]
                                            or ro.get(t + "-cold", {}).get("content_sha256") != ro[t]["content_sha256"])]
    grow = count(on, r"\[kv-reuse\] park-compact grow: ")
    on_resumed = sum(1 for t in xs if cached(rn[t]) > 0)
    rewound = {n: count(n, r"plain-affinity: rewound to") for n in (off, on)}
    non200 = [f"{n}:{t}" for n, r in ((off, ro), (on, rn)) for t, x in r.items() if x["status"] != 200]
    faults = count(off, FAULT) + count(on, FAULT)
    frac = len(both) / len(xs) if xs else 0.0
    ok = (xs and not x_differ and not a_differ and not prompt_mismatch and frac >= 0.8 and grow >= on_resumed
          and all(rewound[n] >= len(a2) for n in (off, on)) and not non200 and faults == 0)
    say(f"DAY38D P1 card={card} order={order} xp_turns={len(xs)} resumed_both={len(both)} frac={frac:.2f} "
        f"xp_door_differ={x_differ[:6]} prompt_mismatch={prompt_mismatch[:4]} a_turn2={len(a2)} a_differ={a_differ[:6]} "
        f"on_resumed={on_resumed} park_compact_grow={grow} affinity_rewound={rewound} non200={non200[:6]} "
        f"faults={faults} -> {'PASS' if ok else 'FAIL'}")
    for n, r in ((off, ro), (on, rn)):
        res = [t for t in xs if cached(r[t]) > 0]
        flips = [t for t in res if r.get(t + "-cold", {}).get("content_sha256") != r[t]["content_sha256"]]
        say(f"DAY38D P1-READING card={card} boot={n} xp_resumed={len(res)} resumed_vs_cold_flips={len(flips)} "
            f"flip_tags={flips[:6]} (near-tie contract, no bound)")

# ---- P2' --------------------------------------------------------------------------------------------------------
for name in ("fault-batch", "fault-nobatch", "fault-nobatch-red"):
    if not have(name):
        continue
    r = rows(name)
    fired = count(name, r"MEMRA_STEP_OOM_FAULT fired")
    ok200 = sum(1 for x in r.values() if x["status"] == 200)
    errored = sorted(t for t, x in r.items() if x["status"] != 200)
    parks = count(name, r"\[kv-reuse\] park-compact: ")
    nxt = []
    for t in errored:
        m = re.match(r"(Xp-\d+-r\d+)-t(\d)$", t)
        if m:
            nt = f"{m.group(1)}-t{int(m.group(2)) + 1}"
            if nt in r:
                cold = r.get(nt + "-cold")
                nxt.append(dict(tag=nt, equal_cold=bool(cold) and cold["content_sha256"] == r[nt]["content_sha256"],
                                pool_resumed=hits(r[nt]), cached=cached(r[nt])))
    faults = count(name, FAULT)
    green = (fired == 1 and len(errored) == 1 and parks == ok200 and bool(nxt)
             and all(n["pool_resumed"] == 0 and n["cached"] == 0 and n["equal_cold"] for n in nxt) and faults == 0)
    red = name.endswith("red")
    say(f"DAY38D P2 card={card} boot={name} fired={fired} errored_rows={errored} ok200={ok200} park_compact={parks} "
        f"next_turn={nxt} faults={faults} -> {'PASS' if green else 'FAIL'}"
        + (" (red arm: the expected reading is FAIL with park_compact=ok200+1 and a resumed next turn)" if red else ""))

# ---- P3, P4, the pair (1.4, 1.5 unchanged) ------------------------------------------------------------------------
cost = {}
for name in glob.glob(os.path.join(root, "boots", "main-*-on")):
    for m in re.finditer(r"\[kv-reuse\] park-compact: (\d+) of (\d+) rows retained in ([\d.]+)ms", read(os.path.join(name, "server.log"))):
        bucket = min((6144, 30720, 122880), key=lambda L: abs(L - int(m.group(1))))
        cost.setdefault(bucket, []).append(float(m.group(3)))
for L in sorted(cost):
    v = cost[L]
    say(f"DAY38D P3 card={card} fed~{L} N={len(v)} p50_ms={pct(v, .5):.2f} p95_ms={pct(v, .95):.2f} max_ms={max(v):.2f}")
for order in ("O1", "O2"):
    for arm in ("off", "on"):
        n = f"main-{order}-{arm}"
        if not have(n):
            continue
        r = rows(n)
        e2e = [x["done_ms"] - x["submit_ms"] for x in r.values() if not x["cold"] and x["turn"] >= 2 and x["status"] == 200]
        say(f"DAY38D P4 card={card} boot={n} idle driver_free={metric(n, 'metrics-end.json', 'cuda_driver_free_bytes')} "
            f"pool_cached={metric(n, 'metrics-end.json', 'cuda_pool_cached_bytes')} "
            f"continuation_pool_entries={metric(n, 'metrics-end.json', 'continuation_pool_entries')} "
            f"resumed_e2e_ms N={len(e2e)} p50={pct(e2e, .5):.1f} p95={pct(e2e, .95):.1f}")
if have("vmm-off") and have("vmm-on"):
    ro, rn = rows("vmm-off"), rows("vmm-on")
    tags = sorted(set(ro) & set(rn))
    differ = [t for t in tags if ro[t]["content_sha256"] != rn[t]["content_sha256"]]
    say(f"DAY38D VMM-PAIR card={card} rows={len(tags)} differ={differ[:6]} "
        f"vmm_off idle vmm_mapped={metric('vmm-off', 'metrics-end.json', 'kv_vmm_mapped_bytes')} "
        f"vmm_on idle vmm_mapped={metric('vmm-on', 'metrics-end.json', 'kv_vmm_mapped_bytes')} "
        f"park_compact={count('vmm-on', r'\[kv-reuse\] park-compact: ')}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
