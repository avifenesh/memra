#!/usr/bin/env python3
"""WP-B day 43 reader (DAY43.md 1.5): C1 to C5 and the readings from one card's boots.

usage: day43-read.py <card> <card root>
Boot names (fixed by the chains): rx-spec-<O1|O2>-<unset|clamp>, offprev.
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


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")
CLAMP = re.compile(r"\[spec\] budget clamp: round truncated at (\d+) of (\d+) accepted")


def later(r):
    return [t for t, x in r.items() if x["shape"] == "RX" and not x["cold"] and x["turn"] >= 2]


def fresh(r):
    return [t for t, x in r.items() if x["shape"] == "RX" and (x["cold"] or x["turn"] == 1)]


def expected_cached(r, t):
    """The previous turn's prompt plus its public completion (the prompt grows by the completion and 64 ids)."""
    x = r[t]
    base = t.rsplit("-t", 1)[0]
    plen = x["L"]
    for turn in range(1, x["turn"] - 1):
        plen += (r.get(f"{base}-t{turn}", {}).get("completion_tokens") or 0) + 64
    return plen + (r.get(f"{base}-t{x['turn'] - 1}", {}).get("completion_tokens") or 0)


# ---- C4 health --------------------------------------------------------------------------------------------------
names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []
for n in names:
    if not have(n):
        say(f"DAY43 C4 card={card} boot={n} no client rows -> FAIL")
        continue
    t = log(n)
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for x in rows(n).values() if x["status"] == 503)
    ok = oom == 0 and crash == 0 and r503 == 0
    say(f"DAY43 C4 card={card} boot={n} oom_lines={oom} crash_lines={crash} r503={r503} -> {'PASS' if ok else 'FAIL'}")

# ---- C1, C2, C3 and the readings ----------------------------------------------------------------------------------
for order in ("O1", "O2"):
    un, cl = f"rx-spec-{order}-unset", f"rx-spec-{order}-clamp"
    if not (have(un) and have(cl)):
        continue
    ru, rc = rows(un), rows(cl)
    tags = sorted(t for t in fresh(ru) if t in rc)
    differ = [t for t in tags if ru[t]["content_sha256"] != rc[t]["content_sha256"]]
    say(f"DAY43 C1 card={card} order={order} rows={len(tags)} differ={differ[:6]} -> "
        f"{'PASS' if tags and not differ else 'FAIL'}")
    fired = CLAMP.findall(log(cl))
    hit = [t for t in later(rc) if (rc[t].get("cached_tokens") or 0) > 0]
    bad = [t for t in hit if rc[t]["cached_tokens"] != expected_cached(rc, t)]
    say(f"DAY43 C2 card={card} boot={cl} clamp_lines={len(fired)} resumed={len(hit)} cached_mismatch={bad[:6]} -> "
        f"{'PASS' if fired and not bad else 'FAIL'}")
    res = later(rc)
    frac = len(hit) / len(res) if res else 0.0
    say(f"DAY43 C3 card={card} boot={cl} turns={len(res)} resumed={len(hit)} frac={frac:.2f} -> "
        f"{'PASS' if frac >= 0.8 else 'FAIL'}")
    for n, r in ((un, ru), (cl, rc)):
        res = later(r)
        h = [t for t in res if (r[t].get("cached_tokens") or 0) > 0]
        flips = [t for t in h if r.get(t + "-cold", {}).get("content_sha256") != r[t]["content_sha256"]]
        tt = [r[t]["ttft_ms"] for t in res if r[t]["ttft_ms"] is not None]
        v = [x for x in r.values() if x["status"] == 200]
        wall = (max(x["done_ms"] for x in v) - min(x["submit_ms"] for x in v)) / 1000.0 if v else float("nan")
        gen = sum(x.get("completion_tokens") or 0 for x in v)
        say(f"DAY43 READING card={card} boot={n} turns={len(res)} resumed={len(h)} flips_vs_cold={len(flips)} "
            f"later_turn_ttft_ms N={len(tt)} p50={pct(tt, .5):.1f} p95={pct(tt, .95):.1f} generated={gen} "
            f"wall_s={wall:.1f} tok_per_s={gen / wall:.2f} clamp_lines={len(CLAMP.findall(log(n)))}")
        for L in sorted({r[t]["L"] for t in res}):
            for G in sorted({r[t]["G"] for t in res}):
                sub = [t for t in res if r[t]["L"] == L and r[t]["G"] == G]
                if not sub:
                    continue
                hs = [t for t in sub if (r[t].get("cached_tokens") or 0) > 0]
                ts = [r[t]["ttft_ms"] for t in sub if r[t]["ttft_ms"] is not None]
                say(f"DAY43 READING card={card} boot={n} L={L} G={G} turns={len(sub)} resumed={len(hs)} "
                    f"ttft_ms p50={pct(ts, .5):.1f} p95={pct(ts, .95):.1f}")

# ---- C5 door OFF ---------------------------------------------------------------------------------------------------
if have("offprev"):
    rp = rows("offprev")
    for order in ("O1", "O2"):
        un = f"rx-spec-{order}-unset"
        if not have(un):
            continue
        ru = rows(un)
        tags = sorted(t for t, x in rp.items() if x["shape"] == "RX" and t in ru)
        differ = [t for t in tags if rp[t]["content_sha256"] != ru[t]["content_sha256"]]
        say(f"DAY43 C5 card={card} unset={un} rows={len(tags)} differ={differ[:6]} -> "
            f"{'PASS' if tags and not differ else 'FAIL'}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
