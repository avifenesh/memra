#!/usr/bin/env python3
"""WP-B day 41 reader (DAY41.md 1.4, addendum A): R1 to R4 and the price readings from one card's boots.

usage: day41-read.py <card> <card root>
Boot names (fixed by the chains): rx-<plain|spec>-<O1|O2>-<keep|rewind>, fx-<plain|spec>-<keep|rewind>, offprev.
"""
import json, os, re, statistics, sys

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


def metric(n, key):
    try:
        return json.load(open(os.path.join(bdir(n), "metrics-end.json"))).get(key)
    except (OSError, ValueError):
        return None


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")
REWIND = re.compile(r"\[kv-reuse\] grid-rewind: (plain|spec) extension of (\d+) committed rows rewound to (\d+) "
                    r"\(re-priming (\d+) rows")


def resumed(r):
    return [t for t, x in r.items() if x["shape"] == "RX" and not x["cold"] and x["turn"] >= 2]


# ---- R4 health (every boot) ---------------------------------------------------------------------------------------
names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []
r4 = True
for n in names:
    if not have(n):
        say(f"DAY41 R4 card={card} boot={n} no client rows -> FAIL")
        r4 = False
        continue
    t = log(n)
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for x in rows(n).values() if x["status"] == 503)
    ok = oom == 0 and crash == 0 and r503 == 0
    r4 &= ok
    say(f"DAY41 R4 card={card} boot={n} oom_lines={oom} crash_lines={crash} r503={r503} -> {'PASS' if ok else 'FAIL'}")

# ---- R1, R2 and the per-arm readings ------------------------------------------------------------------------------
for route in ("plain", "spec"):
    for order in ("O1", "O2"):
        keep, rew = f"rx-{route}-{order}-keep", f"rx-{route}-{order}-rewind"
        if not (have(keep) and have(rew)):
            continue
        rk, rr = rows(keep), rows(rew)
        for n, r in ((keep, rk), (rew, rr)):
            res = resumed(r)
            hit = [t for t in res if (r[t].get("cached_tokens") or 0) > 0]
            flips = [t for t in hit if r.get(t + "-cold", {}).get("content_sha256") != r[t]["content_sha256"]]
            frac = len(hit) / len(res) if res else 0.0
            say(f"DAY41 R2 card={card} boot={n} turns={len(res)} resumed={len(hit)} frac={frac:.2f} -> "
                f"{'PASS' if frac >= 0.8 else 'FAIL'}")
            say(f"DAY41 FLIPS card={card} boot={n} resumed={len(hit)} flips_vs_cold={len(flips)} tags={flips[:6]} "
                f"(reading; keep carries the near-tie residual)")
        res = [t for t in resumed(rr) if (rr[t].get("cached_tokens") or 0) > 0]
        differ = [t for t in res if rr.get(t + "-cold", {}).get("content_sha256") != rr[t]["content_sha256"]]
        lines = REWIND.findall(log(rew))
        r_ok = all(int(f) - int(p) == int(rr_) for _, f, p, rr_ in lines)
        ok = bool(res) and not differ and len(lines) >= len(res) and r_ok
        say(f"DAY41 R1 card={card} boot={rew} resumed={len(res)} differ_vs_cold={differ[:6]} grid_rewind_lines={len(lines)} "
            f"r_equals_f_minus_p={r_ok} -> {'PASS' if ok else 'FAIL'}")
        # The price: per (L, G), re-primed rows (keep's cached minus rewind's), resumed TTFT and E2E, throughput.
        groups = sorted({(x["L"], x["G"]) for x in rr.values() if x["shape"] == "RX"})
        for L, G in groups:
            tags = [t for t in resumed(rk) if rk[t]["L"] == L and rk[t]["G"] == G and t in rr
                    and (rk[t].get("cached_tokens") or 0) > 0 and (rr[t].get("cached_tokens") or 0) > 0]
            if not tags:
                continue
            reprime = [rk[t]["cached_tokens"] - rr[t]["cached_tokens"] for t in tags]
            tk = [rk[t]["ttft_ms"] for t in tags if rk[t]["ttft_ms"] is not None]
            tr = [rr[t]["ttft_ms"] for t in tags if rr[t]["ttft_ms"] is not None]
            ek = [rk[t]["e2e_ms"] for t in tags]
            er = [rr[t]["e2e_ms"] for t in tags]
            ratio = pct(tr, .5) / pct(tk, .5) if tk and tr and pct(tk, .5) else float("nan")
            say(f"DAY41 PRICE card={card} route={route} order={order} L={L} G={G} N={len(tags)} "
                f"reprimed_rows p50={pct(reprime, .5):.0f} max={max(reprime)} "
                f"ttft_ms keep p50={pct(tk, .5):.1f} p95={pct(tk, .95):.1f} rewind p50={pct(tr, .5):.1f} "
                f"p95={pct(tr, .95):.1f} ratio_p50={ratio:.3f} e2e_ms keep p50={pct(ek, .5):.1f} rewind p50={pct(er, .5):.1f}")
        for n, r in ((keep, rk), (rew, rr)):
            v = [x for x in r.values() if x["status"] == 200]
            if v:
                wall = (max(x["done_ms"] for x in v) - min(x["submit_ms"] for x in v)) / 1000.0
                gen = sum(x.get("completion_tokens") or 0 for x in v)
                say(f"DAY41 THROUGHPUT card={card} boot={n} generated={gen} wall_s={wall:.1f} tok_per_s={gen / wall:.2f} "
                    f"idle driver_free={metric(n, 'cuda_driver_free_bytes')} pool_cached={metric(n, 'cuda_pool_cached_bytes')}")

# ---- R3 door OFF -------------------------------------------------------------------------------------------------
if have("offprev"):
    rp = rows("offprev")
    for order in ("O1", "O2"):
        keep = f"rx-plain-{order}-keep"
        if not have(keep):
            continue
        rk = rows(keep)
        tags = sorted(t for t, x in rp.items() if x["shape"] == "RX" and x["L"] == 6144 and t in rk)
        differ = [t for t in tags if rp[t]["content_sha256"] != rk[t]["content_sha256"]]
        ok = bool(tags) and not differ
        say(f"DAY41 R3 card={card} keep={keep} rows={len(tags)} differ={differ[:6]} -> {'PASS' if ok else 'FAIL'}")

# ---- FX readings -----------------------------------------------------------------------------------------------
for route in ("plain", "spec"):
    for arm in ("keep", "rewind"):
        n = f"fx-{route}-{arm}"
        if not have(n):
            continue
        r = rows(n)
        v = [x for x in r.values() if x["status"] == 200]
        tt = [x["ttft_ms"] for x in v if x["ttft_ms"] is not None]
        cached = [x.get("cached_tokens") or 0 for x in v]
        t = log(n)
        say(f"DAY41 FX card={card} boot={n} ok={len(v)} ttft_ms p50={pct(tt, .5):.1f} p95={pct(tt, .95):.1f} "
            f"cached_tokens_sum={sum(cached)} prefix_hit_lines={len(re.findall(r'\\[prefix-cache\\] hit', t))} "
            f"fanout_lines={len(re.findall(r'fanout', t))}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
