#!/usr/bin/env python3
"""WP-B day 44 reader (DAY44.md 1.7): E1 to E6 and the readings from one card's boots.

usage: day44-read.py <card> <card root>
Boot names (fixed by the chains): rx-<plain|spec>-<rx|rxg>-<O1|O2>-<keep|exact>, offprev, fault-plain-rxg6.
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


def ordered(n):
    return [json.loads(l) for l in read(os.path.join(bdir(n), "client.jsonl")).splitlines() if l.strip()]


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
RESUME = re.compile(r"\[kv-reuse\] exact: (plain|spec) resume from (\d+) of (\d+) committed rows \(priming (\d+) rows, "
                    r"(settled|checkpoint)")
SETTLE = re.compile(r"\[kv-reuse\] exact: settle (\d+) (\d+) -> (\d+) \((\d+) rows, ([\d.]+) ms\)")
TTFT = re.compile(r"\[ttft\] .*? prompt_tokens=(\d+) outcome=(\S+) .*? prime_ms=([\d.]+|-) ")


def later(r):
    return [t for t, x in r.items() if x["shape"] == "RX" and not x["cold"] and x["turn"] >= 2]


def resumed(r):
    return [t for t in later(r) if (r[t].get("cached_tokens") or 0) > 0]


def prime_ms_by_tag(n):
    """The [ttft] lines in request order mapped onto the client rows in request order."""
    lines = TTFT.findall(log(n))
    rs = ordered(n)
    if len(lines) != len(rs):
        return {}
    return {r["tag"]: (None if p == "-" else float(p)) for r, (_, _, p) in zip(rs, lines)}


names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []

# ---- E5 health -------------------------------------------------------------------------------------------------
for n in names:
    if not have(n):
        say(f"DAY44 E5 card={card} boot={n} no client rows -> FAIL")
        continue
    t = log(n)
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for x in rows(n).values() if x["status"] == 503)
    ok = oom == 0 and crash == 0 and r503 == 0
    say(f"DAY44 E5 card={card} boot={n} oom_lines={oom} crash_lines={crash} r503={r503} -> {'PASS' if ok else 'FAIL'}")

# ---- E1 to E4 and the readings, per route, shape and order -----------------------------------------------------
for route in ("plain", "spec"):
    for shape in ("rx", "rxg"):
        for order in ("O1", "O2"):
            keep, ex = f"rx-{route}-{shape}-{order}-keep", f"rx-{route}-{shape}-{order}-exact"
            if not (have(keep) and have(ex)):
                continue
            rk, re_ = rows(keep), rows(ex)
            res = resumed(re_)
            differ = [t for t in res if re_.get(t + "-cold", {}).get("content_sha256") != re_[t]["content_sha256"]]
            lines = RESUME.findall(log(ex))
            ok = bool(res) and not differ and len(lines) >= len(res)
            say(f"DAY44 E1 card={card} boot={ex} resumed={len(res)} differ_vs_cold={differ[:6]} exact_lines={len(lines)} "
                f"-> {'PASS' if ok else 'FAIL'}")
            lat = later(re_)
            frac = len(res) / len(lat) if lat else 0.0
            say(f"DAY44 E3 card={card} boot={ex} turns={len(lat)} resumed={len(res)} frac={frac:.2f} -> "
                f"{'PASS' if frac >= 0.95 else 'FAIL'}")
            if shape == "rxg":
                settled = sum(1 for x in lines if x[4] == "settled")
                sfrac = settled / len(lines) if lines else 0.0
                say(f"DAY44 E4 card={card} boot={ex} resume_lines={len(lines)} settled={settled} frac={sfrac:.2f} "
                    f"settles={len(SETTLE.findall(log(ex)))} -> {'PASS' if sfrac >= 0.8 else 'FAIL'}")
            # E2 and the price, per (L, G), over turns resumed on both arms.
            pk, pe = prime_ms_by_tag(keep), prime_ms_by_tag(ex)
            groups = sorted({(x["L"], x["G"]) for x in re_.values() if x["shape"] == "RX"})
            for L, G in groups:
                tags = [t for t in later(rk) if rk[t]["L"] == L and rk[t]["G"] == G and t in re_
                        and (rk[t].get("cached_tokens") or 0) > 0 and (re_[t].get("cached_tokens") or 0) > 0]
                if not tags:
                    continue
                tk = [rk[t]["ttft_ms"] for t in tags if rk[t]["ttft_ms"] is not None]
                te = [re_[t]["ttft_ms"] for t in tags if re_[t]["ttft_ms"] is not None]
                ek = [rk[t]["e2e_ms"] for t in tags]
                ee = [re_[t]["e2e_ms"] for t in tags]
                rt = pct(te, .5) / pct(tk, .5) if tk and te else float("nan")
                re2 = pct(ee, .5) / pct(ek, .5) if ek and ee else float("nan")
                ok = rt <= 1.05 and re2 <= 1.05
                primed = [rk[t]["cached_tokens"] - re_[t]["cached_tokens"] for t in tags]
                pmk = [pk[t] for t in tags if pk.get(t) is not None]
                pme = [pe[t] for t in tags if pe.get(t) is not None]
                say(f"DAY44 E2 card={card} route={route} shape={shape} order={order} L={L} G={G} N={len(tags)} "
                    f"ttft_ms keep p50={pct(tk, .5):.1f} p95={pct(tk, .95):.1f} exact p50={pct(te, .5):.1f} "
                    f"p95={pct(te, .95):.1f} ratio={rt:.3f} e2e_ms keep p50={pct(ek, .5):.1f} exact p50={pct(ee, .5):.1f} "
                    f"ratio={re2:.3f} extra_primed_rows p50={pct(primed, .5):.0f} max={max(primed)} "
                    f"prime_ms keep p50={pct(pmk, .5):.1f} exact p50={pct(pme, .5):.1f} -> {'PASS' if ok else 'FAIL'}")
            for n, r in ((keep, rk), (ex, re_)):
                v = [x for x in r.values() if x["status"] == 200]
                if v:
                    wall = (max(x["done_ms"] for x in v) - min(x["submit_ms"] for x in v)) / 1000.0
                    gen = sum(x.get("completion_tokens") or 0 for x in v)
                    flips = [t for t in resumed(r) if r.get(t + "-cold", {}).get("content_sha256") != r[t]["content_sha256"]]
                    walls = [float(x[4]) for x in SETTLE.findall(log(n))]
                    say(f"DAY44 READING card={card} boot={n} resumed={len(resumed(r))} flips_vs_cold={len(flips)} "
                        f"generated={gen} wall_s={wall:.1f} tok_per_s={gen / wall:.2f} settles={len(walls)} "
                        f"settle_ms p50={pct(walls, .5):.1f} max={max(walls) if walls else float('nan'):.1f} "
                        f"idle driver_free={metric(n, 'cuda_driver_free_bytes')} pool_cached={metric(n, 'cuda_pool_cached_bytes')}")

# ---- E4 fault ----------------------------------------------------------------------------------------------------
f = "fault-plain-rxg6"
if have(f):
    r = rows(f)
    failed = len(re.findall(r"\[kv-reuse\] exact: settle failed \(fault injection", log(f)))
    lat = later(r)
    resumed_after = [t for t in lat if (r[t].get("cached_tokens") or 0) > 0]
    cold_differ = [t for t in lat if (r[t].get("cached_tokens") or 0) == 0
                   and r.get(t + "-cold", {}).get("content_sha256") != r[t]["content_sha256"]]
    ok = failed >= 1 and not resumed_after and not cold_differ
    say(f"DAY44 E4-FAULT card={card} boot={f} settle_failed_lines={failed} later_turns={len(lat)} "
        f"resumed_after_drop={resumed_after[:6]} cold_differ={cold_differ[:6]} -> {'PASS' if ok else 'FAIL'}")

# ---- E6 door OFF ---------------------------------------------------------------------------------------------------
if have("offprev"):
    rp = rows("offprev")
    for order in ("O1", "O2"):
        keep = f"rx-plain-rx-{order}-keep"
        if not have(keep):
            continue
        rk = rows(keep)
        tags = sorted(t for t, x in rp.items() if x["shape"] == "RX" and x["L"] == 6144 and t in rk)
        differ = [t for t in tags if rp[t]["content_sha256"] != rk[t]["content_sha256"]]
        say(f"DAY44 E6 card={card} keep={keep} rows={len(tags)} differ={differ[:6]} -> "
            f"{'PASS' if tags and not differ else 'FAIL'}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
