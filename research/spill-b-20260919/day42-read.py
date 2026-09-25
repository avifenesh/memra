#!/usr/bin/env python3
"""WP-B day 42 reader (DAY42.md 1.5): F1 to F4 and the readings from one card's boots.

usage: day42-read.py <card> <card root>
Boot names (fixed by the chains): ontick-O1, offtick-O1, offtick-O2, ontick-O2, ontick-nocontracts, fault-d2h-delay,
fault-d2h-source-flip (addendum A), fault-sources-helper-gone.
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


def marks(n):
    try:
        return json.load(open(os.path.join(bdir(n), "marks.json")))
    except (OSError, ValueError):
        return {}


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")


def health(n):
    t, r = log(n), rows(n)
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for x in r.values() if x["status"] == 503)
    bad429 = [x["tag"] for x in r.values() if x["status"] == 429
              and not (x.get("retry_after") and str(x["retry_after"]).isdigit() and 1 <= int(x["retry_after"]) <= 60)]
    return oom, crash, r503, bad429


def counts(n):
    t = log(n)
    return dict(ontick_demoted=len(re.findall(r"\[admit-mem\] reclaim demoted", t)),
                plans=len(re.findall(r"\[admit-mem\] reclaim off-tick: plan", t)),
                submitted=len(re.findall(r"\[admit-mem\] reclaim off-tick: submitted", t)),
                published=len(re.findall(r"\[prefix-host\] demote: ", t)),
                landing_defers=len(re.findall(r"verdict=defer .*reason=reclaim-landing\b", t)),
                landing_refusals=len(re.findall(r"reason=reclaim-landing-timeout", t)))


# ---- F3 health and the per-boot counts ------------------------------------------------------------------------
names = ["ontick-O1", "offtick-O1", "offtick-O2", "ontick-O2", "ontick-nocontracts", "fault-d2h-delay",
         "fault-d2h-source-flip", "fault-sources-helper-gone"]
for n in names:
    if not have(n):
        continue
    oom, crash, r503, bad429 = health(n)
    ok = oom == 0 and crash == 0 and r503 == 0 and not bad429
    status = {}
    for x in rows(n).values():
        if x["phase"] == "burst":
            status[x["status"]] = status.get(x["status"], 0) + 1
    say(f"DAY42 F3 card={card} boot={n} oom_lines={oom} crash_lines={crash} r503={r503} bad_429={bad429[:4]} "
        f"burst_status={dict(sorted(status.items(), key=str))} counts={counts(n)} -> {'PASS' if ok else 'FAIL'}")

# ---- P0 exercised (addendum C): the warm entries published before the burst, and the reclaim passes --------------
def exercised(n):
    t, m = log(n), marks(n)
    b0 = m.get("burst_start_ms")
    warm_pub = 0
    for line in t.splitlines():
        mm = re.match(r"(\d+) \[prefix-cache\] insert \([^)]*\): \d+ tokens.*ns \"warm\"\)", line)
        if mm and b0 and int(mm.group(1)) < b0:
            warm_pub += 1
    passes = len(re.findall(r"verdict=(?:defer|refuse|demote-then-admit)", t))
    c = counts(n)
    ran = c["ontick_demoted"] + c["plans"]
    return warm_pub, passes, ran


for n in names:
    if have(n):
        w, p, ran = exercised(n)
        say(f"DAY42 P0 card={card} boot={n} warm_published_before_burst={w} memory_verdict_lines={p} flush_runs={ran} "
            f"-> {'EXERCISED' if w > 0 and ran > 0 else 'NOT-EXERCISED'}")

# ---- F1 identity and F2 the flush ran ---------------------------------------------------------------------------
for order in ("O1", "O2"):
    on, off = f"ontick-{order}", f"offtick-{order}"
    if not (have(on) and have(off)):
        continue
    ro, rf = rows(on), rows(off)
    tags = sorted(t for t, x in ro.items() if x["phase"] in ("tenant", "warmth") and t in rf)
    differ = [t for t in tags if ro[t]["content_sha256"] != rf[t]["content_sha256"]]
    cold = []
    for n, r in ((on, ro), (off, rf)):
        for t, x in r.items():
            if x["phase"] == "warmth" and not t.endswith("-cold"):
                twin = r.get(t + "-cold")
                if twin is None or twin["content_sha256"] != x["content_sha256"]:
                    cold.append(f"{n}:{t}")
    ok = bool(tags) and not differ and not cold
    say(f"DAY42 F1 card={card} order={order} rows={len(tags)} arm_differ={differ[:6]} cold_differ={cold[:6]} "
        f"-> {'PASS' if ok else 'FAIL'}")
    co, cf = counts(on), counts(off)
    f2 = co["ontick_demoted"] >= 1 and cf["plans"] >= 1 and cf["submitted"] >= 1 and cf["published"] >= 1 \
        and cf["ontick_demoted"] == 0
    say(f"DAY42 F2 card={card} order={order} ontick={co} offtick={cf} -> {'PASS' if f2 else 'FAIL'}")

# ---- F4 faults --------------------------------------------------------------------------------------------------
for n in ("fault-d2h-delay", "fault-d2h-source-flip", "fault-sources-helper-gone"):
    if not have(n):
        continue
    t, r = log(n), rows(n)
    oom, crash, r503, bad429 = health(n)
    burst = [x for x in r.values() if x["phase"] == "burst"]
    settled = all(x["status"] in (200, 429) for x in burst)
    if n == "fault-d2h-delay":
        extra = f"landing_refusals={counts(n)['landing_refusals']} armed={bool(re.search(r'MEMRA_KV_HOST_FAULT=d2h-delay', t))}"
        ok = settled
    elif n == "fault-d2h-source-flip":
        c = counts(n)
        armed = bool(re.search(r"MEMRA_KV_HOST_FAULT=d2h-source-flip\): one byte", t))
        extra = f"armed={armed} submitted={c['submitted']} published={c['published']}"
        ok = settled and armed and c["published"] < c["submitted"]
    else:
        extra = f"latch_lines={len(re.findall(r'latched off|latches off|latch', t))}"
        ok = settled
    ok = ok and oom == 0 and crash == 0 and r503 == 0 and not bad429
    say(f"DAY42 F4 card={card} boot={n} burst={len(burst)} all_200_or_429={settled} {extra} -> {'PASS' if ok else 'FAIL'}")

# ---- readings ---------------------------------------------------------------------------------------------------
for n in names:
    if not have(n):
        continue
    r, m = rows(n), marks(n)
    b0, b1 = m.get("burst_start_ms"), m.get("burst_end_ms")
    gaps = []
    for x in r.values():
        if x["phase"] == "tenant" and x.get("gaps_ms"):
            # The gaps inside the burst window: reconstructed from the first token time and the gap sequence.
            ts = x["submit_ms"] + (x["ttft_ms"] or 0)
            for g in x["gaps_ms"]:
                ts += g
                if b0 and b1 and b0 <= ts <= b1:
                    gaps.append(g)
    ttft = [x["ttft_ms"] for x in r.values() if x["phase"] == "burst" and x["ttft_ms"] is not None]
    warm = [x for t, x in r.items() if x["phase"] == "warmth" and not t.endswith("-cold")]
    hits = sum(1 for x in warm if (x.get("cached_tokens") or 0) > 0)
    t = log(n)
    demoted_mb = sum(float(v) for v in re.findall(r"\[prefix-host\] demote: \d+ tokens, ([\d.]+)MB", t))
    say(f"DAY42 READING card={card} boot={n} tenant_gap_ms_in_burst N={len(gaps)} p50={pct(gaps, .5):.1f} "
        f"p99={pct(gaps, .99):.1f} max={max(gaps) if gaps else float('nan'):.1f} burst_ttft_ms N={len(ttft)} "
        f"p50={pct(ttft, .5):.1f} p95={pct(ttft, .95):.1f} demoted_MB={demoted_mb:.0f} warmth_hits={hits}/{len(warm)}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
