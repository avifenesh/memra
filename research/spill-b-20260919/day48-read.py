#!/usr/bin/env python3
"""WP-B day 48 reader (DAY48.md 1.3 and addendum A): V1 to V4 and the readings from one card's boots.

usage: day48-read.py <card> <card root>
Boot names (fixed by the chains): <O1|O2>-<enforce|enforce-vg>.
V1 the verify-graph pool engages (the ENGAGED line and a physical `dspark verify-graph pool debt` line on every boot, and
on enforce-vg at least one `vg_debt=` above 0); V2 no OOM, parked OOM, 503 or crash; V3 every refusal a typed 429 with
Retry-After in 1 to 60 and its own `reject-kv ... enforce=1` line; V4 the requests 200 on both arms of an order have
equal digests.
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


def rows(n):
    return [json.loads(l) for l in read(os.path.join(root, "boots", n, "client.jsonl")).splitlines() if l.strip()]


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")
PARKED = re.compile(r"\[admit-mem\] prefill OOM parked|step OOM parked")
PRED = re.compile(r"\[admit-predict\] id=(\S+) tenant=\"([^\"]*)\" model=\S+ verdict=(\S+) reason=\S+ prompt=\S+ "
                  r"predicted_completion=\S+ kv_hat=(\S+) booked_bytes=(\d+) booked_real=(\d+) .*?budget_bytes=(\S+) "
                  r"retry_after_s=\S+ exempt=\d enforce=(\d)(?: live_free_bytes=\S+)?(?: vg_debt=(\d+))?")
PHYS = re.compile(r"\[admission\] dspark verify-graph pool debt: \+(\d+)MB")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else float("nan")


def salt_of(tenant):
    m = re.search(r"d48-([A-Za-z0-9]+)", tenant)
    return m.group(1) if m else None


names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []
by = {}
for n in names:
    arm = n.split("-", 1)[1]
    rs = rows(n)
    t = read(os.path.join(root, "boots", n, "server.log"))
    by[n] = rs
    if not rs:
        say(f"DAY48 V2 card={card} boot={n} no client rows -> FAIL")
        continue
    preds = PRED.findall(t)
    engaged = len(re.findall(r"MTP verify-graph pool ENGAGED", t))
    phys = [int(x) for x in PHYS.findall(t)]
    vg = [int(p[8]) for p in preds if p[8]]
    ok1 = engaged >= 1 and bool(phys) and (arm != "enforce-vg" or any(v > 0 for v in vg))
    say(f"DAY48 V1 card={card} boot={n} pool_engaged_lines={engaged} physical_debt_lines={len(phys)} "
        f"physical_debt_mb_max={max(phys, default=0)} vg_debt_lines={len(vg)} vg_debt_max={max(vg, default=0)} -> "
        + ("PASS" if ok1 else "FAIL (no reading of the arm)"))
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    parked = len(PARKED.findall(t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for r in rs if r["status"] == 503)
    ok2 = oom == 0 and parked == 0 and crash == 0 and r503 == 0
    say(f"DAY48 V2 card={card} boot={n} oom_lines={oom} parked_oom_lines={parked} crash_lines={crash} r503={r503} -> "
        f"{'PASS' if ok2 else 'FAIL'}")
    by_salt = {}
    for p in preds:
        by_salt.setdefault(salt_of(p[1]), []).append(p)
    bad_status = [r["tag"] for r in rs if r["status"] not in (200, 429) or (r["status"] == 200 and r["error"])]
    r429 = [r for r in rs if r["status"] == 429]
    no_line = [r["tag"] for r in r429 if not any(p[2] == "reject-kv" and p[7] == "1" for p in by_salt.get(r["salt"], []))]
    bad_ra = [r["tag"] for r in r429 if not (r["retry_after"] and r["retry_after"].isdigit()
                                              and 1 <= int(r["retry_after"]) <= 60)]
    ok3 = not bad_status and not no_line and not bad_ra
    say(f"DAY48 V3 card={card} boot={n} r429={len(r429)} without_reject_line={no_line[:4]} "
        f"retry_after_out_of_1_60={bad_ra[:4]} other_non200={bad_status[:4]} -> {'PASS' if ok3 else 'FAIL'}")
    for wave in ("burst", "wave2"):
        wr = [r for r in rs if r["phase"] == wave]
        ok200 = [r for r in wr if r["status"] == 200 and not r["error"]]
        ttft = [r["ttft_ms"] for r in ok200 if r["ttft_ms"] is not None]
        say(f"DAY48 READING card={card} boot={n} wave={wave} n={len(wr)} ok200={len(ok200)} "
            f"r429={sum(1 for r in wr if r['status'] == 429)} ttft_ms p50={pct(ttft, 0.5):.1f} p95={pct(ttft, 0.95):.1f} "
            f"N={len(ttft)}")
for order in ("O1", "O2"):
    a, b = f"{order}-enforce", f"{order}-enforce-vg"
    if a in by and b in by:
        ra = {r["tag"]: r for r in by[a]}
        rb = {r["tag"]: r for r in by[b]}
        common = sorted(set(ra) & set(rb))
        both = [t for t in common if ra[t]["status"] == 200 and rb[t]["status"] == 200 and not ra[t]["error"]
                and not rb[t]["error"]]
        mismatch = [t for t in common if ra[t]["status"] != rb[t]["status"]]
        differ = [t for t in both if ra[t]["content_sha256"] != rb[t]["content_sha256"]]
        say(f"DAY48 V4 card={card} order={order} rows_200_both={len(both)} status_mismatch={len(mismatch)} "
            f"differ={differ[:6]} -> {'PASS' if both and not differ else 'FAIL'}")
with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
