#!/usr/bin/env python3
"""WP-B day 46 reader (DAY46.md 1.3 and addendum A): P1 to P5 and the readings from one card's boots.

usage: day46-read.py <card> <card root>
Boot names (fixed by the chains): <O1|O2>-<shadow|enforce|enforce-wrel>.
P1 and P2 apply to the two enforcing arms; on `shadow` the OOM, 503 and crash counts are the before reading (addendum
A). P4 compares, per order, each enforcing arm with `shadow` on the requests that are 200 on both.
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


def rows(n):
    return [json.loads(l) for l in read(os.path.join(bdir(n), "client.jsonl")).splitlines() if l.strip()]


CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel")
PARKED = re.compile(r"\[admit-mem\] prefill OOM parked|step OOM parked")
PRED = re.compile(r"\[admit-predict\] id=(\S+) tenant=\"([^\"]*)\" model=\S+ verdict=(\S+) reason=\S+ prompt=\S+ "
                  r"predicted_completion=\S+ kv_hat=(\S+) booked_bytes=(\d+) booked_real=(\d+) .*?budget_bytes=(\S+) "
                  r"retry_after_s=\S+ exempt=\d enforce=(\d)")
BOOT_BUDGET = re.compile(r"\[admit-predict\] shadow armed: budget_bytes=(\d+)")
BOOKED = re.compile(r"\[admit-book\] w-booked id=(\S+) model=\S+ bytes=(\d+)")
RELEASE = re.compile(r"\[admit-book\] w-release id=(\S+) model=\S+ bytes=(\d+)")
UNRELEASED = re.compile(r"\[admit-book\] w-retire-unreleased id=(\S+) bytes=(\d+) reason=(\S+)")


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else float("nan")


def salt_of(tenant):
    m = re.search(r"d46-([A-Za-z0-9]+)", tenant)
    return m.group(1) if m else None


names = sorted(os.path.basename(p)[:-len(".arm.txt")] for p in os.listdir(os.path.join(root, "boots"))
               if p.endswith(".arm.txt")) if os.path.isdir(os.path.join(root, "boots")) else []
by = {}
for n in names:
    arm = n.split("-", 1)[1]
    rs = rows(n)
    t = read(os.path.join(bdir(n), "server.log"))
    by[n] = (arm, rs, t)
    if not rs:
        say(f"DAY46 P2 card={card} boot={n} no client rows -> FAIL")
        continue
    enforcing = arm != "shadow"
    oom = len(re.findall(r"CUDA_ERROR_OUT_OF_MEMORY", t))
    parked = len(PARKED.findall(t))
    crash = len(CRASH.findall(t))
    r503 = sum(1 for r in rs if r["status"] == 503)
    tag = "P2" if enforcing else "P2-READING"
    ok = oom == 0 and parked == 0 and crash == 0 and r503 == 0
    say(f"DAY46 {tag} card={card} boot={n} oom_lines={oom} parked_oom_lines={parked} crash_lines={crash} r503={r503} -> "
        + (("PASS" if ok else "FAIL") if enforcing else "READING (the before)"))
    preds = PRED.findall(t)
    by_salt = {}
    for p in preds:
        by_salt.setdefault(salt_of(p[1]), []).append(p)
    if enforcing:
        bad_status = [r["tag"] for r in rs if r["status"] not in (200, 429) or (r["status"] == 200 and r["error"])]
        r429 = [r for r in rs if r["status"] == 429]
        no_line = [r["tag"] for r in r429
                   if not any(p[2] == "reject-kv" and p[7] == "1" for p in by_salt.get(r["salt"], []))]
        bad_ra = [r["tag"] for r in r429 if not (r["retry_after"] and r["retry_after"].isdigit()
                                                  and 1 <= int(r["retry_after"]) <= 60)]
        ok1 = not bad_status and not no_line and not bad_ra
        say(f"DAY46 P1 card={card} boot={n} r429={len(r429)} without_reject_line={no_line[:4]} "
            f"retry_after_out_of_1_60={bad_ra[:4]} other_non200={bad_status[:4]} -> {'PASS' if ok1 else 'FAIL'}")
        over = [p[0] for p in preds if p[2] == "admit" and p[3] != "-" and p[6] != "unset"
                and int(p[4]) + int(p[3]) > int(p[6])]
        admits = sum(1 for p in preds if p[2] == "admit")
        say(f"DAY46 P3 card={card} boot={n} admit_lines={admits} over_budget={over[:4]} -> {'PASS' if admits and not over else 'FAIL'}")
    probe = by_salt.get("probe", [])
    probe_ok = bool(probe) and probe[-1][4] == "0" and probe[-1][5] == "0"
    say(f"DAY46 P5-PROBE card={card} boot={n} probe_booked={probe[-1][4] if probe else None} "
        f"probe_booked_real={probe[-1][5] if probe else None} -> {'PASS' if probe_ok else 'FAIL'}")
    if arm == "enforce-wrel":
        booked = dict(BOOKED.findall(t))
        rel = RELEASE.findall(t)
        unrel = UNRELEASED.findall(t)
        ends = [(i, b) for i, b in rel] + [(i, b) for i, b, _ in unrel]
        end_ids = [i for i, _ in ends]
        twice = sorted({i for i in end_ids if end_ids.count(i) > 1})
        wrong = [i for i, b in ends if booked.get(i) != b]
        missing = sorted(set(booked) - set(end_ids))
        ok5 = bool(booked) and not twice and not wrong and not missing
        say(f"DAY46 P5 card={card} boot={n} w_booked={len(booked)} w_release={len(rel)} w_retire_unreleased={len(unrel)} "
            f"twice={twice[:4]} bytes_mismatch={wrong[:4]} neither={missing[:4]} -> {'PASS' if ok5 else 'FAIL'}")
    # readings
    budget = BOOT_BUDGET.search(t)
    peak = max((int(p[4]) for p in preds), default=0)
    seq_ok = sum(1 for r in rs if r["phase"].startswith("seq") and r["status"] == 200 and not r["error"])
    seq_n = sum(1 for r in rs if r["phase"].startswith("seq"))
    say(f"DAY46 READING card={card} boot={n} budget_bytes={budget.group(1) if budget else None} peak_booked_bytes={peak} "
        f"sequence_admitted={seq_ok}/{seq_n}")
    for wave in ("burst", "wave2"):
        wr = [r for r in rs if r["phase"] == wave]
        ok200 = [r for r in wr if r["status"] == 200 and not r["error"]]
        rej = [r for r in wr if r["status"] == 429]
        ttft = [r["ttft_ms"] for r in ok200 if r["ttft_ms"] is not None]
        first = None
        for p in preds:
            s = salt_of(p[1]) or ""
            if s.startswith(wave):
                first = p
                break
        say(f"DAY46 READING card={card} boot={n} wave={wave} n={len(wr)} ok200={len(ok200)} r429={len(rej)} "
            f"booked_at_first_line={first[4] if first else None} ttft_ms p50={pct(ttft, 0.5):.1f} p95={pct(ttft, 0.95):.1f} "
            f"N={len(ttft)} time_to_429_ms p50={pct([r['e2e_ms'] for r in rej], 0.5):.1f} "
            f"max={max((r['e2e_ms'] for r in rej), default=float('nan')):.1f}")
for order in ("O1", "O2"):
    base = f"{order}-shadow"
    if base not in by:
        continue
    sb = {r["tag"]: r for r in by[base][1]}
    for arm in ("enforce", "enforce-wrel"):
        n = f"{order}-{arm}"
        if n not in by:
            continue
        eb = {r["tag"]: r for r in by[n][1]}
        common = sorted(set(sb) & set(eb))
        both = [t for t in common if sb[t]["status"] == 200 and eb[t]["status"] == 200 and not sb[t]["error"]
                and not eb[t]["error"]]
        mismatch = [t for t in common if sb[t]["status"] != eb[t]["status"]]
        differ = [t for t in both if sb[t]["content_sha256"] != eb[t]["content_sha256"]]
        say(f"DAY46 P4 card={card} order={order} arm={arm} rows_200_both={len(both)} status_mismatch={len(mismatch)} "
            f"differ={differ[:6]} -> {'PASS' if both and not differ else 'FAIL'}")
with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
