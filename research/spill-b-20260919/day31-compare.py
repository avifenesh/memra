#!/usr/bin/env python3
"""Day 31 cross-boot verdicts (DAY31.md section 1.6). Reads the per-boot rows.jsonl and shape.txt written by
run-day26-cell.sh + day31-parse.py, and prints the pre-registered verdict lines. The rules are fixed in DAY31.md
section 1.6 before the first boot and this script implements them as written; it never edits its inputs.

usage: day31-compare.py --card <label> --served-ctx <tokens> --model-ctx <tokens> <order>:<boot-dir> ...
       <order> is O1 or O2; each boot's arm is read from its shape.txt (door env) and its V-BOOT from its REPORT.txt.
"""
import argparse, collections, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("--served-ctx", type=int, required=True)
ap.add_argument("--model-ctx", type=int, required=True)
ap.add_argument("--registry-value", type=int, default=32768, help="R3: the omitted-output default the registries pin")
ap.add_argument("--survey", default="unfilled", help="R4: the survey convention, 'context' or a token count")
ap.add_argument("boots", nargs="+")
a = ap.parse_args()
OPEN = ("i", "iii", "iv-a", "iv-b")
BOUNDED = ("warmup", "ii")
SLACK = 8

boots = []
for spec in a.boots:
    order, d = spec.split(":", 1)
    shape = dict(re.findall(r"(\w+)=(\S+)", open(os.path.join(d, "shape.txt")).read()))
    on = shape.get("admit_by_memory") == "1"
    v = int(shape["open_output_tokens"]) if on else None
    rep = open(os.path.join(d, "REPORT.txt")).read()
    vb = re.search(r"^DAY31 V-BOOT .*$", rep, re.M)
    rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
    boots.append(dict(order=order, dir=d, name=os.path.basename(d.rstrip("/")), on=on, v=v,
                      arm=f"on{v}" if on else "off", vboot=vb.group(0) if vb else None,
                      boot_pass=bool(vb and vb.group(0).endswith("PASS")), rows=rows, report=rep))

print(f"== card {a.card} served_ctx={a.served_ctx} model_ctx={a.model_ctx}")
for b in boots:
    print(b["vboot"] or f"DAY31 V-BOOT boot={b['name']} arm={b['arm']} -> FAIL (no V-BOOT line in REPORT.txt)")
good = [b for b in boots if b["boot_pass"]]
excluded = [b["name"] for b in boots if not b["boot_pass"]]
if excluded:
    print(f"excluded boots (V-BOOT FAIL): {excluded}")


def by_tag(b):
    return {r["tag"]: r for r in b["rows"] if r["class"] != "burst"}


# V-ALLOC: every own cost line against the arithmetic of section 1.6
alloc_lines = []
for b in good:
    n = ok = 0; bad = []
    for r in b["rows"]:
        if r["class"] == "burst" or r.get("cost_src") != "own" or r.get("P") is None or r.get("booked_ctx") is None:
            continue
        P = r["P"]
        if r["class"] in BOUNDED:
            exp = P + r["max_tokens"] + 64
        elif b["on"]:
            exp = max(min(P + b["v"] + SLACK, a.model_ctx), P + SLACK) + 64
        else:
            exp = (a.served_ctx if P + 16 <= a.served_ctx else P + a.served_ctx) + 64
        n += 1
        if r["booked_ctx"] == exp:
            ok += 1
        else:
            bad.append(f"{r['tag']}:{r['booked_ctx']}!={exp}")
    alloc_lines.append(f"DAY31 V-ALLOC card={a.card} boot={b['name']} arm={b['arm']} judged={n} match={ok} "
                       f"mismatch={n - ok}{' ' + ','.join(bad[:6]) if bad else ''} -> {'PASS' if n and ok == n else 'FAIL'}")
print("== V-ALLOC"); print("\n".join(alloc_lines))

# V-ID and V-TRUNC: every ON boot against the OFF boot of the same order
print("== V-ID / V-TRUNC")
trunc_by_v = collections.defaultdict(lambda: collections.Counter())
id_pass_all = True; truncg_pass_all = True
for o in ("O1", "O2"):
    offs = [b for b in good if b["order"] == o and not b["on"]]
    if not offs:
        print(f"order {o}: no OFF boot passed V-BOOT; V-ID not evaluable for this order")
        id_pass_all = False
        continue
    off = by_tag(offs[0])
    for b in [x for x in good if x["order"] == o and x["on"]]:
        on = by_tag(b); v = b["v"]
        elig = eq = 0; differ = []; slack_eq = slack_diff = 0
        tw = nt = both_len = cut = 0; truncg_bad = []
        for tag, r1 in on.items():
            r0 = off.get(tag)
            if r1["status"] == 200 and r1.get("finish_reason") is None:
                cut += 1
            if r0 is None or r0["status"] != 200 or r1["status"] != 200:
                continue
            same_prompt = r0["prompt_sha256"] == r1["prompt_sha256"]
            if r1["class"] in OPEN and r1.get("finish_reason") == "length":
                if not same_prompt:
                    nt += 1
                elif r0.get("finish_reason") == "stop":
                    tw += 1
                elif r0.get("finish_reason") == "length":
                    both_len += 1
                if r1["G"] is not None and not (v <= r1["G"] <= v + SLACK):
                    truncg_bad.append(f"{tag}:G={r1['G']}")
            if not same_prompt:
                continue
            match = (r0["message_sha256"] == r1["message_sha256"] and r0["G"] == r1["G"]
                     and r0.get("finish_reason") == r1.get("finish_reason"))
            if r1["class"] in BOUNDED or (r1["class"] in OPEN and r0.get("finish_reason") == "stop" and r0["G"] <= v):
                elig += 1
                if match:
                    eq += 1
                else:
                    differ.append(tag)
            elif r1["class"] in OPEN and r0.get("finish_reason") == "stop" and v < r0["G"] <= v + SLACK:
                if match:
                    slack_eq += 1
                else:
                    slack_diff += 1
        ok = elig > 0 and eq == elig
        id_pass_all &= ok
        truncg_pass_all &= not truncg_bad
        print(f"DAY31 V-ID card={a.card} order={o} arm={b['arm']} vs=off eligible={elig} equal={eq} differ={len(differ)}"
              f"{' ' + ','.join(differ[:8]) if differ else ''} slack_equal={slack_eq} slack_differ={slack_diff} -> {'PASS' if ok else 'FAIL'}")
        print(f"DAY31 V-TRUNC card={a.card} order={o} arm={b['arm']} truncated_with_twin={tw} length_without_twin={nt} "
              f"length_both={both_len} deadline_cut={cut} truncated_G_outside_[v,v+8]={len(truncg_bad)}"
              f"{' ' + ','.join(truncg_bad[:6]) if truncg_bad else ''}")
        trunc_by_v[v]["with_twin"] += tw; trunc_by_v[v]["without_twin"] += nt
        trunc_by_v[v]["seen"] += 1

# natural G under OFF: every open-class OFF row, both orders, status 200
print("== G-NATURAL (door OFF, open classes, both orders)")
gall = []
for b in [x for x in good if not x["on"]]:
    for r in b["rows"]:
        if r["class"] in OPEN and r["status"] == 200 and r.get("G") is not None:
            gall.append((r["class"], r["G"], r.get("finish_reason"), b["order"], r["tag"]))


def dist(gs):
    gs = sorted(gs)
    if not gs:
        return "n=0"
    q = lambda p: gs[min(len(gs) - 1, int(p * len(gs)))]
    return (f"n={len(gs)} min={gs[0]} p50={q(0.5)} p90={q(0.9)} p99={q(0.99)} max={gs[-1]} "
            f"over2048={sum(g > 2048 for g in gs)} over8192={sum(g > 8192 for g in gs)} over32768={sum(g > 32768 for g in gs)}")


for cls in OPEN:
    print(f"DAY31 G-NATURAL card={a.card} class={cls} {dist([g for c, g, _, _, _ in gall if c == cls])}")
print(f"DAY31 G-NATURAL card={a.card} class=all-open {dist([g for _, g, _, _, _ in gall])} "
      f"finish={dict(collections.Counter(f for _, _, f, _, _ in gall))}")

# burst concurrency and the refusal contract
print("== V-CONC / V-RETRY")
retry_pass_all = True
for b in good:
    m = re.search(r"^window_ms=.*$", b["report"], re.M); m2 = re.search(r"^active_sessions_max=.*$", b["report"], re.M)
    m3 = re.search(r"^admit_mem_id_lines=.*$", b["report"], re.M)
    burst = [r for r in b["rows"] if r["class"] == "burst"]
    n429 = [r for r in burst if r["status"] == 429]
    ra_ok = all(r.get("retry_after") and r["retry_after"].isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in n429)
    refuse = int(re.search(r"'refuse': (\d+)", m3.group(0)).group(1)) if m3 and "'refuse'" in m3.group(0) else 0
    other = [r for r in burst if r["status"] not in (200, 429)]
    ok = ra_ok and len(n429) == refuse and (b["on"] or not n429)
    retry_pass_all &= ok
    print(f"DAY31 V-CONC card={a.card} order={b['order']} arm={b['arm']} {m.group(0) if m else ''} {m2.group(0) if m2 else ''}")
    print(f"DAY31 V-RETRY card={a.card} order={b['order']} arm={b['arm']} r429={len(n429)} refuse_lines={refuse} "
          f"retry_after_in_1_60={ra_ok} other_non200={len(other)} -> {'PASS' if ok else 'FAIL'}")

# selection rules: stated, never applied to a default
print("== SELECT (stated, not chosen)")
vs = sorted(trunc_by_v)
r1 = next((v for v in vs if trunc_by_v[v]["seen"] == 2 and trunc_by_v[v]["with_twin"] + trunc_by_v[v]["without_twin"] == 0), None)
stop_g = [g for _, g, f, _, _ in gall if f == "stop"]
gmax = max(stop_g) if stop_g else None
r2 = next((v for v in vs if gmax is not None and v >= gmax), None)
print(f"DAY31 SELECT card={a.card} R1_smallest_zero_truncation={r1 if r1 is not None else 'none of ' + str(vs)} "
      f"R2_smallest_v_ge_max_natural_G={r2 if r2 is not None else 'none of ' + str(vs)} (max_natural_G={gmax}) "
      f"R3_registry={a.registry_value} R4_survey={a.survey}")
allp = (not excluded) and id_pass_all and truncg_pass_all and retry_pass_all and all(l.endswith("PASS") for l in alloc_lines)
print(f"DAY31 V-DOOR card={a.card} boots={len(boots)} excluded={len(excluded)} v_id_all={id_pass_all} "
      f"v_alloc_all={all(l.endswith('PASS') for l in alloc_lines)} v_trunc_g_all={truncg_pass_all} v_retry_all={retry_pass_all} "
      f"-> {'PASS' if allp else 'FAIL'}")
