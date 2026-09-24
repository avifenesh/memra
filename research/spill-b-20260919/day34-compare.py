#!/usr/bin/env python3
"""Day 34 cross-boot verdicts (DAY34.md section 1.6): day32-compare.py with four changes, each marked "day 34":
V-ALLOC reads the engine's booking `max(ctx_cap, P + budget + 64)`; V-OOM (no client-visible CUDA OOM, no 503 in any
row); G-BOOK (every door admit line within its booked device_free, with pending_prime); PARK (a prefill- or step-OOM
park is counted and reported; when one fires every row of the boot must end 200 or a typed 429). Labels read DAY34.

Day 32 cross-boot verdicts (DAY32.md section 1.6). Reads the per-boot rows.jsonl, shape.txt, REPORT.txt and
server.log (or server.log.gz) written by run-day26-cell.sh + day31-parse.py, and prints the pre-registered verdict
lines. The rules are fixed in DAY32.md section 1.6 before the first boot and this script implements them as written;
it never edits its inputs.

Against day31-compare.py (DAY32.md 1.6 lists each change): the truncation band is G = v exactly; every ON open
`length` row lands in exactly one V-TRUNC term, with a conservation check; V-CRASH scans every boot's server.log;
V-RETRY needs at least one burst response; V-DOOR carries V-CRASH; R1 and R2 read only admissible boots and print
why a value is not admissible.

usage: day32-compare.py --card <label> --served-ctx <tokens> --model-ctx <tokens> \
         --registry-value 32768 --survey context <order>:<boot-dir> ...
       <order> is O1 or O2; each boot's arm is read from its shape.txt (door env) and its V-BOOT from its REPORT.txt.
"""
import argparse, collections, gzip, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("--served-ctx", type=int, required=True)
ap.add_argument("--model-ctx", type=int, required=True)
ap.add_argument("--registry-value", type=int, required=True, help="R3: the omitted-output default the registries pin")
ap.add_argument("--survey", required=True, help="R4: the survey convention, 'context' or a token count")
ap.add_argument("boots", nargs="+")
a = ap.parse_args()
OPEN = ("i", "iii", "iv-a", "iv-b")
BOUNDED = ("warmup", "ii")
SLACK = 8
CRASH = [("panicked", re.compile(r"panicked")), ("worker_PANIC", re.compile(r"\[worker\] PANIC")),
         ("worker_FATAL", re.compile(r"\[worker\] FATAL")), ("respawn", re.compile(r"\[worker\] respawn")),
         ("argmax_sentinel", re.compile(r"argmax sentinel")), ("spec_verify_refused", re.compile(r"spec verify refused"))]


def open_log(d):
    p = os.path.join(d, "server.log")
    if os.path.exists(p):
        return "server.log", open(p, encoding="utf-8", errors="replace")
    return "server.log.gz", gzip.open(p + ".gz", "rt", encoding="utf-8", errors="replace")


PARK = re.compile(r"\[admit-mem\] prefill OOM parked|\[admit-oom\] step OOM parked")


def crash_scan(d):
    name, f = open_log(d)
    counts = collections.Counter(); first = None
    oom = 0; first_oom = None; parks = collections.Counter(); admits = []; mem_id = 0
    with f:
        for n, line in enumerate(f, 1):
            for key, rx in CRASH:
                if rx.search(line):
                    counts[key] += 1
                    if first is None:
                        # quote up to the first em dash, as the engine's longer lines carry one
                        first = f"{name}:{n} {line.rstrip().split(chr(0x2014))[0].strip()[:220]}"
            # day 34: a park receipt carries the OOM text but is a recovery, not a client-visible OOM
            pm = PARK.search(line)
            if pm:
                parks["prefill" if "prefill OOM" in pm.group(0) else "step"] += 1
            elif "CUDA_ERROR_OUT_OF_MEMORY" in line:
                oom += 1
                first_oom = first_oom or f"{name}:{n} {line.rstrip().split(chr(0x2014))[0].strip()[:220]}"
            if "[admit-mem] id=" in line:
                mem_id += 1
                if "verdict=admit " in line:
                    kv = dict(re.findall(r"(\w+)=(\S+)", line)); kv["at"] = f"{name}:{n}"; admits.append(kv)
    return counts, first, dict(oom=oom, first_oom=first_oom, parks=parks, admits=admits, mem_id=mem_id)


boots = []
for spec in a.boots:
    order, d = spec.split(":", 1)
    shape = dict(re.findall(r"(\w+)=(\S+)", open(os.path.join(d, "shape.txt")).read()))
    on = shape.get("admit_by_memory") == "1"
    v = int(shape["open_output_tokens"]) if on else None
    rep = open(os.path.join(d, "REPORT.txt")).read()
    vb = re.search(r"^DAY31 V-BOOT .*$", rep, re.M)  # the per-boot parser is day31-parse.py, unchanged
    rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
    counts, first, scan = crash_scan(d)
    boots.append(dict(order=order, dir=d, name=os.path.basename(d.rstrip("/")), on=on, v=v,
                      arm=f"on{v}" if on else "off", vboot=vb.group(0) if vb else None,
                      boot_pass=bool(vb and vb.group(0).endswith("PASS")), rows=rows, report=rep,
                      crash=counts, crash_first=first, crash_pass=not counts, scan=scan))

print(f"== card {a.card} served_ctx={a.served_ctx} model_ctx={a.model_ctx}")
for b in boots:
    print((b["vboot"] or f"DAY31 V-BOOT boot={b['name']} arm={b['arm']} -> FAIL (no V-BOOT line in REPORT.txt)")
          .replace("DAY31 V-BOOT", "DAY34 V-BOOT", 1))
good = [b for b in boots if b["boot_pass"]]
excluded = [b["name"] for b in boots if not b["boot_pass"]]
if excluded:
    print(f"excluded boots (V-BOOT FAIL): {excluded}")

print("== V-CRASH")
for b in boots:
    print(f"DAY34 V-CRASH card={a.card} boot={b['name']} arm={b['arm']} lines={sum(b['crash'].values())} "
          f"by_pattern={dict(b['crash'])} first={b['crash_first'] or 'none'} -> {'PASS' if b['crash_pass'] else 'FAIL'}")
crash_pass_all = all(b["crash_pass"] for b in boots)

# day 34: V-OOM, G-BOOK, PARK
print("== V-OOM / G-BOOK / PARK")
oom_pass_all = book_pass_all = park_pass_all = True
for b in boots:
    sc = b["scan"]; st = collections.Counter(r["status"] for r in b["rows"])
    ok = sc["oom"] == 0 and st.get(503, 0) == 0
    oom_pass_all &= ok
    print(f"DAY34 V-OOM card={a.card} boot={b['name']} arm={b['arm']} oom_lines={sc['oom']} status_503={st.get(503, 0)} "
          f"first_oom={sc['first_oom'] or 'none'} -> {'PASS' if ok else 'FAIL'}")
    adm = sc["admits"]
    bad = [x["at"] for x in adm if not (x.get("est_bytes", "").isdigit() and x.get("device_free", "").isdigit()
                                        and int(x["est_bytes"]) <= int(x["device_free"]) and "pending_prime" in x)]
    if b["on"]:
        ok = bool(adm) and not bad
    else:
        ok = sc["mem_id"] == 0
    book_pass_all &= ok
    pp = [int(x["pending_prime"]) for x in adm if x.get("pending_prime", "").isdigit()]
    print(f"DAY34 G-BOOK card={a.card} boot={b['name']} arm={b['arm']} admit_lines={len(adm)} mem_id_lines={sc['mem_id']} "
          f"est_over_booked_free={len(bad)}{' ' + ','.join(bad[:4]) if bad else ''} pending_prime_max={max(pp) if pp else 0} "
          f"-> {'PASS' if ok else 'FAIL'}")
    n = sum(sc["parks"].values())
    ends = collections.Counter(r["status"] for r in b["rows"])
    burst429 = [r for r in b["rows"] if r["status"] == 429]
    typed = all(str(r.get("retry_after") or "").isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in burst429)
    ok = n == 0 or (set(ends) <= {200, 429} and typed)
    park_pass_all &= ok
    print(f"DAY34 PARK card={a.card} boot={b['name']} arm={b['arm']} prefill_parks={sc['parks'].get('prefill', 0)} "
          f"step_parks={sc['parks'].get('step', 0)} row_statuses={dict(ends)} -> {'PASS' if ok else 'FAIL'}")


def by_tag(b):
    return {r["tag"]: r for r in b["rows"] if r["class"] != "burst"}


# V-ALLOC: every own cost line against the arithmetic of section 1.6 (the code's form of the ON charge)
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
            # day 34: the engine books max(ctx_cap, need), need = P + budget + 64, budget = ctx_cap - P - 8 on the
            # door-ON open arm (memra#659), so the line reads max(ctx_cap, ctx_cap + 56)
            cap = min(P + b["v"] + SLACK, a.model_ctx)
            exp = max(cap, P + (cap - P - SLACK if cap - P > SLACK else cap - P) + 64)
        else:
            # door OFF: budget = ctx_cap - P, so max(ctx_cap, ctx_cap + 64) = ctx_cap + 64 (day 32's form)
            exp = (a.served_ctx if P + 16 <= a.served_ctx else P + a.served_ctx) + 64
        n += 1
        if r["booked_ctx"] == exp:
            ok += 1
        else:
            bad.append(f"{r['tag']}:{r['booked_ctx']}!={exp}")
    alloc_lines.append(f"DAY34 V-ALLOC card={a.card} boot={b['name']} arm={b['arm']} judged={n} match={ok} "
                       f"mismatch={n - ok}{' ' + ','.join(bad[:6]) if bad else ''} -> {'PASS' if n and ok == n else 'FAIL'}")
print("== V-ALLOC"); print("\n".join(alloc_lines))

# V-ID and V-TRUNC: every ON boot against the OFF boot of the same order
print("== V-ID / V-TRUNC")
trunc = {}          # (order, v) -> counters
id_pass_all = True; band_pass_all = True; conserve_all = True
for o in ("O1", "O2"):
    offs = [b for b in good if b["order"] == o and not b["on"]]
    if not offs:
        print(f"order {o}: no OFF boot passed V-BOOT; V-ID not evaluable for this order")
        id_pass_all = False
        continue
    off = by_tag(offs[0])
    for b in [x for x in good if x["order"] == o and x["on"]]:
        on = by_tag(b); v = b["v"]
        elig = eq = 0; differ = []; bnd_eq = bnd_diff = 0
        c = collections.Counter(); band_bad = []; cut = 0; lens = []
        for tag, r1 in on.items():
            r0 = off.get(tag)
            if r1["status"] == 200 and r1.get("finish_reason") is None:
                cut += 1
            # every ON open `length` row: the band, then exactly one term
            if r1["class"] in OPEN and r1["status"] == 200 and r1.get("finish_reason") == "length":
                lens.append(tag)
                if r1["G"] != v:
                    band_bad.append(f"{tag}:G={r1['G']}")
                if r0 is None or r0["status"] != 200:
                    c["length_twin_failed"] += 1
                elif r0["prompt_sha256"] != r1["prompt_sha256"]:
                    c["length_prompt_differs"] += 1
                elif r0.get("finish_reason") == "stop":
                    c["truncated_with_twin"] += 1
                elif r0.get("finish_reason") == "length":
                    c["length_both"] += 1
                else:
                    c["length_twin_other_finish"] += 1
            if r0 is None or r0["status"] != 200 or r1["status"] != 200 or r0["prompt_sha256"] != r1["prompt_sha256"]:
                continue
            match = (r0["message_sha256"] == r1["message_sha256"] and r0["G"] == r1["G"]
                     and r0.get("finish_reason") == r1.get("finish_reason"))
            if r1["class"] in BOUNDED or (r1["class"] in OPEN and r0.get("finish_reason") == "stop" and r0["G"] < v):
                elig += 1
                if match:
                    eq += 1
                else:
                    differ.append(tag)
            elif r1["class"] in OPEN and r0.get("finish_reason") == "stop" and r0["G"] == v:
                if r0["message_sha256"] == r1["message_sha256"] and r0["G"] == r1["G"]:
                    bnd_eq += 1
                else:
                    bnd_diff += 1
        # the burst: every ON burst `length` row is in the band too
        for r in b["rows"]:
            if r["class"] == "burst" and r["status"] == 200 and r.get("finish_reason") == "length" and r["G"] != v:
                band_bad.append(f"{r['tag']}:G={r['G']}")
        terms = ("truncated_with_twin", "length_both", "length_prompt_differs", "length_twin_failed",
                 "length_twin_other_finish")
        conserve = sum(c[t] for t in terms) == len(lens)
        non200 = [f"{t}:{r['status']}" for t, r in on.items() if r["class"] in OPEN and r["status"] != 200]
        ok = elig > 0 and eq == elig
        id_pass_all &= ok; band_pass_all &= not band_bad; conserve_all &= conserve
        trunc[(o, v)] = dict(c, non200=len(non200), crash=not b["crash_pass"], boot=b["name"],
                             off_boot_pass=True)
        print(f"DAY34 V-ID card={a.card} order={o} arm={b['arm']} vs=off eligible={elig} equal={eq} differ={len(differ)}"
              f"{' ' + ','.join(differ[:8]) if differ else ''} boundary_equal={bnd_eq} boundary_differ={bnd_diff} "
              f"-> {'PASS' if ok else 'FAIL'}")
        print(f"DAY34 V-TRUNC card={a.card} order={o} arm={b['arm']} on_length_rows={len(lens)} "
              + " ".join(f"{t}={c[t]}" for t in terms)
              + f" length_without_twin={c['length_prompt_differs'] + c['length_twin_failed']} conserved={conserve} "
              f"deadline_cut={cut} open_non200={len(non200)}{' ' + ','.join(non200[:8]) if non200 else ''} "
              f"G_outside_band_v={len(band_bad)}{' ' + ','.join(band_bad[:6]) if band_bad else ''}")

# natural G under OFF: every open-class row of the OFF boots that passed V-BOOT and V-CRASH, status 200
print("== G-NATURAL (door OFF, open classes, both orders, admissible OFF boots)")
off_adm = [x for x in good if not x["on"] and x["crash_pass"]]
gall = []
for b in off_adm:
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


print(f"DAY34 G-NATURAL card={a.card} off_boots={[b['name'] for b in off_adm]}")
for cls in OPEN:
    print(f"DAY34 G-NATURAL card={a.card} class={cls} {dist([g for c, g, _, _, _ in gall if c == cls])} "
          f"finish={dict(collections.Counter(f for c, _, f, _, _ in gall if c == cls))}")
print(f"DAY34 G-NATURAL card={a.card} class=all-open {dist([g for _, g, _, _, _ in gall])} "
      f"finish={dict(collections.Counter(f for _, _, f, _, _ in gall))}")

# burst concurrency and the refusal contract
print("== V-CONC / V-RETRY")
retry_pass_all = True
for b in good:
    m = re.search(r"^window_ms=.*$", b["report"], re.M); m2 = re.search(r"^active_sessions_max=.*$", b["report"], re.M)
    m3 = re.search(r"^admit_mem_id_lines=.*$", b["report"], re.M)
    burst = [r for r in b["rows"] if r["class"] == "burst"]
    responded = [r for r in burst if r["status"] is not None]
    n429 = [r for r in burst if r["status"] == 429]
    ra_ok = all(r.get("retry_after") and r["retry_after"].isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in n429)
    refuse = int(re.search(r"'refuse': (\d+)", m3.group(0)).group(1)) if m3 and "'refuse'" in m3.group(0) else 0
    other = [r for r in burst if r["status"] not in (200, 429)]
    ok = bool(responded) and ra_ok and len(n429) == refuse and (b["on"] or not n429)
    retry_pass_all &= ok
    print(f"DAY34 V-CONC card={a.card} order={b['order']} arm={b['arm']} {m.group(0) if m else ''} {m2.group(0) if m2 else ''}")
    print(f"DAY34 V-RETRY card={a.card} order={b['order']} arm={b['arm']} burst_responses={len(responded)}/{len(burst)} "
          f"r429={len(n429)} refuse_lines={refuse} retry_after_in_1_60={ra_ok} other_non200={len(other)} "
          f"-> {'PASS' if ok else 'FAIL'}")

# selection rules: stated, never applied to a default
print("== SELECT (stated, not chosen)")
vs = sorted({v for (_, v) in trunc} | {b["v"] for b in boots if b["on"]})
off_pass = {b["order"] for b in good if not b["on"]}
adm, inadm, truncating = [], [], []
for v in vs:
    why = []
    for o in ("O1", "O2"):
        t = trunc.get((o, v))
        if o not in off_pass:
            why.append(f"{o}:off_boot_not_passed")
        elif t is None:
            why.append(f"{o}:on_boot_not_passed")
        else:
            if t["crash"]:
                why.append(f"{o}:crash")
            if t["non200"]:
                why.append(f"{o}:open_non200={t['non200']}")
    if why:
        inadm.append(f"{v}:{'+'.join(why)}")
        continue
    adm.append(v)
    n = sum(trunc[(o, v)].get(k, 0) for o in ("O1", "O2") for k in
            ("truncated_with_twin", "length_prompt_differs", "length_twin_failed", "length_twin_other_finish"))
    if n:
        truncating.append(f"{v}:{n}")
zero = [v for v in adm if not any(t.startswith(f"{v}:") for t in truncating)]
r1 = (str(zero[0]) if zero else "none") + f" admissible={adm} inadmissible={inadm} truncating={truncating}"
stop_g = [g for _, g, f, _, _ in gall if f == "stop"]
gmax = max(stop_g) if stop_g else None
if not off_adm:
    r2 = "none (no admissible OFF boot)"
else:
    r2v = next((v for v in vs if gmax is not None and v >= gmax), None)
    r2 = f"{r2v if r2v is not None else 'none of ' + str(vs)} (max_natural_G={gmax})"
print(f"DAY34 SELECT card={a.card} R1_smallest_zero_truncation={r1}")
print(f"DAY34 SELECT card={a.card} R2_smallest_v_ge_max_natural_G={r2}")
print(f"DAY34 SELECT card={a.card} R3_registry={a.registry_value} R4_survey={a.survey}")
alloc_all = all(l.endswith("PASS") for l in alloc_lines)
allp = (not excluded) and crash_pass_all and id_pass_all and band_pass_all and conserve_all and retry_pass_all and alloc_all \
    and oom_pass_all and book_pass_all and park_pass_all
print(f"DAY34 V-DOOR card={a.card} boots={len(boots)} excluded={len(excluded)} v_crash_all={crash_pass_all} "
      f"v_id_all={id_pass_all} v_alloc_all={alloc_all} v_trunc_band_all={band_pass_all} v_trunc_conserved_all={conserve_all} "
      f"v_retry_all={retry_pass_all} v_oom_all={oom_pass_all} g_book_all={book_pass_all} park_all={park_pass_all} "
      f"-> {'PASS' if allp else 'FAIL'}")
