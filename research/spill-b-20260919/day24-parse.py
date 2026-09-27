#!/usr/bin/env python3
"""Day 24 memra#476 cell parser: joins server.log (stamped), client.jsonl and samples.csv into the pre-registered rows
(DAY24.md 1.1) and evaluates A (per-request arithmetic residual), B (device delta versus the real book against the
calibrated floor) and C (Overloaded/OOM count). Prints verbatim rows; never edits inputs."""
import csv, json, os, re, sys
C = sys.argv[1]
log = [l.rstrip("\n") for l in open(os.path.join(C, "server.log"), encoding="utf-8", errors="replace")]
client = [json.loads(l) for l in open(os.path.join(C, "client.jsonl"))]
samples = list(csv.DictReader(open(os.path.join(C, "samples.csv"))))
ready_ms = int(open(os.path.join(C, "ready_ms.txt")).read().strip()); ready_ms = ready_ms // 10**6 if ready_ms > 10**15 else ready_ms
def stamp(l):
    m = re.match(r"(\d+) (.*)", l); return (int(m.group(1)), m.group(2)) if m else (None, l)
# --- boot facts
floor = None; budget_line = None
for _, l in map(stamp, log):
    m = re.search(r"boot calibration done: .*transient floor (\d+)MB", l)
    if m: floor = int(m.group(1)) * (1 << 20)
    if "[admit-predict] shadow armed" in l: budget_line = l
    m = re.search(r"boot calibration (disarmed|skipped|FAILED)", l)
    if m and floor is None: floor = ("static", l)
# --- request-cost lines keyed by ctx (= prompt + max_tokens): bpt, prefill, fixed, cost_pre_draft
cost_lines = {}; draft_lines = []; restore_lines = []
for t, l in map(stamp, log):
    m = re.search(r"\[admission\] request cost: model=\S+ ctx=(\d+) path=(\w+) = (\d+) B/token x ctx \+ (\d+)MB prefill-workspace \+ (\d+)MB fixed = (\d+)MB", l)
    if m: cost_lines[int(m.group(1))] = dict(path=m.group(2), bpt=int(m.group(3)), line=l, t=t)
    if "per-session draft-state charge" in l: draft_lines.append((t, l))
    if "retained prefix plan" in l: restore_lines.append((t, l))
# the MB-rounded line loses bytes; exact per-request cost comes from the admit-predict line pairs below.
admits = []; pending_cost = None; last_draft = None
for t, l in map(stamp, log):
    if "[admission] request cost:" in l: pending_cost = l
    if "per-session draft-state charge" in l:
        md = re.search(r"\+(\d+)MB", l); last_draft = int(md.group(1)) * 10**6 if md else None
    if l.startswith("[admit-predict] id="):
        kv = dict(re.findall(r"(\w+)=(\S+)", l)); kv["t"] = t; kv["line"] = l
        kv["cost_line"] = pending_cost; kv["draft_bytes"] = last_draft; pending_cost = None; admits.append(kv)
ttft = {}
for t, l in map(stamp, log):
    if l.startswith("[ttft] id="):
        kv = dict(re.findall(r"(\w+)=(\S+)", l)); ttft[kv["id"]] = (t, kv)
# --- per-request rows. cost(r) is recovered exactly from consecutive admit lines' booked_real deltas when no retire
# intervened; otherwise from the request-cost line (MB-rounded, flagged '~').
print("== boot"); print(budget_line or "(no shadow armed line)")
print(f"floor={floor}")
print("== request-cost lines (verbatim, one per distinct (ctx, path, cost))")
for ctx in sorted(cost_lines): print(cost_lines[ctx]["line"])
for _, l in draft_lines: print(l)
for _, l in restore_lines: print(l)
print("== admit-predict lines (verbatim)")
for a in admits: print(a["line"])
# shadow book over time: +kv_hat at admit line time, -kv_hat at the [ttft]... completion is client done_ms; match by order
# of admit lines to client rows by submit order (same order on a FIFO admission; verified by prompt token count monotone
# with prompt_chars within a burst).
client_sorted = sorted(client, key=lambda r: r["submit_ms"]); used_rows = set()
def match(a, i):
    P = int(a["prompt"])
    for j, r in enumerate(client_sorted):
        if j in used_rows: continue
        u = r.get("usage") or {}
        if u.get("prompt_tokens") == P: used_rows.add(j); return r
    for j, r in enumerate(client_sorted):
        if j not in used_rows: used_rows.add(j); return r
    return None
events = []; real_events = []
for i, a in enumerate(admits):
    r = match(a, i); a["client"] = r
    # exact physical cost from the next admit line's book delta when it is the very next in-flight entry
    nxt = admits[i + 1] if i + 1 < len(admits) else None
    a["cost_exact"] = None
    if nxt and int(nxt["inflight"]) == int(a["inflight"]) + 1 and int(nxt["booked_bytes"]) == int(a["booked_bytes"]) + int(a["kv_hat"]):
        a["cost_exact"] = int(nxt["booked_real"]) - int(a["booked_real"])
    if r is None: continue
    kv = int(a["kv_hat"]); events.append((a["t"], +kv)); events.append((r["done_ms"], -kv))
    cl = a.get("cost_line"); cost_line_total = None
    if cl:
        mm = re.search(r"= (\d+)MB$", cl); cost_line_total = int(mm.group(1)) * 10**6 if mm else None
        if a.get("draft_bytes") and " path=spec " in cl: cost_line_total += a["draft_bytes"]
    a["cost_used"] = a["cost_exact"] if a["cost_exact"] is not None else cost_line_total
    if a["cost_used"] is not None:
        real_events.append((a["t"], +a["cost_used"])); real_events.append((r["done_ms"], -a["cost_used"]))
real_events.sort()
def real_at(t):
    return sum(d for (tt, d) in real_events if tt <= t)
def inflight_at(t):
    return sum(1 for (tt, d) in events if tt <= t and d > 0) - sum(1 for (tt, d) in events if tt <= t and d < 0)
events.sort()
def shadow_at(t):
    return sum(d for (tt, d) in events if tt <= t)
# idle baseline: samples between warm done and the first 'a' submit
warm_done = next(r["done_ms"] for r in client if r["tag"] == "warm"); a_submit = min(r["submit_ms"] for r in client if r["tag"].startswith("a"))
base = [int(s["mem_used_mib"]) for s in samples if s["mem_used_mib"] and warm_done < int(s["epoch_ms"]) < a_submit]
baseline = max(base) if base else None
print(f"== idle baseline mem.used MiB: n={len(base)} min={min(base) if base else None} max={baseline}")
print("== samples (epoch_ms, mem_used_mib, delta_bytes, real_book(reconstructed), shadow_book(reconstructed), under_real, under_shadow, inflight); every 4th sample printed, all kept in samples.csv")
max_under_real = max_under_shadow = None; peak = None; max_under_real_inflight = max_under_shadow_inflight = None
for i, s in enumerate(samples):
    if not s["mem_used_mib"] or baseline is None: continue
    t = int(s["epoch_ms"]); used = int(s["mem_used_mib"]); delta = (used - baseline) * (1 << 20)
    real = real_at(t)   # reconstructed from the admit lines (the /metrics snapshot is a throttled publish: REPORT-v1)
    sh = shadow_at(t)
    ur = (delta - real) if real is not None else None; us = delta - sh
    if ur is not None and (max_under_real is None or ur > max_under_real[0]): max_under_real = (ur, t, used, real)
    if max_under_shadow is None or us > max_under_shadow[0]: max_under_shadow = (us, t, used, sh)
    if peak is None or used > peak[1]: peak = (t, used)
    if inflight_at(t) > 0:
        if max_under_real_inflight is None or ur > max_under_real_inflight[0]: max_under_real_inflight = (ur, t, used, real)
        if max_under_shadow_inflight is None or us > max_under_shadow_inflight[0]: max_under_shadow_inflight = (us, t, used, sh)
    if i % 4 == 0 and t >= warm_done: print(f"{t} {used} {delta} {real} {sh} {ur} {us} {inflight_at(t)}")
print(f"== peak mem.used MiB={peak[1] if peak else None} at {peak[0] if peak else None}")
print(f"max_underbook_real  = {max_under_real}   (bytes, epoch_ms, mem_used_mib, real_book)")
print(f"max_underbook_shadow = {max_under_shadow}   (bytes, epoch_ms, mem_used_mib, shadow_book)")
print(f"(second, labeled row, not the pre-registered headline) while inflight > 0: max_underbook_real={max_under_real_inflight} max_underbook_shadow={max_under_shadow_inflight}")
print("== per-request physical cost: exact from the next admit line's booked_real delta where available, else the MB-rounded line (~)")
for a in admits:
    r = a.get("client"); print(f"{r['tag'] if r else '?'} P={a['prompt']} cost_exact={a.get('cost_exact')} cost_used={a.get('cost_used')} kv_hat={a['kv_hat']} real/shadow={(a.get('cost_used') or 0) / int(a['kv_hat']):.2f}x")
# --- A: per request residual = (cost - kv_hat) - (ctx(C) - ctx(P+L+8)) under flat geometry, from the request-cost
# line's exact bpt and the admit line's exact P and L. cost is MB-rounded on the line; exact cost = bpt*C + prefill + fixed
# + draft where prefill/fixed are MB-rounded: residual is therefore reported to +-2 MB and compared against 0 at that grain.
print("== A: per request (tag, P, L, C, kv_hat, cost~, gap~, bracket=bpt*(C-(P+L+8)), residual~ = gap - bracket, W~+D~ from line)")
resid = []
for a in admits:
    r = a.get("client"); P = int(a["prompt"]); L = int(a["predicted_completion"]); kv = int(a["kv_hat"])
    tag = r["tag"] if r else "?"
    cl = a.get("cost_line")
    if not cl:
        print(f"{tag} P={P} L={L} kv_hat={kv} (no request-cost line in this admission block: same (ctx, path, cost) key as an earlier request)"); continue
    m = re.search(r"ctx=(\d+) path=(\w+) = (\d+) B/token x ctx \+ (\d+)MB prefill-workspace \+ (\d+)MB fixed = (\d+)MB", cl)
    Cc, path, bpt, W, A, cost = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)) * 10**6, int(m.group(5)) * 10**6, int(m.group(6)) * 10**6
    D = a.get("draft_bytes") or 0 if path == "spec" else 0
    cost_total = cost + D
    gap = cost_total - kv; bracket = bpt * (Cc - (P + L + 8)); res = gap - bracket
    flat = abs((cost - W - A) - bpt * Cc) <= 3 * 10**6  # three MB-rounded terms on the line
    resid.append(res)
    print(f"{tag} P={P} L={L} C={Cc} path={path} bpt={bpt} kv_hat={kv} cost~={cost_total} gap~={gap} bracket={bracket} residual~={res} W~={W} A~={A} D~={D} flat_geometry={flat}")
print(f"A: max |residual~| = {max(abs(x) for x in resid) if resid else None} bytes (tolerance 0 at the line's 1 MB grain, so |residual~| <= 2e6 passes)")
if isinstance(floor, int) and max_under_real:
    print(f"B: max_underbook_real={max_under_real[0]} <= floor={floor}: {'PASS' if max_under_real[0] <= floor else 'FAIL'}")
else:
    print(f"B: not evaluable (floor={floor}, max_under_real={max_under_real})")
ov = sum(1 for l in log if "Overloaded" in l or "CUDA_ERROR_OUT_OF_MEMORY" in l or "out of memory" in l)
print(f"C: Overloaded/OOM lines = {ov}: {'PASS' if ov == 0 else 'FAIL'}")

# --- memra#524: /readyz against the first successful completion (ready_ms is the first 200 from a 0.5 s poll)
print("== readiness (memra#524): /readyz first 200 versus the warm request; boot calibration line; warm [ttft] line")
warm = next((r for r in client if r["tag"] == "warm"), None)
cal = [l for _, l in map(stamp, log) if "[admit-cal] boot calibration" in l]
for l in cal: print(l)
eng = [(t, l) for t, l in map(stamp, log) if "[worker] Engine ready" in l or "[worker] loaded" in l or "[server] listening" in l or "[server] ready" in l]
for t, l in eng[:6]: print(f"{t} {l[:160]}")
if warm:
    print(f"ready_ms={ready_ms} warm_submit_ms={warm['submit_ms']} warm_done_ms={warm['done_ms']} ready_to_first_completion_ms={warm['done_ms'] - ready_ms} (poll grain 500 ms)")
    tw = [(t, kv) for t, (tt, kv) in ((k, v) for k, v in ttft.items()) if int(kv.get("prompt_tokens", -1)) == int((warm.get("usage") or {}).get("prompt_tokens", -2))]
    for t, kv in ttft.values():
        if int(kv.get("prompt_tokens", -1)) == int((warm.get("usage") or {}).get("prompt_tokens", -2)):
            print(f"warm ttft: prime_wait_ms={kv.get('prime_wait_ms')} prime_ms={kv.get('prime_ms')} decode_wait_ms={kv.get('decode_wait_ms')} first_decode_ms={kv.get('first_decode_ms')} total_ms={kv.get('total_ms')}")
print(f"requests={len(client)} non-200={sum(1 for r in client if r['status'] != 200)} admits={len(admits)} ready_ms={ready_ms}")
