#!/usr/bin/env python3
"""Day 28 memra#476 shape-walk parser (DAY28.md 1.4): joins server.log (stamped), client.jsonl and samples.csv into the
pre-registered rows and evaluates G1 (growth attributable to the walk's shapes against 10 % of the floor), G2 (in-flight
device delta versus the real book against the floor), G3 (per-admit arithmetic residual) and G4 (Overloaded/OOM count).
Prints verbatim rows; never edits inputs."""
import csv, json, os, re, sys
C = sys.argv[1]
log = [l.rstrip("\n") for l in open(os.path.join(C, "server.log"), encoding="utf-8", errors="replace")]
client = [json.loads(l) for l in open(os.path.join(C, "client.jsonl"))]
samples = list(csv.DictReader(open(os.path.join(C, "samples.csv"))))
ready_ms = int(open(os.path.join(C, "ready_ms.txt")).read().strip()); ready_ms = ready_ms // 10**6 if ready_ms > 10**15 else ready_ms
def stamp(l):
    m = re.match(r"(\d+) (.*)", l); return (int(m.group(1)), m.group(2)) if m else (None, l)
S = list(map(stamp, log))
# --- boot facts
floor = None; budget_line = None; cal_line = None
for _, l in S:
    m = re.search(r"boot calibration done: .*transient floor (\d+)MB", l)
    if m: floor = int(m.group(1)) * (1 << 20); cal_line = l
    if "[admit-predict] shadow armed" in l: budget_line = l
    m = re.search(r"boot calibration (disarmed|skipped|FAILED)", l)
    if m and floor is None: floor = ("static", l)
print("== boot"); print(budget_line or "(no shadow armed line)"); print(cal_line or "(no calibration line)"); print(f"floor={floor}")
# --- pool grows and draft-state flips (the growth term), verbatim, with bytes
print("== [fa-pool] grow lines (verbatim) with g_i = 4 x (o_i + 2 x m_i) and the cumulative G_fa after each")
grows = []; G = 0
for t, l in S:
    m = re.search(r"\[fa-pool\] grow #(\d+) dev=(\d+) o_len (\d+) -> (\d+) ml_len (\d+) -> (\d+)", l)
    if m:
        o, ml = int(m.group(4)), int(m.group(6)); g = 4 * (o + 2 * ml); G += g
        grows.append((t, g, G, l)); print(f"{t} {l}  g_i={g} G_fa={G}")
flips = []
for t, l in S:
    m = re.search(r"draft-session state high-water: (\d+)MB \(max of parked delta (\d+)MB and capture-time pool peak (\d+)MB", l)
    if m: flips.append((t, int(m.group(1)) * (1 << 20), l)); print(f"{t} {l}")
    if "capture appetite floor raised" in l: print(f"{t} {l}")
vg = [l for _, l in S if "[spec-vg]" in l]
print(f"[spec-vg] lines: {len(vg)}" + (f"  first: {vg[0]}" if vg else ""))
G_ready = sum(g for t, g, _, _ in grows if t is not None and t <= ready_ms)
def growth_at(t):
    gf = sum(g for tg, g, _, _ in grows if tg is not None and ready_ms < tg <= t)
    hw = [b for tf, b, _ in flips if tf is not None and tf <= t]
    hw_ready = [b for tf, b, _ in flips if tf is not None and tf <= ready_ms]
    dD = (max(hw) - max(hw_ready)) if hw and hw_ready and max(hw) > max(hw_ready) else (max(hw) if hw and not hw_ready else 0)
    return gf + dD, gf, dD
print(f"G_fa at ready (the probe's grows, inside the floor's measurement) = {G_ready}; grows after ready = {sum(1 for t, *_ in grows if t and t > ready_ms)}")
# --- request-cost and admit lines
admits = []; pending_cost = None; last_draft = None; cost_lines = []
for t, l in S:
    if "[admission] request cost:" in l: pending_cost = l; cost_lines.append(l)
    if "per-session draft-state charge" in l:
        md = re.search(r"\+(\d+)MB", l); last_draft = int(md.group(1)) * 10**6 if md else None
    if l.startswith("[admit-predict] id="):
        kv = dict(re.findall(r"(\w+)=(\S+)", l)); kv["t"] = t; kv["line"] = l
        kv["cost_line"] = pending_cost; kv["draft_bytes"] = last_draft; pending_cost = None; admits.append(kv)
print("== request-cost lines (verbatim)");  [print(l) for l in cost_lines]
print("== admit-predict lines (verbatim)"); [print(a["line"]) for a in admits]
client_sorted = sorted(client, key=lambda r: r["submit_ms"]); used_rows = set()
def match(a):
    P = int(a["prompt"])
    for j, r in enumerate(client_sorted):
        if j in used_rows: continue
        if (r.get("usage") or {}).get("prompt_tokens") == P: used_rows.add(j); return r
    for j, r in enumerate(client_sorted):
        if j not in used_rows: used_rows.add(j); return r
    return None
events = []; real_events = []
for i, a in enumerate(admits):
    r = match(a); a["client"] = r
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
real_events.sort(); events.sort()
real_at = lambda t: sum(d for (tt, d) in real_events if tt <= t)
shadow_at = lambda t: sum(d for (tt, d) in events if tt <= t)
inflight_at = lambda t: sum(1 for (tt, d) in events if tt <= t and d > 0) - sum(1 for (tt, d) in events if tt <= t and d < 0)
warm_done = next(r["done_ms"] for r in client if r["tag"] == "warm"); s1_submit = min(r["submit_ms"] for r in client if r["tag"].startswith("S1"))
base = [int(s["mem_used_mib"]) for s in samples if s["mem_used_mib"] and warm_done < int(s["epoch_ms"]) < s1_submit]
baseline = max(base) if base else None
print(f"== idle baseline mem.used MiB: n={len(base)} min={min(base) if base else None} max={baseline}")
print("== samples (epoch_ms, mem_used_mib, delta_bytes, real_book, shadow_book, under_real, under_shadow, inflight, growth(t), pool_reserved, pool_cached); every 4th printed, all kept in samples.csv")
mur = mus = None; peak = None; mur_in = mus_in = None; mur_in_g = None
for i, s in enumerate(samples):
    if not s["mem_used_mib"] or baseline is None: continue
    t = int(s["epoch_ms"]); used = int(s["mem_used_mib"]); delta = (used - baseline) * (1 << 20)
    real = real_at(t); sh = shadow_at(t); ur = delta - real; us = delta - sh; gr, _, _ = growth_at(t); inf = inflight_at(t)
    if mur is None or ur > mur[0]: mur = (ur, t, used, real)
    if mus is None or us > mus[0]: mus = (us, t, used, sh)
    if peak is None or used > peak[1]: peak = (t, used)
    if inf > 0:
        if mur_in is None or ur > mur_in[0]: mur_in = (ur, t, used, real)
        if mus_in is None or us > mus_in[0]: mus_in = (us, t, used, sh)
        if mur_in_g is None or (ur - gr) > mur_in_g[0]: mur_in_g = (ur - gr, t, used, real, gr)
    if i % 4 == 0 and t >= warm_done: print(f"{t} {used} {delta} {real} {sh} {ur} {us} {inf} {gr} {s.get('pool_reserved','')} {s.get('pool_cached','')}")
print(f"== peak mem.used MiB={peak[1] if peak else None} at {peak[0] if peak else None}")
print(f"max_underbook_real   = {mur}   (bytes, epoch_ms, mem_used_mib, real_book), all samples")
print(f"max_underbook_shadow = {mus}   (bytes, epoch_ms, mem_used_mib, shadow_book), all samples")
print(f"while inflight > 0: max_underbook_real={mur_in} max_underbook_shadow={mus_in}; real minus growth(t): {mur_in_g}")
# --- device step across each grow after ready: the sample right before and right after
print("== device step across each post-ready grow (MiB used before -> after, growth bytes g_i)")
for tg, g, Gc, l in grows:
    if tg is None or tg <= ready_ms: continue
    before = [int(s["mem_used_mib"]) for s in samples if s["mem_used_mib"] and int(s["epoch_ms"]) <= tg][-1:] or [None]
    after = [int(s["mem_used_mib"]) for s in samples if s["mem_used_mib"] and int(s["epoch_ms"]) > tg][:1] or [None]
    print(f"grow at {tg}: used {before[0]} -> {after[0]} MiB; g_i={g} ({g / (1 << 20):.1f} MiB); inflight={inflight_at(tg)}")
# --- G3: per-admit residual
print("== per-request physical cost and G3 residual (tag, P, L, C, path, kv_hat, booked_real delta / cost~, residual)")
resid = []
for a in admits:
    r = a.get("client"); P = int(a["prompt"]); L = int(a["predicted_completion"]); kv = int(a["kv_hat"]); tag = r["tag"] if r else "?"
    cl = a.get("cost_line")
    if not cl:
        print(f"{tag} P={P} L={L} kv_hat={kv} cost_exact={a.get('cost_exact')} (no request-cost line in this block)")
        if a.get("cost_exact") is not None:
            print(f"   (exact-only row; bracket needs bpt and C from a cost line: skipped in G3)")
        continue
    m = re.search(r"ctx=(\d+) path=(\w+) = (\d+) B/token x ctx \+ (\d+)MB prefill-workspace \+ (\d+)MB fixed = (\d+)MB", cl)
    Cc, path, bpt, W, A, cost = int(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4)) * 10**6, int(m.group(5)) * 10**6, int(m.group(6)) * 10**6
    D = (a.get("draft_bytes") or 0) if path == "spec" else 0
    bracket = bpt * (Cc - (P + L + 8))
    if a.get("cost_exact") is not None:
        res = (a["cost_exact"] - kv) - bracket; kind = "exact"
    else:
        res = ((cost + D) - kv) - bracket; kind = "line~"
    resid.append((abs(res), kind))
    print(f"{tag} P={P} L={L} C={Cc} path={path} bpt={bpt} kv_hat={kv} cost_exact={a.get('cost_exact')} cost~={cost + D} bracket={bracket} residual({kind})={res} W~={W} A~={A} D~={D}")
ex = [r for r, k in resid if k == "exact"]; ln = [r for r, k in resid if k == "line~"]
print(f"G3: exact rows max |residual| = {max(ex) if ex else None} (tolerance 0); line~ rows max |residual| = {max(ln) if ln else None} (tolerance 2e6 at the 1 MB grain)")
g_end, gf_end, dD_end = growth_at(10**15)
if isinstance(floor, int):
    print(f"G1: growth(end) = {g_end} (G_fa after ready {gf_end} + delta_D {dD_end}) <= 10% floor = {floor // 10}: {'PASS' if g_end <= floor // 10 else 'FAIL'}")
    print(f"G2: while inflight > 0, max_underbook_real = {mur_in[0] if mur_in else None} <= floor = {floor}: {'PASS' if mur_in and mur_in[0] <= floor else 'FAIL'}")
else:
    print(f"G1/G2: not evaluable (floor={floor})")
ov = sum(1 for l in log if "Overloaded" in l or "CUDA_ERROR_OUT_OF_MEMORY" in l or "out of memory" in l)
print(f"G4: Overloaded/OOM lines = {ov}: {'PASS' if ov == 0 else 'FAIL'}")
paths = {}
for a in admits:
    cl = a.get("cost_line") or ""; m = re.search(r"path=(\w+)", cl); r = a.get("client")
    if r and m: paths[r["tag"]] = m.group(1)
print("== path per request (from the request-cost line in its block): " + " ".join(f"{k}={v}" for k, v in sorted(paths.items())))
print(f"requests={len(client)} non-200={sum(1 for r in client if r['status'] != 200)} admits={len(admits)} ready_ms={ready_ms}")
