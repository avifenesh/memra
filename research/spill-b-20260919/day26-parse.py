#!/usr/bin/env python3
"""Day 26 memra#539 cell parser: joins server.log (stamped), client.jsonl, samples.csv and the metrics snapshots into
the pre-registered per-request rows (DAY26.md section 2) and the per-arm allocated-over-used summary. Prints verbatim
rows; never edits inputs.

Per request: P = usage.prompt_tokens, G = usage.completion_tokens, cached = prompt_tokens_details.cached_tokens;
the `[admission] request cost` line stamped inside [submit, done] gives path, B/token, the booked ctx (admission_cap)
and the booked MB. Allocated KV = B/token x ctx_alloc where ctx_alloc is the cache's max_ctx (`ctx_cap`): the booked
ctx minus 64 on the open arms (request_ctx_cap's server-context arm; admission_cap = need = ctx_cap + 64 there because
budget = ctx_cap - P), and P + max_tokens + 8 on the bounded arm (admission_cap = need = P + max_tokens + 64). Used KV =
B/token x (P + G). A request whose own line was deduplicated by the worker inherits the last line with the expected
booked ctx and is flagged `inherited`.
"""
import csv, json, os, re, statistics, sys

C = sys.argv[1]
log = [l.rstrip("\n") for l in open(os.path.join(C, "server.log"), encoding="utf-8", errors="replace")]
client = [json.loads(l) for l in open(os.path.join(C, "client.jsonl"))]
samples = list(csv.DictReader(open(os.path.join(C, "samples.csv"))))
shape = open(os.path.join(C, "shape.txt")).read().strip()
ready = json.load(open(os.path.join(C, "metrics-ready.json"))) if os.path.exists(os.path.join(C, "metrics-ready.json")) else {}
SLACK = 8
SPEC_SHRINK_SLACK = 64


def stamp(l):
    m = re.match(r"(\d+) (.*)", l)
    return (int(m.group(1)), m.group(2)) if m else (None, l)


COST = re.compile(r"\[admission\] request cost: model=\S+ ctx=(\d+) path=(\w+) = (\d+) B/token x ctx \+ (\d+)MB "
                  r"prefill-workspace \+ (\d+)MB fixed = (\d+)MB")
cost_lines = []
for t, l in map(stamp, log):
    m = COST.search(l)
    if m:
        cost_lines.append(dict(t=t, ctx=int(m.group(1)), path=m.group(2), bpt=int(m.group(3)),
                               prefill_mb=int(m.group(4)), fixed_mb=int(m.group(5)), total_mb=int(m.group(6)), line=l))
predict = []
for t, l in map(stamp, log):
    if l.startswith("[admit-predict] id="):
        kv = dict(re.findall(r"(\w+)=(\S+)", l)); kv["t"] = t; kv["line"] = l; predict.append(kv)
boot = [l for _, l in map(stamp, log) if re.search(r"\[admit-mem\] door=|shadow armed|ctx=\d+ .*(window|context)|MEMRA_CTX|session ctx", l)]

print("== shape"); print(shape)
print("== boot lines (context, doors)")
for l in boot[:8]:
    print(l)
gpu_before = open(os.path.join(C, "gpu-before.csv")).read().strip().splitlines()
gpu_after = open(os.path.join(C, "gpu-after.csv")).read().strip().splitlines()
print("== rig"); print(gpu_before[-1]); print(gpu_after[-1])
print(f"ready: cuda_driver_free_bytes={ready.get('cuda_driver_free_bytes')} cuda_pool_used_bytes={ready.get('cuda_pool_used_bytes')} "
      f"cuda_pool_reserved_bytes={ready.get('cuda_pool_reserved_bytes')} prefix_cache_bytes={ready.get('prefix_cache_bytes')}")
print("== request-cost lines (verbatim, in log order)")
for c in cost_lines:
    print(c["line"])

server_ctx = None
open_rows = [r for r in client if r["arm"] in ("i", "iii") and r["status"] == 200]
# the open arm's booked ctx is the server context whenever prompt + 16 <= ctx; take the modal ctx among lines that
# fall inside an open request's window
cands = []
for r in open_rows:
    for c in cost_lines:
        if r["submit_ms"] <= c["t"] <= r["done_ms"]:
            cands.append(c["ctx"])
if cands:
    server_ctx = statistics.mode(cands)
print(f"== open-arm booked ctx (admission_cap) from the lines: {server_ctx}; allocated ctx_cap = {None if server_ctx is None else server_ctx - SPEC_SHRINK_SLACK}")

rows = []
last_by_ctx = {}
for c in cost_lines:
    pass
print("== per-request rows")
print("tag | arm | P | G | cached | path | B/token | ctx_booked | ctx_alloc | allocated_kv_B | used_kv_B | alloc/used | "
      "booked_MB | drv_free_before | drv_free_after | pool_used_before | pool_used_after | ms | line")
for r in sorted(client, key=lambda r: r["submit_ms"]):
    if r["arm"] == "warmup" or r["status"] != 200:
        continue
    u = r["usage"] or {}
    P = u.get("prompt_tokens"); G = u.get("completion_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    mine = [c for c in cost_lines if r["submit_ms"] <= c["t"] <= r["done_ms"]]
    inherited = False
    if r["arm"] == "ii":
        expect = P + r["max_tokens"] + SPEC_SHRINK_SLACK
        exact = [c for c in mine if c["ctx"] == expect] or [c for c in mine if c["ctx"] == P + r["max_tokens"] + SLACK]
        c = exact[-1] if exact else (mine[-1] if mine else None)
        if c is None:
            prev = [x for x in cost_lines if x["t"] <= r["done_ms"] and x["ctx"] in (expect, P + r["max_tokens"] + SLACK)]
            c = prev[-1] if prev else None; inherited = c is not None
        ctx_alloc = P + r["max_tokens"] + SLACK
    else:
        exact = [c for c in mine if c["ctx"] == server_ctx]
        c = exact[-1] if exact else (mine[-1] if mine else None)
        if c is None:
            prev = [x for x in cost_lines if x["t"] <= r["done_ms"] and x["ctx"] == server_ctx]
            c = prev[-1] if prev else None; inherited = c is not None
        # open arm: budget = ctx_cap - P, need = P + budget + SPEC_SHRINK_SLACK = ctx_cap + 64, so the booked
        # admission_cap is ctx_cap + 64 while the cache is allocated at ctx_cap (worker.rs:21876-21879, 2773-2779)
        ctx_alloc = c["ctx"] - SPEC_SHRINK_SLACK if c else None
    if c is None or P is None or G is None:
        print(f"{r['tag']} | {r['arm']} | {P} | {G} | {cached} | (no request-cost line matched) | usage={u}")
        continue
    alloc_b = c["bpt"] * ctx_alloc
    used_b = c["bpt"] * (P + G)
    ratio = alloc_b / used_b if used_b else float("inf")
    row = dict(tag=r["tag"], arm=r["arm"], length_idx=r["length_idx"], P=P, G=G, cached=cached, path=c["path"], bpt=c["bpt"],
               ctx_booked=c["ctx"], ctx_alloc=ctx_alloc, alloc_b=alloc_b, used_b=used_b, ratio=ratio, booked_mb=c["total_mb"],
               inherited=inherited, ms=r["done_ms"] - r["submit_ms"],
               free_before=(r.get("before") or {}).get("cuda_driver_free_bytes"), free_after=(r.get("after") or {}).get("cuda_driver_free_bytes"),
               pool_before=(r.get("before") or {}).get("cuda_pool_used_bytes"), pool_after=(r.get("after") or {}).get("cuda_pool_used_bytes"))
    rows.append(row)
    print(f"{row['tag']} | {row['arm']} | {P} | {G} | {cached} | {c['path']} | {c['bpt']} | {c['ctx']} | {ctx_alloc} | {alloc_b} | {used_b} | "
          f"{ratio:.2f} | {c['total_mb']} | {row['free_before']} | {row['free_after']} | {row['pool_before']} | {row['pool_after']} | "
          f"{row['ms']} | {'inherited' if inherited else 'own'}")

print("== per-arm allocated over used (verbatim per row, then N, min, median, max)")
for arm in ("i", "ii", "iii"):
    for li in sorted({x["length_idx"] for x in rows}):
        sel = [x for x in rows if x["arm"] == arm and x["length_idx"] == li]
        if not sel:
            continue
        rs = [x["ratio"] for x in sel]
        Ps = [x["P"] for x in sel]; Gs = [x["G"] for x in sel]; cs = [x["cached"] for x in sel]
        print(f"arm={arm} L{li} N={len(sel)} P={Ps} G={Gs} cached={cs} ratios={[round(v, 2) for v in rs]} "
              f"min={min(rs):.2f} median={statistics.median(rs):.2f} max={max(rs):.2f} "
              f"alloc_B={sel[0]['alloc_b']} used_B_median={int(statistics.median([x['used_b'] for x in sel]))} "
              f"booked_MB={[x['booked_mb'] for x in sel]} inherited={sum(1 for x in sel if x['inherited'])}")
print("== driver free (bytes) across the run")
fr = [int(s["cuda_driver_free_bytes"]) for s in samples if s.get("cuda_driver_free_bytes")]
pu = [int(s["cuda_pool_used_bytes"]) for s in samples if s.get("cuda_pool_used_bytes")]
if fr:
    print(f"samples={len(samples)} driver_free first={fr[0]} min={min(fr)} last={fr[-1]}; pool_used first={pu[0] if pu else None} "
          f"max={max(pu) if pu else None} last={pu[-1] if pu else None}")
print("== concurrency arithmetic at the served context (idle free at ready divided by the booked cost; not an admission run)")
first_req = min((r["submit_ms"] for r in client if r["arm"] != "warmup"), default=None)
idle = [int(s["cuda_driver_free_bytes"]) for s in samples
        if s.get("cuda_driver_free_bytes") and int(s["cuda_driver_free_bytes"]) > 0 and first_req and int(s["epoch_ms"]) < first_req]
free_ready = max(idle) if idle else None
print(f"idle driver free before the first request (max of {len(idle)} nonzero samples; the ready-time gauges read 0 until the first tick): {free_ready}")
for li in sorted({x["length_idx"] for x in rows}):
    o = [x for x in rows if x["arm"] == "i" and x["length_idx"] == li]
    b = [x for x in rows if x["arm"] == "ii" and x["length_idx"] == li]
    if o and b and free_ready:
        co = statistics.median([x["booked_mb"] for x in o]) * 1e6
        cb = statistics.median([x["booked_mb"] for x in b]) * 1e6
        print(f"L{li}: P~{int(statistics.median([x['P'] for x in o]))} open booked_MB={co / 1e6:.0f} -> {int(free_ready // co)} sessions; "
              f"bounded booked_MB={cb / 1e6:.0f} -> {int(free_ready // cb)} sessions (free_ready={free_ready})")
print("== prefix-cache and park lines (census by kind, verbatim first of each)")
kinds = {}
for _, l in map(stamp, log):
    m = re.match(r"(\[prefix-cache\] [a-z]+(?: \([a-zA-Z -]+\))?|\[worker\] (?:reuse|dspark-park|spec park)[^:]*|\[admit-oom\][^:]*)", l)
    if m:
        kinds.setdefault(m.group(1), [0, l]); kinds[m.group(1)][0] += 1
for k, (n, first) in sorted(kinds.items(), key=lambda kv: -kv[1][0]):
    print(f"{n:5d}  {k}  first: {first[:200]}")
print(f"prefix-cache hit lines: {sum(n for k, (n, _) in kinds.items() if k.startswith('[prefix-cache] hit'))}")
print("== admit-predict lines (verbatim, first 6 and last 3)")
for kv in predict[:6] + predict[-3:]:
    print(kv["line"])
print("== day 27: retained at idle, pools and the park door (metrics-end.json, the last sample, the server lines)")
end = json.load(open(os.path.join(C, "metrics-end.json"))) if os.path.exists(os.path.join(C, "metrics-end.json")) else {}
last = samples[-1] if samples else {}
def gi(d, k):
    v = d.get(k); return int(v) if v not in (None, "") else None
free_end = gi(last, "cuda_driver_free_bytes"); pool_used_end = gi(last, "cuda_pool_used_bytes"); pool_res_end = gi(last, "cuda_pool_reserved_bytes")
# pool baseline: the pool used at idle before the first mix request (the ready-time gauges read 0 until the first tick,
# so take the max nonzero idle sample, like free_ready above; the warmup's own gauges are 0 for the same reason)
idle_pool = [int(s["cuda_pool_used_bytes"]) for s in samples
             if s.get("cuda_pool_used_bytes") and int(s["cuda_pool_used_bytes"]) > 0 and first_req and int(s["epoch_ms"]) < first_req]
pool_after_warmup = max(idle_pool) if idle_pool else None
print(f"idle driver free before the first request={free_ready} last sample={free_end} retained_by_process={None if None in (free_ready, free_end) else free_ready - free_end}")
print(f"pool used after warmup={pool_after_warmup} last={pool_used_end} delta={None if None in (pool_after_warmup, pool_used_end) else pool_used_end - pool_after_warmup}; "
      f"pool reserved last={pool_res_end} cached last={end.get('cuda_pool_cached_bytes')}")
for k in ("continuation_pool_entries", "continuation_pool_hits", "continuation_pool_evictions", "spec_pool_entries", "spec_pool_hits",
          "spec_pool_evictions", "spec_pool_misses", "prefix_cache_entries", "prefix_cache_bytes", "prefix_cache_hits", "prefix_cache_hit_tokens",
          "prefix_cache_inserts", "prefix_cache_evictions", "prefix_cache_misses", "step_oom_parks", "admission_vram_defers"):
    print(f"{k}={end.get(k)}")
park = {}
for _, l in map(stamp, log):
    m = re.match(r"(\[kv-reuse\] park-compact[a-z :]*?)(?=[:( ]|$)", l)
    if m:
        park.setdefault(m.group(1).strip(), [0, l]); park[m.group(1).strip()][0] += 1
print(f"park-compact lines: {sum(n for n, _ in park.values())}")
for k, (n, first) in sorted(park.items()):
    print(f"{n:5d}  {k}  first: {first[:220]}")
aff = {}
for _, l in map(stamp, log):
    m = re.search(r"\[worker\] ((?:plain|spec)-affinity: [a-z]+)", l)
    if m:
        aff[m.group(1)] = aff.get(m.group(1), 0) + 1
print("affinity lines: " + ", ".join(f"{k}={v}" for k, v in sorted(aff.items())))
print("== day 27: completion digests (tag, P, G, cached, finish_reason, sha256)")
for r in sorted(client, key=lambda r: r["submit_ms"]):
    if r["arm"] == "warmup":
        continue
    u = r["usage"] or {}
    print(f"{r['tag']} P={u.get('prompt_tokens')} G={u.get('completion_tokens')} cached={(u.get('prompt_tokens_details') or {}).get('cached_tokens')} "
          f"finish={r.get('finish_reason')} sha256={r.get('content_sha256')}")
print(f"requests_ok={len(rows)} non200={sum(1 for r in client if r['status'] != 200 and r['arm'] != 'warmup')}")
