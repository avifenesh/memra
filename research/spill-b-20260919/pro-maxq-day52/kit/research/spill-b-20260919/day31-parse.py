#!/usr/bin/env python3
"""Day 31 per-boot parser (DAY31.md section 1). Reads one boot's server.log (stamped), client.jsonl, samples.csv,
marks.json and shape.txt; writes <boot>/rows.jsonl (one derived row per request) and prints the boot report. Never
edits its inputs. The cross-boot verdicts are day31-compare.py's job.

Per sequential request: the first `[admission] request cost` line stamped inside [submit, done] is its own line
(`own`); a request with none inherits the last line stamped before its submit (`inherited`, reported, never judged).
Booked ctx = that line's ctx (admission_cap). Allocated KV = B/token x (booked ctx - 64); used KV = B/token x (P + G).
Burst: statuses, Retry-After values, `[admit-mem] id=` lines by verdict inside the burst window, and the maximum
`active_sessions` and `admission_booked_bytes` of the 250 ms samples inside it.
"""
import collections, csv, json, os, re, statistics, sys

C = sys.argv[1]
log = [l.rstrip("\n") for l in open(os.path.join(C, "server.log"), encoding="utf-8", errors="replace")]
client = [json.loads(l) for l in open(os.path.join(C, "client.jsonl"))]
samples = list(csv.DictReader(open(os.path.join(C, "samples.csv"))))
shape = open(os.path.join(C, "shape.txt")).read().strip()
marks = json.load(open(os.path.join(C, "marks.json"))) if os.path.exists(os.path.join(C, "marks.json")) else {}
sh = dict(re.findall(r"(\w+)=(\S+)", shape))


def stamp(l):
    m = re.match(r"(\d+) (.*)", l)
    return (int(m.group(1)), m.group(2)) if m else (None, l)


stamped = [stamp(l) for l in log]
COST = re.compile(r"\[admission\] request cost: model=\S+ ctx=(\d+) path=(\w+) = (\d+) B/token x ctx \+ (\d+)MB "
                  r"prefill-workspace \+ (\d+)MB fixed = (\d+)MB")
cost = []
for t, l in stamped:
    m = COST.search(l)
    if m:
        cost.append(dict(t=t, ctx=int(m.group(1)), path=m.group(2), bpt=int(m.group(3)), prefill_mb=int(m.group(4)),
                         fixed_mb=int(m.group(5)), total_mb=int(m.group(6))))
door = [l for _, l in stamped if l.startswith("[admit-mem] door=")]
shadow = [l for _, l in stamped if l.startswith("[admit-predict] shadow armed")]
admit_mem = []
for t, l in stamped:
    if l.startswith("[admit-mem] id="):
        kv = dict(re.findall(r"(\w+)=(\S+)", l)); kv["t"] = t; kv["line"] = l; admit_mem.append(kv)
reclaim = [(t, l) for t, l in stamped if l.startswith("[admit-mem] reclaim")]
fatal = [l for _, l in stamped if re.search(r"panicked|FATAL|CUDA_ERROR|out of memory", l)]

want_on = sh.get("admit_by_memory") == "1"
want_v = sh.get("open_output_tokens")
arm = f"on{want_v}" if want_on else "off"
boot_ok = False
if door:
    m = re.match(r"\[admit-mem\] door=(ON|OFF) open_output_tokens=(\d+)", door[0])
    if m:
        boot_ok = (m.group(1) == "ON" and want_on and m.group(2) == want_v) or (m.group(1) == "OFF" and not want_on)

out = []
seq = [r for r in client if r["class"] != "burst"]
for r in seq:
    own = [c for c in cost if r["submit_ms"] <= c["t"] <= r["done_ms"]]
    src = "own" if own else "inherited"
    c = own[0] if own else ([x for x in cost if x["t"] < r["submit_ms"]] or [None])[-1]
    u = r.get("usage") or {}
    P = u.get("prompt_tokens"); G = u.get("completion_tokens")
    row = {k: r.get(k) for k in ("tag", "class", "status", "finish_reason", "prompt_sha256", "message_sha256",
                                "content_sha256", "request_id", "retry_after", "max_tokens", "submit_ms", "done_ms",
                                "reasoning_chars", "content_chars")}
    row.update(P=P, G=G, cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens"),
               cost_src=src, booked_ctx=c and c["ctx"], path=c and c["path"], bpt=c and c["bpt"],
               cost_total_mb=c and c["total_mb"])
    if c and P is not None and G is not None:
        row["alloc_bytes"] = c["bpt"] * (c["ctx"] - 64)
        row["used_bytes"] = c["bpt"] * (P + G)
    out.append(row)
b0, b1 = marks.get("burst_start_ms"), marks.get("burst_end_ms")
burst = [r for r in client if r["class"] == "burst"]
for r in burst:
    u = r.get("usage") or {}
    out.append({"tag": r["tag"], "class": "burst", "status": r["status"], "finish_reason": r.get("finish_reason"),
                "request_id": r.get("request_id"), "retry_after": r.get("retry_after"), "err": r.get("err"),
                "P": u.get("prompt_tokens"), "G": u.get("completion_tokens"), "submit_ms": r["submit_ms"],
                "done_ms": r["done_ms"], "prompt_sha256": r.get("prompt_sha256"), "message_sha256": r.get("message_sha256")})
with open(os.path.join(C, "rows.jsonl"), "w") as f:
    for row in out:
        f.write(json.dumps(row) + "\n")


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


win = [s for s in samples if b0 and b1 and b0 <= int(s["epoch_ms"]) <= b1]
act = [fnum(s["active_sessions"]) for s in win if fnum(s["active_sessions"]) is not None]
booked = [fnum(s["admission_booked_bytes"]) for s in win if fnum(s["admission_booked_bytes"]) is not None]
am_win = [x for x in admit_mem if b0 and b1 and b0 <= x["t"] <= b1 + 60000]
burst_cost = [c for c in cost if b0 and b1 and b0 <= c["t"] <= b1]
st = collections.Counter(r["status"] for r in burst)
fin = collections.Counter(r.get("finish_reason") for r in burst if r["status"] == 200)
ra = collections.Counter(r.get("retry_after") for r in burst if r["status"] != 200)
budget = None
if shadow:
    m = re.search(r"budget_bytes=(\d+)", shadow[0])
    budget = int(m.group(1)) if m else None
med_total = statistics.median(c["total_mb"] for c in burst_cost) if burst_cost else None

print("== shape"); print(shape)
print(f"== arm {arm}")
print("== boot door line (verbatim)"); print(door[0] if door else "(none)")
print(f"DAY31 V-BOOT boot={os.path.basename(C)} arm={arm} door_line={'present' if door else 'absent'} -> {'PASS' if boot_ok else 'FAIL'}")
print("== shadow budget line (verbatim)"); print(shadow[0] if shadow else "(none)")
print("== fatal lines"); print("\n".join(fatal[:10]) if fatal else "(none)")
print("== sequential requests: tag | status | finish | P | G | cached | path | bpt | booked_ctx | alloc_bytes | used_bytes | cost_src | message_sha256[:16]")
for r in out:
    if r["class"] == "burst":
        continue
    print(" | ".join(str(x) for x in (r["tag"], r["status"], r["finish_reason"], r["P"], r["G"], r["cached"], r["path"],
                                       r["bpt"], r["booked_ctx"], r.get("alloc_bytes"), r.get("used_bytes"), r["cost_src"],
                                       (r["message_sha256"] or "-")[:16])))
print("== burst")
print(f"window_ms={None if not (b0 and b1) else b1 - b0} B={len(burst)} status={dict(st)} finish={dict(fin)} retry_after={dict(ra)}")
print(f"active_sessions_max={max(act) if act else None} admission_booked_bytes_max={max(booked) if booked else None} "
      f"samples={len(win)} shadow_budget_bytes={budget} burst_cost_lines={len(burst_cost)} burst_cost_total_mb_median={med_total} "
      f"arith_sessions={None if not (budget and med_total) else int(budget // (med_total * 1e6))}")
vc = collections.Counter(x.get("verdict") for x in am_win)
print(f"admit_mem_id_lines={len(am_win)} by_verdict={dict(vc)} reclaim_lines={len(reclaim)}")
for x in am_win[:6]:
    print(x["line"])
if len(am_win) > 6:
    print(f"... {len(am_win) - 6} more [admit-mem] id= lines in server.log")
open_ok = [r for r in out if r["class"] in ("i", "iii", "iv-a", "iv-b") and r["status"] == 200 and r["G"] is not None]
if open_ok:
    gs = sorted(r["G"] for r in open_ok)
    print(f"== open-class G (this boot): n={len(gs)} min={gs[0]} p50={gs[len(gs) // 2]} max={gs[-1]} "
          f"stop={sum(1 for r in open_ok if r['finish_reason'] == 'stop')} length={sum(1 for r in open_ok if r['finish_reason'] == 'length')} "
          f"other={sum(1 for r in open_ok if r['finish_reason'] not in ('stop', 'length'))}")
