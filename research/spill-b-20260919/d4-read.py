#!/usr/bin/env python3
"""Day 31 D4 readings (DAY31-D4.md section 1.3). Reads the D4-host boot and its control boot (rows.jsonl, marks.json,
stamped server.log); prints D4-B, D4-C, D4-ID and the burst side by side. Never edits its inputs.

usage: d4-read.py <D4-host boot dir> <control boot dir>
"""
import collections, json, os, re, sys

D4, CTL = sys.argv[1], sys.argv[2]


def load(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
    marks = json.load(open(os.path.join(d, "marks.json"))) if os.path.exists(os.path.join(d, "marks.json")) else {}
    log = []
    for l in open(os.path.join(d, "server.log"), encoding="utf-8", errors="replace"):
        m = re.match(r"(\d+) (.*)", l.rstrip("\n"))
        if m:
            log.append((int(m.group(1)), m.group(2)))
    return rows, marks, log


def inwin(t, marks):
    b0, b1 = marks.get("burst_start_ms"), marks.get("burst_end_ms")
    return b0 is not None and b1 is not None and b0 <= t <= b1


def kv(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


rows, marks, log = load(D4)
crows, cmarks, clog = load(CTL)

# D4-B: every reclaim-demoted line, its flush window and the copy/settle ms printed inside it
print("== D4-B")
tot = collections.Counter(); n = 0
for i, (t, l) in enumerate(log):
    if not l.startswith("[admit-mem] reclaim demoted"):
        continue
    m = re.match(r"\[admit-mem\] reclaim demoted (\d+) of (\d+) device prefix entries to the host tier "
                 r"\((\d+)MB kept warm; host (\d+)MB of (\d+)MB resident\)", l)
    j = i - 1
    while j >= 0 and log[j][1].startswith("[prefix-host]"):
        j -= 1
    t0 = log[j][0] if j >= 0 else t
    copy_ms = sum(float(x) for _, s in log[j + 1:i] for x in re.findall(r"([\d.]+)ms from submission to completion", s))
    settle_ms = sum(float(x) for _, s in log[j + 1:i] for x in re.findall(r"([\d.]+)ms since\s+the hand-off", s))
    n += 1
    D, E, M, H, B = (int(x) for x in m.groups()) if m else (None,) * 5
    print(f"DAY31 D4-B reclaim#{n} t={t} burst={inwin(t, marks)} demoted={D} of={E} kept_warm_MB={M} host_MB={H}/{B} "
          f"flush_window_ms={t - t0} prefix_host_lines={i - j - 1} copy_ms_sum={copy_ms:.1f} settle_ms_sum={settle_ms:.1f}")
    if m:
        tot["demoted"] += D; tot["evicted"] += E; tot["mb"] += M
    tot["window_ms"] += t - t0
print(f"DAY31 D4-B totals reclaim_lines={n} demoted={tot['demoted']} evicted={tot['evicted']} kept_warm_MB={tot['mb']} "
      f"flush_window_ms_sum={tot['window_ms']}")
for name, lg, mk in (("D4-host", log, marks), ("control", clog, cmarks)):
    vd = sum(1 for t, l in lg if l.startswith("[admit-oom] VRAM defer") and inwin(t, mk))
    print(f"DAY31 D4-B {name} vram_defer_lines_in_burst={vd}")

# D4-C: the refusal contract with the tier armed
print("== D4-C")
am = [(t, kv(l)) for t, l in log if l.startswith("[admit-mem] id=")]
refuse = [f for t, f in am if f.get("verdict") == "refuse" and inwin(t, marks)]
host_seen = any(int(f.get("host_free", "0")) > 0 for _, f in am)
burst = [r for r in rows if r["class"] == "burst"]
r429 = [r for r in burst if r["status"] == 429]
ra_ok = all(r.get("retry_after") and str(r["retry_after"]).isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in r429)
pred_ok = all(int(f["waited_ms"]) >= 8000 and int(f["short_by"]) > min(int(f["demotable"]), int(f["host_free"]))
              for f in refuse)
logged_ra = collections.Counter(f.get("retry_after_s") for f in refuse)
sent_ra = collections.Counter(str(r.get("retry_after")) for r in r429)
if not r429:
    verdict = "NOT-OBSERVED"
else:
    verdict = "PASS" if (ra_ok and len(r429) == len(refuse) and pred_ok and host_seen) else "FAIL"
print(f"DAY31 D4-C r429={len(r429)} refuse_lines={len(refuse)} retry_after_in_1_60={ra_ok} "
      f"refuse_predicate_all={pred_ok} host_tier_seen_armed={host_seen} sent_retry_after={dict(sent_ra)} "
      f"logged_retry_after_s={dict(logged_ra)} -> {verdict}")
for f in refuse[:4]:
    print("  refuse:", " ".join(f"{k}={f.get(k)}" for k in ("id", "short_by", "device_free", "demotable", "host_free",
                                                          "waited_ms", "retry_after_s")))

# D4-ID: arming the host tier against the control, non-burst twins
print("== D4-ID")
a = {r["tag"]: r for r in rows if r["class"] != "burst"}
b = {r["tag"]: r for r in crows if r["class"] != "burst"}
tw = eq = 0; differ = []
for tag, r1 in a.items():
    r0 = b.get(tag)
    if r0 is None or r0["status"] != 200 or r1["status"] != 200 or r0["prompt_sha256"] != r1["prompt_sha256"]:
        continue
    tw += 1
    if r0["message_sha256"] == r1["message_sha256"] and r0["G"] == r1["G"] and r0.get("finish_reason") == r1.get("finish_reason"):
        eq += 1
    else:
        differ.append(tag)
print(f"DAY31 D4-ID twins={tw} equal={eq} differ={len(differ)}{' ' + ','.join(differ[:10]) if differ else ''} "
      f"-> {'PASS' if tw and eq == tw else 'FAIL'}")

# burst side by side
print("== burst")
for name, rs in (("D4-host", rows), ("control", crows)):
    bs = [r for r in rs if r["class"] == "burst"]
    print(f"DAY31 D4-BURST {name} B={len(bs)} status={dict(collections.Counter(r['status'] for r in bs))} "
          f"retry_after={dict(collections.Counter(str(r.get('retry_after')) for r in bs if r['status'] != 200))}")
