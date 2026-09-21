#!/usr/bin/env python3
"""Replay the day-15 OFF/ON comparisons from the mirrored receipts (lead rulings 14 and 15, Option B).

Reads research/spill-c-20260919/pro-single-day15 (the BOX3 mirror) and asserts what Option B's
admissibility gate requires: with the pageable-tier D2H routed through the TransferEngine under
MEMRA_KV_HOST_CONTRACTS=1, every gate prints the same verdict lines OFF and ON (host-spill identity
and failure gates under the DEFAULT spec environment, the plain identity pair, lane A's tenant
reclaim gate fix arm, lane B's two prefix gates, serve-smoke), `verify ok` on every ON promote,
equal `[prefix-host] demote:` byte counts, and one D2H contract receipt (ticket, items, completion,
require=ok, checksum digest) before every ON demote. Integrity only, N=1, executed-not-qualified;
nothing here is a support state.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day15"
DRAFT_CLASS = "server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B+mtp-draft-kv-q8_0-34B-q5_1-24B"
RECEIPT = re.compile(
    r"\[prefix-host\] contracts door D2H receipt: ticket issuer=(\d+) seq=(\d+) epochs=(\d+)/(\d+)/(\d+) "
    r"items=(\d+) \((\d+) KV planes(, draft)?\) complete=(\d+) require=ok checksums_sha256=([0-9a-f]{64}) "
    r"retired acknowledged"
)
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def text(path):
    return Path(path).read_text(errors="replace")


def lines(path, pattern):
    return [l for l in text(path).splitlines() if re.search(pattern, l)]


def verdicts(cell):
    return lines(ROOT / cell / "command.log", r"^\s+(ok|FAIL): |GATE: |cache-meter-gate: |^serve-smoke: |-> (PASS|FAIL)|REFUSED")


def cell_status(cell):
    rows = [json.loads(l) for l in (ROOT / cell / "CELL.jsonl").read_text().splitlines() if l.strip()]
    end = [r for r in rows if r.get("event") == "end"][-1]
    return end["status"], end["exit_code"], end["gpu_power_limits"]


def strip_ms(l):
    return re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1])


def demotes(log):
    return [strip_ms(l) for l in lines(log, r"\[prefix-host\] demote: ")]


def count(log, pattern):
    return len(lines(log, pattern))


def receipts(log):
    return [RECEIPT.search(l).groups() for l in lines(log, r"contracts door D2H receipt: ") if RECEIPT.search(l)]


def receipt_law(label, log, expect_draft):
    """Every ON demote is preceded by exactly one well-formed contract receipt."""
    rs = receipts(log)
    raw = count(log, r"contracts door D2H receipt: ticket")
    check(f"{label}: every receipt line parses", raw == len(rs) and raw >= 1, f"lines={raw} parsed={len(rs)}")
    check(f"{label}: one receipt per demote", len(rs) == len(demotes(log)), f"receipts={len(rs)} demotes={len(demotes(log))}")
    body = text(log).splitlines()
    idx_r = [i for i, l in enumerate(body) if "contracts door D2H receipt: ticket" in l]
    idx_d = [i for i, l in enumerate(body) if "[prefix-host] demote: " in l]
    check(f"{label}: each receipt precedes its demote line", all(r < d for r, d in zip(idx_r, idx_d)) and len(idx_r) == len(idx_d))
    for issuer, seq, st, sg, dg, items, planes, draft, complete, digest in rs:
        planes_total = int(planes) + (1 if draft else 0)
        check(f"{label}: receipt seq {seq}: epochs 0/1/1, items = 2 x planes, complete = items, draft plane {'present' if expect_draft else 'absent'}",
              (st, sg, dg) == ("0", "1", "1") and int(items) == 2 * planes_total and int(complete) == int(items) and bool(draft) == expect_draft,
              f"items={items} planes={planes} draft={bool(draft)} complete={complete}")
    seqs = [int(r[1]) for r in rs]
    check(f"{label}: ticket sequence strictly increasing, one issuer", seqs == sorted(seqs) and len(set(seqs)) == len(seqs) and len({r[0] for r in rs}) == 1)
    check(f"{label}: checksum digests distinct per entry", len({r[9] for r in rs}) == len(rs))
    check(f"{label}: no receipt mismatch refusal", count(log, r"differs from its D2H contract receipt") == 0)


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
check("build exit 0", (ROOT / "build" / "exit").read_text().strip() == "0")
cells = sorted(p.name for p in ROOT.iterdir() if (p / "CELL.jsonl").exists())
print("  cells:", ", ".join(cells))
for cell in cells:
    status, code, power = cell_status(cell)
    check(f"{cell}: collector captured the cell at 600 W", status in ("executed-not-qualified", "failed") and power[0]["power.limit"] == "600.00 W", f"status={status} exit={code}")
    check(f"{cell}: ran the day-15 binary", binary in text(ROOT / cell / "command.log"))


def identity_pair(spec, label):
    print(f"== kv-host-spill-identity-gate, {label}, door OFF vs ON")
    off_cell, on_cell = f"hostgate-identity-off-{spec}", f"hostgate-identity-on-{spec}"
    if not (ROOT / off_cell / "CELL.jsonl").exists() or not (ROOT / on_cell / "CELL.jsonl").exists():
        check(f"[{spec}] both cells present", False)
        return None, None
    off, on = verdicts(off_cell), verdicts(on_cell)
    check(f"[{spec}] verdict lines identical", off == on, f"{len(off)} lines")
    check(f"[{spec}] both arms ALL GREEN", all(any("IDENTITY GATE: ALL GREEN" in l for l in v) for v in (off, on)))
    off_log = ROOT / off_cell / "ev/host-on-server.log"
    on_log = ROOT / on_cell / "ev/host-on-server.log"
    check(f"[{spec}] [prefix-host] demote: byte counts equal", demotes(off_log) == demotes(on_log) and len(demotes(on_log)) >= 1, " | ".join(demotes(on_log)))
    promotes = count(on_log, r"\[prefix-host\] promote: ")
    verify_ok = count(on_log, r"\[prefix-host\] verify ok")
    skips = count(on_log, r"\[prefix-cache\] skip pinned host-promote insert")
    check(f"[{spec}] ON: verify ok on every promote, every other verify ok is the named device-side skip", promotes >= 1 and verify_ok == promotes + skips, f"promotes={promotes} verify_ok={verify_ok} skips={skips}")
    check(f"[{spec}] promote, verify and skip counts equal across arms", (count(off_log, r"\[prefix-host\] promote: "), count(off_log, r"\[prefix-host\] verify ok"), count(off_log, r"\[prefix-cache\] skip pinned host-promote insert")) == (promotes, verify_ok, skips))
    events = lambda log: [strip_ms(l) for l in lines(log, r"\[prefix-(cache|host)\] ") if "contracts door" not in l]
    check(f"[{spec}] prefix-cache and prefix-host event sequence identical across arms (timings stripped, door lines excluded)", events(off_log) == events(on_log) and len(events(on_log)) >= 8, f"{len(events(on_log))} events")
    check(f"[{spec}] OFF: no contract receipt line (the route is door-only)", count(off_log, r"contracts door") == 0)
    check(f"[{spec}] ON: door ON line names Option B and the in-flight bound", count(on_log, r"contracts door ON \(MEMRA_KV_HOST_CONTRACTS=1\): 1 model program identities.*in-flight \d+; host tier armed; KV plane D2H through the transfer engine on the pageable tier \(Option B\)") == 1)
    check(f"[{spec}] ON: no refusal line of any kind", count(on_log, r"contracts door\): |REFUSED|refused|TIER DISABLED") == 0)
    receipt_law(f"[{spec}] ON", on_log, expect_draft=(spec == "default"))
    check(f"[{spec}] OFF twin boots print no [prefix-host] line in either arm", count(ROOT / off_cell / "ev/host-off-server.log", r"prefix-host\]") == 0 and count(ROOT / on_cell / "ev/host-off-server.log", r"prefix-host\]") == 0)
    texts = {}
    for arm, cell in (("off", off_cell), ("on", on_cell)):
        for r in ("r1", "r3"):
            texts[(arm, r)] = json.loads((ROOT / cell / f"ev/host-on-{r}.json").read_text())["choices"][0]["text"]
        r3 = json.loads((ROOT / cell / "ev/host-on-r3.json").read_text())
        check(f"[{spec}] {arm}: r3 served a strict-prefix hit (0 < cached < prompt)", 0 < r3["usage"]["prompt_tokens_details"]["cached_tokens"] < r3["usage"]["prompt_tokens"], f"cached={r3['usage']['prompt_tokens_details']['cached_tokens']} prompt={r3['usage']['prompt_tokens']}")
    check(f"[{spec}] r1 and r3 texts identical across door arms", texts[("off", "r1")] == texts[("on", "r1")] and texts[("off", "r3")] == texts[("on", "r3")])
    return off_log, on_log


off_log, on_log = identity_pair("default", "DEFAULT spec environment (draft-bearing entries over Option B's D2H)")
if on_log is not None:
    check("[default] both arms: the demoted entries were spec-boundary (draft-bearing) inserts", count(on_log, r"insert probation \(spec-boundary\)") >= 1 and count(off_log, r"insert probation \(spec-boundary\)") >= 1)
identity_pair("plain", "MEMRA_SERVE_SPEC=0 (the day-13 plain surface, regression)")


def failure_pair(spec, label):
    print(f"== kv-host-spill-failure-gate, {label}, door OFF vs ON")
    off_cell, on_cell = f"hostgate-failure-off-{spec}", f"hostgate-failure-on-{spec}"
    if not (ROOT / off_cell / "CELL.jsonl").exists() or not (ROOT / on_cell / "CELL.jsonl").exists():
        check(f"[{spec}] both cells present", False)
        return
    off, on = verdicts(off_cell), verdicts(on_cell)
    check(f"[{spec}] verdict lines identical", off == on, f"{len(off)} lines")
    check(f"[{spec}] both arms: exactly one FAIL, the pre-existing pool-full LOUD-and-named assertion", all(sum(l.strip().startswith("FAIL:") for l in v) == 1 and any("FAIL: pool-full refusal is LOUD and named" in l for l in v) for v in (off, on)))
    for cell in ("poolfull", "digest", "alloc"):
        a = [strip_ms(l) for l in lines(ROOT / off_cell / f"ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)") if "contracts door" not in l]
        b = [strip_ms(l) for l in lines(ROOT / on_cell / f"ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)") if "contracts door" not in l]
        check(f"[{spec}] {cell} cell: tier event lines identical (timings stripped, door lines excluded)", a == b and len(a) >= 1, f"{len(a)} lines")
    on_digest = ROOT / on_cell / "ev/digest-server.log"
    on_alloc = ROOT / on_cell / "ev/alloc-server.log"
    check(f"[{spec}] digest cell: the flipped byte is caught at promote in both arms", all(count(ROOT / c / "ev/digest-server.log", r"VERIFY FAILED") >= 1 and count(ROOT / c / "ev/digest-server.log", r"FAULT: flipped one demoted K byte") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] digest cell ON: the receipt precedes the flip and the bind names the injected difference, no refusal", count(on_digest, r"contracts door D2H receipt: ticket") >= 1 and count(on_digest, r"differs from its D2H receipt as injected \(MEMRA_KV_HOST_FAULT=flip-demote\)") >= 1 and count(on_digest, r"differs from its D2H contract receipt") == 0)
    check(f"[{spec}] alloc cell: the tier latches off in both arms", all(count(ROOT / c / "ev/alloc-server.log", r"TIER DISABLED: pinned host alloc") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] alloc cell ON: the fault fires before any plane leaves the entry (no receipt, no quarantine)", count(on_alloc, r"contracts door D2H receipt: ticket") == 0 and count(on_alloc, r"quarantined|no longer whole") == 0)
    check(f"[{spec}] pool-full: the tenant share cap evaporates before the D2H in both arms (pre-existing)", all(count(ROOT / c / "ev/poolfull-server.log", r"demote evaporated at the tenant share cap") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] ON: no refusal line in any cell", sum(count(ROOT / on_cell / f"ev/{c}-server.log", r"contracts door\): |REFUSED|refused") for c in ("poolfull", "digest", "alloc")) == 0)


failure_pair("default", "DEFAULT spec environment")


def tenant_pair():
    print("== lane A kv-host-tenant-reclaim-gate (fix arm), door OFF vs ON")
    off_cell, on_cell = "tenant-off", "tenant-on"
    if not (ROOT / off_cell / "CELL.jsonl").exists() or not (ROOT / on_cell / "CELL.jsonl").exists():
        check("tenant cells present", False)
        return
    off, on = verdicts(off_cell), verdicts(on_cell)
    check("verdict lines identical", off == on, f"{len(off)} lines")
    check("both arms PASS", all(any("GATE: kv-host-tenant-reclaim (fix arm) PASS" in l for l in v) for v in (off, on)))
    off_log, on_log = ROOT / off_cell / "ev/server.log", ROOT / on_cell / "ev/server.log"
    check("[prefix-host] demote: byte counts equal", demotes(off_log) == demotes(on_log) and len(demotes(on_log)) >= 3, " | ".join(demotes(on_log)))
    ev = lambda log: [strip_ms(l) for l in lines(log, r"\[prefix-host\] (demote|evict|promote|verify)") if "contracts door" not in l]
    check("demote, reclaim-evict and promote sequence identical across arms", ev(off_log) == ev(on_log) and len(ev(on_log)) >= 5, f"{len(ev(on_log))} events")
    check("ON: reclaim evictions run after the transfer (a receipt precedes every tenant-share evict line's demote)", count(on_log, r"evict \(tenant share\)") >= 1)
    check("ON: no refusal line of any kind", count(on_log, r"contracts door\): |REFUSED|refused|TIER DISABLED") == 0)
    receipt_law("tenant ON", on_log, expect_draft=True)


tenant_pair()

print("== serve-smoke (plain + cache-metering arms), door OFF vs ON")
if (ROOT / "smoke-off" / "CELL.jsonl").exists() and (ROOT / "smoke-on" / "CELL.jsonl").exists():
    off, on = verdicts("smoke-off"), verdicts("smoke-on")
    check("verdict lines identical", off == on, f"{len(off)} lines")
    check("both arms `serve-smoke: 0 failed`", all(any(l.strip() == "serve-smoke: 0 failed" for l in v) for v in (off, on)))
    check("both arms carry the cache-metering PASS line", all(any("cache-metering accounting exact" in l for l in v) for v in (off, on)))
    check("OFF server log has no [prefix-host] or door line", count(ROOT / "smoke-off-server.log", r"prefix-host\]|kv-host-contracts") == 0)
    check("ON server log: no host tier, one door line under its own tag, no [prefix-host] line", count(ROOT / "smoke-on-server.log", r"\[kv-host-contracts\] MEMRA_KV_HOST_CONTRACTS=1 with no host tier") == 1 and count(ROOT / "smoke-on-server.log", r"prefix-host\]") == 0)
else:
    check("smoke cells present", False)


def b_pair(gate, verdict_re):
    print(f"== lane B {gate} gate, door OFF vs ON")
    off_cell, on_cell = f"b{gate}-off", f"b{gate}-on"
    if not (ROOT / off_cell / "CELL.jsonl").exists() or not (ROOT / on_cell / "CELL.jsonl").exists():
        check(f"{gate} cells present", False)
        return
    off, on = verdicts(off_cell), verdicts(on_cell)
    check(f"{gate}: verdict lines identical", off == on, f"{len(off)} lines")
    v = [l for l in on if re.search(verdict_re, l)]
    check(f"{gate}: verdict line present", len(v) >= 1, v[-1].strip() if v else "none")
    logs = lambda cell: sorted((ROOT / cell / "ev").rglob("*.log"))
    check(f"{gate}: no [prefix-host] line in either arm (no host tier on these boots)", all(count(l, r"prefix-host\]") == 0 for c in (off_cell, on_cell) for l in logs(c)))
    check(f"{gate}: ON boots announce the door with nothing to route", any(count(l, r"\[kv-host-contracts\] MEMRA_KV_HOST_CONTRACTS=1 with no host tier") >= 1 for l in logs(on_cell)))


b_pair("evict", r"V1=\w+ V2=\w+ V3=\w+ V4=\w+ -> (PASS|FAIL)")
b_pair("newest", r"V1=\w+ V2=\w+ V3=\w+ V4=\w+ -> (PASS|FAIL)")

print()
print("DAY15 REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
