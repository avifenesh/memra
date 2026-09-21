#!/usr/bin/env python3
"""Replay the day-14 OFF/ON admissibility comparisons from the mirrored receipts (lead ruling 16).

Reads research/spill-c-20260919/pro-single-day14 (the BOX3 mirror) and asserts what ruling 16
requires as the exit criterion for the draft-bearing surface: under the gates' DEFAULT spec
environment (this MTP artifact serves speculatively, every prefix insert is a spec-boundary capture
with a draft plane) the identity and failure gates print identical verdict lines with the door OFF
and ON, `verify ok` on every ON promote, equal `[prefix-host] demote:` byte counts, and no refusal
line of any kind in the ON arm; plus the day-13 plain-surface pairs and serve-smoke unchanged.
Integrity only, N=1, executed-not-qualified; nothing here is a support state.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day14"
DRAFT_CLASS = "server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B+mtp-draft-kv-q8_0-34B-q5_1-24B"
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def lines(path, pattern):
    text = Path(path).read_text(errors="replace")
    return [l for l in text.splitlines() if re.search(pattern, l)]


def verdicts(cell):
    return lines(ROOT / cell / "command.log", r"^\s+(ok|FAIL): |GATE: |cache-meter-gate: |^serve-smoke: ")


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


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
check("build exit 0", (ROOT / "build" / "exit").read_text().strip() == "0")
cells = sorted(p.name for p in ROOT.iterdir() if (p / "CELL.jsonl").exists())
check("ten cells captured", len(cells) == 10, ", ".join(cells))
for cell in cells:
    status, code, power = cell_status(cell)
    # `failed` is the collector's word for a nonzero command exit (a red gate), not a torn cell;
    # which arms are red, and why, is asserted per gate below.
    check(f"{cell}: collector captured the cell at 600 W", status in ("executed-not-qualified", "failed") and power[0]["power.limit"] == "600.00 W", f"status={status} exit={code}")
    check(f"{cell}: ran the day-14 binary", binary in (ROOT / cell / "command.log").read_text(errors="replace"))

print("== serve-smoke (plain + cache-metering arms), door OFF vs ON")
off, on = verdicts("smoke-off"), verdicts("smoke-on")
check("verdict lines identical", off == on, f"{len(off)} lines")
check("both arms `serve-smoke: 0 failed`", all(any(l.strip() == "serve-smoke: 0 failed" for l in v) for v in (off, on)))
check("both arms carry the cache-metering PASS line", all(any("cache-metering accounting exact" in l for l in v) for v in (off, on)))
check("OFF server log has no [prefix-host] or door line", count(ROOT / "smoke-off-server.log", r"prefix-host\]|kv-host-contracts") == 0)
check("ON server log: no host tier, one door line under its own tag, no [prefix-host] line",
      count(ROOT / "smoke-on-server.log", r"\[kv-host-contracts\] MEMRA_KV_HOST_CONTRACTS=1 with no host tier") == 1 and count(ROOT / "smoke-on-server.log", r"prefix-host\]") == 0)


def identity_pair(spec, label):
    print(f"== kv-host-spill-identity-gate, {label}, door OFF vs ON")
    off_cell, on_cell = f"hostgate-identity-off-{spec}", f"hostgate-identity-on-{spec}"
    off, on = verdicts(off_cell), verdicts(on_cell)
    check(f"[{spec}] verdict lines identical", off == on, f"{len(off)} lines")
    check(f"[{spec}] both arms ALL GREEN", all(any("IDENTITY GATE: ALL GREEN" in l for l in v) for v in (off, on)))
    off_log = ROOT / off_cell / "ev/host-on-server.log"
    on_log = ROOT / on_cell / "ev/host-on-server.log"
    check(f"[{spec}] [prefix-host] demote: byte counts equal", demotes(off_log) == demotes(on_log) and len(demotes(on_log)) >= 1, " | ".join(demotes(on_log)))
    promotes = count(on_log, r"\[prefix-host\] promote: ")
    verify_ok = count(on_log, r"\[prefix-host\] verify ok")
    # A promote that passes verify can still be declined by the DEVICE cache's protected-share
    # rule (`[prefix-cache] skip pinned host-promote insert: ... would evict protected bytes`),
    # the spec environment's own behavior at MEMRA_HOSTGATE_CACHE_MB=256 (identical in the OFF
    # arm and in day 13's OFF-default cell). Every verify ok is therefore a promote or that
    # named skip; nothing else may absorb one.
    skips = count(on_log, r"\[prefix-cache\] skip pinned host-promote insert")
    check(f"[{spec}] ON: verify ok on every promote, every other verify ok is the named device-side skip", promotes >= 1 and verify_ok == promotes + skips, f"promotes={promotes} verify_ok={verify_ok} skips={skips}")
    check(f"[{spec}] promote, verify and skip counts equal across arms", (count(off_log, r"\[prefix-host\] promote: "), count(off_log, r"\[prefix-host\] verify ok"), count(off_log, r"\[prefix-cache\] skip pinned host-promote insert")) == (promotes, verify_ok, skips))
    events = lambda log: [strip_ms(l) for l in lines(log, r"\[prefix-(cache|host)\] ") if "contracts door" not in l]
    check(f"[{spec}] prefix-cache and prefix-host event sequence identical across arms (timings stripped)", events(off_log) == events(on_log) and len(events(on_log)) >= 8, f"{len(events(on_log))} events")
    check(f"[{spec}] ON: door announced with the plain program identity", count(on_log, r"contracts door: model gate program identity artifact_sha256=[0-9a-f]{64} plan_debug_sha256=[0-9a-f]{64} numeric=server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B template=gguf-jinja device=0") == 1)
    check(f"[{spec}] ON: door announced the draft program identity (embedded MTP head)", count(on_log, r"contracts door: model gate draft program identity source=embedded numeric=" + re.escape(DRAFT_CLASS)) == 1)
    check(f"[{spec}] ON: door ON line", count(on_log, r"contracts door ON \(MEMRA_KV_HOST_CONTRACTS=1\): 1 model program identities") == 1)
    check(f"[{spec}] ON: no refusal line of any kind", count(on_log, r"contracts door\): |REFUSED|refused") == 0)
    check(f"[{spec}] OFF twin boots print no [prefix-host] line in either arm", count(ROOT / off_cell / "ev/host-off-server.log", r"prefix-host\]") == 0 and count(ROOT / on_cell / "ev/host-off-server.log", r"prefix-host\]") == 0)
    for arm, cell in (("off", off_cell), ("on", on_cell)):
        r1 = json.loads((ROOT / cell / "ev/host-on-r1.json").read_text())
        r3 = json.loads((ROOT / cell / "ev/host-on-r3.json").read_text())
        check(f"[{spec}] {arm}: r3 served a strict-prefix hit (0 < cached < prompt)", 0 < r3["usage"]["prompt_tokens_details"]["cached_tokens"] < r3["usage"]["prompt_tokens"], f"cached={r3['usage']['prompt_tokens_details']['cached_tokens']} prompt={r3['usage']['prompt_tokens']} r1_prompt={r1['usage']['prompt_tokens']}")
    texts = {}
    for arm, cell in (("off", off_cell), ("on", on_cell)):
        for r in ("r1", "r3"):
            texts[(arm, r)] = json.loads((ROOT / cell / f"ev/host-on-{r}.json").read_text())["choices"][0]["text"]
    check(f"[{spec}] r1 and r3 texts identical across door arms", texts[("off", "r1")] == texts[("on", "r1")] and texts[("off", "r3")] == texts[("on", "r3")])
    return off_log, on_log


off_log, on_log = identity_pair("default", "DEFAULT spec environment (the ruling-16 exit criterion: draft-bearing entries)")
spec_inserts = count(on_log, r"insert probation \(spec-boundary\)")
check("[default] ON: the demoted entries were spec-boundary (draft-bearing) inserts", spec_inserts >= 1 and count(off_log, r"insert probation \(spec-boundary\)") >= 1, f"on={spec_inserts}")
check("[default] both arms: zero plain-published inserts (the surface exercised is the draft-bearing one)", all(count(l, r"insert probation \((plain|prime|split)") == 0 for l in (off_log, on_log)))
identity_pair("plain", "MEMRA_SERVE_SPEC=0 (the day-13 plain surface, regression)")


def failure_pair(spec, label):
    print(f"== kv-host-spill-failure-gate, {label}, door OFF vs ON")
    off_cell, on_cell = f"hostgate-failure-off-{spec}", f"hostgate-failure-on-{spec}"
    off, on = verdicts(off_cell), verdicts(on_cell)
    check(f"[{spec}] verdict lines identical", off == on, f"{len(off)} lines")
    check(f"[{spec}] both arms: exactly one FAIL, the pre-existing pool-full LOUD-and-named assertion", all(sum(l.strip().startswith("FAIL:") for l in v) == 1 and any("FAIL: pool-full refusal is LOUD and named" in l for l in v) for v in (off, on)))
    for cell in ("poolfull", "digest", "alloc"):
        a = [strip_ms(l) for l in lines(ROOT / off_cell / f"ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)")]
        b = [strip_ms(l) for l in lines(ROOT / on_cell / f"ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)")]
        check(f"[{spec}] {cell} cell: tier event lines identical (timings stripped)", a == b and len(a) >= 1, f"{len(a)} lines")
    check(f"[{spec}] digest cell: the flipped byte is caught in both arms", all(count(ROOT / c / "ev/digest-server.log", r"VERIFY FAILED") >= 1 and count(ROOT / c / "ev/digest-server.log", r"FAULT: flipped one demoted K byte") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] alloc cell: the tier latches off in both arms", all(count(ROOT / c / "ev/alloc-server.log", r"TIER DISABLED: pinned host alloc") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] pool-full: the tenant share cap evaporates before the D2H in both arms (pre-existing, not a door effect)", all(count(ROOT / c / "ev/poolfull-server.log", r"demote evaporated at the tenant share cap") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] ON: no refusal line in any cell", sum(count(ROOT / on_cell / f"ev/{c}-server.log", r"contracts door\): |REFUSED|refused") for c in ("poolfull", "digest", "alloc")) == 0)


failure_pair("default", "DEFAULT spec environment (ruling-16 exit criterion)")
failure_pair("plain", "MEMRA_SERVE_SPEC=0 (day-13 regression)")

print()
print("DAY14 REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
