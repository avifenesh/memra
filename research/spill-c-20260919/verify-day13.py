#!/usr/bin/env python3
"""Replay the day-13 OFF/ON admissibility comparisons from the mirrored receipts (ruling 15).

Reads research/spill-c-20260919/pro-single-day13 (the BOX3 mirror) and asserts, per gate,
what the lead's ruling requires: identical verdict lines across the door OFF and door ON
arms where the gate's surface is inside Option A, `verify ok` on every ON promote, equal
`[prefix-host] demote:` byte counts, and, where an arm differs, that every missing demote
is accounted for by a named `demote refused (contracts door)` line. Integrity only, N=1,
executed-not-qualified; nothing here is a support state.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day13"
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def lines(path, pattern):
    text = Path(path).read_text(errors="replace")
    return [l for l in text.splitlines() if re.search(pattern, l)]


def verdicts(cell):
    return lines(ROOT / cell / "command.log", r"^\s+(ok|FAIL): |GATE: |^PREFIX-EVICT-RECLAIM: |cache-meter-gate: |^serve-smoke: ")


def cell_status(cell):
    rows = [json.loads(l) for l in (ROOT / cell / "CELL.jsonl").read_text().splitlines() if l.strip()]
    end = [r for r in rows if r.get("event") == "end"][-1]
    return end["status"], end["exit_code"], end["gpu_power_limits"]


def demotes(log):
    return [re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1]) for l in lines(log, r"\[prefix-host\] demote: ")]


def count(log, pattern):
    return len(lines(log, pattern))


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
for cell in sorted(p.name for p in ROOT.iterdir() if (p / "CELL.jsonl").exists()):
    status, code, power = cell_status(cell)
    # `failed` is the collector's word for a nonzero command exit (a red gate), not a torn cell;
    # which arms are red, and why, is asserted per gate below.
    check(f"{cell}: collector captured the cell at 600 W", status in ("executed-not-qualified", "failed") and power[0]["power.limit"] == "600.00 W", f"status={status} exit={code}")
    rig = ROOT / cell / "command.log"
    check(f"{cell}: ran the pass-2 binary", binary in rig.read_text(errors="replace"))

print("== serve-smoke (plain + cache-metering arms), door OFF vs ON")
off, on = verdicts("smoke-off"), verdicts("smoke-on")
check("verdict lines identical", off == on, f"{len(off)} lines")
check("both arms `serve-smoke: 0 failed`", any(l.strip() == "serve-smoke: 0 failed" for l in off) and any(l.strip() == "serve-smoke: 0 failed" for l in on))
check("both arms carry the cache-metering PASS line", all(any("cache-metering accounting exact" in l for l in v) for v in (off, on)))
check("OFF server log has no [prefix-host] or door line", count(ROOT / "smoke-off-server.log", r"prefix-host\]|kv-host-contracts") == 0)
check("ON server log: no host tier, one door line under its own tag, no [prefix-host] line",
      count(ROOT / "smoke-on-server.log", r"\[kv-host-contracts\] MEMRA_KV_HOST_CONTRACTS=1 with no host tier") == 1 and count(ROOT / "smoke-on-server.log", r"prefix-host\]") == 0)

print("== kv-host-spill-identity-gate, MEMRA_SERVE_SPEC=0 (the contract-routed surface), door OFF vs ON")
off, on = verdicts("hostgate-identity-off-plain"), verdicts("hostgate-identity-on-plain")
check("verdict lines identical", off == on, f"{len(off)} lines")
check("both arms ALL GREEN", all(any("IDENTITY GATE: ALL GREEN" in l for l in v) for v in (off, on)))
off_log = ROOT / "hostgate-identity-off-plain/ev/host-on-server.log"
on_log = ROOT / "hostgate-identity-on-plain/ev/host-on-server.log"
check("[prefix-host] demote: byte counts equal", demotes(off_log) == demotes(on_log), " | ".join(demotes(on_log)))
promotes = count(on_log, r"\[prefix-host\] promote: ")
check("ON: verify ok on every promote", count(on_log, r"\[prefix-host\] verify ok") == promotes and promotes >= 1, f"promotes={promotes}")
check("ON: door announced with the program identity", count(on_log, r"contracts door: model gate program identity artifact_sha256=[0-9a-f]{64} plan_debug_sha256=[0-9a-f]{64} numeric=server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B template=gguf-jinja device=0") == 1 and count(on_log, r"contracts door ON \(MEMRA_KV_HOST_CONTRACTS=1\): 1 model program identities") == 1)
check("ON: no refusal line of any kind", count(on_log, r"contracts door\): |REFUSED|refused") == 0)
check("OFF twin boots print no [prefix-host] line in either arm", count(ROOT / "hostgate-identity-off-plain/ev/host-off-server.log", r"prefix-host\]") == 0 and count(ROOT / "hostgate-identity-on-plain/ev/host-off-server.log", r"prefix-host\]") == 0)
for arm in ("off", "on"):
    r1 = json.loads((ROOT / f"hostgate-identity-{arm}-plain/ev/host-on-r1.json").read_text())
    r3 = json.loads((ROOT / f"hostgate-identity-{arm}-plain/ev/host-on-r3.json").read_text())
    check(f"{arm}: r3 served a strict-prefix hit (0 < cached < prompt)", 0 < r3["usage"]["prompt_tokens_details"]["cached_tokens"] < r3["usage"]["prompt_tokens"], f"cached={r3['usage']['prompt_tokens_details']['cached_tokens']} prompt={r3['usage']['prompt_tokens']} r1_prompt={r1['usage']['prompt_tokens']}")
texts = {}
for arm in ("off", "on"):
    for r in ("r1", "r3"):
        texts[(arm, r)] = json.loads((ROOT / f"hostgate-identity-{arm}-plain/ev/host-on-{r}.json").read_text())["choices"][0]["text"]
check("r1 and r3 texts identical across door arms", texts[("off", "r1")] == texts[("on", "r1")] and texts[("off", "r3")] == texts[("on", "r3")])

print("== kv-host-spill-identity-gate, spec serving (gate default env), door OFF vs ON")
off, on = verdicts("hostgate-identity-off-default"), verdicts("hostgate-identity-on-default")
check("OFF arm ALL GREEN", any("IDENTITY GATE: ALL GREEN" in l for l in off))
on_log = ROOT / "hostgate-identity-on-default/ev/host-on-server.log"
off_log = ROOT / "hostgate-identity-off-default/ev/host-on-server.log"
refused = count(on_log, r"demote refused \(contracts door\): entry carries TP, latent or draft planes")
check("ON arm differs: every entry was a spec-boundary insert refused by name", refused >= 1 and count(on_log, r"\[prefix-host\] demote: ") == 0 and count(on_log, r"insert probation \(spec-boundary\)") >= 2, f"refused={refused} off_demotes={len(demotes(off_log))}")
check("ON arm: 5 named FAILs, r1/r3 byte identity still holds", any("5 FAILURE(S)" in l for l in on) and any("ok: r3 ON == OFF byte identity" in l for l in on))

print("== kv-host-spill-failure-gate, MEMRA_SERVE_SPEC=0, door OFF vs ON")
off, on = verdicts("hostgate-failure-off-plain"), verdicts("hostgate-failure-on-plain")
check("verdict lines identical", off == on, f"{len(off)} lines")
check("both arms: exactly one FAIL, the pool-full LOUD-and-named assertion", all(sum(l.strip().startswith("FAIL:") for l in v) == 1 and any("FAIL: pool-full refusal is LOUD and named" in l for l in v) for v in (off, on)))
for cell in ("poolfull", "digest", "alloc"):
    a = [re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1]) for l in lines(ROOT / f"hostgate-failure-off-plain/ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)")]
    b = [re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1]) for l in lines(ROOT / f"hostgate-failure-on-plain/ev/{cell}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)")]
    check(f"{cell} cell: tier event lines identical (timings stripped)", a == b and len(a) >= 1, f"{len(a)} lines")
check("pool-full: the tenant share cap evaporates before the D2H in both arms (pre-existing, not a door effect)", all(count(ROOT / f"hostgate-failure-{arm}-plain/ev/poolfull-server.log", r"demote evaporated at the tenant share cap") == 2 for arm in ("off", "on")))

print("== kv-host-spill-failure-gate, spec serving, door OFF vs ON")
off, on = verdicts("hostgate-failure-off-default"), verdicts("hostgate-failure-on-default")
check("OFF arm: the same single pool-full FAIL as the plain arm", sum(l.strip().startswith("FAIL:") for l in off) == 1)
refused = sum(count(ROOT / f"hostgate-failure-on-default/ev/{c}-server.log", r"demote refused \(contracts door\)") for c in ("poolfull", "digest", "alloc"))
check("ON arm differs by the surface refusal only (spec-boundary entries refused by name)", refused >= 4 and any("6 FAILURE(S)" in l for l in on), f"refused={refused}")

print("== prefix-evict-reclaim-gate (lane B), door OFF vs ON")
off, on = verdicts("reclaim-off"), verdicts("reclaim-on")
v_off = [l for l in off if l.startswith("PREFIX-EVICT-RECLAIM:")]
v_on = [l for l in on if l.startswith("PREFIX-EVICT-RECLAIM:")]
check("verdict line identical", v_off == v_on and len(v_off) == 1, v_on[0] if v_on else "missing")
check("V4 identity digest equal across arms", bool(v_on) and re.search(r"identity=([0-9a-f]+)", v_on[0]).group(1) == re.search(r"identity=([0-9a-f]+)", v_off[0]).group(1))
check("both arms red on this tree (V1/V3 FAIL: lane B's fix is not on this branch), same digest", bool(v_on) and "V1=FAIL V2=ok V3=FAIL V4=ok -> FAIL" in v_on[0])
on_logs = list((ROOT / "reclaim-on/cell").rglob("*.log"))
check("ON: no host tier in this gate, door line under its own tag, no [prefix-host] line", sum(count(p, r"kv-host-contracts") for p in on_logs) >= 1 and sum(count(p, r"prefix-host\]") for p in on_logs) == 0)

print("== attempt 1 (MEMRA_HOSTGATE_CACHE_MB=128, binary e7e23dcf4): the tuning pass, kept as evidence")
a1 = ROOT / "attempt1-cache128"
check("every host gate cell failed its feed assertion in BOTH arms (no entry fit 128 MB)", all(any("no device eviction fired" in l or "FAIL: device budget forced an eviction" in l for l in lines(a1 / c / "command.log", r"FAIL")) for c in ("hostgate-identity-off-plain", "hostgate-identity-on-plain", "hostgate-failure-off-plain", "hostgate-failure-on-plain")))
check("attempt 1 exposed the OFF-twin line the b81881dad fix removed", count(a1 / "hostgate-identity-on-plain/command.log", r"FAIL: OFF boot never touches the tier") == 1)

print()
print("DAY13 REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
