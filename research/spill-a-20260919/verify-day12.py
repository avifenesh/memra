#!/usr/bin/env python3
"""Replay the day-12 receipts from the BOX3 mirror (memra#384 gate, base versus fix; memra#385 harness).

Reads research/spill-a-20260919/pro-single-day12 and asserts, from the bytes on disk and never
from a live box: both gate cells ended `executed-not-qualified` with exit 0 at the card's 600 W
limit; the base arm printed the evaporation line and evicted nothing of the tenant's own; the fix
arm printed the tenant-share eviction of the tenant's OWN entry followed by that tenant's demote
and no evaporation; the demote bytes are equal across arms for every prompt that demoted in both,
and the fix arm's reclaimed demote carries exactly the bytes the base arm evaporated; every
request text is byte-identical across arms; the other tenant's row is identical across arms; the
arena harness cell ended with every allocation freed. Integrity only, N=1 for the gate and N=5
pairs per order for the harness; nothing here is a support state or a #385 decision.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day12"
# Attempt 1 (`tenant-base`, `tenant-fix`, `arena`) died on an unauthenticated /metrics scrape and a
# stale harness; attempt 2 (`tenant-*-r2`) ran the whole cell but the gate's basic-regex matchers
# never matched the `[prefix-host]` tag (two FAILs, two vacuous passes). Every attempt's dir stays
# as the record. Attempt 3 is the gate evidence; the arena harness cell of attempt 2 stands.
CELLS = {"base": "tenant-base-r3", "fix": "tenant-fix-r3", "arena": "arena-r2"}
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def cell_end(cell):
    rows = [json.loads(l) for l in (ROOT / cell / "CELL.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if r.get("event") == "end"][-1]


def summary(cell):
    return json.loads((ROOT / cell / "ev" / "SUMMARY.json").read_text())


def mb_of(line):
    m = re.search(r"(\d+) tokens, ([\d.]+)MB", line)
    return (int(m.group(1)), m.group(2)) if m else None


def verdicts(cell):
    text = (ROOT / cell / "command.log").read_text(errors="replace")
    return [l for l in text.splitlines() if re.match(r"^\s+(ok|FAIL): |^GATE: ", l)]


print("== build and card")
for arm in ("base", "fix"):
    src = (ROOT / "build" / f"source-{arm}.txt").read_text().strip()
    exit_code = (ROOT / "build" / f"exit-{arm}").read_text().strip()
    sha = (ROOT / "build" / f"binary-{arm}.sha256").read_text().split()[0]
    print(f"  {arm}: source {src} exit {exit_code} binary sha256 {sha}")
    check(f"{arm} build exit 0", exit_code == "0")
    check(f"{arm} tree clean at build", (ROOT / "build" / f"dirty-{arm}.txt").read_text().strip() == "")
base_src = (ROOT / "build" / "source-base.txt").read_text().strip()
check("base is origin/main be07f2d36", base_src.startswith("be07f2d36"))
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)

print("== gate cells")
S = {}
for arm in ("base", "fix"):
    end = cell_end(CELLS[arm])
    check(f"tenant-{arm} status executed-not-qualified, exit 0",
          end["status"] == "executed-not-qualified" and end["exit_code"] == 0,
          f"status={end['status']} exit={end['exit_code']}")
    check(f"tenant-{arm} power limits 600 W",
          all(p["power.limit"] == "600.00 W" for p in end["gpu_power_limits"]))
    lock = json.loads((ROOT / CELLS[arm] / "lock.json").read_text())
    check(f"tenant-{arm} collector lock is the canonical rig lock",
          lock.get("lock") == "/tmp/memra-gpu.lock" and lock.get("acquired") is True and "seam" not in lock)
    S[arm] = summary(CELLS[arm])
    check(f"tenant-{arm} SUMMARY names its arm and binary",
          S[arm]["arm"] == arm and S[arm]["binary_sha256"] == (ROOT / "build" / f"binary-{arm}.sha256").read_text().split()[0])
    v = verdicts(CELLS[arm])
    check(f"tenant-{arm} gate PASS line present and no FAIL line",
          any(l.startswith(f"GATE: kv-host-tenant-reclaim ({arm} arm) PASS") for l in v) and not any("FAIL" in l for l in v))
    for l in v:
        print(f"    {l}")

print("== base arm: the pre-#384 behavior")
b, f = S["base"], S["fix"]
check("base printed the evaporation line", len(b["evaporations"]) >= 1)
for l in b["evaporations"]:
    print(f"    {l}")
check("base evaporation names acme's row", all("t:acme" in l for l in b["evaporations"]))
check("base evicted nothing of acme's own", b["reclaims"] == [])
check("base r8 cold: the evaporated entry is gone", b["requests"]["r8"]["cached_tokens"] == 0)
check("base metrics: rejects >= 1, no reclaims",
      (b["metrics"]["prefix_host_tenant_rejects"] or 0) >= 1 and not b["metrics"].get("prefix_host_tenant_reclaims"))

print("== fix arm: the reclaim")
check("fix printed no evaporation line", f["evaporations"] == [])
check("fix evicted acme's own entries at the cap", len(f["reclaims"]) >= 1)
for l in f["reclaims"]:
    print(f"    {l}")
check("every fix reclaim names acme in the row and the namespace, never beta",
      all('tenant "t:acme"' in l and 'ns "t:acme' in l and "t:beta" not in l for l in f["reclaims"]))
check("fix r8 promotes the reclaimed admission",
      f["requests"]["r8"]["cached_tokens"] == f["requests"]["r6"]["prompt_tokens"] > 0)
check("no device-tier promote-insert skip in either arm", not b.get("device_promote_skips") and not f.get("device_promote_skips"))
check("fix metrics: reclaims >= 1, rejects == 0",
      (f["metrics"]["prefix_host_tenant_reclaims"] or 0) >= 1 and (f["metrics"]["prefix_host_tenant_rejects"] or 0) == 0)
# The reclaim is an eviction, so the pool's entry count must not exceed base's at the end.
check("fix ends with the pool no fuller than base (the cap held)",
      f["metrics"]["prefix_host_entries"] <= (b["metrics"]["prefix_host_entries"] or 0) + 1)

print("== across arms: same bytes, same texts, other tenant untouched")
same_prompts = {}
for arm in ("base", "fix"):
    for l in S[arm]["demotes"]:
        tok, mb = mb_of(l)
        same_prompts.setdefault((tok, "acme" if "t:acme" in l else "beta"), {})[arm] = mb
for (tok, tenant), arms in sorted(same_prompts.items()):
    if len(arms) == 2:
        check(f"{tenant} {tok}-token demote bytes equal across arms", arms["base"] == arms["fix"], f"{arms}")
# The base arm's evaporated images are exactly the bytes the fix arm demoted for the same token counts.
evap = {mb_of(l) for l in b["evaporations"]}
fix_acme = {mb_of(l) for l in f["demotes"] if "t:acme" in l}
check("every image base evaporated, fix demoted at the same byte count (equal D2H payload)",
      evap and evap <= fix_acme, f"evaporated={sorted(evap)} fix acme demotes={sorted(fix_acme)}")
for i in range(1, 9):
    rb, rf = b["requests"][f"r{i}"], f["requests"][f"r{i}"]
    check(f"r{i} HTTP 200 in both arms", rb["http"] == 200 and rf["http"] == 200)
    check(f"r{i} text byte-identical across arms", rb["text_sha256"] == rf["text_sha256"])
    check(f"r{i} prompt_tokens equal across arms", rb["prompt_tokens"] == rf["prompt_tokens"])
for i in (1, 7):
    check(f"r{i} (beta) cached_tokens equal across arms", b["requests"][f"r{i}"]["cached_tokens"] == f["requests"][f"r{i}"]["cached_tokens"])
check("beta's r7 promote in both arms", b["requests"]["r7"]["cached_tokens"] == b["requests"]["r1"]["prompt_tokens"] > 0
      and f["requests"]["r7"]["cached_tokens"] == f["requests"]["r1"]["prompt_tokens"] > 0)
check("beta's promote lines equal across arms",
      [l for l in b["promotes"] if "t:beta" in l] == [l for l in f["promotes"] if "t:beta" in l])
check("no LRU eviction in either arm (the pool never filled)", b["lru_evictions"] == [] and f["lru_evictions"] == [])

print("== arena harness cell (memra#385, harness receipt only)")
end = cell_end(CELLS["arena"])
check("arena status executed-not-qualified, exit 0", end["status"] == "executed-not-qualified" and end["exit_code"] == 0,
      f"status={end['status']} exit={end['exit_code']}")
rec = json.loads((ROOT / CELLS["arena"] / "receipt.json").read_text())
check("arena receipt status executed-not-qualified", rec["status"] == "executed-not-qualified")
check("arena: every row ok (allocated and freed)", all(r["ok"] for r in rec["rows"]))
check("arena: 5 pairs per order, both orders, two arms",
      sum(1 for r in rec["rows"] if r["order"] == "AB") == 10 and sum(1 for r in rec["rows"] if r["order"] == "BA") == 10)
basis = rec["mem_free_at_start"] if rec.get("basis") == "free" else rec["mem_available_at_start"]
check("arena: size is 75 percent of its basis (rounded to 2 MiB) and below MemAvailable at start",
      abs(rec["bytes"] - basis * 0.75) < (2 << 20) and rec["bytes"] < rec["mem_available_at_start"],
      f"basis={rec.get('basis')} {basis} bytes={rec['bytes']}")
for arm, s in rec["summary"].items():
    print(f"    {arm}: n={s['n']} median {s['alloc_ms_median']:.1f} ms (AB {s.get('alloc_ms_median_AB', float('nan')):.1f}, BA {s.get('alloc_ms_median_BA', float('nan')):.1f}) "
          f"min {s['alloc_ms_min']:.1f} max {s['alloc_ms_max']:.1f} free median {s['free_ms_median']:.1f} ms, {s['alloc_gib_per_s_median']:.2f} GiB/s")
print(f"    bytes {rec['bytes']} chunks {rec['chunks']} gpu start [{rec['gpu_at_start']}] end [{rec['gpu_at_end']}] loadavg start {rec['loadavg_at_start']} end {rec['loadavg_at_end']}")

print()
if fails:
    print(f"DAY12 FAIL ({len(fails)}): " + "; ".join(fails))
    sys.exit(1)
print("DAY12 PASS: base evaporates, fix reclaims the tenant's own entry and demotes the same bytes; texts identical; beta untouched; arena harness ran")
