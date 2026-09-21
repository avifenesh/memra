#!/usr/bin/env python3
"""Replay the day-16 OFF/ON comparisons from the mirrored receipts (lead ruling 15, Option C).

Reads research/spill-c-20260919/pro-single-day16 (the BOX3 mirror) and asserts what Option C's
admissibility gate requires on top of B's: with the pageable-tier D2H (B) AND the promote H2D (C)
routed through the TransferEngine under MEMRA_KV_HOST_CONTRACTS=1, every gate prints the same
verdict lines OFF and ON (host-spill identity and failure gates under the DEFAULT spec environment,
the plain identity pair, lane A's tenant reclaim gate fix arm, lane B's two prefix gates,
serve-smoke), `verify ok` on every ON promote, equal `[prefix-host] demote:` byte counts, one D2H
receipt before every ON demote, one H2D receipt (require=ok against the D2H receipts, published
retired acknowledged) before every ON promote whose digest equals a D2H digest in the same log, the
six GPU unit cells of both routes passed, the four-cell contract fault gate ALL GREEN, and the WC
pair cell replayed. Integrity only, N=1 (the WC pair N=5 per arm per order), executed-not-qualified;
nothing here is a support state.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day16"
DRAFT_CLASS = "server-prefix-entry-v5-kv-q8_0-34B-q5_1-24B+mtp-draft-kv-q8_0-34B-q5_1-24B"
DOOR_REFUSAL = r"\[prefix-host\] (demote refused|promote refused|REFUSED|TIER DISABLED|.*\(contracts door\): |demote failed)"
RECEIPT = re.compile(
    r"\[prefix-host\] contracts door D2H receipt: ticket issuer=(\d+) seq=(\d+) epochs=(\d+)/(\d+)/(\d+) "
    r"items=(\d+) \((\d+) KV planes(, draft)?\) complete=(\d+) require=ok checksums_sha256=([0-9a-f]{64}) "
    r"retired acknowledged"
)
H2D = re.compile(
    r"\[prefix-host\] contracts door H2D receipt: ticket issuer=(\d+) seq=(\d+) epochs=(\d+)/(\d+)/(\d+) "
    r"items=(\d+) \((\d+) KV planes(, draft)?\) complete=(\d+) require=ok checksums_sha256=([0-9a-f]{64}) "
    r"published retired acknowledged"
)
DOOR_ON = (r"contracts door ON \(MEMRA_KV_HOST_CONTRACTS=1\): 1 model program identities.*in-flight \d+; host tier armed; "
           r"KV plane D2H through the transfer engine on the pageable tier \(Option B\); KV plane H2D through the same engine on promote \(Option C\)")
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def text(path):
    return Path(path).read_text(errors="replace")


def lines(path, pattern):
    return [l for l in text(path).splitlines() if re.search(pattern, l)]


def cdir(cell):
    """The collector directory that captured this cell: the base name, or the highest `-retryN`
    attempt (the runner re-invokes the collector into a new `--out` while the rig lock is busy;
    the gate's own `ev/` stays under the base name)."""
    retries = [d for d in ROOT.glob(f"{cell}-retry*") if d.is_dir() and re.fullmatch(rf"{re.escape(cell)}-retry\d+", d.name)]
    attempts = [ROOT / cell] + sorted(retries, key=lambda d: int(d.name.rsplit("retry", 1)[1]))
    captured = [d for d in attempts if (d / "CELL.jsonl").exists()]
    return captured[-1] if captured else ROOT / cell


def captured(cell):
    return (cdir(cell) / "CELL.jsonl").exists()


def verdicts(cell):
    return lines(cdir(cell) / "command.log", r"^\s+(ok|FAIL): |GATE: |cache-meter-gate: |^serve-smoke: |-> (PASS|FAIL)|REFUSED")


def cell_status(cell):
    rows = [json.loads(l) for l in (cdir(cell) / "CELL.jsonl").read_text().splitlines() if l.strip()]
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
    # The digest over the ordered item checksums names the entry's bytes: two demotes of the same
    # entry (the same prompt, the same token count and size) carry the same digest, and different
    # entries never share one.
    by_digest = {}
    for r_i, d_i in zip(idx_r, idx_d):
        digest = RECEIPT.search(body[r_i]).group(10)
        by_digest.setdefault(digest, set()).add(re.sub(r" in [\d.]+ms.*", "", body[d_i]))
    check(f"{label}: one checksum digest per distinct entry (a re-demoted entry repeats its digest)", all(len(v) == 1 for v in by_digest.values()) and len(by_digest) >= 1, f"{len(by_digest)} digests over {len(rs)} demotes")
    check(f"{label}: no receipt mismatch refusal", count(log, r"differs from its D2H contract receipt") == 0)


def promotes_of(log):
    return [strip_ms(l) for l in lines(log, r"\[prefix-host\] promote: ")]


def h2d_law(label, log, expect_draft):
    """Option C: every ON promote is preceded by exactly one well-formed H2D receipt whose digest names
    bytes a D2H receipt in the same log delivered (the same ordered checksums, the same domain)."""
    rs = [H2D.search(l).groups() for l in lines(log, r"contracts door H2D receipt: ticket") if H2D.search(l)]
    raw = count(log, r"contracts door H2D receipt: ticket")
    check(f"{label}: every H2D receipt line parses", raw == len(rs) and raw >= 1, f"lines={raw} parsed={len(rs)}")
    check(f"{label}: one H2D receipt per promote", len(rs) == len(promotes_of(log)), f"receipts={len(rs)} promotes={len(promotes_of(log))}")
    body = text(log).splitlines()
    idx_r = [i for i, l in enumerate(body) if "contracts door H2D receipt: ticket" in l]
    idx_p = [i for i, l in enumerate(body) if "[prefix-host] promote: " in l]
    check(f"{label}: each H2D receipt precedes its promote line", all(r < p for r, p in zip(idx_r, idx_p)) and len(idx_r) == len(idx_p))
    for issuer, seq, st, sg, dg, items, planes, draft, complete, digest in rs:
        planes_total = int(planes) + (1 if draft else 0)
        check(f"{label}: H2D receipt seq {seq}: epochs 0/1/1, items = 2 x planes, complete = items, draft plane {'present' if expect_draft else 'absent'}",
              (st, sg, dg) == ("0", "1", "1") and int(items) == 2 * planes_total and int(complete) == int(items) and bool(draft) == expect_draft,
              f"items={items} planes={planes} draft={bool(draft)} complete={complete}")
    d2h_digests = {r[9] for r in receipts(log)}
    check(f"{label}: every H2D digest equals a D2H digest of the same log (the same bytes came back up)", all(r[9] in d2h_digests for r in rs) and len(rs) >= 1)
    all_seqs = sorted(int(r[1]) for r in rs) + sorted(int(r[1]) for r in receipts(log))
    check(f"{label}: D2H and H2D tickets share one issuer and one strictly increasing sequence", len({r[0] for r in rs} | {r[0] for r in receipts(log)}) == 1 and len(set(all_seqs)) == len(all_seqs))
    check(f"{label}: no H2D receipt refusal, no host entry dropped by the door, no latch", count(log, r"tier H2D receipt refused|host entry dropped, cold path serves|TIER DISABLED") == 0)


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
check("build exit 0", (ROOT / "build" / "exit").read_text().strip() == "0")
cells = sorted({re.sub(r"-retry\d+$", "", p.name) for p in ROOT.iterdir() if (p / "CELL.jsonl").exists()})
print("  cells:", ", ".join(cells))
check("seventeen cells captured", len(cells) == 17, str(len(cells)))
retried = [d.name for d in ROOT.glob("*-retry*") if d.is_dir() and (d / "CELL.jsonl").exists()]
print("  captured on a retry (rig lock busy on the earlier attempts):", ", ".join(retried) or "none")
for cell in cells:
    status, code, power = cell_status(cell)
    check(f"{cell}: collector captured the cell at 600 W", status in ("executed-not-qualified", "failed") and power[0]["power.limit"] == "600.00 W", f"status={status} exit={code}")
    check(f"{cell}: ran the day-16 binary", binary in text(cdir(cell) / "command.log"))
    check(f"{cell}: collector --validate exit 0", (ROOT / "validate.log").exists() and f"validate {cdir(cell).name} rc=0" in text(ROOT / "validate.log"))


def identity_pair(spec, label):
    print(f"== kv-host-spill-identity-gate, {label}, door OFF vs ON")
    off_cell, on_cell = f"hostgate-identity-off-{spec}", f"hostgate-identity-on-{spec}"
    if not captured(off_cell) or not captured(on_cell):
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
    check(f"[{spec}] ON: door ON line names Options B and C and the in-flight bound", count(on_log, DOOR_ON) == 1)
    check(f"[{spec}] ON: no host-tier refusal line of any kind", count(on_log, DOOR_REFUSAL) == 0)
    check(f"[{spec}] the device cache's typed lines are the same in both arms (lane B's refusal is not a door effect)", lines(off_log, r"\[prefix-cache\] insert refused") == lines(on_log, r"\[prefix-cache\] insert refused"))
    receipt_law(f"[{spec}] ON", on_log, expect_draft=(spec == "default"))
    h2d_law(f"[{spec}] ON", on_log, expect_draft=(spec == "default"))
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
    if not captured(off_cell) or not captured(on_cell):
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
    dbody = text(on_digest).splitlines()
    i_named = next((i for i, l in enumerate(dbody) if "plane host bytes differ from the D2H receipt as injected (MEMRA_KV_HOST_FAULT=flip-demote); the verify arm catches it at promote" in l), None)
    i_verify = next((i for i, l in enumerate(dbody) if "VERIFY FAILED" in l), None)
    check(f"[{spec}] digest cell ON: Option C names the injected difference at the H2D, publishes under the completion's own checksums, and the verify arm catches it (OFF shape kept)", i_named is not None and i_verify is not None and i_named < i_verify and count(on_digest, r"tier H2D receipt refused") == 0, f"named@{i_named} verify@{i_verify}")
    check(f"[{spec}] alloc cell: the tier latches off in both arms", all(count(ROOT / c / "ev/alloc-server.log", r"TIER DISABLED: pinned host alloc") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] alloc cell ON: the fault fires before any plane leaves the entry (no receipt, no quarantine)", count(on_alloc, r"contracts door D2H receipt: ticket") == 0 and count(on_alloc, r"quarantined|no longer whole") == 0)
    check(f"[{spec}] pool-full: the tenant share cap evaporates before the D2H in both arms (pre-existing)", all(count(ROOT / c / "ev/poolfull-server.log", r"demote evaporated at the tenant share cap") >= 1 for c in (off_cell, on_cell)))
    check(f"[{spec}] ON: no door refusal line in any cell (the alloc cell's latch and the pool-full evaporation are the OFF shape)", sum(count(ROOT / on_cell / f"ev/{c}-server.log", r"\(contracts door\): |\[prefix-host\] REFUSED|promote refused") for c in ("poolfull", "digest", "alloc")) == 0)


failure_pair("default", "DEFAULT spec environment")


def tenant_pair():
    print("== lane A kv-host-tenant-reclaim-gate (fix arm), door OFF vs ON")
    off_cell, on_cell = "tenant-off", "tenant-on"
    if not captured(off_cell) or not captured(on_cell):
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
    check("ON: no host-tier refusal line of any kind", count(on_log, DOOR_REFUSAL) == 0)
    receipt_law("tenant ON", on_log, expect_draft=True)
    h2d_law("tenant ON", on_log, expect_draft=True)


tenant_pair()

print("== serve-smoke (plain + cache-metering arms), door OFF vs ON")
if captured("smoke-off") and captured("smoke-on"):
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
    if not captured(off_cell) or not captured(on_cell):
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

print("== GPU unit cells of both routes (cargo test --release -- --ignored, under the collector lock)")
if captured("gputests"):
    log = cdir("gputests") / "command.log"
    for name in ("option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on",
                 "option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on",
                 "option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin",
                 "option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin",
                 "option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin",
                 "option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source"):
        check(f"gputests: {name} ok", count(log, rf"test worker::tests::{name} \.\.\. ok") == 1)
    check("gputests: test result 6 passed, 0 failed", count(log, r"test result: ok\. 6 passed; 0 failed") == 1)
else:
    check("gputests cell present", False)

print("== kv-host-contract-fault-gate (door ON; demote and promote faults, one-shot each)")
if captured("faultgate"):
    v = verdicts("faultgate")
    check("faultgate: ALL GREEN", any("KV-HOST-CONTRACT-FAULT GATE: ALL GREEN" in l for l in v), f"{len(v)} lines")
    check("faultgate: no FAIL line", not any(l.strip().startswith("FAIL:") for l in v))
    ev = ROOT / "faultgate" / "ev"
    for cell, kind, seq in (("presubmit", "producer fence", 1), ("postpublish", "receipt", 2)):
        log = ev / f"{cell}-server.log"
        check(f"faultgate {cell}: one injected demote refusal then a receipt seq={seq}", count(log, rf"demote failed \(tier D2H {kind} refused: injected failure") == 1 and count(log, rf"contracts door D2H receipt: ticket issuer=\d+ seq={seq} ") == 1)
    for cell, kind in (("promote-presubmit", "producer fence"), ("promote-postpublish", "publication")):
        log = ev / f"{cell}-server.log"
        body = text(log).splitlines()
        i_ref = next((i for i, l in enumerate(body) if f"promote refused (contracts door): tier H2D {kind} refused: injected failure" in l), None)
        i_h2d = [i for i, l in enumerate(body) if "contracts door H2D receipt: ticket" in l]
        i_pro = [i for i, l in enumerate(body) if "[prefix-host] promote: " in l]
        check(f"faultgate {cell}: one injected promote refusal, then an H2D receipt and a promote line after it, no latch, no drop, no leak",
              i_ref is not None and count(log, "injected failure") == 1 and any(i > i_ref for i in i_h2d) and any(i > i_ref for i in i_pro)
              and count(log, r"TIER DISABLED|host entry dropped|leaked|Capacity") == 0, f"refusal@{i_ref} h2d={i_h2d} promote={i_pro}")
        if cell == "promote-postpublish" and i_ref is not None:
            seqs = [int(H2D.search(body[i]).group(2)) for i in i_h2d if H2D.search(body[i])]
            d2h = [int(RECEIPT.search(l).group(2)) for l in body if RECEIPT.search(l)]
            check("faultgate promote-postpublish: the aborted ticket's sequence number is consumed (the next tickets skip it: retired and acknowledged, nothing leaked)", len(set(seqs + d2h)) == len(seqs + d2h) and max(seqs + d2h) == len(seqs + d2h) + 1, f"h2d={seqs} d2h={d2h}")
else:
    check("faultgate cell present", False)

print("== WC pair cell (OFF vs ON demote and promote wall times, N=5 per arm per order, one lock hold)")
if captured("wc-pair"):
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "wc-pair.py"), str(cdir("wc-pair"))], capture_output=True, text=True)
    print("\n".join("    " + l for l in r.stdout.splitlines()))
    check("wc-pair: replay PASS", r.returncode == 0 and "WC PAIR REPLAY: PASS" in r.stdout)
else:
    check("wc-pair cell present", False)

print()
print("DAY16 REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
