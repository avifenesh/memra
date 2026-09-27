#!/usr/bin/env python3
"""Replay the day-15 REVIEW receipts (PR #599 findings 1 and 2 on the Option B unwind).

Reads research/spill-c-20260919/pro-single-day15-review (the BOX3 mirror) and asserts: the two GPU
unit cells of the unwind pass on the card; the contract fault gate is ALL GREEN with both one-shot
faults (typed refusal, planes back, tier on, the next demote's receipt); and the host-spill identity
and failure gates still print identical verdict lines OFF and ON under the default spec env on the
review binary, with one receipt per ON demote. Integrity only, N=1, executed-not-qualified.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day15-review"
RECEIPT = re.compile(r"contracts door D2H receipt: ticket issuer=(\d+) seq=(\d+) epochs=0/1/1 items=(\d+) \((\d+) KV planes(, draft)?\) complete=(\d+) require=ok checksums_sha256=[0-9a-f]{64} retired acknowledged")
DOOR_REFUSAL = r"\[prefix-host\] (demote refused|promote refused|REFUSED|TIER DISABLED|.*\(contracts door\): |demote failed)"
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def text(p):
    return Path(p).read_text(errors="replace")


def lines(p, pat):
    return [l for l in text(p).splitlines() if re.search(pat, l)]


def count(p, pat):
    return len(lines(p, pat))


def cdir(cell):
    retries = [d for d in ROOT.glob(f"{cell}-retry*") if d.is_dir() and re.fullmatch(rf"{re.escape(cell)}-retry\d+", d.name)]
    attempts = [ROOT / cell] + sorted(retries, key=lambda d: int(d.name.rsplit("retry", 1)[1]))
    captured = [d for d in attempts if (d / "CELL.jsonl").exists()]
    return captured[-1] if captured else ROOT / cell


def status(cell):
    rows = [json.loads(l) for l in (cdir(cell) / "CELL.jsonl").read_text().splitlines() if l.strip()]
    end = [r for r in rows if r.get("event") == "end"][-1]
    return end["status"], end["exit_code"], end["gpu_power_limits"][0]["power.limit"]


def verdicts(cell):
    return lines(cdir(cell) / "command.log", r"^\s+(ok|FAIL): |GATE: |^test result: ")


def strip_ms(l):
    return re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1])


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
check("build exit 0 (release binary and the GPU unit-cell binary)", (ROOT / "build" / "exit").read_text().strip() == "0")
cells = sorted({re.sub(r"-retry\d+$", "", p.name) for p in ROOT.iterdir() if (p / "CELL.jsonl").exists()})
print("  cells:", ", ".join(cells))
check("six cells captured", len(cells) == 6, str(len(cells)))
for cell in cells:
    st, code, power = status(cell)
    check(f"{cell}: collector captured the cell at 600 W", st in ("executed-not-qualified", "failed") and power == "600.00 W", f"status={st} exit={code}")
    check(f"{cell}: --validate exit 0", f"validate {cdir(cell).name} rc=0" in text(ROOT / "validate.log"))

print("== GPU unit cells of the unwind (ignored without a device; run on the card under the rig lock)")
g = cdir("gputests") / "command.log"
check("both cells ran on the review source", source in text(g))
for name in ("option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on", "option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on"):
    check(f"{name} ... ok", count(g, rf"^test worker::tests::{name} \.\.\. ok$") == 1)
check("test result: ok. 2 passed; 0 failed", count(g, r"^test result: ok\. 2 passed; 0 failed") == 1)
check("gputests cell exit 0", status("gputests")[1] == 0)

print("== tools/kv-host-contract-fault-gate.sh (door ON, one-shot contract-presubmit and contract-postpublish)")
f = cdir("faultgate") / "command.log"
check("ALL GREEN", count(f, r"^KV-HOST-CONTRACT-FAULT GATE: ALL GREEN$") == 1)
check("no FAIL line", count(f, r"^\s+FAIL: ") == 0)
for cell, fault, kind, seq in (("presubmit", "contract-presubmit", "producer fence", 1), ("postpublish", "contract-postpublish", "receipt", 2)):
    log = ROOT / "faultgate" / "ev" / f"{cell}-server.log"
    body = text(log).splitlines()
    refused = [i for i, l in enumerate(body) if f"demote failed (tier D2H {kind} refused: injected failure (MEMRA_KV_HOST_FAULT={fault})); nothing demoted" in l]
    receipts = [(i, RECEIPT.search(l)) for i, l in enumerate(body) if RECEIPT.search(l)]
    demotes = [i for i, l in enumerate(body) if "[prefix-host] demote: " in l]
    check(f"{cell}: exactly one injected refusal, typed", len(refused) == 1)
    check(f"{cell}: exactly one receipt and one demote, both after the refusal", len(receipts) == 1 and len(demotes) == 1 and refused and receipts[0][0] > refused[0] and demotes[0] > refused[0])
    check(f"{cell}: the next ticket is seq={seq} ({'no ticket was issued before' if seq == 1 else 'the aborted ticket retired and was acknowledged'})", bool(receipts) and int(receipts[0][1].group(2)) == seq and receipts[0][1].group(6) == receipts[0][1].group(3))
    check(f"{cell}: tier never latched, nothing quarantined, nothing leaked, no Capacity", count(log, r"TIER DISABLED|no longer whole|leaked|Capacity") == 0)
    check(f"{cell}: no host-tier refusal beyond the injected one", count(log, DOOR_REFUSAL) == 1)
    check(f"{cell}: door ON with the transfer engine", count(log, r"contracts door ON \(MEMRA_KV_HOST_CONTRACTS=1\).*Option B") == 1)


def pair(gate, spec, green):
    print(f"== kv-host-spill-{gate}-gate, {spec}, door OFF vs ON (review binary)")
    off, on = f"hostgate-{gate}-off-{spec}", f"hostgate-{gate}-on-{spec}"
    a, b = verdicts(off), verdicts(on)
    check(f"{gate}: verdict lines identical", a == b and len(a) >= 10, f"{len(a)} lines")
    check(f"{gate}: verdict as expected in both arms", all(any(green in l for l in v) for v in (a, b)))
    if gate == "identity":
        lo, ln = ROOT / off / "ev/host-on-server.log", ROOT / on / "ev/host-on-server.log"
        d = lambda p: [strip_ms(l) for l in lines(p, r"\[prefix-host\] demote: ")]
        check("identity: demote byte counts equal", d(lo) == d(ln) and len(d(ln)) >= 1, " | ".join(d(ln)))
        check("identity: one receipt per ON demote", count(ln, r"contracts door D2H receipt: ticket") == len(d(ln)))
        check("identity: verify ok on the promote in both arms", count(lo, r"verify ok") >= 1 and count(lo, r"verify ok") == count(ln, r"verify ok"))
        check("identity: no host-tier refusal ON", count(ln, DOOR_REFUSAL) == 0)
    else:
        for c in ("poolfull", "digest", "alloc"):
            x = [strip_ms(l) for l in lines(ROOT / off / f"ev/{c}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)") if "contracts door" not in l]
            y = [strip_ms(l) for l in lines(ROOT / on / f"ev/{c}-server.log", r"\[prefix-host\] (demote|FAULT|VERIFY|TIER DISABLED|skip)") if "contracts door" not in l]
            check(f"failure: {c} cell tier event lines identical", x == y and len(x) >= 1, f"{len(x)} lines")
        dg = ROOT / on / "ev/digest-server.log"
        check("failure: digest cell ON: receipt, flip, named injected difference, VERIFY FAILED", count(dg, r"D2H receipt: ticket") >= 1 and count(dg, r"as injected \(MEMRA_KV_HOST_FAULT=flip-demote\)") >= 1 and count(dg, r"VERIFY FAILED") >= 1)


pair("identity", "default", "IDENTITY GATE: ALL GREEN")
pair("failure", "default", "FAILURE GATE: 1 FAILURE(S)")
print()
print("DAY15 REVIEW REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
