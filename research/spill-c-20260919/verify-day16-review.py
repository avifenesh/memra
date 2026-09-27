#!/usr/bin/env python3
"""Replay the day-16 review receipts (PR #605 findings 1 and 2 on the Option C unwind) from the BOX3 mirror
research/spill-c-20260919/pro-single-day16-review: the eight GPU unit cells of both contract routes passed,
the six-cell contract fault gate is ALL GREEN with the two new promote cells (a partially accepted batch, a
first ready_view reported failed while the engine had published) ending as plain refusals with the ticket
retired and acknowledged, and the host-spill identity gate is line-equal OFF and ON on the review binary
with one H2D receipt per ON promote. Integrity only, N=1, executed-not-qualified.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "pro-single-day16-review"
fails = []


def check(name, ok, detail=""):
    print(f"  {'ok' if ok else 'FAIL'}: {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


def text(path):
    return Path(path).read_text(errors="replace")


def lines(path, pattern):
    return [l for l in text(path).splitlines() if re.search(pattern, l)]


def count(path, pattern):
    return len(lines(path, pattern))


def cdir(cell):
    retries = [d for d in ROOT.glob(f"{cell}-retry*") if d.is_dir() and re.fullmatch(rf"{re.escape(cell)}-retry\d+", d.name)]
    attempts = [ROOT / cell] + sorted(retries, key=lambda d: int(d.name.rsplit("retry", 1)[1]))
    captured = [d for d in attempts if (d / "CELL.jsonl").exists()]
    return captured[-1] if captured else ROOT / cell


def captured(cell):
    return (cdir(cell) / "CELL.jsonl").exists()


def verdicts(cell):
    return lines(cdir(cell) / "command.log", r"^\s+(ok|FAIL): |GATE: ")


def strip_ms(l):
    return re.sub(r" in [\d.]+ms", "", l.split("] ", 1)[1])


print("== binary and card")
source = (ROOT / "build" / "source.txt").read_text().strip()
binary = (ROOT / "build" / "binary.sha256").read_text().split()[0]
card = (ROOT / "card.csv").read_text().splitlines()[1]
print(f"  source {source}\n  binary sha256 {binary}\n  card {card}")
check("card at its 600 W limit", "600.00 W, 600.00 W" in card)
check("build exit 0", (ROOT / "build" / "exit").read_text().strip() == "0")
cells = sorted({re.sub(r"-retry\d+$", "", p.name) for p in ROOT.iterdir() if (p / "CELL.jsonl").exists()})
print("  cells:", ", ".join(cells))
check("four cells captured", len(cells) == 4, str(len(cells)))
for cell in cells:
    rows = [json.loads(l) for l in (cdir(cell) / "CELL.jsonl").read_text().splitlines() if l.strip()]
    end = [r for r in rows if r.get("event") == "end"][-1]
    check(f"{cell}: collector captured the cell at 600 W", end["status"] in ("executed-not-qualified", "failed") and end["gpu_power_limits"][0]["power.limit"] == "600.00 W", f"status={end['status']} exit={end['exit_code']}")
    check(f"{cell}: ran the review tree", (source in text(cdir(cell) / "command.log")) if cell == "gputests" else binary in text(cdir(cell) / "command.log"))
    check(f"{cell}: collector --validate exit 0", (ROOT / "validate.log").exists() and f"validate {cdir(cell).name} rc=0" in text(ROOT / "validate.log"))

print("== GPU unit cells of both routes (eight)")
if captured("gputests"):
    log = cdir("gputests") / "command.log"
    for name in ("option_b_presubmit_refusal_returns_every_plane_and_keeps_the_tier_on",
                 "option_b_postpublish_refusal_retires_the_ticket_and_keeps_the_tier_on",
                 "option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin",
                 "option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin",
                 "option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin",
                 "option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source",
                 "option_c_partial_acceptance_unwinds_refused_with_every_destination_released",
                 "option_c_first_ready_view_failure_unwinds_through_the_published_arm"):
        check(f"gputests: {name} ok", count(log, rf"test worker::tests::{name} \.\.\. ok") == 1)
    check("gputests: test result 8 passed, 0 failed", count(log, r"test result: ok\. 8 passed; 0 failed") == 1)
else:
    check("gputests cell present", False)

print("== kv-host-contract-fault-gate, six cells")
if captured("faultgate"):
    v = verdicts("faultgate")
    check("faultgate: ALL GREEN", any("KV-HOST-CONTRACT-FAULT GATE: ALL GREEN" in l for l in v), f"{len(v)} lines")
    check("faultgate: no FAIL line", not any(l.strip().startswith("FAIL:") for l in v))
    ev = ROOT / "faultgate" / "ev"
    H2D = re.compile(r"contracts door H2D receipt: ticket issuer=(\d+) seq=(\d+)")
    D2H = re.compile(r"contracts door D2H receipt: ticket issuer=(\d+) seq=(\d+)")
    expected = {
        "promote-presubmit": "tier H2D producer fence refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-presubmit)",
        "promote-postpublish": "tier H2D publication refused: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-postpublish)",
        "promote-reject": "tier H2D batch partially refused: 1 of 34 items (injected failure (MEMRA_KV_HOST_FAULT=contract-promote-reject))",
        "promote-readyview": "tier H2D destination 0 not publishable: injected failure (MEMRA_KV_HOST_FAULT=contract-promote-readyview)",
    }
    for cell, reason in expected.items():
        log = ev / f"{cell}-server.log"
        body = text(log).splitlines()
        refusal = f"[prefix-host] promote refused (contracts door): {reason}; serving without the host entry"
        i_ref = next((i for i, l in enumerate(body) if refusal in l), None)
        i_h2d = [i for i, l in enumerate(body) if "contracts door H2D receipt: ticket" in l]
        i_pro = [i for i, l in enumerate(body) if "[prefix-host] promote: " in l]
        check(f"faultgate {cell}: one typed injected refusal, then an H2D receipt and a promote line, no latch, no drop, no leak, no quarantine",
              i_ref is not None and count(log, "injected failure") == 1 and any(i > i_ref for i in i_h2d) and any(i > i_ref for i in i_pro)
              and count(log, r"TIER DISABLED|host entry dropped|leaked|Capacity|already published|no longer whole") == 0, f"refusal@{i_ref} h2d={i_h2d} promote={i_pro}")
        seqs = sorted([int(H2D.search(l).group(2)) for l in body if H2D.search(l)] + [int(D2H.search(l).group(2)) for l in body if D2H.search(l)])
        # A ticket that was issued and aborted (postpublish, reject, readyview) consumes its sequence number:
        # the visible receipts skip exactly one. The presubmit cell issued no ticket for the refusal.
        skipped = set(range(1, max(seqs) + 1)) - set(seqs)
        want_skip = 0 if cell == "promote-presubmit" else 1
        check(f"faultgate {cell}: the aborted ticket's sequence is consumed ({want_skip} skipped: retired and acknowledged, nothing leaked)", len(skipped) == want_skip and len(set(seqs)) == len(seqs), f"seqs={seqs} skipped={sorted(skipped)}")
else:
    check("faultgate cell present", False)

print("== kv-host-spill-identity-gate, DEFAULT spec env, door OFF vs ON on the review binary")
off_cell, on_cell = "hostgate-identity-off-default", "hostgate-identity-on-default"
if captured(off_cell) and captured(on_cell):
    off, on = verdicts(off_cell), verdicts(on_cell)
    check("verdict lines identical", off == on, f"{len(off)} lines")
    check("both arms ALL GREEN", all(any("IDENTITY GATE: ALL GREEN" in l for l in v) for v in (off, on)))
    off_log, on_log = ROOT / off_cell / "ev/host-on-server.log", ROOT / on_cell / "ev/host-on-server.log"
    demotes = lambda log: [strip_ms(l) for l in lines(log, r"\[prefix-host\] demote: ")]
    check("[prefix-host] demote: byte counts equal", demotes(off_log) == demotes(on_log) and len(demotes(on_log)) >= 1, " | ".join(demotes(on_log)))
    events = lambda log: [strip_ms(l) for l in lines(log, r"\[prefix-(cache|host)\] ") if "contracts door" not in l]
    check("prefix event sequence identical (timings stripped, door lines excluded)", events(off_log) == events(on_log) and len(events(on_log)) >= 8, f"{len(events(on_log))} events")
    promotes = count(on_log, r"\[prefix-host\] promote: ")
    check("ON: verify ok on every promote", promotes >= 1 and count(on_log, r"verify ok") == promotes, f"promotes={promotes}")
    h2d = [re.search(r"checksums_sha256=([0-9a-f]{64})", l).group(1) for l in lines(on_log, r"contracts door H2D receipt: ticket")]
    d2h = {re.search(r"checksums_sha256=([0-9a-f]{64})", l).group(1) for l in lines(on_log, r"contracts door D2H receipt: ticket")}
    check("ON: one H2D receipt per promote, each digest equal to a D2H digest of the same log", len(h2d) == promotes and all(h in d2h for h in h2d))
    check("ON: no host-tier refusal line", count(on_log, r"\[prefix-host\] (demote refused|promote refused|REFUSED|TIER DISABLED|.*\(contracts door\): |demote failed)") == 0)
    check("OFF: no contracts door line", count(off_log, r"contracts door") == 0)
else:
    check("identity cells present", False)

print()
print("DAY16 REVIEW REPLAY:", "PASS" if not fails else f"{len(fails)} FAILURE(S): " + "; ".join(fails))
sys.exit(1 if fails else 0)
