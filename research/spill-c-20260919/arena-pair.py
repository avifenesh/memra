#!/usr/bin/env python3
"""Day-17 arena pair analysis (research/spill-c-20260919/DAY17.md "The arena cell"): demote and promote
wall times for the pageable host tier against the startup pinned arena (MEMRA_GLM5_TP_KV_HOST=1), the
contracts door OFF in both arms, for the same two entries from the mirrored arena-pair cell, N=5 per arm
per order, both orders, one lock hold. Applies the PRE-REGISTERED rule (written before the run) and prints
its verdict line verbatim with every input; integrity and arithmetic only: one card, one window,
executed-not-qualified.
usage: arena-pair.py [cell-dir]   (default: pro-single-day17/arena-pair next to this file)
"""
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "pro-single-day17" / "arena-pair"
BASE = ROOT.parent / re.sub(r"-retry\d+$", "", ROOT.name)
EV = BASE / "ev"
_attempts = [d for d in ROOT.parent.glob(f"{BASE.name}*") if d.is_dir() and re.fullmatch(rf"{re.escape(BASE.name)}(-retry\d+)?", d.name) and (d / "CELL.jsonl").exists()]
CAPTURED = sorted(_attempts, key=lambda d: int(d.name.rsplit("retry", 1)[1]) if "retry" in d.name else -1)[-1] if _attempts else ROOT
DEMOTE = re.compile(r"\[prefix-host\] demote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
PROMOTE = re.compile(r"\[prefix-host\] promote: (\d+) tokens, ([\d.]+)MB in ([\d.]+)ms")
ARENA_BOOT = "[prefix-host DEBUG] arena startup:"
FORBIDDEN = ("TIER DISABLED", "refused", "WARNING", "arena reserve failed", "VERIFY FAILED", "leaked")
ARMS = ["o1-page", "o1-arena", "o2-arena", "o2-page"]
# The rule's thresholds, fixed before the run (DAY17.md): a first-touch step is present at >= 20 ms and
# absent below 10 ms; "not slower" is within +10 % of the pageable arm's pooled median or below.
STEP_PRESENT_MS = 20.0
STEP_ABSENT_MS = 10.0
NOT_SLOWER_FACTOR = 1.10


def parse(label):
    log = (EV / f"{label}-server.log").read_text(errors="replace").splitlines()
    demotes, promotes = [], []
    last_demote = None
    arena_boot = any(ARENA_BOOT in l for l in log)
    forbidden = sorted({w for l in log for w in FORBIDDEN if w in l})
    for l in log:
        m = DEMOTE.search(l)
        if m:
            last_demote = float(m.group(3))
            demotes.append((int(m.group(1)), float(m.group(2)), last_demote))
            continue
        m = PROMOTE.search(l)
        if m:
            ms = float(m.group(3))
            inline = last_demote
            promotes.append((int(m.group(1)), float(m.group(2)), ms, inline, ms - inline if inline is not None else None))
            last_demote = None
    return demotes, promotes, arena_boot, forbidden


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt(xs):
    return f"median {med(xs):.1f} ms (N={len(xs)}, min {min(xs):.1f}, max {max(xs):.1f})" if xs else "none"


def regime():
    sampler = CAPTURED / "command.gpu.csv"
    if not sampler.exists():
        return f"telemetry: command.gpu.csv absent under {CAPTURED.name}", None
    rows = list(csv.DictReader(sampler.open(errors="replace"), skipinitialspace=True))
    def col(name, conv=float):
        out = []
        for r in rows:
            v = r.get(name)
            if v is None:
                continue
            v = v.split()[0]
            try:
                out.append(conv(v))
            except ValueError:
                pass
        return out
    temps = col("temperature.gpu")
    power = col("power.draw [W]")
    sm = col("clocks.current.sm [MHz]")
    limit = {r.get("power.limit [W]", "").strip() for r in rows}
    text = (f"telemetry: {len(rows)} samples at 250 ms; temperature {min(temps):.0f}..{max(temps):.0f} C; "
            f"power draw max {max(power):.0f} W (limit {', '.join(sorted(limit))}); SM clock {min(sm):.0f}..{max(sm):.0f} MHz")
    return text, (min(temps), max(temps), max(power), ", ".join(sorted(limit)))


def identity():
    """r3..r7 texts and cached_tokens must be identical across the four boots (the served bytes)."""
    rows = {}
    for label in ARMS:
        for i in range(3, 8):
            p = EV / f"{label}-r{i}.json"
            if not p.exists():
                return False, f"{label}-r{i}.json missing"
            j = json.loads(p.read_text())
            text = j["choices"][0].get("text")
            cached = (j.get("usage") or {}).get("cached_tokens", (j.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens"))
            rows.setdefault(i, set()).add((text, cached))
    bad = [i for i, s in rows.items() if len(s) != 1]
    return (not bad), ("identical texts and cached_tokens r3..r7 across the four boots" if not bad else f"r{bad} differ across boots")


def main():
    print(f"arena pair cell: evidence {EV}, collector capture {CAPTURED.name}")
    per = {}
    checks = []
    for label in ARMS:
        d, p, arena_boot, forbidden = parse(label)
        per[label] = (d, p)
        arena = label.endswith("-arena")
        dm = [x[2] for x in d[:5]]
        pm = [x[2] for x in p[:5]]
        pe = [x[4] for x in p[:5] if x[4] is not None]
        print(f"\n{label}: {len(d)} demotes, {len(p)} promotes, arena boot line {'present' if arena_boot else 'absent'}")
        print(f"  raw demote ms (r2..r7):  {[x[2] for x in d]}")
        print(f"  raw promote ms (r3..r7): {[x[2] for x in p]}  inline demote ms: {[x[3] for x in p]}")
        print(f"  demote  (r2..r6): {fmt(dm)}")
        print(f"  promote (r3..r7): {fmt(pm)}")
        print(f"  promote minus inline demote: {fmt(pe)}")
        print(f"  entries: " + ", ".join(sorted({f'{t} tok/{mb} MB' for t, mb, *_ in d})))
        checks.append((f"{label}: 6 demotes and 5 promotes", len(d) == 6 and len(p) == 5))
        checks.append((f"{label}: arena boot line {'present' if arena else 'absent'} as the arm requires", arena_boot == arena))
        checks.append((f"{label}: no forbidden wording {list(FORBIDDEN)}", not forbidden))
        checks.append((f"{label}: every promote had an inline demote in its window", all(x[3] is not None for x in p)))
    # the rule's inputs
    step = {}
    steady = {}
    for label in ARMS:
        d = [x[2] for x in per[label][0]]
        if len(d) == 6:
            step[label] = med(d[:3]) - med(d[3:])
            steady[label] = d[3:]
    pooled = {}
    for arm in ("page", "arena"):
        labels = [l for l in ARMS if l.endswith(f"-{arm}")]
        dm = [x[2] for l in labels for x in per[l][0][:5]]
        pm = [x[2] for l in labels for x in per[l][1][:5]]
        pe = [x[4] for l in labels for x in per[l][1][:5] if x[4] is not None]
        sd = [x for l in labels for x in steady.get(l, [])]
        pooled[arm] = (med(dm), med(pm), med(pe), med(sd))
        print(f"\npooled {arm.upper()} ({' + '.join(labels)}):")
        print(f"  demote (r2..r6):  {fmt(dm)}")
        print(f"  promote (r3..r7): {fmt(pm)}")
        print(f"  promote minus inline demote: {fmt(pe)}")
        print(f"  steady-state demote (r5..r7): {fmt(sd)}")
    print()
    for label in ARMS:
        if label in step:
            print(f"first-touch step {label}: median(r2..r4) - median(r5..r7) = {step[label]:.1f} ms")
    text, reg = regime()
    print(text)
    marks = EV / "marks.tsv"
    window = ""
    if marks.exists():
        rows = [l.split("\t") for l in marks.read_text().splitlines() if l.strip()]
        t0 = datetime.fromisoformat(rows[0][0].replace("Z", "+00:00"))
        t1 = datetime.fromisoformat(rows[-1][0].replace("Z", "+00:00"))
        window = f"{(t1 - t0).total_seconds():.0f}s"
        print(f"window: {rows[0][0]} .. {rows[-1][0]} ({window}, one lock hold)")
    ident_ok, ident_text = identity()
    checks.append((f"identity: {ident_text}", ident_ok))
    page_bytes = sorted({(t, mb) for l in ARMS if l.endswith("-page") for t, mb, *_ in per[l][0]})
    arena_bytes = sorted({(t, mb) for l in ARMS if l.endswith("-arena") for t, mb, *_ in per[l][0]})
    checks.append(("demote byte counts equal across arms", page_bytes == arena_bytes and bool(page_bytes)))
    fails = [n for n, ok in checks if not ok]
    for n, ok in checks:
        print(f"  {'ok' if ok else 'FAIL'}: {n}")
    # the pre-registered rule
    complete = all(l in step for l in ARMS)
    page_step_present = complete and all(step[l] >= STEP_PRESENT_MS for l in ("o1-page", "o2-page"))
    arena_step_absent = complete and all(step[l] < STEP_ABSENT_MS for l in ("o1-arena", "o2-arena"))
    if not complete:
        first_touch = "void (incomplete sequences)"
    elif not page_step_present:
        first_touch = "void (pageable step absent this sitting)"
    elif arena_step_absent:
        first_touch = "arena_first_touch_absent"
    else:
        first_touch = "first-touch step present in arena arm"
    not_slower = complete and pooled["arena"][3] <= pooled["page"][3] * NOT_SLOWER_FACTOR and pooled["arena"][2] <= pooled["page"][2] * NOT_SLOWER_FACTOR
    slower_text = "arena_not_slower" if not_slower else "arena_slower"
    verdict = "VOID (integrity)" if fails else f"{first_touch}; {slower_text}"
    line = (f"ARENA-PAIR rule first_touch_page_o1={step.get('o1-page', float('nan')):.1f} first_touch_page_o2={step.get('o2-page', float('nan')):.1f} "
            f"first_touch_arena_o1={step.get('o1-arena', float('nan')):.1f} first_touch_arena_o2={step.get('o2-arena', float('nan')):.1f} "
            f"steady_demote_page={pooled['page'][3]:.1f} steady_demote_arena={pooled['arena'][3]:.1f} "
            f"demote_page={pooled['page'][0]:.1f} demote_arena={pooled['arena'][0]:.1f} "
            f"promote_page={pooled['page'][1]:.1f} promote_arena={pooled['arena'][1]:.1f} "
            f"promote_excl_page={pooled['page'][2]:.1f} promote_excl_arena={pooled['arena'][2]:.1f} "
            f"N=5/arm/order pooled=10 orders=2 window={window} "
            + (f"temp_c={reg[0]:.0f}..{reg[1]:.0f} power_max_w={reg[2]:.0f} power_limit_w={reg[3]} " if reg else "telemetry=absent ")
            + f"identity={'ok' if ident_ok else 'FAIL'} integrity={'ok' if not fails else str(len(fails)) + ' FAIL'} -> {verdict}")
    print(line)
    print(f"ARENA PAIR REPLAY: {'PASS' if not fails else 'FAIL ' + str(len(fails))} ({len(checks)} checks)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
