#!/usr/bin/env python3
"""memra#1299 ring cell: judge base and head (rules in ring-check.sh's header).
usage: ring-compare.py OUT_DIR [--base-only]   (exit 0 = PASS)
--base-only judges the base arm alone, before the head arm is run: the vehicle check."""
import re
import sys
from pathlib import Path

CAP = 4096  # PRIME_CHUNK_MAX_TOKENS
MIN_T = 16  # PRIME_MIN_T
SPLIT = 64
# The shortest continuation whose main schedule has a call wider than the chunk (4,097 rows; the
# ring still takes that one: its append fails once a call exceeds its rows less the kept ones).
SHORTEST = 27_916
RING_ERR = "SWA ring append does not fit"
out = Path(sys.argv[1])
ARMS = ("base",) if "--base-only" in sys.argv[2:] else ("base", "head")
guard = (out / "guard.log").read_text(errors="replace")


def rc(arm: str) -> int | None:
    m = re.search(rf"^\S+ {arm} probe rc=(\d+)", guard, re.M)
    return int(m.group(1)) if m else None


def total(text: str) -> int | None:
    m = re.search(r"tickinv: T=(\d+)", text)
    return int(m.group(1)) if m else None


def walk(text: str, t: int) -> list[int] | None:
    """The continuation's schedule: the last [prime-walk] line with t=<continuation> base=64."""
    found = re.findall(rf"\[prime-walk\] t={t} base={SPLIT} calls=\[([0-9, ]*)\]", text)
    return [int(x) for x in found[-1].split(",") if x.strip()] if found else None


ok = True


def judge(cond: bool, text: str) -> None:
    global ok
    ok = ok and cond
    print(("ok   " if cond else "FAIL ") + text)


logs = {arm: (out / f"{arm}.log").read_text(errors="replace") for arm in ARMS}
for arm in ARMS:
    text = logs[arm]
    t = total(text)
    cont = t - SPLIT if t else None
    print(f"{arm}: T={t} continuation={cont} rc={rc(arm)}")
    judge(cont is not None and cont >= SHORTEST, f"{arm}: the continuation is at least {SHORTEST:,} tokens")
    calls = walk(text, cont) if cont else None
    print(f"{arm}: continuation calls {calls}")
    if arm == "base":
        judge(rc(arm) not in (None, 0, 124), f"base: the probe exits non-zero and not by the timeout (rc {rc(arm)})")
        judge(RING_ERR in text, f"base: the log has `{RING_ERR}`")
        for line in text.splitlines():
            if RING_ERR in line:
                print("      " + line.strip()[:300])
                break
        judge(bool(calls) and max(calls) > CAP, f"base: a call wider than {CAP:,} rows (widest {max(calls) if calls else None})")
        if not ok:
            # Where it stopped instead: the last lines of the log.
            for line in [x for x in text.splitlines() if x.strip()][-3:]:
                print("      " + line.strip()[:300])
    else:
        judge(rc(arm) == 0, f"head: the probe exits 0 (rc {rc(arm)})")
        judge(RING_ERR not in text, "head: no ring error")
        judge(
            bool(calls) and max(calls) <= CAP and min(calls) >= MIN_T and sum(calls) == cont,
            f"head: every call within {CAP:,} rows and at least {MIN_T}, summing to the continuation "
            f"(widest {max(calls) if calls else None}, narrowest {min(calls) if calls else None}, sum {sum(calls) if calls else None})",
        )
        row = re.search(rf"^\s*sp{SPLIT} \|.*$", text, re.M)
        judge(row is not None, "head: the sp64 row is printed")
        ref = re.search(r"^ref budget=0 .*$", text, re.M)
        print("reported, not judged (the walk continuation against the one-call batched prime):")
        print("      " + (ref.group(0) if ref else "no ref line"))
        print("      " + (row.group(0).strip() if row else "no sp64 row"))
print(("BASE " if len(ARMS) == 1 else "VERDICT ") + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
