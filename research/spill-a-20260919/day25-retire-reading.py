#!/usr/bin/env python3
"""WP-A day 25 reading of the retire-settle share (DAY25.md Task 2 pre-registration, fixed before the run).

Input: one or more server logs of the hit gate's ON arm (`qwen-on-server.log`, `qwen-off-server.log`). Per
`[prefix-cache] capture published off the tick (<why>): N tokens complete after P poll(s), C ms from submission
to completion, T ms to publication (<settled_by>; the settle held the owner thread H ms, entered A ms after
submission)` line: why, tokens, polls, completion C, held H, entered-after A, share H / C. Grouped by settled_by:
N, min, median, max of H, A and C, and of the share. A line without the clause (an older binary) is counted as
`no-clause`. Nothing here is tuned; no threshold.

    day25-retire-reading.py <server.log> [<server.log> ...]
"""
import re
import statistics
import sys

LINE = re.compile(
    r"\[prefix-cache\] capture published off the tick \((?P<why>[^)]+)\): (?P<toks>\d+) tokens complete after "
    r"(?P<polls>\d+) poll\(s\), (?P<completion>[0-9.]+)ms from submission to completion, (?P<publication>[0-9.]+)ms "
    r"to publication \((?P<settled>[^;)]+)(?:; the settle held the owner thread (?P<held>[0-9.]+)ms, entered "
    r"(?P<after>[0-9.]+)ms after submission)?\)"
)


def stats(xs):
    if not xs:
        return "n/a"
    return f"N={len(xs)} min={min(xs):.2f} median={statistics.median(xs):.2f} max={max(xs):.2f}"


def main():
    rows = []
    for path in sys.argv[1:]:
        for ln in open(path, errors="replace"):
            m = LINE.search(ln)
            if not m:
                continue
            d = m.groupdict()
            row = {"log": path.rsplit("/", 1)[-1], "why": d["why"], "toks": int(d["toks"]), "polls": int(d["polls"]),
                   "completion": float(d["completion"]), "settled": d["settled"].strip(),
                   "held": float(d["held"]) if d["held"] else None,
                   "after": float(d["after"]) if d["after"] else None}
            row["share"] = row["held"] / row["completion"] if row["held"] is not None and row["completion"] > 0 else None
            rows.append(row)
            print(f"  {row['log']} why={row['why']} toks={row['toks']} polls={row['polls']} completion={row['completion']:.1f} "
                  f"settled_by='{row['settled']}' held={row['held']} entered_after={row['after']} "
                  f"share={'n/a' if row['share'] is None else f'{row['share']:.3f}'}")
    groups = {}
    for r in rows:
        groups.setdefault(r["settled"], []).append(r)
    print(f"DAY25 RETIRE-SETTLE lines={len(rows)} no_clause={sum(r['held'] is None for r in rows)}")
    for settled, rs in sorted(groups.items()):
        held = [r["held"] for r in rs if r["held"] is not None]
        after = [r["after"] for r in rs if r["after"] is not None]
        comp = [r["completion"] for r in rs]
        share = [r["share"] for r in rs if r["share"] is not None]
        print(f"DAY25 RETIRE-SETTLE settled_by='{settled}' N={len(rs)} whys={sorted(set(r['why'] for r in rs))} "
              f"toks={sorted(set(r['toks'] for r in rs))} | held_ms {stats(held)} | entered_after_ms {stats(after)} | "
              f"completion_ms {stats(comp)} | share_held_over_completion {stats(share)}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
