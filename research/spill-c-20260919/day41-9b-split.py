#!/usr/bin/env python3
"""Day 41 reader (DAY41.md section 1, registered before it ran): the 9B entry's conv, ssm and hidden split read off
lane A day 31's `by slot class` copy-complete lines, checked against day 38's registered bounds.

usage: day41-9b-split.py <root> [<root> ...]   (lane A receipt dirs, searched recursively for *.log)
"""
import re
import sys
from collections import Counter
from pathlib import Path

LINE = re.compile(
    r"demote copy complete off the tick: .*?(\d+) heap payloads \(([0-9.]+)MB\).*?by slot class: "
    r"conv (\d+) \((\d+) B\), ssm (\d+) \((\d+) B\), hidden (\d+) \((\d+) B\), logits (\d+) \((\d+) B\)")
KV = 950_272
TOKENS = 256
RECURRENT_HIDDEN = (52_599_728, 52_699_728)  # [lo, hi)
LOGITS = (950_272, 1_099_744)  # (lo, hi)
HEAP = (53_650_000, 53_699_472)  # [lo, hi)
ENTRY = (54_600_528, 54_650_000)  # [lo, hi)


def main():
    tuples = Counter()
    files = Counter()
    for root in sys.argv[1:]:
        for log in sorted(Path(root).rglob("*.log")):
            for line in log.read_text(errors="replace").splitlines():
                m = LINE.search(line)
                if m and int(m.group(1)) == 50:
                    tuples[tuple(m.groups())] += 1
                    files[str(log)] += 1
    print(f"DAY41 9B INPUT files={len(files)} lines={sum(tuples.values())} distinct_tuples={len(tuples)}")
    ok = bool(tuples)
    plain_seen = 0
    for groups, count in sorted(tuples.items()):
        payloads, heap_mb, cn, cb, sn, sb, hn, hb, ln, lb = groups
        cb, sb, hb, lb = int(cb), int(sb), int(hb), int(lb)
        heap = cb + sb + hb + lb
        rh = cb + sb + hb
        entry = KV + TOKENS + heap
        plain = hb == 0
        clauses = {
            "heap_rounds_to_line": round(heap / 1e6, 1) == float(heap_mb),
            "slots": int(cn) + int(sn) == 48 and int(ln) == 1 and int(hn) == 1,
        }
        if plain:
            plain_seen += 1
            clauses.update({
                "recurrent_hidden_bound": RECURRENT_HIDDEN[0] <= rh < RECURRENT_HIDDEN[1],
                "logits_bound": LOGITS[0] < lb < LOGITS[1],
                "heap_bound": HEAP[0] <= heap < HEAP[1],
                "entry_bound": ENTRY[0] <= entry < ENTRY[1],
            })
            ok = ok and all(clauses.values())
        cls = "plain" if plain else "draft-bearing"
        if plain:
            shares = " ".join(f"{name} {100 * b / entry:.3f}%" for name, b in
                              (("kv", KV), ("conv", cb), ("ssm", sb), ("hidden", hb), ("logits", lb)))
            whole = f"entry=kv+tokens+heap={entry} | shares of entry: {shares}"
        else:
            # This class's KV carries the draft plane (day 38 section 3, spec class), which no line here prints.
            shares = " ".join(f"{name} {100 * b / heap:.3f}%" for name, b in
                              (("conv", cb), ("ssm", sb), ("hidden", hb), ("logits", lb)))
            whole = f"entry not computed (KV with the draft plane is not on this line) | shares of heap: {shares}"
        print(f"DAY41 9B TUPLE class={cls} lines={count} conv={cn}x{cb // int(cn)}={cb} ssm={sn}x{sb // int(sn)}={sb} "
              f"hidden={hb} logits={lb} ({lb // 4} f32) heap={heap} (line {heap_mb}MB) conv+ssm+hidden={rh} "
              f"{whole} | " + " ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in clauses.items()))
    ok = ok and plain_seen > 0
    print(f"DAY41 9B SPLIT plain_tuples={plain_seen} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
