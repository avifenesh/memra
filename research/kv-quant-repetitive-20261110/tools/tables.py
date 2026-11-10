#!/usr/bin/env python3
"""Condensed README tables from summarize.py's SUMMARY.jsonl (memra#1177).

usage: tables.py SUMMARY.jsonl
"""
import json
import sys


def f(x):
    return f"{x:.2f}"


def main() -> None:
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    by = {(r["cell"], r["pair"]): r for r in rows}
    cells = sorted({r["cell"] for r in rows}, key=lambda c: (c.rsplit("-", 2)[0], int(c.rsplit("-", 2)[1]), c))
    print(
        "| document | depth | decode | rows | pair | flips | x margin > 1 | > 2 | y margin > 1 | > 2 | "
        "x right | y right | spread median | p90 | max | rows spread > 3 | top-1 = next token (x / y) |"
    )
    print("|" + "---|" * 17)
    for c in cells:
        doc, depth, arm = c.rsplit("-", 2)
        for p in ("ab", "ac", "bc"):
            r = by.get((c, p))
            if r is None:
                continue
            if arm == "eager" and p == "ab":
                continue  # the same two programs as the batch cell's ab row
            x, y = p[0], p[1]
            print(
                f"| {doc} | {depth} | {arm} | {r['rows']} | {p} | {r['flips']} | "
                f"{r[f'flips_margin_{x}_over_1']} | {r[f'flips_margin_{x}_over_2']} | "
                f"{r[f'flips_margin_{y}_over_1']} | {r[f'flips_margin_{y}_over_2']} | "
                f"{r[f'flips_{x}_right']} | {r[f'flips_{y}_right']} | "
                f"{f(r['spread_median'])} | {f(r['spread_p90'])} | {f(r['spread_max'])} | "
                f"{r['rows_spread_over_3']} | {r[f'top1_is_next_{x}']} / {r[f'top1_is_next_{y}']} |"
            )
    print()
    print("| document | depth | margin a p10 | median | margin b p10 | median | margin c p10 | median |")
    print("|" + "---|" * 8)
    for c in cells:
        doc, depth, arm = c.rsplit("-", 2)
        if arm != "batch":
            continue
        ab, bc = by[(c, "ab")], by.get((c, "bc"))
        print(
            f"| {doc} | {depth} | {f(ab['margin_a_p10'])} | {f(ab['margin_a_median'])} | "
            f"{f(ab['margin_b_p10'])} | {f(ab['margin_b_median'])} | "
            + (f"{f(bc['margin_c_p10'])} | {f(bc['margin_c_median'])} |" if bc else "- | - |")
        )


if __name__ == "__main__":
    main()
