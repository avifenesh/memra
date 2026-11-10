#!/usr/bin/env python3
"""The README's condensed tables, from SUMMARY.jsonl (memra#1177).

usage: tables.py SUMMARY.jsonl
"""
import json
import sys


def f(x):
    return f"{x:.2f}"


def main() -> None:
    cells = [json.loads(line) for line in open(sys.argv[1])]
    print("| cell | prompt | answer tokens | answer rows | a differs | b differs | c differs | known values right | default text = plain |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        right = c["served_answer_right"]
        print(
            f"| {c['cell']} | {c['prompt_tokens']} | {c['answer_tokens']} | {c['answer_rows']} | "
            f"{c['a_top1_differs_from_served']} | {c['b_top1_differs_from_served']} | {c['c_top1_differs_from_served']} | "
            f"{sum(right.values())} of {len(right)} | {'yes' if c['default_equals_plain_text'] else 'no'} |"
        )
    print()
    print("| cell | rows | a/c spread median | p90 | max | rows > 1 | rows > 3 | a/c flips |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cells:
        for where, label in (("answer", "answer"), ("question", "question"), ("document", "document")):
            s = c[f"spread_ac_{where}"]
            print(
                f"| {c['cell']} | {label} {s['n']} | {f(s['median'])} | {f(s['p90'])} | {f(s['max'])} | "
                f"{s['over_1']} | {s['over_3']} | {c[f'flips_ac_{where}']} |"
            )
    print()
    print("| cell | a/b answer: median, p90, max | b/c answer: median, p90, max | a/b document: p90, max | b/c document: p90, max |")
    print("|---|---|---|---|---|")
    for c in cells:
        ab, bc = c["spread_ab_answer"], c["spread_bc_answer"]
        abd, bcd = c["spread_ab_document"], c["spread_bc_document"]
        print(
            f"| {c['cell']} | {f(ab['median'])}, {f(ab['p90'])}, {f(ab['max'])} | {f(bc['median'])}, {f(bc['p90'])}, {f(bc['max'])} | "
            f"{f(abd['p90'])}, {f(abd['max'])} | {f(bcd['p90'])}, {f(bcd['max'])} |"
        )
    print()
    print("| cell | smallest answer-row margin a / b / c | answer rows with a margin under 1 | of them inside a known value | value rows | smallest margin at a value row a / b / c | largest a/c spread at a value row |")
    print("|---|---|---|---|---|---|---|")
    for c in cells:
        near = [w for w in c["answer_rows_spread_over_1"] if min(w["a_margin"], w["b_margin"], w["c_margin"]) < 1]
        inside = [w for w in near if w["inside_value"]]
        print(
            f"| {c['cell']} | {f(c['margin_a_answer']['min'])} / {f(c['margin_b_answer']['min'])} / {f(c['margin_c_answer']['min'])} | "
            f"{len(near)} | {len(inside)} | {c['value_rows']} | "
            f"{f(c['value_rows_min_margin_a'])} / {f(c['value_rows_min_margin_b'])} / {f(c['value_rows_min_margin_c'])} | "
            f"{f(c['value_rows_ac_spread_max'])} |"
        )


if __name__ == "__main__":
    main()
