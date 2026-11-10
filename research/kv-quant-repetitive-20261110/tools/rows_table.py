#!/usr/bin/env python3
"""The README's two row tables (memra#1177).

usage: rows_table.py TOKENIZER_JSON PROMPTS_DIR RUNS_DIR
RUNS_DIR holds <cell>/rows.jsonl (gunzip the committed rows.jsonl.gz first).

WHERE: per batch cell, the rows whose a-vs-b spread exceeds 3, how many of them have "\\n\\n"
(id 271) among the contending ids, the rows whose b-vs-c spread exceeds 3 and the overlap, and
how many of all rows have it among the contending ids (the base rate), and the a-vs-b spread
median and p90 over the rows where "\\n\\n" does not contend.
ROWS: every JSONL row where a flip against the f32 reference (a vs b, a vs c batch, a vs c eager)
has an f32 top-2 margin above 1, with each program's top token and margin.
"""
import json
import statistics
import struct
import sys

from tokenizers import Tokenizer

tok = Tokenizer.from_file(sys.argv[1])
PROMPTS, RUNS = sys.argv[2], sys.argv[3]
NL = 271


def f(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def m(t):
    return f(f(t[0][1]) - f(t[1][1]))


def load(c):
    return {json.loads(l)["row"]: json.loads(l) for l in open(f"{RUNS}/{c}/rows.jsonl")}


def q(xs, p):
    """The percentile summarize.py uses: sorted[round((n - 1) * p)]."""
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def ids_of(doc, depth):
    return tok.encode(open(f"{PROMPTS}/{doc}-{depth}.txt").read(), add_special_tokens=False).ids


def contending(r, p):
    return {r[p[0]]["top"][0][0], r[p[0]]["top"][1][0], r[p[1]]["top"][0][0], r[p[1]]["top"][1][0]}


print("| document | depth | rows ab > 3 | with \\n\\n contending | all rows with \\n\\n contending | rows bc > 3 | both | rows without \\n\\n | ab median there | p90 |")
print("|" + "---|" * 10)
for doc in ["jsonl", "csv", "prose"]:
    for depth in ["32768", "65535"]:
        rows = load(f"{doc}-{depth}-batch")
        big = [r for r in rows.values() if r["ab"]["delta"] > 3]
        nl = sum(1 for r in big if NL in contending(r, "ab"))
        bigbc = [r for r in rows.values() if r["bc"]["delta"] > 3]
        ov = len({r["row"] for r in big} & {r["row"] for r in bigbc})
        rest = sorted(r["ab"]["delta"] for r in rows.values() if NL not in contending(r, "ab"))
        nl_all = sum(1 for r in rows.values() if NL in contending(r, "ab"))
        print(
            f"| {doc} | {depth} | {len(big)} | {nl} | {nl_all} | {len(bigbc)} | {ov} | "
            f"{len(rest)} | {statistics.median(rest):.2f} | {q(rest, 0.9):.2f} |"
        )
print()
print("| depth | row | text before | true next | a | b | c batch | c eager | spread ab | spread bc |")
print("|" + "---|" * 10)
for depth in ["32768", "65535"]:
    B = load(f"jsonl-{depth}-batch")
    E = load(f"jsonl-{depth}-eager") if depth == "32768" else None
    ids = ids_of("jsonl", depth)
    sel = set()
    for r in B.values():
        for p in ("ab", "ac"):
            if r[p]["flip"] and m(r["a"]["top"]) > 1:
                sel.add(r["row"])
    if E:
        for r in E.values():
            if r["ac"]["flip"] and m(r["a"]["top"]) > 1:
                sel.add(r["row"])
    for i in sorted(sel):
        r = B[i]
        ctx = tok.decode(ids[max(0, i - 14) : i + 1])
        nxt = tok.decode([ids[i + 1]])

        def cell(t):
            return f"`{json.dumps(tok.decode([t[0][0]]))[1:-1]}` {m(t):.2f}"

        ce = cell(E[i]["c"]["top"]) if E else "not run"
        print(
            f"| {depth} | {i} | `...{json.dumps(ctx[-40:])[1:-1]}` | `{json.dumps(nxt)[1:-1]}` | "
            f"{cell(r['a']['top'])} | {cell(r['b']['top'])} | {cell(r['c']['top'])} | {ce} | "
            f"{r['ab']['delta']:.1f} | {r['bc']['delta']:.1f} |"
        )
