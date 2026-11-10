#!/usr/bin/env python3
"""Summaries for the memra#1154 prototype cells written by pk_ab.py.

usage: pk_summary.py inv RUNS_JSONL     one row per prompt, one column per (arm, K, context)
       pk_summary.py perf RUNS_JSONL [BASE_ARM CAND_ARM]
"""
import json
import statistics
import sys
from collections import defaultdict


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def inv(rows):
    arms = sorted({r["arm"] for r in rows})
    out = {}
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        prompts = sorted({r["prompt"] for r in mine})
        print(f"\n## {arm}")
        print("| prompt | K | context | tokens | acceptance | sha |")
        print("|---|---|---|---:|---:|---|")
        verdict = {}
        for p in prompts:
            shas = set()
            for r in sorted((r for r in mine if r["prompt"] == p), key=lambda r: (r["spec_k_pin"], r["class"])):
                acc = "" if r["acceptance"] is None else f'{r["acceptance"]:.3f}'
                print(f'| {p} | {r["spec_k_pin"]} | {r["class"]} | {r["completion_tokens"]} | {acc} | {r["sha"]} |')
                shas.add(r["sha"])
                if r["error"]:
                    print(f'| {p} | error | {r["error"]} | | | |')
            verdict[p] = len(shas)
        print("texts per prompt:", ", ".join(f"{p}: {n}" for p, n in verdict.items()))
        by_k = defaultdict(lambda: defaultdict(set))
        for r in mine:
            by_k[r["spec_k_pin"]][r["prompt"]].add(r["sha"])
        for k in sorted(by_k):
            print(f"  K={k}: texts per prompt within the K:", ", ".join(f"{p}: {len(s)}" for p, s in sorted(by_k[k].items())))
        out[arm] = verdict
    return out


def u_count(a, b):
    """Pairs (x in a, y in b) with x > y, ties as half."""
    return sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)


def perf(rows, base="base", cand="cand2"):
    classes = []
    for r in rows:
        if r["class"] not in classes:
            classes.append(r["class"])
    print("| class | arm | n | K served | TTFT median (s) | decode median, range (tok/s) | acceptance per drafted token (mean) | texts |")
    print("|---|---|---:|---|---:|---|---:|---:|")
    med = {}
    for c in classes:
        for arm in (base, cand):
            rs = [r for r in rows if r["class"] == c and r["arm"] == arm and not r["error"]]
            if not rs:
                continue
            dec = [r["decode_tok_s"] for r in rs if r["decode_tok_s"]]
            ks = defaultdict(int)
            for r in rs:
                ks[(r["k"] or ["?"])[0]] += 1
            rates = [r["acceptance"] for r in rs if r["acceptance"] is not None]
            acc = f"{statistics.mean(rates):.3f}" if rates else ""
            med[(c, arm)] = dec
            print(f'| {c} | {arm} | {len(rs)} | {", ".join(f"K={k} x{n}" for k, n in sorted(ks.items()))} | '
                  f'{statistics.median(r["ttft_s"] for r in rs):.3f} | {statistics.median(dec):.2f}, {min(dec):.2f}-{max(dec):.2f} | '
                  f'{acc} | {len({r["sha"] for r in rs})} |')
    print("\n| class | decode ratio (cand over base, medians) | U | acceptance base | acceptance cand | tokens per round base | cand |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c in classes:
        if (c, base) not in med or (c, cand) not in med:
            continue
        b, k = med[(c, base)], med[(c, cand)]

        def acc(arm):
            rs = [r for r in rows if r["class"] == c and r["arm"] == arm and not r["error"] and r["acceptance"] is not None]
            if not rs:
                return float("nan"), float("nan")
            a = statistics.mean(r["acceptance"] for r in rs)
            # tokens a round emits: the accepted drafts plus the one the verify adds
            per = statistics.mean(1 + int(r["k"][0]) * r["acceptance"] for r in rs if r["k"])
            return a, per
        ab, tb = acc(base)
        ak, tk = acc(cand)
        print(f"| {c} | {statistics.median(k) / statistics.median(b):.3f} | {u_count(k, b):g}/{len(k) * len(b)} | {ab:.3f} | {ak:.3f} | {tb:.2f} | {tk:.2f} |")
    errs = [r for r in rows if r["error"]]
    print(f"\nrequests: {len(rows)}, errors: {len(errs)}")
    for c in classes:
        if c.startswith("greedy"):
            shas = {arm: sorted({r["sha"] for r in rows if r["class"] == c and r["arm"] == arm}) for arm in (base, cand)}
            print(f"{c}: {shas}")


def main():
    mode, path = sys.argv[1], sys.argv[2]
    rows = load(path)
    if mode == "inv":
        inv(rows)
    else:
        perf(rows, *sys.argv[3:5])


main()
