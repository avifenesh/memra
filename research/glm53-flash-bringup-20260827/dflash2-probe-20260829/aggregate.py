#!/usr/bin/env python3
"""Aggregate probe cycles into the decision tables: acceptance-at-k (k=1..7) by traffic
class, mean accepted length, tokens per verify cycle, and the speedup arithmetic under
(a) today's sequential verify and (b) T-parallel verify at one decode-step-class cost."""
import json
import sys

DECODE_TOK_S = json.load(open("/root/dfp2/decode_rate.json"))[0]["decode_tok_s"]


def table(cycles, label):
    out = {"label": label}
    for cls in ["all", "tool", "prose"]:
        rows = [c for c in cycles if cls == "all" or c["class"] == cls]
        if not rows:
            continue
        n = len(rows)
        acc_at_k = []
        for k in range(1, 8):
            acc_at_k.append(round(sum(1 for c in rows if all(c["hits"][:k]) and len(c["hits"]) >= k) / n, 4))
        mean_acc = sum(c["accepted"] for c in rows) / n
        mean_prod = sum(c["produced"] for c in rows) / n
        out[cls] = {
            "cycles": n,
            "acc_at_k": acc_at_k,
            "mean_accepted": round(mean_acc, 3),
            "tokens_per_cycle": round(mean_prod, 3),
        }
    return out


def arithmetic(tokens_per_cycle, decode=DECODE_TOK_S):
    t = 1.0 / decode
    seq = {
        "note": "sequential verify: each drafted token still runs the decode program one at a "
                "time, so a cycle of L accepted + 1 bonus costs (L+1) decode steps plus draft "
                "overhead; throughput <= plain decode regardless of acceptance",
        "speedup": "<= 1.0x (none)",
    }
    rows = {}
    for verify_cost, draft_cost in [(1.0, 0.0), (1.0, 0.2), (1.5, 0.2)]:
        rate = tokens_per_cycle / (t * (verify_cost + draft_cost))
        rows[f"verify={verify_cost}t draft={draft_cost}t"] = {
            "projected_tok_s": round(rate, 1),
            "speedup_vs_plain": round(rate / decode, 2),
        }
    return {"decode_tok_s_measured": decode, "sequential_verify": seq, "t_parallel_verify": rows}


results = {}
for src, label in [("cycles_dflash2.json", "dflash2"), ("cycles_ngram.json", "ngram-floor")]:
    try:
        cycles = json.load(open(f"/root/dfp2/{src}"))
    except FileNotFoundError:
        continue
    tab = table(cycles, label)
    tab["arithmetic"] = arithmetic(tab["all"]["tokens_per_cycle"])
    results[label] = tab

json.dump(results, open("/root/dfp2/summary.json", "w"), indent=1)
print(json.dumps(results, indent=1))
