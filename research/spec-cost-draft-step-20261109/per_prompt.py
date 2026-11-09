#!/usr/bin/env python3
"""Per prompt: K=2 against K=3 from a fixed-K A/B (memra#1093).

Reads a run directory of research/spec-k-cold-long-20261109/ (arms k3 and k2, one K per boot,
the same prompts in every boot) and prints, per prompt, each arm's median decode rate, the ratio,
the acceptance per drafted token, the round time (tokens per round over the decode rate, with
tokens per round = 1 + K x acceptance) and the third draft position's acceptance implied if the
first two positions accept alike under both depths (3 x acc(K=3) - 2 x acc(K=2)).

usage: per_prompt.py RUN_DIR
"""
import json
import statistics
import sys

rows = [json.loads(line) for line in open(sys.argv[1] + "/runs.jsonl")]
prompts = []
for r in rows:
    if r["prompt"] not in prompts:
        prompts.append(r["prompt"])
print("| prompt | prompt tokens | K=3 decode (tok/s) | K=2 decode (tok/s) | K=2 over K=3 | acceptance K=3 / K=2 "
      "| round K=3 / K=2 (ms) | implied third-position acceptance |")
print("|---|---:|---:|---:|---:|---|---|---:|")
for p in prompts:
    arm = {a: [r for r in rows if r["prompt"] == p and r["arm"] == a] for a in ("k3", "k2")}
    rate = {a: statistics.median(r["decode_tok_s"] for r in v) for a, v in arm.items()}
    acc = {a: statistics.median(r["acceptance"] for r in v) for a, v in arm.items()}
    ms = {a: statistics.median(1000 * (1 + int(r["k"][0]) * r["acceptance"]) / r["decode_tok_s"] for r in v)
          for a, v in arm.items()}
    ks = {a: sorted({r["k"][0] for r in v}) for a, v in arm.items()}
    note = "" if ks == {"k3": ["3"], "k2": ["2"]} else f" (K {ks['k3']}/{ks['k2']})"
    print(f"| {p}{note} | {arm['k3'][0]['prompt_tokens']} | {rate['k3']:.1f} | {rate['k2']:.1f} | "
          f"{rate['k2'] / rate['k3']:.3f} | {acc['k3']:.3f} / {acc['k2']:.3f} | {ms['k3']:.1f} / {ms['k2']:.1f} | "
          f"{3 * acc['k3'] - 2 * acc['k2']:.2f} |")
