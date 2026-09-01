#!/usr/bin/env python3
"""Prompt-SHAPE cells for qwen4_exp spec acceptance (mtp9 residual item 2).

mtp9 found that every mtp2..mtp8 perf row shared one prompt file whose full-vocab draft
acceptance is 0.840, while chat-template renders of real tasks accept 0.290-0.588. Since the
spec round is ~80% verify, committed-tokens-per-round is the whole multiplier — so what moves
acceptance across prompt SHAPES is worth more than any kernel left in the residual.

The confound to separate is thinking prose. memory `reasoning-effort-unpinned-decode-cell`:
on qwen38 the drafter accepts think-prose at ~0.76 versus the claim shape at 1.00, a silent
-40% that presents as a fleet-wide regression. This emits the SAME four tasks in three
shapes so acceptance can be attributed to the shape rather than to the task:

  thinkon   default render (this model's default is `Reasoning effort ... xhigh`)
  thinkoff  chat_template_kwargs={"enable_thinking": False}
  efflow    chat_template_kwargs={"reasoning_effort": "low"}

All four tasks are from the HELD-OUT set (absent from the rank corpus), so these cells stay
comparable with the mtp9 held-out numbers.

Usage: python3 make-shape-prompts.py <artifact_dir> <out_dir>
Output: <out_dir>/{thinkon,thinkoff,efflow}-prompts.tsv in the real-gate prompts.tsv shape
        (`index<TAB>ids<TAB>ids`; column 3 is a placeholder — the spec instruments compare
        plain against spec, never against a golden).
"""
import os
import sys

from transformers import AutoTokenizer

src, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
tok = AutoTokenizer.from_pretrained(src)

# The four held-out tasks (CODE[-2], CODE[-1], REASONING[-2], REASONING[-1] in
# make-corpus-prompts.py), verbatim so the two packs stay aligned.
TASKS = [
    "Convert this callback-based Node function to async/await, preserving the error semantics exactly, and say what changes for a caller that used to pass a callback.",
    "Write a Python generator that reads a JSONL file lazily, validates each record against a pydantic model, and yields (line_number, error) for the invalid ones instead of raising.",
    "A cache has a 92% hit rate. A hit costs 0.2 ms, a miss costs 40 ms. What is the mean latency, and how much does raising the hit rate to 96% save in percentage terms?",
    "I have two servers. One does 120 tokens/s and costs 2.10/hour; the other does 79 tokens/s and costs 1.30/hour. Which is cheaper per million tokens, and by how much?",
]

SHAPES = {
    "thinkon": {},
    "thinkoff": {"enable_thinking": False},
    "efflow": {"reasoning_effort": "low"},
}

for name, ck in SHAPES.items():
    rows = []
    for text in TASKS:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
            **ck,
        )
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        rows.append(list(ids))
    path = os.path.join(out_dir, f"{name}-prompts.tsv")
    with open(path, "w") as f:
        for i, ids in enumerate(rows):
            csv_ids = ",".join(str(t) for t in ids)
            f.write(f"{i}\t{csv_ids}\t{csv_ids}\n")
    print(f"{name}: {len(rows)} prompts, tokens {[len(r) for r in rows]} -> {path}")
