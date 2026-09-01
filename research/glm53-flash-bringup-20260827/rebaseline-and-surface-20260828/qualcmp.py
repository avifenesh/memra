#!/usr/bin/env python3
"""Old-render (ChatML) vs new-render (native glm5 dialect) vendor-default sampled quality."""
import json

R = "/home/ubuntu/lane-rebaseline-20260828"
old = [json.loads(l) for l in open(f"{R}/04-old-render-sampled-quality.jsonl") if l.startswith("{")]
new = [json.loads(l) for l in open(f"{R}/12-new-render-quality-and-coldrestore.txt")
       if l.startswith("{") and '"NEWRENDER"' in l]

print("     |        OLD-RENDER (ChatML lookalike)   |        NEW-RENDER (native glm5)")
print(" idx | ptok ctok finish think cont  loopR loopC | ptok ctok finish think cont  loopR loopC")
print("-" * 96)
for o, n in zip(old, new):
    assert o["prompt_idx"] == n["prompt_idx"], (o["prompt_idx"], n["prompt_idx"])
    print(f" {o['prompt_idx']:>3} | {o['prompt_tokens']:>4} {o['completion_tokens']:>4} "
          f"{str(o['finish_reason'])[:6]:>6} {o['reasoning_tok']:>5} {o['content_tok']:>4} "
          f"{o['loop_score_reasoning']:>6} {o['loop_score_content']:>5} "
          f"| {n['prompt_tokens']:>4} {n['completion_tokens']:>4} "
          f"{str(n['finish_reason'])[:6]:>6} {n['reasoning_tok']:>5} {n['content_tok']:>4} "
          f"{n['loop_score_reasoning']:>6} {n['loop_score_content']:>5}")
print()
print(f"OLD render, prompts that produced ZERO answer text in 1024 tokens: "
      f"{sum(1 for r in old if r['content_chars'] == 0)}/{len(old)}")
print(f"NEW render, prompts that produced ZERO answer text in 1024 tokens: "
      f"{sum(1 for r in new if r['content_chars'] == 0)}/{len(new)}")
print(f"OLD render, median thinking tokens: "
      f"{sorted(r['reasoning_tok'] for r in old)[len(old)//2]}")
print(f"NEW render, median thinking tokens: "
      f"{sorted(r['reasoning_tok'] for r in new)[len(new)//2]}")
print()
print("NEW-render answer heads (the product shape, vendor-default sampled, effort pinned low):")
for n in new:
    flag = "  <-- DEGENERATE" if max(n["loop_score_content"], n["loop_score_reasoning"]) >= 0.15 else ""
    head = n["content_head"][:110] if n["content_chars"] else "(no answer text: " + n["reasoning_head"][:80] + ")"
    print(f"  p{n['prompt_idx']} loopC={n['loop_score_content']:<5} {head!r}{flag}")
print()
print("OLD-render answer heads, same prompts:")
for o in old:
    head = o["content_head"][:110] if o["content_chars"] else "(no answer text: " + o["reasoning_head"][:80] + ")"
    print(f"  p{o['prompt_idx']} loopC={o['loop_score_content']:<5} {head!r}")
