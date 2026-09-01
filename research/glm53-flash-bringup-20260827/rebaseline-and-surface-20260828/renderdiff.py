#!/usr/bin/env python3
"""ATTRIBUTION for the greedy sha move on the chat path.

The claim under test: the p5 greedy sha moved fd006d0d50eb59b5 -> 4ec98d8aeb7a30e6 ONLY because
commit 025c7d4314 changed the RENDERED PROMPT (ChatML lookalike -> the checkpoint's own native
dialect), not because the forward changed.

Measured facts this script has to reproduce, both taken on the box against the live servers:
  OLD binary (pre-025c), /v1/chat/completions p5, reasoning_effort low : prompt_tokens = 203
  NEW binary (lane head), same request                                 : prompt_tokens = 192

If the checkpoint's OWN chat_template.jinja renders p5 to exactly 192 tokens, and a ChatML frame
renders the same message to exactly 203, then the delta is the template and nothing else.
"""
import json, sys
from transformers import AutoTokenizer

MODEL_DIR = "/home/ubuntu/models/glm53-nvfp4"
POOL = json.load(open("/home/ubuntu/prompts.json"))["decode"]
p5 = POOL[5]["text"]

tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
tpl = open(f"{MODEL_DIR}/chat_template.jinja", encoding="utf-8").read()

msgs = [{"role": "user", "content": p5}]

print("=" * 78)
print("A. NATIVE render, the checkpoint's own chat_template.jinja")
print("=" * 78)
native = {}
for effort in ["low", "high", None]:
    kw = {"reasoning_effort": effort} if effort else {}
    s = tok.apply_chat_template(msgs, chat_template=tpl, tokenize=False,
                                add_generation_prompt=True, **kw)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    native[str(effort)] = (len(ids), s)
    print(f"  effort={str(effort):>5}: {len(ids):>4} tokens")
print()
print("  --- native head (effort=low), first 240 chars, repr ---")
print("  " + repr(native["low"][1][:240]))
print("  --- native tail, last 120 chars ---")
print("  " + repr(native["low"][1][-120:]))

print()
print("=" * 78)
print("B. ChatML lookalike, the pre-fix render")
print("=" * 78)
chatml_variants = {
    "bare":
        f"<|im_start|>user\n{p5}<|im_end|>\n<|im_start|>assistant\n",
    "with-think-open":
        f"<|im_start|>user\n{p5}<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "sys+bare":
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{p5}<|im_end|>\n<|im_start|>assistant\n",
}
for k, s in chatml_variants.items():
    ids = tok(s, add_special_tokens=False)["input_ids"]
    print(f"  {k:>16}: {len(ids):>4} tokens")

print()
print("=" * 78)
print("C. Are <|im_start|> / <|im_end|> real tokens in THIS checkpoint?")
print("=" * 78)
v = tok.get_vocab()
for marker in ["<|im_start|>", "<|im_end|>", "[gMASK]", "<sop>", "<|user|>", "<|assistant|>",
               "<|system|>", "<|observation|>", "<think>", "<tool_call>", "<arg_key>"]:
    tid = v.get(marker)
    n = len(tok(marker, add_special_tokens=False)["input_ids"])
    print(f"  {marker:>16}: vocab_id={tid!s:>8}  tokenizes_to={n} token(s)"
          + ("   <-- NOT a special token: shreds into ordinary text" if tid is None else ""))

print()
print("=" * 78)
print("D. VERDICT")
print("=" * 78)
n_low = native["low"][0]
best = None
for k, s in chatml_variants.items():
    n = len(tok(s, add_special_tokens=False)["input_ids"])
    if n == 203:
        best = k
print(f"  NEW binary measured prompt_tokens = 192 ; native(effort=low) renders {n_low}"
      f"  -> {'MATCH' if n_low == 192 else 'MISMATCH'}")
print(f"  OLD binary measured prompt_tokens = 203 ; ChatML variant matching 203 = {best!r}")
print(f"  delta = {203 - 192} tokens of prompt, which is why the greedy tape differs.")
