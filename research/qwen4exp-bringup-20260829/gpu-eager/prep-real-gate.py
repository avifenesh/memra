#!/usr/bin/env python3
"""Dump the transformers goldens into the neutral format qwen4exp_real_gate reads.

From <goldens_dir> (hidden-goldens.pt + greedy-goldens.json, made by make-goldens.py on
the pinned BF16) into <out_dir>:

  manifest.tsv     name<TAB>rows<TAB>cols<TAB>file  (layer0..N, exit_mixer, logits)
  *.bin            raw little-endian f32, row-major, batch dim squeezed
  input-ids.txt    the hidden-goldens probe prompt token ids (never re-tokenized)
  prompts.tsv      idx<TAB>prompt-ids-csv<TAB>golden-continuation-csv (greedy gate)
  prompt-texts.json  idx -> prompt text (human reference; the Rust gate reads only ids)

Prompt ids come from the SAME tokenizer call make-goldens.py used (tok(text).input_ids).

Usage: prep-real-gate.py <bf16_model_dir> <goldens_dir> <out_dir>
"""
import json
import os
import sys

import torch

src, gold, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)

cap = torch.load(os.path.join(gold, "hidden-goldens.pt"), map_location="cpu", weights_only=True)
ids = cap.pop("input_ids").flatten().tolist()
with open(f"{out}/input-ids.txt", "w") as f:
    f.write(" ".join(str(i) for i in ids))
rows = []
for name, t in sorted(cap.items()):
    t = t.float()
    if t.dim() == 3:
        assert t.shape[0] == 1, (name, t.shape)
        t = t.squeeze(0)
    assert t.dim() == 2, (name, t.shape)
    fn = f"{name}.bin"
    t.contiguous().numpy().astype("<f4").tofile(f"{out}/{fn}")
    rows.append(f"{name}\t{t.shape[0]}\t{t.shape[1]}\t{fn}")
with open(f"{out}/manifest.tsv", "w") as f:
    f.write("\n".join(rows) + "\n")

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(src)
greedy = json.load(open(os.path.join(gold, "greedy-goldens.json")))
lines, texts = [], {}
for i, (text, golden) in enumerate(greedy.items()):
    pids = tok(text)["input_ids"]
    lines.append(f"{i}\t{','.join(map(str, pids))}\t{','.join(map(str, golden))}")
    texts[str(i)] = text
with open(f"{out}/prompts.tsv", "w") as f:
    f.write("\n".join(lines) + "\n")
json.dump(texts, open(f"{out}/prompt-texts.json", "w"), indent=1)
print(f"dumped {len(rows)} records, probe T={len(ids)}, {len(lines)} prompts -> {out}")
