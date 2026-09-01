#!/usr/bin/env python3
"""Teacher-forced DFlash2 acceptance scoring for GLM-5.3-Flash on real agent prompts.

WHAT RUNS WHERE (stated per the external-implementation law: reference code outside
serving, creating pinned evidence only):
  - TARGET path: the memra SERVING binary's greedy continuations (rollouts.json), plus
    per-position context features captured from the memra engine forward
    (MEMRA_TRACE_LAYER_ROWS, stream-mean of completed layer outputs at layers
    5,14,24,33,42, the exact aux-hidden definition the SGLang glm5_next DFlash2
    integration pins in its unit test).
  - DRAFTER: incoai/GLM-5.3-Flash-DFlash2 (revision dc77ff1c99eeb2df044ee3d4f0094eb0
    33fee410), run with the z-lab reference implementation (dflash/model.py, the code
    memra's own q38 DFlash2 port was parity-gated against). The drafter's noise
    embeddings and output head are the TARGET's own embed_tokens / lm_head, loaded
    unquantized from the artifact shards (both sit in the quantization ignore list).

THE LOOP mirrors dflash_generate cycle-for-cycle (block_size 8 = anchor + 7 drafts,
incremental drafter KV cache with crop-to-start, position ids over [new_lo, start+8)),
with one substitution: committed tokens come from the serving binary's greedy path
instead of a co-evolving transformers target. Greedy is the instrument; the acceptance
rule is the DFlash2 greedy rule: longest drafted prefix equal to the target's greedy
tokens, and the cycle then advances produced = accepted + 1 (the bonus token), exactly
as production verify would.
"""
import json
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/root/dflash")
from dflash.model import DFlash2DraftModel, _crop_to, _make_cache  # noqa: E402

from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

ART = "/root/models/glm53-nvfp4"
DRAFT = "/root/models/glm53-dflash2"
CAP = "/root/dfp2/cap"
LAYERS = [5, 14, 24, 33, 42]
DEV = "cuda:0"  # run under CUDA_VISIBLE_DEVICES to pick the physical card

torch.set_grad_enabled(False)

# --- target-side embed_tokens / lm_head, unquantized from the artifact shards ---
index = json.load(open(f"{ART}/model.safetensors.index.json"))["weight_map"]


def load_tensor(name):
    with safe_open(f"{ART}/{index[name]}", framework="pt") as f:
        return f.get_tensor(name)


EMBED_NAME = next(n for n in index if "embed_tokens.weight" in n)
HEAD_NAME = next(n for n in index if n.startswith("lm_head") and n.endswith(".weight"))
w_embed = load_tensor(EMBED_NAME).to(DEV, torch.bfloat16)
w_head = load_tensor(HEAD_NAME).to(DEV, torch.bfloat16)
print(f"embed {EMBED_NAME} {tuple(w_embed.shape)}  head {HEAD_NAME} {tuple(w_head.shape)}")

head = torch.nn.Linear(w_head.shape[1], w_head.shape[0], bias=False, dtype=torch.bfloat16, device=DEV)
head.weight.data.copy_(w_head)

# --- drafter ---
model = DFlash2DraftModel.from_pretrained(DRAFT, dtype=torch.bfloat16).to(DEV).eval()
block = model.block_size
mask_id = model.mask_token_id
assert model.target_layer_ids == LAYERS, (model.target_layer_ids, LAYERS)
print(f"drafter loaded: block={block} mask={mask_id} layers={model.target_layer_ids} "
      f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

# --- traffic-class spans: tool wire (tool_call blocks, {"...": JSON objects, fences) ---
def tool_spans(text):
    spans = []
    for m in re.finditer(r"<tool_call>", text):
        end = text.find("</tool_call>", m.start())
        spans.append((m.start(), len(text) if end == -1 else end + len("</tool_call>")))
    for m in re.finditer(r"```", text):
        end = text.find("```", m.end())
        if end != -1:
            spans.append((m.start(), end + 3))
    for m in re.finditer(r'\{"', text):
        depth, i = 0, m.start()
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        spans.append((m.start(), min(i + 1, len(text))))
    return spans


def classify(offset, spans):
    return "tool" if any(a <= offset < b for a, b in spans) else "prose"


tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
prompts = {p["name"]: p for p in json.load(open("/root/dfp2/scoring_prompts.json"))}
rollouts = json.load(open("/root/dfp2/rollouts.json"))

all_cycles = []
for r in rollouts:
    name = r["name"]
    p = prompts[name]
    S = p["ids"] + r["cont_ids"]
    P = p["n_ids"]
    N = len(S)

    feats = []
    for L in LAYERS:
        a = np.fromfile(f"{CAP}/{name}/layer{L}.f32", dtype="<f4")
        assert a.size == N * 4096, (name, L, a.size, N * 4096)
        feats.append(torch.from_numpy(a.reshape(N, 4096)))
    F_feat = torch.cat(feats, dim=-1).to(DEV, torch.bfloat16)  # [N, 5*4096]

    enc = tok.encode(r["text"], add_special_tokens=False)
    assert enc.ids == r["cont_ids"]
    offsets = enc.offsets
    spans = tool_spans(r["text"])

    S_t = torch.tensor(S, dtype=torch.long, device=DEV)
    pos_full = torch.arange(N + block, dtype=torch.long, device=DEV)

    cache = _make_cache(model.config)
    start, new_lo = P, 0
    cycles = []
    while N - start >= block:
        verify = block
        block_ids = torch.full((1, verify), mask_id, dtype=torch.long, device=DEV)
        block_ids[0, 0] = S[start]
        noise = torch.nn.functional.embedding(block_ids, w_embed)
        th = F_feat[new_lo:start].unsqueeze(0)
        pos = pos_full[new_lo:start + verify].unsqueeze(0)
        hidden = model(
            target_hidden=th,
            noise_embedding=noise,
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )[:, 1 - verify:, :]
        _crop_to(cache, start)
        draft_tokens, _, _ = model.propose(hidden, block_ids[:, 0], head, 0.0)
        draft = draft_tokens[0].tolist()
        truth = S[start + 1:start + verify]
        L_acc = 0
        for d, t in zip(draft, truth):
            if d != t:
                break
            L_acc += 1
        produced = L_acc + 1
        j = start + 1 - P  # first drafted position, continuation-relative
        cls = classify(offsets[j][0], spans) if 0 <= j < len(offsets) else "prose"
        cycles.append({
            "name": name, "start": start, "class": cls, "accepted": L_acc,
            "produced": produced, "hits": [int(d == t) for d, t in zip(draft, truth)],
        })
        new_lo = start
        start += produced

    n_tok = sum(1 for c in cycles if c["class"] == "tool")
    mean_acc = sum(c["accepted"] for c in cycles) / max(len(cycles), 1)
    print(f"{name}: cycles={len(cycles)} (tool={n_tok}) mean_accepted={mean_acc:.2f} "
          f"tokens_scored={start - P}")
    all_cycles.extend(cycles)
    del F_feat
    torch.cuda.empty_cache()

json.dump(all_cycles, open("/root/dfp2/cycles_dflash2.json", "w"))
print(f"total cycles: {len(all_cycles)}")
