#!/usr/bin/env python3
"""Golden generation for qwen4_exp parity gates (phase 4).

Runs on the mint box AFTER the NVFP4 mint frees the GPUs, against the pinned BF16
checkpoint (rev de4b8e4d) with transformers main. Produces, under <outdir>:

  template-goldens.json   chat-template renders -> token ids (plain, tools, thinking
                          kwargs matrix, multimodal marker case)
  greedy-goldens.json     greedy continuations (64 new tokens) for fixed prompts,
                          with per-step argmax token ids
  hidden-goldens.pt       per-layer wide-stream hidden states + exit mixer output for
                          one short prompt (layer-by-layer parity anchors)

Usage: python3 make-goldens.py <model_dir> <outdir>
"""
import json, sys, time
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

src, out = sys.argv[1], sys.argv[2]
import os
os.makedirs(out, exist_ok=True)
t0 = time.time()

tok = AutoTokenizer.from_pretrained(src)

# --- template goldens (no model needed) ---
msgs = [{"role": "user", "content": "What is 15% of 240?"}]
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get the current weather for a location",
    "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                   "required": ["location"]}}}]
cases = {
    "plain": dict(),
    "thinking_off": dict(chat_template_kwargs={"enable_thinking": False}),
    "preserve_off": dict(chat_template_kwargs={"preserve_thinking": False}),
    "effort_low": dict(chat_template_kwargs={"reasoning_effort": "low"}),
    "with_tools": dict(tools=tools),
    "multiturn": dict(messages_override=[
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"}]),
}
tpl = {}
for name, kw in cases.items():
    m = kw.pop("messages_override", msgs)
    ck = kw.pop("chat_template_kwargs", {})
    try:
        ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True, **kw, **ck)
        if hasattr(ids, "keys"):  # BatchEncoding on some kwarg paths
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        txt = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, **kw, **ck)
        tpl[name] = {"ids": list(ids), "text": txt}
    except Exception as e:  # bank the failure shape too — a gate needs to know
        tpl[name] = {"error": repr(e)}
json.dump(tpl, open(f"{out}/template-goldens.json", "w"), indent=1)
print(f"[{time.time()-t0:.0f}s] template goldens: {list(tpl)}", flush=True)

# --- model goldens ---
# Same load strategy as the mint (auto device-map dies on the fused ~95 GiB ngram param):
# hand-built map, 14 layers per GPU, rest in RAM; ngram gather executes natively on CPU.
from transformers import AutoConfig
from accelerate import init_empty_weights
from accelerate.hooks import remove_hook_from_module

cfg_hf = AutoConfig.from_pretrained(src)
with init_empty_weights():
    shell = AutoModelForImageTextToText.from_config(cfg_hf)
shell.tie_weights()
GPU0_LAYERS, GPU1_LAYERS = range(0, 14), range(14, 28)
dm = {
    "model.language_model.embed_tokens": 0,
    "model.language_model.hyper_connection_mixer": 1,
    "model.visual": "cpu",
    "mtp": "cpu",
    "lm_head": 1,
}
layers = dict(shell.get_submodule("model.language_model.layers").named_children())
for i_s, layer in layers.items():
    i = int(i_s)
    dev = 0 if i in GPU0_LAYERS else (1 if i in GPU1_LAYERS else "cpu")
    kids = dict(layer.named_children())
    if "ple" in kids:
        for kid in kids:
            dm[f"model.language_model.layers.{i_s}.{kid}"] = "cpu" if kid == "ple" else dev
        for pn, _ in layer.named_parameters(recurse=False):
            dm[f"model.language_model.layers.{i_s}.{pn}"] = dev
    else:
        dm[f"model.language_model.layers.{i_s}"] = dev
for name, _ in shell.named_children():
    if name not in ("model", "lm_head", "mtp"):
        dm.setdefault(name, "cpu")
for name, _ in shell.get_submodule("model").named_children():
    full = f"model.{name}"
    if not any(k == full or k.startswith(full + ".") for k in dm):
        dm[full] = "cpu"
for name, _ in shell.get_submodule("model.language_model").named_children():
    full = f"model.language_model.{name}"
    if name != "layers" and not any(k == full or k.startswith(full + ".") for k in dm):
        dm[full] = "cpu"

model = AutoModelForImageTextToText.from_pretrained(
    src, dtype=torch.bfloat16, device_map=dm, low_cpu_mem_usage=True,
)
model.eval()
ng_name = next(n for n, m in model.named_modules()
               if n.endswith("ngram_embedding") and isinstance(m, torch.nn.Embedding))
ng = model.get_submodule(ng_name)
remove_hook_from_module(ng, recurse=True)
ng.to("cpu")
_orig_fwd = ng.forward
ng.forward = lambda ids: _orig_fwd(ids.to("cpu")).to(ids.device)
print(f"[{time.time()-t0:.0f}s] model loaded (explicit map, ngram on cpu)", flush=True)

PROMPTS = [
    "Write a Python function to merge two sorted linked lists.",
    "The capital of Australia is",
    "def fib(n):\n    ",
    "Translate to French: the weather is nice today.",
]
greedy = {}
for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to("cuda:0")
    with torch.no_grad():
        seq = model.generate(ids, max_new_tokens=64, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    greedy[p] = seq[0, ids.shape[1]:].tolist()
    print(f"[{time.time()-t0:.0f}s] greedy: {p[:40]!r}", flush=True)
json.dump(greedy, open(f"{out}/greedy-goldens.json", "w"), indent=1)

# --- per-layer hidden captures (wide stream after each decoder layer) ---
probe = "The quick brown fox jumps over the lazy dog."
ids = tok(probe, return_tensors="pt").input_ids.to("cuda:0")
captures = {}
hooks = []
lm = model.model.language_model
for i, layer in enumerate(lm.layers):
    hooks.append(layer.register_forward_hook(
        lambda mod, a, o, i=i: captures.__setitem__(f"layer{i}", (o[0] if isinstance(o, tuple) else o).detach().float().cpu())))
hooks.append(lm.hyper_connection_mixer.register_forward_hook(
    lambda mod, a, o: captures.__setitem__("exit_mixer", (o[0] if isinstance(o, tuple) else o).detach().float().cpu())))
with torch.no_grad():
    logits = model(ids).logits.detach().float().cpu()
for h in hooks:
    h.remove()
captures["logits"] = logits
captures["input_ids"] = ids.cpu()
torch.save(captures, f"{out}/hidden-goldens.pt")
print(f"[{time.time()-t0:.0f}s] hidden goldens: {len(captures)} tensors -> {out}", flush=True)
