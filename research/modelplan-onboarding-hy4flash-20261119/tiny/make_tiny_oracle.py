"""Tiny-gate oracle for the hy_v4_flash pack: the pinned transformers program on memra's fixture.

Runs `HYV4FlashTextModel` from transformers PR #50903 at b865850d0532 (modeling, configuration,
conversion mapping, cache and masking files checked against their git blobs; no line edited) on
CPU in float64 over the checkpoint image `memra model verify tiny --against hy_v4_flash --out OUT`
writes (`OUT/tiny-checkpoint/`, and `OUT/tiny-checkpoint-<variant>/` per variant):

- `config.json`: the pack's tiny config, the artifact's config.json at hidden 64 (its
  `quantization_config` the artifact's ModelOpt map at that width);
- `model.safetensors`: the text program's tensors in the artifact's storage: BF16, F32, the I32
  lookup tables, FP8 128x128 blocks (`weight_scale_inv`, multiplied in), NVFP4 (U8 e2m1 pairs,
  element 2i in the low nibble, an F8_E4M3 `weight_scale` per 16 inputs, an F32
  `weight_scale_2` per matrix or per expert) and the Engram tables' FP8 rows with one F32
  `weight_scale` each;
- `fixture.json`: the token ids and where the prefill ends.

Loading: every tensor is dequantized exactly in float64 (`e2m1 * e4m3 * scale_2`, `e4m3 *
block scale`, `e4m3 * row scale`), renamed by the pinned `conversion_mapping` rules for
`hy_v4_flash`, and the separate gate and up expert banks are concatenated into the module's
`gate_up_proj` (gate first, as `_apply_gate` chunks it). The Engram multipliers are redrawn from
the loaded lookup table, as `_init_weights` does after loading.

Protocol, as `generate` runs it: the prefill `token_ids[:decode_from]` at positions 0.., one
`DynamicCache`, then one teacher-forced step per later token at its position, each step passing
`ngram_tokens` and `engram_mask` built by the pinned `build_engram_inputs` over the whole
history (`prepare_inputs_for_generation`). The mask drops the pad id and the image and video
placeholders (`HYV4FlashModel.build_engram_inputs`, run on a stand-in that carries the config's
`image_token_id` and `video_token_id`). Logits are `model.norm` output times `lm_head` in float64,
rounded once to f32: the TSV memra's tiny gate pins.

Process-level substitutions:

- float32 islands -> float64 (`torch.float32`, `torch.float` and `Tensor.float` in the modeling
  module), and `topk` under the CUDA rule (a stable descending sort: the values above the k-th,
  then ties in ascending index), as in indexpool/gen_indexpool.py.
- The four Engram norms (`nn.RMSNorm(eps=None)`) get `eps = 2^-23`, the op-math epsilon the
  program adds on BF16 activations (engram/), not the F64 epsilon ATen would add here.
- The constants the program computes with in F32 (`rms_norm_eps` 1e-5, `hc_eps` 1e-6,
  `routed_scaling_factor` 3.998, `swiglu_limit`, `hc_magnitude`) are fed as their f32 values:
  a float64 run would otherwise read the decimal literals, which no F32/BF16 run of the program
  sees. `--raw-constants` keeps the literals (a measurement of that difference, not the oracle).
- The pinned `_fp8_round_trip` stays on: on rows that are already `q * scale` with the largest
  code 448 it is the identity in float64, which the run asserts.

Recorded beside the TSV (`--trace`): per forward call, each indexed layer's recalled groups per
query (the top-k indices plus the query's own group, sorted) and each Engram layer's table rows
per token (hash ids plus the head offsets), as FNV-1a 64 digests of their text, the same text
memra's gate hashes.

usage: OMP_NUM_THREADS=1 venv/bin/python -I make_tiny_oracle.py CHECKPOINT_DIR OUT_TSV
           [--trace OUT_JSON] [--raw-constants]
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import types

import numpy as np
import torch

import transformers
import transformers.cache_utils as cache_mod
import transformers.conversion_mapping as conv_mod
import transformers.masking_utils as mask_mod
import transformers.models.hy_v4_flash.configuration_hy_v4_flash as conf_mod
import transformers.models.hy_v4_flash.modeling_hy_v4_flash as model_mod


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


PINS = {
    model_mod: "2d796211a4f867bd4c46d4a952cfee0375929f72",
    conf_mod: "16cb099b7991c190c4d5d85dd30b78e0b4c62770",
    conv_mod: "bc06957034df1155cbdfe626f828cfbc63c9011d",
    cache_mod: "a1ccfd37cd07c34dddb0a4e3ce0c3c196a697cb2",
    mask_mod: "2831f1fee95f2106d374e22e886844ca5cf9d471",
}
for module, pin in PINS.items():
    assert blob(module.__file__) == pin, f"{module.__name__} is not the pinned revision"

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("out")
ap.add_argument("--trace", default="")
ap.add_argument("--raw-constants", action="store_true")
a = ap.parse_args()


def cuda_topk(self, k, dim=-1, largest=True, sorted=True):
    assert largest, "only largest-first selection is used"
    values, indices = torch.sort(self, dim=dim, descending=True, stable=True)
    values, indices = values.narrow(dim, 0, k), indices.narrow(dim, 0, k)
    if RECORD["indexer"] is not None:
        RECORD["indexer"].append(indices)
    return torch.return_types.topk((values, indices))


RECORD = {"indexer": None}
proxy = types.ModuleType("torch_f64")
proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
proxy.float32 = torch.float64
proxy.float = torch.float64
proxy.topk = lambda x, k, dim=-1, largest=True, sorted=True: cuda_topk(x, k, dim, largest, sorted)
model_mod.torch = proxy
torch.Tensor.float = lambda self: self.to(torch.float64)
torch.Tensor.topk = cuda_topk
torch.set_default_dtype(torch.float64)

# ---- the checkpoint image ----------------------------------------------------------------------
top = json.load(open(os.path.join(a.checkpoint, "config.json")))
fixture = json.load(open(os.path.join(a.checkpoint, "fixture.json")))
raw = open(os.path.join(a.checkpoint, "model.safetensors"), "rb").read()
n = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8 : 8 + n])
body = raw[8 + n :]
DT = {"BF16": torch.bfloat16, "F32": torch.float32, "I32": torch.int32, "F8_E4M3": torch.float8_e4m3fn,
      "U8": torch.uint8}


def stored(name):
    h = header[name]
    s, e = h["data_offsets"]
    return torch.frombuffer(bytearray(body[s:e]), dtype=DT[h["dtype"]]).reshape(h["shape"])


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
tensors = {}
for name in sorted(header):
    if name.endswith((".weight_scale", ".weight_scale_inv", ".weight_scale_2")):
        continue
    dtype = header[name]["dtype"]
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    value = stored(name)
    if dtype == "U8":
        lo, hi = (value & 0x0F).long(), (value >> 4).long()
        codes = torch.stack([E2M1[lo], E2M1[hi]], dim=-1).flatten(-2)
        scale = stored(stem + ".weight_scale").to(torch.float64).repeat_interleave(16, dim=-1)
        scale_2 = stored(stem + ".weight_scale_2").to(torch.float64)
        value = codes * scale * scale_2.reshape(*scale_2.shape, 1, 1)
    elif dtype == "F8_E4M3" and stem + ".weight_scale_inv" in header:
        grid = stored(stem + ".weight_scale_inv").to(torch.float64)
        full = grid.repeat_interleave(128, 0).repeat_interleave(128, 1)[: value.shape[0], : value.shape[1]]
        value = value.to(torch.float64) * full
    elif dtype == "F8_E4M3":
        value = value.to(torch.float64) * stored(stem + ".weight_scale").to(torch.float64)[:, None]
    elif dtype == "I32":
        value = value.clone()
    else:
        value = value.to(torch.float64)
    tensors[name] = value
lm_head = tensors.pop("lm_head.weight")

# The family's own text renames: the mapping appends the HY4-VL tower and placement rules after
# them, which a text model does not read.
mapping = conv_mod._build_checkpoint_conversion_mapping()
full, tower = mapping["hy_v4_flash"], mapping["hy_v4_vl"]
rules = full[: len(full) - len(tower)]
assert [repr(r) for r in full[len(rules):]] == [repr(r) for r in tower]
renamed = {}
for name, value in tensors.items():
    new = name
    for rule in rules:
        for src, dst in zip(rule.source_patterns, rule.target_patterns):
            new = re.sub(src, dst, new)
    assert new.startswith("model."), new
    renamed[new[len("model."):]] = value
for key in [k for k in renamed if k.endswith("mlp.experts.gate_proj.weight")]:
    stem = key[: -len("gate_proj.weight")]
    gate, up = renamed.pop(key), renamed.pop(stem + "up_proj.weight")
    renamed[stem + "gate_up_proj"] = torch.cat([gate, up], dim=-2)
    renamed[stem + "down_proj"] = renamed.pop(stem + "down_proj.weight")

# ---- the pinned model ----------------------------------------------------------------------------
text = {k: v for k, v in top["text_config"].items() if k != "model_type"}
F32_CONSTANTS = ["rms_norm_eps", "hc_eps", "routed_scaling_factor", "swiglu_limit", "hc_magnitude"]
if not a.raw_constants:
    for key in F32_CONSTANTS:
        text[key] = float(np.float32(text[key]))
cfg = conf_mod.HYV4FlashTextConfig(**text)
cfg._attn_implementation = "eager"
cfg._experts_implementation = "eager"
model = model_mod.HYV4FlashTextModel(cfg).to(torch.float64).eval()
missing, unexpected = model.load_state_dict(renamed, strict=False)
assert not unexpected, unexpected
assert not missing, missing
OP_MATH_EPS = 2.0**-23
engrams = {}
with torch.no_grad():
    for il, layer in enumerate(model.layers):
        engram = layer.engram_layer
        if engram is None:
            continue
        engram.hash_multipliers.copy_(engram.draw_hash_multipliers(int(engram.lookup_table.max()) + 1))
        for norm in [*engram.lookup_per_head_norms, *engram.norm_key, *engram.norm_query, engram.unified_rmsnorm]:
            assert norm.eps is None
            norm.eps = OP_MATH_EPS
        engrams[il] = engram
assert sorted(il + 1 for il in engrams) == sorted(cfg.engram_layer_ids), "Engram sits at id - 1"

# ---- recording ---------------------------------------------------------------------------------------
CALL = {}
indexer_forward = model_mod.HYV4FlashCompressedIndexer.forward


def recording_indexer(self, hidden_states, q_resid, attention_mask, position_ids, indexer_position_ids, cache,
                      kv_length):
    RECORD["indexer"] = []
    try:
        out = indexer_forward(self, hidden_states, q_resid, attention_mask, position_ids, indexer_position_ids,
                              cache, kv_length)
    finally:
        picks, RECORD["indexer"] = RECORD["indexer"], None
    if picks:
        assert len(picks) == 1
        seq = hidden_states.shape[1]
        slots = torch.arange(seq) + (cache.get_seq_length(self.layer_idx) - seq)
        own = (slots // self.index_kpool)[None, :, None]
        groups = torch.cat([picks[0], own], dim=-1)[0]
        CALL["recalled"].append((self.layer_idx, [sorted(g.tolist()) for g in groups]))
    return out


model_mod.HYV4FlashCompressedIndexer.forward = recording_indexer
hash_ngrams = model_mod.HYV4FlashEngramLayer.hash_ngrams


def recording_hash(self, ngram_tokens):
    out = hash_ngrams(self, ngram_tokens)
    il = self.layer_id - 1
    offsets = self.multi_head_embedding.offsets
    CALL["engram"].append((il, (out + offsets)[0].reshape(-1).tolist()))
    return out


model_mod.HYV4FlashEngramLayer.hash_ngrams = recording_hash
round_trip = model_mod._fp8_round_trip


def checked_round_trip(x):
    out = round_trip(x)
    assert torch.equal(out, x), "the stored Engram rows are fixed points of the FP8 round trip"
    return out


model_mod._fp8_round_trip = checked_round_trip

# ---- the request -------------------------------------------------------------------------------------
ids = torch.tensor([fixture["token_ids"]])
prefill = fixture["decode_from"]
mask_stub = types.SimpleNamespace(
    config=types.SimpleNamespace(image_token_id=top["image_token_id"], video_token_id=top["video_token_id"]),
    language_model=model,
)
cache = cache_mod.DynamicCache(config=cfg)
rows, trace = [], []
with torch.no_grad():
    for start, end in [(0, prefill)] + [(t, t + 1) for t in range(prefill, ids.shape[1])]:
        CALL.clear()
        CALL.update({"start": start, "tokens": end - start, "recalled": [], "engram": []})
        history = ids[:, :end]
        ngram, emask = model_mod.HYV4FlashModel.build_engram_inputs(mask_stub, history, None)
        hidden = model(
            input_ids=ids[:, start:end],
            position_ids=torch.arange(start, end)[None],
            cache=cache,
            use_cache=True,
            ngram_tokens=ngram[:, start:end],
            engram_mask=emask[:, start:end],
        ).last_hidden_state
        rows.append(hidden[0] @ lm_head.T)
        trace.append(dict(CALL))
logits = torch.cat(rows).to(torch.float32)


def fnv1a(text):
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


with open(a.out, "w") as f:
    f.write("stream\tposition\ttoken\tlogit_f32_bits\n")
    for p, row in enumerate(logits):
        for t, v in enumerate(row.tolist()):
            f.write(f"main\t{p}\t{t}\t{struct.unpack('<I', struct.pack('<f', v))[0]:08x}\n")
digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()

calls = []
for call in trace:
    calls.append({
        "start": call["start"],
        "tokens": call["tokens"],
        "recalled": {str(il): fnv1a("\n".join(",".join(map(str, g)) for g in groups))
                     for il, groups in call["recalled"]},
        "engram_rows": {str(il): fnv1a(",".join(map(str, r))) for il, r in call["engram"]},
    })
if a.trace:
    json.dump({
        "transformers_modeling_blob": PINS[model_mod],
        "dtype": "float64",
        "raw_constants": a.raw_constants,
        "token_ids": fixture["token_ids"],
        "decode_from": prefill,
        "tsv_sha256": digest,
        "calls": calls,
    }, open(a.trace, "w"), indent=1)
print(json.dumps({
    "tokens": ids.shape[1],
    "prefill": prefill,
    "decode_steps": ids.shape[1] - prefill,
    "indexer_calls": sum(len(c["recalled"]) for c in trace),
    "engram_calls": sum(len(c["engram"]) for c in trace),
    "raw_constants": a.raw_constants,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "out": os.path.basename(a.out),
    "sha256": digest,
}))
