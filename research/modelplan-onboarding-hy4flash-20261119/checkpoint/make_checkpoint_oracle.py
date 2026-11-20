"""CheckpointParity oracle for the hy_v4_flash pack: the pinned transformers program on the artifact.

Runs `HYV4FlashTextModel` from transformers PR #50903 at b865850d0532 (modeling, configuration,
conversion mapping, cache and masking files checked against their git blobs; no line edited) on
CPU over the artifact's own shards (Avifenesh/Hy4-flash-NVFP4@f9eac06cf022), one request at a
time, as `generate` runs it: a prefill of `token_ids[:decode_from]`, one `DynamicCache`, then one
teacher-forced step per later token, each step passing `ngram_tokens` and `engram_mask` from the
pinned `build_engram_inputs` over the whole history (the mask drops the pad id and the image and
video placeholders).

Three backends of the same program (`--dtype`):

- `f64`, the oracle: every tensor dequantized exactly in float64 (`e2m1 * e4m3 * weight_scale_2`
  per expert, `e4m3 * weight_scale_inv` per 128x128 block), float32 islands mapped to float64,
  the constants the program computes with in F32 fed as their f32 values, the Engram norms at
  `eps = 2^-23`, as in the tiny oracle (`../tiny/make_tiny_oracle.py`);
- `f32`, the second backend the native class's band is set from: the program at
  `dtype=float32`, every parameter the exactly dequantized weight rounded once to F32, F32
  activations;
- `bf16`, a diagnostic: the program at the dtype `from_pretrained` gives it, BF16 parameters and
  activations, the `_keep_in_fp32_modules_strict` modules and the head in F32, the exactly
  dequantized weights rounded once to BF16;
- `bf16a`, the backend the GPU route's class (`hy-v4-flash-bf16-activations`, step 8) is banded
  from: the `f32` backend with the input of every linear layer the program runs at its
  activation dtype rounded to BF16, as a W4A16 or BF16 kernel takes it (every `nn.Linear` outside
  `_keep_in_fp32_modules_strict` and the routed experts' two `F.linear`s); the weights stay the
  exactly dequantized values in F32, and the program's F32 islands (router, hyper-connections,
  indexer pooling, head) stay F32;
- `bf16x2a`: as `bf16a`, the input carried as two BF16 terms instead (`hi = bf16(x)`,
  `lo = bf16(x - hi)`, the GEMM reads `hi + lo`: 16 significant bits), the representation the GPU
  route's GEMMs take.

Both read the Engram tables as stored: a row is `f32(e4m3 * row scale)`, the pinned
`_fp8_round_trip` runs on it in F32 (where it is the identity, asserted) and the row then meets
the activations' dtype. A BF16 model would otherwise re-quantize every row with a BF16 scale
(step 5), an artefact of loading the table into BF16, not the program the artifact stores.
Ids past the lookup table read the pad id's entry (pinned `hash_ngrams`). The top-k of the
indexer and the router uses the CUDA tie rule (values above the k-th, then ties in ascending
index), as on every memra surface (step 4).

Memory: the trunk is resident (about 56 GB in float64); the routed experts are dequantized per
expert when the eager experts loop reaches them, and the Engram rows are gathered from the
memory-mapped shards, so the 334 GB a dequantized model would need never exists.

Writes the `memra-checkpoint-tape-v2` directory memra's gate reads (`tape.json`, `logits.f32`
with one f32 row of the vocabulary per position, `indexer_sel.i32` and `indexer_logits.f32` with
every indexed layer's recalled groups and group logits per forward call, `router_sel.i32` and
`router_scores.f32` with every MoE layer's selected experts and selection scores (sigmoid plus
the correction bias) per call, and per call each Engram layer's table rows as an FNV-1a 64
digest), and with `--dump-layers` each decoder layer's output streams of the first call (f32)
for layer-by-layer diagnosis.

The program has two discrete surfaces, the indexer's group pick and the router's expert pick.
A near tie there flips between two correct backends and the flip then moves every later
position (the f32 backend flips a router pick at a gap of 5.4e-7 on the `long` request).
`--force-selections TAPE` takes every pick from the f64 oracle's tape (the router's weights
follow the forced experts through the pinned lines, the indexer's groups are its forced top-k;
the tape still records the backend's own picks), so the f32 backend measures the program's numeric spread under the same selections: the band.

usage: venv/bin/python -I make_checkpoint_oracle.py CKPT_DIR REQUEST_JSON OUT_DIR
           [--dtype f64|f32|bf16] [--force-selections TAPE_DIR] [--dump-layers DIR] [--layers N]
REQUEST_JSON: {"name": ..., "token_ids": [...], "decode_from": N}
"""

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

F_REAL = F

import transformers
import transformers.cache_utils as cache_mod
import transformers.conversion_mapping as conv_mod
import transformers.masking_utils as mask_mod
import transformers.models.hy_v4_flash.configuration_hy_v4_flash as conf_mod
import transformers.models.hy_v4_flash.modeling_hy_v4_flash as model_mod

REAL_F32 = torch.float32


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
ap.add_argument("request")
ap.add_argument("out")
ap.add_argument("--dtype", choices=["f64", "f32", "bf16", "bf16a", "bf16x2a"], default="f64")
ap.add_argument("--dump-layers", default="")
ap.add_argument("--layers", type=int, default=0, help="diagnostic: run only the first N decoder layers")
ap.add_argument("--force-selections", default="",
                help="band backend: take every router and indexer pick from this tape (the f64 oracle's)")
a = ap.parse_args()
F64 = a.dtype == "f64"
ACT = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16, "bf16a": torch.float32,
       "bf16x2a": torch.float32}[a.dtype]
A16 = a.dtype in ("bf16a", "bf16x2a")
T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:8.1f}s]", *args, file=sys.stderr, flush=True)


# ---- dtype program --------------------------------------------------------------------------------
RECORD = {"indexer": None}


def cuda_topk(self, k, dim=-1, largest=True, sorted=True):
    assert largest, "only largest-first selection is used"
    values, indices = torch.sort(self, dim=dim, descending=True, stable=True)
    values, indices = values.narrow(dim, 0, k), indices.narrow(dim, 0, k)
    if RECORD["indexer"] is not None:
        RECORD["indexer"].append((indices, self))
        if RECORD.get("force_indexer") is not None:
            indices = RECORD["force_indexer"]
            assert indices.shape == values.shape, "the forced picks are not this call's query geometry"
            values = torch.gather(self, dim, indices)
            FORCED["used"].add(RECORD["force_key"])
    return torch.return_types.topk((values, indices))


proxy = types.ModuleType("torch_program")
proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
proxy.topk = lambda x, k, dim=-1, largest=True, sorted=True: cuda_topk(x, k, dim, largest, sorted)
if F64:
    proxy.float32 = torch.float64
    proxy.float = torch.float64
    torch.Tensor.float = lambda self: self.to(torch.float64)
model_mod.torch = proxy
torch.Tensor.topk = cuda_topk

# ---- the artifact ----------------------------------------------------------------------------------
top = json.load(open(os.path.join(a.checkpoint, "config.json")))
index = json.load(open(os.path.join(a.checkpoint, "model.safetensors.index.json")))["weight_map"]
DT = {"BF16": (np.uint16, torch.bfloat16), "F32": (np.float32, torch.float32), "I32": (np.int32, torch.int32),
      "F8_E4M3": (np.uint8, torch.float8_e4m3fn), "U8": (np.uint8, torch.uint8)}
SHARDS = {}


def shard(file):
    if file not in SHARDS:
        path = os.path.join(a.checkpoint, file)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        SHARDS[file] = (np.memmap(path, dtype=np.uint8, mode="r"), 8 + n, header)
    return SHARDS[file]


def stored_np(name):
    """The tensor's bytes as stored, a NumPy view of the memory-mapped shard (no copy)."""
    mm, base, header = shard(index[name])
    h = header[name]
    s, e = h["data_offsets"]
    return mm[base + s : base + e].view(DT[h["dtype"]][0]).reshape(h["shape"])


def stored(name):
    """The tensor as stored, a torch view of the memory-mapped shard (no copy)."""
    t_dtype = DT[header_dtype(name)][1]
    t = torch.from_numpy(stored_np(name))
    return t.view(t_dtype) if t_dtype in (torch.bfloat16, torch.float8_e4m3fn) else t


def header_dtype(name):
    return shard(index[name])[2][name]["dtype"]


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


def nvfp4(stem, expert):
    """One expert's matrix, dequantized exactly in float64: e2m1 * e4m3 * weight_scale_2."""
    packed = stored(stem + ".weight")[expert].to(torch.int64)
    codes = torch.stack([E2M1[packed & 0x0F], E2M1[packed >> 4]], dim=-1).flatten(-2)
    scale = stored(stem + ".weight_scale")[expert].to(torch.float64).repeat_interleave(16, dim=-1)
    scale_2 = stored(stem + ".weight_scale_2")[expert].to(torch.float64)
    return codes * scale * scale_2


def dense(name):
    """A non-expert tensor dequantized exactly in float64 (FP8 blocks with weight_scale_inv)."""
    stem = name[: -len(".weight")] if name.endswith(".weight") else name
    dtype = header_dtype(name)
    value = stored(name)
    if dtype == "F8_E4M3":
        grid = stored(stem + ".weight_scale_inv").to(torch.float64)
        full = grid.repeat_interleave(128, 0).repeat_interleave(128, 1)[: value.shape[0], : value.shape[1]]
        return value.to(torch.float64) * full
    if dtype == "I32":
        return value.clone()
    assert dtype in ("BF16", "F32"), (name, dtype)
    return value.to(torch.float64)


# The family's own text renames: the mapping appends the HY4-VL tower and placement rules after
# them, which a text model does not read.
mapping = conv_mod._build_checkpoint_conversion_mapping()
full_rules, tower = mapping["hy_v4_flash"], mapping["hy_v4_vl"]
rules = full_rules[: len(full_rules) - len(tower)]
assert [repr(r) for r in full_rules[len(rules):]] == [repr(r) for r in tower]


def rename(name):
    new = name
    for rule in rules:
        for src, dst in zip(rule.source_patterns, rule.target_patterns):
            new = re.sub(src, dst, new)
    assert new.startswith("model."), new
    return new[len("model."):]


# what `HYV4FlashForConditionalGeneration` keeps in F32: the base list plus the head
KEEP_F32 = model_mod.HYV4FlashForConditionalGeneration._keep_in_fp32_modules_strict
assert "lm_head" in KEEP_F32


def program_dtype(module_name, value):
    """The dtype `from_pretrained(dtype=bfloat16)` gives a parameter, or float64 for the oracle."""
    if value.dtype == torch.int32:
        return value
    if F64:
        return value.to(torch.float64)
    if a.dtype in ("f32", "bf16a", "bf16x2a") or any(k in module_name for k in KEEP_F32):
        return value.to(REAL_F32)
    return value.to(torch.bfloat16)


# ---- the pinned model ------------------------------------------------------------------------------
text = {k: v for k, v in top["text_config"].items() if k != "model_type"}
if F64:
    for key in ["rms_norm_eps", "hc_eps", "routed_scaling_factor", "swiglu_limit", "hc_magnitude"]:
        text[key] = float(np.float32(text[key]))
if a.layers:
    text["num_hidden_layers"] = a.layers
    for key in ["layer_types", "mlp_layer_types", "swa_pattern", "indexer_types"]:
        if isinstance(text.get(key), list):
            text[key] = text[key][: a.layers]
    text["engram_layer_ids"] = [i for i in text["engram_layer_ids"] if i <= a.layers]
cfg = conf_mod.HYV4FlashTextConfig(**text)
cfg._attn_implementation = "eager"
cfg._experts_implementation = "eager"
torch.set_default_dtype(torch.float64 if F64 else REAL_F32)
with torch.device("meta"):
    model = model_mod.HYV4FlashTextModel(cfg)
model.eval()


ROUND_TRIP = model_mod._fp8_round_trip


class LazyBank:
    """`gate_up_proj` / `down_proj` of the eager experts loop, one expert dequantized when it is read."""

    def __init__(self, stems, dtype):
        self.stems, self.dtype = stems, dtype

    def __getitem__(self, expert):
        e = int(expert)
        w = torch.cat([nvfp4(s, e) for s in self.stems], dim=-2).to(self.dtype)
        # `bf16a`: the experts' `F.linear` rounds its input (the program's activation dtype)
        w._a16 = A16
        return w


class LazyRows(torch.nn.Module):
    """The Engram table as stored: `f32(e4m3 * row scale)`, gathered from the memory-mapped shard."""

    def __init__(self, name):
        super().__init__()
        self.codes, self.scale = stored_np(name), stored_np(name + "_scale")
        self.weight = torch.empty(0)  # `HYV4FlashEngramEmbedding.forward` reads its device

    def forward(self, indices):
        flat = indices.reshape(-1).numpy()
        rows = torch.from_numpy(np.ascontiguousarray(self.codes[flat])).view(torch.float8_e4m3fn)
        scale = torch.from_numpy(np.ascontiguousarray(self.scale[flat]))
        out = rows.to(REAL_F32) * scale[:, None]
        # the pinned round trip, in F32 on the stored rows, is the identity
        assert torch.equal(ROUND_TRIP(out), out), "a stored Engram row is not a fixed point of the FP8 round trip"
        return out.reshape(*indices.shape, -1)


state = {}
lazy = {}
n_layers = cfg.num_hidden_layers
for name in sorted(index):
    if name.startswith(("vit.", "model.dfly_layers.")) or name.endswith((".weight_scale", ".weight_scale_inv",
                                                                          ".weight_scale_2")):
        continue
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if (m and int(m.group(1)) >= n_layers) or name == "lm_head.weight":
        continue
    new = rename(name)
    if ".mlp.experts." in new or "multi_head_embedding.embedding" in new:
        lazy[new] = name
        continue
    state[new] = program_dtype(new, dense(name))
lm_head = program_dtype("lm_head", dense("lm_head.weight"))
log(f"resident tensors {len(state)}, {sum(v.numel() * v.element_size() for v in state.values()) / 1e9:.1f} GB")

missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
assert not unexpected, unexpected
del state
# What the loop and the rows read lazily.
for il, layer in enumerate(model.layers):
    if cfg.mlp_layer_types[il] == "sparse":
        stem = f"model.layers.{il}.mlp.experts."
        experts = layer.mlp.experts
        del experts.gate_up_proj, experts.down_proj
        experts.gate_up_proj = LazyBank([stem + "gate_proj", stem + "up_proj"], ACT)
        experts.down_proj = LazyBank([stem + "down_proj"], ACT)
        missing = [k for k in missing if not k.startswith(f"layers.{il}.mlp.experts.")]
    if layer.engram_layer is not None:
        layer.engram_layer.multi_head_embedding.embedding = LazyRows(
            f"model.layers.{il}.engram_layer.multi_head_embedding.embedding.weight")
        missing = [k for k in missing if k != f"layers.{il}.engram_layer.multi_head_embedding.embedding.weight"]
assert not missing, missing

# Non-persistent buffers, rebuilt off the meta device as `__init__` and `_init_weights` build them.
OP_MATH_EPS = 2.0**-23
with torch.no_grad():
    rope_dtype = torch.float64 if F64 else REAL_F32

    def rebuild_rotary(rot):
        for layer_type, fn in rot.rope_init_fns.items():
            inv_freq, _ = fn(cfg.per_layer_config[layer_type], None, layer_type=layer_type)
            setattr(rot, f"{layer_type}_inv_freq", inv_freq.to(rope_dtype))
            setattr(rot, f"{layer_type}_original_inv_freq", inv_freq.to(rope_dtype).clone())

    rebuild_rotary(model.rotary_emb)
    engrams = {}
    for il, layer in enumerate(model.layers):
        if layer.self_attn.compressed_indexer is not None:
            rebuild_rotary(layer.self_attn.compressed_indexer.rotary_emb)
        engram = layer.engram_layer
        if engram is None:
            continue
        engram.head_vocab_sizes = torch.tensor(engram.config_head_vocab_sizes)
        engram.multi_head_embedding.offsets = engram.multi_head_embedding.head_offsets()
        engram.hash_multipliers = engram.draw_hash_multipliers(int(engram.lookup_table.max()) + 1)
        for norm in [*engram.lookup_per_head_norms, *engram.norm_key, *engram.norm_query, engram.unified_rmsnorm]:
            assert norm.eps is None
            norm.eps = OP_MATH_EPS
        engrams[il] = engram
for n, t in list(model.named_parameters()) + list(model.named_buffers()):
    assert not t.is_meta, f"{n} is still on the meta device"
assert sorted(il + 1 for il in engrams) == sorted(cfg.engram_layer_ids), "Engram sits at id - 1"
A16_COUNT = {"modules": 0, "expert_calls": 0}
if A16:
    def a16(x):
        hi = x.to(torch.bfloat16).to(REAL_F32)
        if a.dtype == "bf16a":
            return hi
        return hi + (x - hi).to(torch.bfloat16).to(REAL_F32)

    def round_input(module, args):
        return (a16(args[0]),) + tuple(args[1:])

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(k in name for k in KEEP_F32):
            module.register_forward_pre_hook(round_input)
            A16_COUNT["modules"] += 1
    functional = types.ModuleType("functional_program")
    functional.__dict__.update({k: getattr(F_REAL, k) for k in dir(F_REAL) if not k.startswith("__")})

    def a16_linear(x, w, b=None):
        if getattr(w, "_a16", False):
            A16_COUNT["expert_calls"] += 1
            x = a16(x)
        return F_REAL.linear(x, w, b)

    functional.linear = a16_linear
    model_mod.F = functional
    log(f"{a.dtype}: {A16_COUNT['modules']} linear layers round their input")
log(f"model ready ({a.dtype}), {n_layers} layers, Engram at decoder layers {sorted(engrams)}")

# ---- recording -------------------------------------------------------------------------------------
CALL = {}
indexer_forward = model_mod.HYV4FlashCompressedIndexer.forward


def recording_indexer(self, hidden_states, q_resid, attention_mask, position_ids, indexer_position_ids, cache,
                      kv_length):
    RECORD["indexer"] = []
    RECORD["force_indexer"] = None
    key = (CALL["index"], self.layer_idx)
    if FORCED is not None and key in FORCED["indexer"]:
        # the forced tape's recalled groups less one occurrence of the own group: the top-k picks
        # (consumed only when `cuda_topk` substitutes them)
        sel = FORCED["indexer"][(CALL["index"], self.layer_idx)]
        seq = hidden_states.shape[1]
        assert len(sel) == seq, "the forced picks are not this call's queries"
        RECORD["force_key"] = ("indexer",) + key
        start = cache.get_seq_length(self.layer_idx) - seq  # the attention has cached this call
        picks = []
        for q in range(seq):
            row = list(sel[q])
            row.remove((start + q) // self.index_kpool)
            picks.append(row)
        RECORD["force_indexer"] = torch.tensor([picks], dtype=torch.int64)
    try:
        out = indexer_forward(self, hidden_states, q_resid, attention_mask, position_ids, indexer_position_ids,
                              cache, kv_length)
    finally:
        picks, RECORD["indexer"] = RECORD["indexer"], None
        RECORD["force_indexer"] = None
    assert not (FORCED is not None and picks and ("indexer",) + key not in FORCED["used"]), \
        f"the forcing tape has no indexer record for call {key[0]} layer {key[1]}"
    if picks:
        assert len(picks) == 1
        indices, logits = picks[0]
        seq = hidden_states.shape[1]
        slots = torch.arange(seq) + (cache.get_seq_length(self.layer_idx) - seq)
        own = (slots // self.index_kpool)[None, :, None]
        groups = torch.cat([indices, own], dim=-1)[0]
        CALL["recalled"].append((self.layer_idx, torch.sort(groups, dim=-1).values.to(torch.int32),
                                 logits[0].to(REAL_F32)))
    return out


model_mod.HYV4FlashCompressedIndexer.forward = recording_indexer
for il, layer in enumerate(model.layers):
    if cfg.mlp_layer_types[il] == "sparse":
        layer.mlp.gate.layer_idx = il


def recording_router(self, hidden_states):
    """The pinned `HYV4FlashTopkRouter.forward` (n_group 1, topk_group 1: the group mask keeps every
    expert), recording the selection scores and the picks; under `--force-selections` the picks
    are the forced tape's and the weights follow them through the same lines."""
    hidden_states = hidden_states.view(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.type(proxy.float32), self.weight.type(proxy.float32))
    scores = router_logits.sigmoid()
    scores_for_choice = scores + self.e_score_correction_bias
    assert self.num_group == 1 and self.topk_group == 1
    topk_indices = proxy.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
    # recorded: the backend's own picks (under forcing, where it would have flipped)
    CALL["routes"].append((self.layer_idx, torch.sort(topk_indices, dim=-1).values.to(torch.int32),
                           scores_for_choice.to(REAL_F32)))
    key = (CALL["index"], self.layer_idx)
    if FORCED is not None:
        assert key in FORCED["router"], f"the forcing tape has no router record for call {key[0]} layer {key[1]}"
        FORCED["used"].add(("router",) + key)
        topk_indices = torch.tensor(FORCED["router"][key], dtype=torch.int64)
        assert topk_indices.shape == (hidden_states.shape[0], self.top_k)
    topk_weights = scores.gather(1, topk_indices)
    if self.norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights /= denominator
    topk_weights = topk_weights * self.routed_scaling_factor
    return router_logits, topk_weights, topk_indices


model_mod.HYV4FlashTopkRouter.forward = recording_router
hash_ngrams = model_mod.HYV4FlashEngramLayer.hash_ngrams


def recording_hash(self, ngram_tokens):
    out = hash_ngrams(self, ngram_tokens)
    CALL["engram"].append((self.layer_id - 1, (out + self.multi_head_embedding.offsets)[0].reshape(-1).tolist()))
    return out


model_mod.HYV4FlashEngramLayer.hash_ngrams = recording_hash


def applied_round_trip(x):
    # applied (and asserted the identity) on the F32 rows in `LazyRows.forward`, before
    # `HYV4FlashEngramLayer.forward` casts them to the activations' dtype
    return x


model_mod._fp8_round_trip = applied_round_trip

DUMP = {"on": bool(a.dump_layers), "rows": []}
if DUMP["on"]:
    os.makedirs(a.dump_layers, exist_ok=True)

    def dump_hook(module, args, output):
        if CALL.get("index") == 0:
            np.asarray(output[0].to(REAL_F32).numpy(), dtype="<f4").tofile(
                os.path.join(a.dump_layers, f"layer{module.layer_idx:02d}.f32"))

    for layer in model.layers:
        layer.register_forward_hook(dump_hook)


FORCED = None
FORCING_DIGEST = None


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


if a.force_selections:
    tape_path = os.path.join(a.force_selections, "tape.json")
    FORCING_DIGEST = file_sha(tape_path)
    ft = json.load(open(tape_path))
    req0 = json.load(open(a.request))
    assert (ft["token_ids"], ft["decode_from"]) == (req0["token_ids"], req0["decode_from"]), \
        "the forcing tape scored another request"
    assert not ft.get("forced_selections"), "the forcing tape is itself forced"
    for part, key in [("indexer", "sel"), ("router", "sel")]:
        meta = ft[part]
        assert file_sha(os.path.join(a.force_selections, meta[key + "_file"])) == meta[key + "_sha256"], \
            f"the forcing tape's {part} picks are not the ones it declares"
    fsel = np.fromfile(os.path.join(a.force_selections, ft["indexer"]["sel_file"]), dtype="<i4")
    frs = np.fromfile(os.path.join(a.force_selections, ft["router"]["sel_file"]), dtype="<i4")
    FORCED = {"indexer": {}, "router": {}, "used": set()}
    for part, flat in [("indexer", fsel), ("router", frs)]:
        for r in ft[part]["records"]:
            k = (r["call"], r["layer"])
            assert k not in FORCED[part], f"two {part} records for {k}"
            FORCED[part][k] = flat[r["sel_offset"]:r["sel_offset"] + r["queries"] * r["k"]] \
                .reshape(r["queries"], r["k"]).tolist()
    log(f"forcing {len(FORCED['indexer'])} indexer and {len(FORCED['router'])} router calls from {a.force_selections}")

# ---- the request -----------------------------------------------------------------------------------
req = json.load(open(a.request))
ids = torch.tensor([req["token_ids"]])
prefill = req["decode_from"]
assert 0 < prefill <= ids.shape[1]
mask_stub = types.SimpleNamespace(
    config=types.SimpleNamespace(image_token_id=top["image_token_id"], video_token_id=top["video_token_id"]),
    language_model=model,
)
os.makedirs(a.out, exist_ok=True)
cache = cache_mod.DynamicCache(config=cfg)
lf = open(os.path.join(a.out, "logits.f32"), "wb")
sel_f = open(os.path.join(a.out, "indexer_sel.i32"), "wb")
lg_f = open(os.path.join(a.out, "indexer_logits.f32"), "wb")
rs_f = open(os.path.join(a.out, "router_sel.i32"), "wb")
rc_f = open(os.path.join(a.out, "router_scores.f32"), "wb")
router_index, rs_off, rc_off = [], 0, 0
logits_sha = hashlib.sha256()
indexer_index, engram_index = [], []
sel_off = lg_off = 0
argmax = []


def fnv1a(text):
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


with torch.no_grad():
    calls = [(0, prefill)] + [(t, t + 1) for t in range(prefill, ids.shape[1])]
    for ci, (start, end) in enumerate(calls):
        CALL.clear()
        CALL.update({"index": ci, "recalled": [], "engram": [], "routes": []})
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
        # the head in F32 (`enable_lm_head_fp32`), float64 in the oracle; logits rounded once to f32
        rows = (hidden[0].to(lm_head.dtype) @ lm_head.T).to(REAL_F32).contiguous()
        data = rows.numpy().astype("<f4").tobytes()
        lf.write(data)
        logits_sha.update(data)
        argmax += [int(i) for i in torch.argmax(rows, dim=-1)]
        for il, groups, glogits in CALL["recalled"]:
            gb = groups.numpy().astype("<i4").tobytes()
            lb = glogits.numpy().astype("<f4").tobytes()
            sel_f.write(gb)
            lg_f.write(lb)
            indexer_index.append({"call": ci, "start": start, "tokens": end - start, "layer": il,
                                  "queries": int(groups.shape[0]), "k": int(groups.shape[1]),
                                  "groups": int(glogits.shape[1]), "sel_offset": sel_off, "logit_offset": lg_off})
            sel_off += groups.numel()
            lg_off += glogits.numel()
        for il, picks, scores in CALL["routes"]:
            rs_f.write(picks.numpy().astype("<i4").tobytes())
            rc_f.write(scores.numpy().astype("<f4").tobytes())
            router_index.append({"call": ci, "layer": il, "queries": int(picks.shape[0]), "k": int(picks.shape[1]),
                                 "experts": int(scores.shape[1]), "sel_offset": rs_off, "score_offset": rc_off})
            rs_off += picks.numel()
            rc_off += scores.numel()
        for il, r in CALL["engram"]:
            engram_index.append({"call": ci, "layer": il, "rows": fnv1a(",".join(map(str, r)))})
        log(f"call {ci + 1}/{len(calls)} tokens {start}..{end} indexer_layers={len(CALL['recalled'])} "
            f"engram_layers={len(CALL['engram'])}")
if FORCED is not None:
    want = {("indexer",) + k for k in FORCED["indexer"]} | {("router",) + k for k in FORCED["router"]}
    assert FORCED["used"] == want, f"{len(want - FORCED['used'])} forced records were never consumed"
lf.close()
sel_f.close()
lg_f.close()
rs_f.close()
rc_f.close()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


assert not A16 or (A16_COUNT["modules"] > 0 and A16_COUNT["expert_calls"] > 0), "bf16a rounded nothing"
tape = {
    "format": "memra-checkpoint-tape-v2",
    "engine": f"transformers-pr50903@b865850d0532-cpu-{a.dtype}",
    "numeric_class": "hy-v4-flash-source-program",
    "backend_dtype": a.dtype,
    "request": req["name"],
    "token_ids": req["token_ids"],
    "decode_from": prefill,
    "vocab": int(lm_head.shape[0]),
    "positions": int(ids.shape[1]),
    "layers": n_layers,
    "argmax": argmax,
    "logits": {"file": "logits.f32", "sha256": logits_sha.hexdigest()},
    "indexer": {"sel_file": "indexer_sel.i32", "sel_sha256": sha(os.path.join(a.out, "indexer_sel.i32")),
                "logit_file": "indexer_logits.f32",
                "logit_sha256": sha(os.path.join(a.out, "indexer_logits.f32")), "records": indexer_index},
    "router": {"sel_file": "router_sel.i32", "sel_sha256": sha(os.path.join(a.out, "router_sel.i32")),
               "score_file": "router_scores.f32", "score_sha256": sha(os.path.join(a.out, "router_scores.f32")),
               "records": router_index},
    "engram": engram_index,
    "forced_selections": bool(a.force_selections),
    "a16": A16_COUNT if A16 else None,
    "forcing_tape_sha256": FORCING_DIGEST,
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "transformers_modeling_blob": PINS[model_mod],
    "threads": torch.get_num_threads(),
    "seconds": round(time.time() - T0, 1),
}
json.dump(tape, open(os.path.join(a.out, "tape.json"), "w"), indent=1)
print(json.dumps({k: tape[k] for k in ["engine", "request", "positions", "decode_from", "seconds"]} |
                 {"logits_sha256": tape["logits"]["sha256"], "indexer_records": len(indexer_index),
                  "engram_records": len(engram_index)}))
