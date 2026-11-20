"""Oracle for the Hy4-flash Engram memory (`HYV4FlashEngramLayer`), two parts.

1. `artifact_hash`: the integer program on the artifact's own config and lookup table. The
   pinned config class reads the pinned `config.json`; the lookup table of `model.layers.1` is
   range-read from the artifact (the four are byte-identical, sources/lookup-tables-nvfp4.json).
   The pinned `draw_hash_multipliers` (NumPy `default_rng`), `build_ngram_tokens` and
   `hash_ngrams` run as unbound methods on stand-in objects, so no 63 GB table is built. The
   tokens cover the identity prefix, random ids, the pad id, the 14 placeholders past the table
   and a packed second document. Recorded: the multipliers per layer, the n-gram tokens, the
   mask and the hash ids, plus the table entries those tokens read (not the table).

2. `tiny`: the pinned model on the step 4 tiny config (41 tokens past the indexer budget) with
   one Engram layer (`engram_layer_ids` [2], so the module sits in decoder layer 1), 2 orders of
   2 heads of 4 dims, primes 29, 31, 37, 41, a 60-entry lookup table on a 64-token vocabulary
   (ids 60-63 lie past it) and seed 3. Recorded as gen_indexpool.py records, plus the Engram
   module's inputs and output, its hash ids, and its table as FP8 codes with one scale per row.

Process-level substitutions, no line of the pinned code edited:

- float32 islands -> float64 and the CUDA `topk` rule, as in indexpool/gen_indexpool.py.
- The four Engram norms (`nn.RMSNorm(eps=None)`) get `eps = 2^-23`. ATen's default is the
  epsilon of the op-math type: F32 for the program's BF16 activations, F64 in this float64 run.
  Setting it keeps the program's value.
- The lookup table is set after construction and the multipliers redrawn from it, which is what
  loading a checkpoint does (`_init_weights` draws them from the loaded map).
- `engram_fp8_embedding` stays on: the table rows pass the pinned `_fp8_round_trip` in float64.
  The record carries that round trip's codes and scales; `q * scale` equals its output exactly
  (asserted), which is how the artifact stores the rows.

    OMP_NUM_THREADS=1 venv/bin/python -I gen_engram.py CONFIG_JSON OUT.json
"""

import array
import hashlib
import json
import struct
import sys
import types
import urllib.request

import torch

import transformers.models.hy_v4_flash.configuration_hy_v4_flash as conf_mod
import transformers.models.hy_v4_flash.modeling_hy_v4_flash as model_mod


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


assert blob(model_mod.__file__) == "2d796211a4f867bd4c46d4a952cfee0375929f72", "modeling pin"
assert blob(conf_mod.__file__) == "16cb099b7991c190c4d5d85dd30b78e0b4c62770", "configuration pin"

CONFIG_SHA256 = "c88f950d8050be7402a9d969a28fa2af8d09fa46bce04920a3a14ae0dbf4b999"
TABLE_SHA256 = "87c7b2e1c2fc227ba7fdec4dd7a867533072eb2b73a85cc54189174ccfaf5f3f"
ARTIFACT = ("Avifenesh/Hy4-flash-NVFP4", "f9eac06cf022cfd3dc2e8ed423922466b2adcd87")
Layer = model_mod.HYV4FlashEngramLayer


def flat(t):
    return t.detach().reshape(-1).tolist()


# ---- part 1: the artifact's integer program ----------------------------------------------
raw_config = open(sys.argv[1], "rb").read()
assert hashlib.sha256(raw_config).hexdigest() == CONFIG_SHA256, "config pin"
text = json.loads(raw_config)["text_config"]
cfg = conf_mod.HYV4FlashTextConfig(**{k: v for k, v in text.items() if k != "model_type"})


def get(path, a, b):
    url = f"https://huggingface.co/{ARTIFACT[0]}/resolve/{ARTIFACT[1]}/{path}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "engram-oracle"})
    return urllib.request.urlopen(req, timeout=120).read()


index = json.loads(
    urllib.request.urlopen(
        urllib.request.Request(
            f"https://huggingface.co/{ARTIFACT[0]}/resolve/{ARTIFACT[1]}/model.safetensors.index.json",
            headers={"User-Agent": "engram-oracle"},
        ),
        timeout=60,
    ).read()
)
name = "model.layers.1.engram_layer.lookup_table"
shard = index["weight_map"][name]
n = struct.unpack("<Q", get(shard, 0, 7))[0]
hdr = json.loads(get(shard, 8, 8 + n - 1))
s, e = hdr[name]["data_offsets"]
raw_table = get(shard, 8 + n + s, 8 + n + e - 1)
assert hashlib.sha256(raw_table).hexdigest() == TABLE_SHA256, "lookup table pin"
values = array.array("i")
values.frombytes(raw_table)
table = torch.tensor(values.tolist(), dtype=torch.int32)
assert table.shape[0] == cfg.engram_lookup_table_size
V = int(table.max()) + 1

gen = torch.Generator().manual_seed(20261122)
first = torch.cat(
    [
        torch.arange(0, 70),  # the identity prefix and the first ids past it
        torch.randint(0, cfg.vocab_size, (60,), generator=gen),
        torch.tensor([cfg.engram_pad_id, 17, 18, 19, cfg.engram_pad_id, cfg.engram_pad_id]),
        torch.arange(120818, 120832),  # the 14 placeholders past the table
        torch.tensor([17, 18, 19, 17, 18, 19, 120831, 0, 120002, 92619, 120817]),
    ]
)
second = torch.cat([torch.tensor([120818, 17, 18, 19]), torch.randint(0, cfg.vocab_size, (40,), generator=gen)])
ids = torch.cat([first, second])[None]
positions = torch.cat([torch.arange(first.shape[0]), torch.arange(second.shape[0])])[None]
model_stub = types.SimpleNamespace(config=cfg)
ngram = model_mod.HYV4FlashTextModel.build_ngram_tokens(model_stub, ids, positions)
mask = ids != cfg.engram_pad_id

layers = {}
for layer_id in cfg.engram_layer_ids:
    stub = types.SimpleNamespace(seed=cfg.engram_seed, layer_id=layer_id, max_ngram_size=cfg.engram_max_ngram_size)
    multipliers = Layer.draw_hash_multipliers(stub, V)
    sizes = [size for order in cfg.engram_layer_head_vocab_sizes[str(layer_id)] for size in order]
    hasher = types.SimpleNamespace(
        lookup_table=table,
        pad_id=cfg.engram_pad_id,
        hash_multipliers=multipliers,
        max_ngram_size=cfg.engram_max_ngram_size,
        n_head_per_ngram=cfg.engram_n_head_per_ngram,
        head_vocab_sizes=torch.tensor(sizes),
    )
    hashes = Layer.hash_ngrams(hasher, ngram)
    layers[str(layer_id)] = {
        "multipliers": multipliers.tolist(),
        # Bounded by the table length instead (the `arange` default's bound): must differ.
        "multipliers_table_length_bound": Layer.draw_hash_multipliers(stub, table.shape[0]).tolist(),
        "moduli": sizes,
        "hash_ids": flat(hashes),
    }
used = sorted(set(int(t) for t in ngram.reshape(-1).tolist() if t < table.shape[0]) | {cfg.engram_pad_id})
artifact_hash = {
    "config_sha256": CONFIG_SHA256,
    "lookup_table_sha256": TABLE_SHA256,
    "lookup_table_size": table.shape[0],
    "compressed_vocab": V,
    "engram_seed": cfg.engram_seed,
    "ids": flat(ids),
    "positions": flat(positions),
    "ngram_tokens": flat(ngram),
    "mask": [int(v) for v in flat(mask)],
    "lookup_entries": {str(i): int(table[i]) for i in used},
    "layers": layers,
}

# ---- part 2: the tiny model ------------------------------------------------------------------
NATIVE_TOPK = torch.Tensor.topk


def cuda_topk(self, k, dim=-1, largest=True, sorted=True):
    assert largest, "only largest-first selection is used"
    values, indices = torch.sort(self, dim=dim, descending=True, stable=True)
    return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))


proxy = types.ModuleType("torch_f64")
proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
proxy.float32 = torch.float64
proxy.float = torch.float64
proxy.topk = lambda x, k, dim=-1, largest=True, sorted=True: cuda_topk(x, k, dim, largest, sorted)
model_mod.torch = proxy
torch.Tensor.float = lambda self: self.to(torch.float64)
torch.Tensor.topk = cuda_topk
torch.set_default_dtype(torch.float64)

L, S, PAD, TABLE = 4, 41, 5, 60
PRIMES = [[29, 31], [37, 41]]
OP_MATH_EPS = 2.0**-23
config = conf_mod.HYV4FlashTextConfig(
    vocab_size=64,
    hidden_size=16,
    intermediate_size=24,
    num_hidden_layers=L,
    num_attention_heads=2,
    num_key_value_heads=2,
    swa_num_attention_heads=4,
    q_lora_rank=12,
    kv_lora_rank=8,
    qk_nope_head_dim=6,
    qk_rope_head_dim=8,
    v_head_dim=5,
    swa_qk_nope_head_dim=6,
    swa_qk_rope_head_dim=8,
    swa_v_head_dim=5,
    index_topk=12,
    index_head_dim=12,
    index_n_heads=2,
    compressed_indexer_compress_ratio=4,
    routed_expert_hidden_size=8,
    moe_intermediate_size=6,
    n_routed_experts=6,
    num_experts_per_tok=2,
    n_shared_experts=1,
    routed_scaling_factor=1.75,
    norm_topk_prob=True,
    n_group=1,
    topk_group=1,
    swiglu_limit=1.5,
    sliding_window=3,
    hc_mult=4,
    hc_magnitude=2.0,
    hc_eps=2.0**-20,
    rms_norm_eps=2.0**-17,
    learnable_sink_init=0.0,
    engram_layer_ids=[2],
    engram_max_ngram_size=3,
    engram_n_head_per_ngram=2,
    engram_n_embed_per_ngram=8,
    engram_layer_head_vocab_sizes={"2": PRIMES},
    engram_pad_id=PAD,
    engram_fp8_embedding=True,
    engram_seed=3,
    engram_lookup_table_size=TABLE,
    layer_types=["deepseek_sparse_attention", "deepseek_sparse_attention", "sliding_attention", "sliding_attention"],
    mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
    rope_parameters={
        "sliding_attention": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_theta": 10000.0, "rope_type": "default"},
        "deepseek_sparse_attention": {"mrope_interleaved": True, "mrope_section": [2, 1, 1], "rope_theta": 10000000.0, "rope_type": "default"},
    },
    pad_token_id=None,
    bos_token_id=None,
    eos_token_id=None,
)
config._attn_implementation = "eager"
config._experts_implementation = "eager"

SEED = 20261122
torch.manual_seed(SEED)
model = model_mod.HYV4FlashTextModel(config).to(torch.float64).eval()
engram = model.layers[1].engram_layer
assert engram is not None and all(layer.engram_layer is None for i, layer in enumerate(model.layers) if i != 1)
with torch.no_grad():
    for pname, p in model.named_parameters():
        if pname.endswith("norm.weight") or "layernorm" in pname or pname.endswith("norm_weight"):
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(0.3 * torch.randn_like(p))
    for bname, b in model.named_buffers():
        if bname.endswith("e_score_correction_bias"):
            b.copy_(0.05 * torch.randn_like(b))
    # A map that is not the identity, with fewer compressed ids (40) than entries (60).
    tiny_table = torch.tensor([(7 * i + 3) % 40 for i in range(TABLE)], dtype=torch.int32)
    engram.lookup_table.copy_(tiny_table)
    engram.hash_multipliers.copy_(engram.draw_hash_multipliers(int(tiny_table.max()) + 1))
for norm in [*engram.lookup_per_head_norms, *engram.norm_key, *engram.norm_query, engram.unified_rmsnorm]:
    assert norm.eps is None
    norm.eps = OP_MATH_EPS

weight = engram.multi_head_embedding.embedding.weight.detach()
info = torch.finfo(torch.float8_e4m3fn)
row_scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12) / info.max
codes = (weight / row_scale).clamp(-info.max, info.max).to(torch.float8_e4m3fn)
assert torch.equal(codes.to(torch.float64) * row_scale, model_mod._fp8_round_trip(weight)), "q * scale is the round trip"

gen = torch.Generator().manual_seed(SEED)
input_ids = torch.randint(0, config.vocab_size, (1, S), generator=gen)
input_ids[0, 10] = PAD
input_ids[0, 20] = 61
input_ids[0, 30] = 63
input_ids[0, 31] = 60
input_ids[0, 32:35] = input_ids[0, 2:5]  # a repeated trigram
position_ids = torch.arange(S)[None]
ngram_tokens = model.build_ngram_tokens(input_ids, position_ids)
engram_mask = input_ids != PAD

captured = {}


def hook(hname):
    def fn(module, args, kwargs, output):
        rec = {"in": [a for a in args if isinstance(a, torch.Tensor)]}
        if "hidden_states" in kwargs:
            rec["in"].append(kwargs["hidden_states"])
        rec["out"] = list(output) if isinstance(output, tuple) else [output]
        captured[hname] = {k: [flat(t) for t in v if isinstance(t, torch.Tensor)] for k, v in rec.items()}

    return fn


for il, layer in enumerate(model.layers):
    for sub in ["attn_hc", "self_attn", "ffn_hc", "mlp"]:
        getattr(layer, sub).register_forward_hook(hook(f"layers.{il}.{sub}"), with_kwargs=True)
    layer.register_forward_hook(hook(f"layers.{il}"), with_kwargs=True)
engram.register_forward_hook(hook("layers.1.engram_layer"), with_kwargs=True)

with torch.no_grad():
    out = model(
        input_ids=input_ids, position_ids=position_ids, ngram_tokens=ngram_tokens, engram_mask=engram_mask
    ).last_hidden_state
    tiny_hashes = engram.hash_ngrams(ngram_tokens)

weights = {
    wname: {"shape": list(t.shape), "data": flat(t)}
    for wname, t in model.state_dict().items()
    if not wname.endswith("multi_head_embedding.embedding.weight")
}
tiny = {
    "seed": SEED,
    "op_math_eps": OP_MATH_EPS,
    "config": {"engram_layer_ids": [2], "max_ngram": 3, "heads_per_order": 2, "head_dim": 4,
               "moduli": [p for order in PRIMES for p in order], "pad_id": PAD, "lookup_table_size": TABLE,
               "engram_seed": 3},
    "lookup_table": tiny_table.tolist(),
    "multipliers": engram.hash_multipliers.tolist(),
    "input_ids": flat(input_ids),
    "positions": flat(position_ids),
    "ngram_tokens": flat(ngram_tokens),
    "mask": [int(v) for v in flat(engram_mask)],
    "hash_ids": flat(tiny_hashes),
    "table_codes": flat(codes.view(torch.uint8)),
    "table_scales": flat(row_scale),
    "weights": weights,
    "modules": captured,
    "last_hidden_state": flat(out),
}
record = {
    "transformers_modeling_blob": blob(model_mod.__file__),
    "dtype": "float64",
    "artifact_hash": artifact_hash,
    "tiny": tiny,
}
json.dump(record, open(sys.argv[2], "w"))
print(json.dumps({
    "compressed_vocab": V,
    "multipliers": {k: v["multipliers"] for k, v in layers.items()},
    "tokens": ids.shape[1],
    "tiny_weights": len(weights),
    "tiny_modules": len(captured),
    "tiny_masked": int((~engram_mask).sum()),
}))
