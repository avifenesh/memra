"""Module-level oracle for the Hy4-flash pooled indexer (`HYV4FlashCompressedIndexer`).

Runs the pinned transformers modeling code (PR #50903 at b865850d0532, installed from that
commit; the modeling and configuration files are checked against their git blobs) on a tiny
config in float64, with more tokens than the indexer budget so the indexed layers select groups.
Records, per indexed layer, the indexer's inputs (`hidden_states`, `q_resid`), the cached token
keys, the pooled group keys, the group logits, the recalled groups and the visible-key mask it
lays over the causal mask, and the inputs and outputs of every module the reference executor
reimplements (as modules/gen_modules.py does), then the model's last hidden state.

Two process-level substitutions, no line of the pinned code edited:

- float32 islands -> float64, as in modules/gen_modules.py (the modeling module's `torch.float32`
  and `torch.float`, and `Tensor.float`).
- `topk` follows the CUDA rule. The program runs on CUDA, where `torch.topk` (PyTorch v2.15.0,
  aten/src/ATen/native/cuda/TensorTopK.cu, `sbtopk::gatherTopK` and `mbtopk::gatherTopK`)
  writes every value strictly above the k-th value, then the values equal to it in ascending
  index order until k are written. CPU `topk` leaves the order of ties unspecified. The ReLU
  scores make exact ties common (a group no head scores above zero has logit 0), so the oracle
  selects with a stable descending sort: the same set, lower index first among equal values.
  The record counts how many queries select a different set under the CPU kernel and under the
  flipped tie order (higher index first).

    venv/bin/python -I gen_indexpool.py OUT.json
"""

import hashlib
import json
import sys
import types

import torch

import transformers.models.hy_v4_flash.configuration_hy_v4_flash as conf_mod
import transformers.models.hy_v4_flash.modeling_hy_v4_flash as model_mod


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


assert blob(model_mod.__file__) == "2d796211a4f867bd4c46d4a952cfee0375929f72", "modeling pin"
assert blob(conf_mod.__file__) == "16cb099b7991c190c4d5d85dd30b78e0b4c62770", "configuration pin"

NATIVE_TOPK = torch.Tensor.topk


def cuda_topk(self, k, dim=-1, largest=True, sorted=True):
    """torch.topk with the CUDA selection: the values above the k-th, then ties by ascending index."""
    assert largest, "only largest-first selection is used"
    values, indices = torch.sort(self, dim=dim, descending=True, stable=True)
    return torch.return_types.topk((values.narrow(dim, 0, k), indices.narrow(dim, 0, k)))


# float32 islands -> float64 (see the module docstring).
proxy = types.ModuleType("torch_f64")
proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
proxy.float32 = torch.float64
proxy.float = torch.float64
proxy.topk = lambda x, k, dim=-1, largest=True, sorted=True: cuda_topk(x, k, dim, largest, sorted)
model_mod.torch = proxy
torch.Tensor.float = lambda self: self.to(torch.float64)
torch.Tensor.topk = cuda_topk
torch.set_default_dtype(torch.float64)

L = 4
S = 41
INDEX_TOPK = 12
INDEX_HEAD_DIM = 12
INDEX_HEADS = 2
KPOOL = 4
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
    index_topk=INDEX_TOPK,
    index_head_dim=INDEX_HEAD_DIM,
    index_n_heads=INDEX_HEADS,
    compressed_indexer_compress_ratio=KPOOL,
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
    # Epsilons exactly representable in float32: the typed plan carries them as f32.
    hc_eps=2.0**-20,
    rms_norm_eps=2.0**-17,
    learnable_sink_init=0.0,
    engram_layer_ids=None,
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
assert config.index_kpool == KPOOL and config.index_topk == INDEX_TOPK

SEED = 20261121
torch.manual_seed(SEED)
model = model_mod.HYV4FlashTextModel(config).to(torch.float64).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("norm.weight") or "layernorm" in name or name.endswith("norm_weight"):
            p.copy_(1.0 + 0.1 * torch.randn_like(p))
        else:
            p.copy_(0.3 * torch.randn_like(p))
    for name, b in model.named_buffers():
        if name.endswith("e_score_correction_bias"):
            b.copy_(0.05 * torch.randn_like(b))

input_ids = torch.randint(0, config.vocab_size, (1, S))
captured = {}
indexer = {}


def flat(t):
    return t.detach().reshape(-1).tolist()


def hook(name):
    def fn(module, args, kwargs, output):
        rec = {"in": [a for a in args if isinstance(a, torch.Tensor)]}
        if "hidden_states" in kwargs:
            rec["in"].append(kwargs["hidden_states"])
        rec["out"] = list(output) if isinstance(output, tuple) else [output]
        captured[name] = {k: [flat(t) for t in v if isinstance(t, torch.Tensor)] for k, v in rec.items()}
        captured[name]["in_shapes"] = [list(t.shape) for t in rec["in"]]
        captured[name]["out_shapes"] = [list(t.shape) for t in rec["out"] if isinstance(t, torch.Tensor)]

    return fn


def indexer_hook(il):
    def fn(module, args, kwargs, output):
        hidden_states, q_resid, attention_mask = args[0], args[1], args[2]
        rec = indexer.setdefault(il, {})
        rec["x"] = flat(hidden_states)
        rec["q_resid"] = flat(q_resid)
        rec["kv_length"] = int(args[6])
        # The causal mask is 0 where visible and finfo.min elsewhere; the indexer fills finfo.min
        # on the keys it does not recall.
        rec["visible"] = [[int(v) for v in row] for row in (output[0, 0] == 0).tolist()]
        rec["causal"] = [[int(v) for v in row] for row in (attention_mask[0, 0] == 0).tolist()]

    return fn


original_pool = model_mod.HYV4FlashCompressedIndexer.pool_groups


def pool_groups(self, token_keys):
    out = original_pool(self, token_keys)
    rec = indexer.setdefault(self.layer_idx, {})
    rec["token_keys"] = flat(token_keys)
    rec["group_keys"] = flat(out)
    return out


model_mod.HYV4FlashCompressedIndexer.pool_groups = pool_groups

# The group logits as they reach the selection, and the selected sets under the three tie orders.
topk_calls = []


def recording_topk(self, k, dim=-1, largest=True, sorted=True):
    if self.dim() == 3 and self.shape[1] == S and k == INDEX_TOPK // KPOOL - 1:
        topk_calls.append(self.detach().clone())
    return cuda_topk(self, k, dim, largest, sorted)


torch.Tensor.topk = recording_topk

for il, layer in enumerate(model.layers):
    for sub in ["attn_hc", "input_layernorm", "self_attn", "ffn_hc", "post_attention_layernorm", "mlp", "post_mlp_layernorm"]:
        getattr(layer, sub).register_forward_hook(hook(f"layers.{il}.{sub}"), with_kwargs=True)
    layer.register_forward_hook(hook(f"layers.{il}"), with_kwargs=True)
    if layer.self_attn.compressed_indexer is not None:
        layer.self_attn.compressed_indexer.register_forward_hook(indexer_hook(il), with_kwargs=True)
model.hc_head.register_forward_hook(hook("hc_head"), with_kwargs=True)
model.norm.register_forward_hook(hook("norm"), with_kwargs=True)

with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=torch.arange(S)[None]).last_hidden_state

indexed_layers = sorted(indexer)
assert len(topk_calls) == len(indexed_layers), (len(topk_calls), indexed_layers)
k = INDEX_TOPK // KPOOL - 1


def selected(logits, order):
    """Selected group set per query under a tie order: 'cuda', 'flipped' or 'cpu'."""
    sets = []
    for row in logits[0]:
        if order == "cuda":
            idx = cuda_topk(row, k).indices
        elif order == "flipped":
            # Higher index first among equal values: reverse, stable sort, map back.
            rev = torch.flip(row, dims=[0])
            idx = row.shape[0] - 1 - cuda_topk(rev, k).indices
        else:
            idx = NATIVE_TOPK(row, k).indices
        sets.append(sorted(int(i) for i in idx))
    return sets


tie_stats = {}
for il, logits in zip(indexed_layers, topk_calls):
    rec = indexer[il]
    num_groups = logits.shape[-1]
    finite = torch.isfinite(logits[0])
    # Signed zeros at a tie must agree within a query (the CUDA radix key orders -0.0 below 0.0).
    for row, ok in zip(logits[0], finite):
        zeros = row[ok][row[ok] == 0]
        assert len(set(torch.signbit(zeros).tolist())) <= 1, "mixed signed zeros in one query"
    rec["groups"] = num_groups
    rec["logits"] = [[float(v) if f else None for v, f in zip(row.tolist(), fr.tolist())] for row, fr in zip(logits[0], finite)]
    cuda = selected(logits, "cuda")
    rec["selected"] = cuda
    flipped = selected(logits, "flipped")
    cpu = selected(logits, "cpu")
    # A query whose closed groups do not fill k also takes masked (-inf) groups; those keys are
    # future or its own group, so they never change the visible set. Count only real differences.
    def closed(sets, q):
        return sorted(g for g in sets[q] if g < q // KPOOL)

    tie_stats[il] = {
        "queries": S,
        "flipped_differs": sum(closed(cuda, q) != closed(flipped, q) for q in range(S)),
        "cpu_differs": sum(closed(cuda, q) != closed(cpu, q) for q in range(S)),
        "zero_logit_closed_groups": int(sum(((row == 0) & fr).sum() for row, fr in zip(logits[0], finite))),
    }

weights = {name: {"shape": list(t.shape), "data": flat(t)} for name, t in model.state_dict().items()}
record = {
    "transformers_modeling_blob": blob(model_mod.__file__),
    "dtype": "float64",
    "topk": "cuda rule: values above the k-th, then ties by ascending index (PyTorch v2.15.0 TensorTopK.cu)",
    "seed": SEED,
    "config": {
        "hidden_size": 16, "num_hidden_layers": L, "heads": {"indexed": 2, "sliding": 4},
        "q_lora_rank": 12, "kv_lora_rank": 8, "qk_nope_head_dim": 6, "qk_rope_head_dim": 8, "v_head_dim": 5,
        "rope_theta": {"indexed": 1e7, "sliding": 1e4}, "sliding_window": 3,
        "index_topk": INDEX_TOPK, "index_head_dim": INDEX_HEAD_DIM, "index_n_heads": INDEX_HEADS, "index_kpool": KPOOL,
        "intermediate_size": 24, "routed_expert_hidden_size": 8, "moe_intermediate_size": 6,
        "n_routed_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1, "routed_scaling_factor": 1.75,
        "swiglu_limit": 1.5, "hc_mult": 4, "hc_magnitude": 2.0, "hc_eps": 2.0**-20, "rms_norm_eps": 2.0**-17,
        "layer_kinds": ["indexed", "indexed", "sliding", "sliding"], "dense_layers": [True, False, False, False],
    },
    "input_ids": input_ids.reshape(-1).tolist(),
    "positions": list(range(S)),
    "weights": weights,
    "modules": captured,
    "indexer": {str(il): indexer[il] for il in indexed_layers},
    "tie_stats": {str(il): v for il, v in tie_stats.items()},
    "last_hidden_state": flat(out),
}
json.dump(record, open(sys.argv[1], "w"))
print(json.dumps({"weights": len(weights), "modules": len(captured), "out": list(out.shape), "tie_stats": tie_stats}))
