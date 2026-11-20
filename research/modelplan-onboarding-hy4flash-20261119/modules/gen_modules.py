"""Module-level oracle for the Hy4-flash attention, residual and MoE operations.

Runs the pinned transformers modeling code (PR #50903 at b865850d0532, installed from that
commit; the modeling and configuration files are checked against their git blobs) on a tiny
config in float64, and records the inputs and outputs of every module the reference executor
reimplements: per layer the attention hyper-connection, `input_layernorm`, `self_attn`, the MLP
hyper-connection, `post_attention_layernorm`, the MLP, `post_mlp_layernorm` and the decoder
layer itself; then `hc_head` and the final norm.

The pinned code casts to float32 inside RMSNorm, RoPE, the router and the hyper-connections.
This process maps those casts to float64 (the modeling module's `torch.float32` and
`torch.float`, and `Tensor.float`), so both sides compute in float64 and the comparison bound can be tight. No
line of the pinned code is edited.

    venv/bin/python -I gen_modules.py OUT.json
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

# float32 islands -> float64 (see the module docstring).
proxy = types.ModuleType("torch_f64")
proxy.__dict__.update({k: getattr(torch, k) for k in dir(torch) if not k.startswith("__")})
proxy.float32 = torch.float64
proxy.float = torch.float64
model_mod.torch = proxy
torch.Tensor.float = lambda self: self.to(torch.float64)
torch.set_default_dtype(torch.float64)

L = 4
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
    index_topk=8,
    index_head_dim=8,
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

torch.manual_seed(20261120)
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

S = 7
input_ids = torch.randint(0, config.vocab_size, (1, S))
captured = {}


def hook(name):
    def fn(module, args, kwargs, output):
        rec = {"in": [a for a in args if isinstance(a, torch.Tensor)]}
        if "hidden_states" in kwargs:
            rec["in"].append(kwargs["hidden_states"])
        rec["out"] = list(output) if isinstance(output, tuple) else [output]
        captured[name] = {
            k: [t.detach().reshape(-1).tolist() for t in v if isinstance(t, torch.Tensor)]
            for k, v in rec.items()
        }
        captured[name]["in_shapes"] = [list(t.shape) for t in rec["in"]]
        captured[name]["out_shapes"] = [list(t.shape) for t in rec["out"] if isinstance(t, torch.Tensor)]

    return fn


for il, layer in enumerate(model.layers):
    for sub in ["attn_hc", "input_layernorm", "self_attn", "ffn_hc", "post_attention_layernorm", "mlp", "post_mlp_layernorm"]:
        getattr(layer, sub).register_forward_hook(hook(f"layers.{il}.{sub}"), with_kwargs=True)
    layer.register_forward_hook(hook(f"layers.{il}"), with_kwargs=True)
model.hc_head.register_forward_hook(hook("hc_head"), with_kwargs=True)
model.norm.register_forward_hook(hook("norm"), with_kwargs=True)

with torch.no_grad():
    out = model(input_ids=input_ids, position_ids=torch.arange(S)[None]).last_hidden_state

weights = {name: {"shape": list(t.shape), "data": t.detach().reshape(-1).tolist()} for name, t in model.state_dict().items()}
record = {
    "transformers_modeling_blob": blob(model_mod.__file__),
    "dtype": "float64",
    "seed": 20261120,
    "config": {
        "hidden_size": 16, "num_hidden_layers": L, "heads": {"indexed": 2, "sliding": 4},
        "q_lora_rank": 12, "kv_lora_rank": 8, "qk_nope_head_dim": 6, "qk_rope_head_dim": 8, "v_head_dim": 5,
        "rope_theta": {"indexed": 1e7, "sliding": 1e4}, "sliding_window": 3, "index_topk": 8,
        "intermediate_size": 24, "routed_expert_hidden_size": 8, "moe_intermediate_size": 6,
        "n_routed_experts": 6, "num_experts_per_tok": 2, "n_shared_experts": 1, "routed_scaling_factor": 1.75,
        "swiglu_limit": 1.5, "hc_mult": 4, "hc_magnitude": 2.0, "hc_eps": 2.0**-20, "rms_norm_eps": 2.0**-17,
        "layer_kinds": ["indexed", "indexed", "sliding", "sliding"], "dense_layers": [True, False, False, False],
    },
    "input_ids": input_ids.reshape(-1).tolist(),
    "positions": list(range(S)),
    "weights": weights,
    "modules": captured,
    "last_hidden_state": out.reshape(-1).tolist(),
}
json.dump(record, open(sys.argv[1], "w"))
print(len(weights), "weights;", len(captured), "modules;", "out", list(out.shape))
