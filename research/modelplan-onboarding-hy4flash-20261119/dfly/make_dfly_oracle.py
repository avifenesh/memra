"""DFly2 drafter oracle for the hy_v4_flash pack: the pinned vLLM program on the artifact, on CPU.

The drafter's program is vLLM PR #68296 at 8cbb80efa397 (`vllm/models/hy_v4/nvidia/dfly.py`,
`dfly_speculator.py`, `dfly_inputs.py`, `dfly_target_reuse.py`, `vllm/models/hy_v4/dfly_config.py`,
`vllm/models/hy_v4_flash/dfly_config.py`, and the decoder layer, attention and feed-forward it builds
on in `vllm/models/hy_v4/nvidia/{model,attention,moe}.py`). transformers PR #50903 does not run the
drafter (`HYV4FlashPreTrainedModel._keys_to_ignore_on_load_unexpected` drops `dfly_layers`). vLLM's
modules need CUDA, its custom ops and a KV-cache runtime, so this file transcribes the program onto
torch on CPU: every step below names the pinned lines it follows, `--vllm-src` must hold the pinned
files (their git blobs are checked), and the one piece of the pinned tree that runs on CPU as
written, the test suite's torch reference of the sparse index rows
(`tests/v1/spec_decode/test_dfly_prepare_inputs.py::_reference_reuse_indices`), is executed from
that file to build every query's key list.

One draft round from source position `p` (the last accepted target row), `k` drafts:

- the query block: ids `[anchor, mask x k]` at positions `p + 1 .. p + 1 + k`, where `anchor` is the
  token the target chose at `p` (`_prepare_dfly_inputs_kernel`: `input_ids = where(offsets == 0,
  bonus, mask_token_id)`, `positions = anchor + offsets`, `anchor = target_position + 1`);
- slot 0's embedding fused with the target's final hidden row at `p` (the head's input;
  `prepare_dfly_inputs` asserts "the target's final hidden states"): `e + gate * eh_proj(cat(enorm(e),
  hnorm(h)))` (`HYV4DFlyFusion.forward`, `HYV4DFlyQueryModel.forward`);
- `dfly_num_hidden_layers` decoder layers, single-stream residual (`HYV4DecoderLayer._forward_normal`,
  `enable_ihc` false), each a gated MLA with its learnable sink and `post_proj_norm`
  (`HYV4MLAAttention.forward`; no indexer: `dfly_external_topk`) and a dense SwiGLU without a clamp
  (`HYV4FeedForward`). Draft layer `i` attends, per block row `j`, the keys of
  `_reference_reuse_indices`: the target index layer's top-k row of `p` (its snapshot, read from
  the target KV layer's cache: the shared storage of `HYV4DFlyTargetReuse`), filtered to slots
  before `anchor - context_local_window`, the local window (0 here), then the block rows `0..j`,
  whose latent and rope key the draft layer itself writes. The MLA runs absorbed, as the sparse
  FlashMLA backend computes it: `q_lat = W_UK^T q_nope` per head, score `(q_lat . latent + q_rope .
  k_rope) * 256^-0.5`, the sink in the max and the denominator, output `W_UV (sum p * latent)`;
- `final_norm(hidden + residual)` (`HYV4DFlyModel.forward`), then for step `s = 0 .. k - 1`:
  `c = hidden[s + 1] + down(silu(gate) * up)(cat(hidden_norm(hidden[s + 1]), embed_norm(e(prev))))`
  (`HYV4DFlyHiddenCorrection`), the target's head (F32 there, float64 here) and the draft
  `argmax` (`greedy_argmax`), `prev` the anchor for step 0 and then each step's draft.

The MLP width is the checkpoint's (`HYV4DFlyModel.materialize_ffn` from the gate/up/down shapes;
the config's `dfly_intermediate_size`, when declared, must agree, and every layer must agree); the
derived config's `intermediate_size` default (the trunk's 13,824) is read only for dummy loads.

Backends (`--dtype`): `f64`, the oracle: every tensor dequantized exactly in float64 (NVFP4:
`e2m1 * e4m3 * weight_scale_2`; FP8: `e4m3 * weight_scale_inv` per 128x128 block), the target's
inputs as the f64 target oracle computed them, the f32 constants (`rms_norm_eps`) at their f32
values; `bf16x2a`, the band backend of the GPU route's class (`hy-v4-flash-bf16x2-activations`):
F32 arithmetic, the target's inputs rounded once to F32 (the route's cache), and the input of
every weight GEMM carried as two BF16 terms (`hi = bf16(x)`, `lo = bf16(x - hi)`), the absorbed
MLA's two batched products included, as the route computes them; the head stays F32 over F32
inputs. `--force-drafts TAPE` takes every step's `prev` from that tape (the f64 oracle's), so a
band backend is compared along the same drafts; its own argmax is still recorded.

usage: python -I make_dfly_oracle.py CKPT_DIR REQUEST_JSON TARGET_DIR OUT_DIR --vllm-src DIR
           [--dtype f64|bf16x2a] [--k 5] [--anchors all|LIST|SPEC] [--force-drafts TAPE] [--red ARM]
TARGET_DIR: the target oracle's `--dump-dfly` directory for the same request.
"""

import argparse
import ast
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np
import torch

T0 = time.time()


def log(*args):
    print(f"[{time.time() - T0:8.1f}s]", *args, file=sys.stderr, flush=True)


def blob(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


# the pinned files this transcription follows (vLLM PR #68296 head 8cbb80efa3974d10bb09c87580627bd874a3606b)
PINS = {
    "vllm/models/hy_v4/nvidia/dfly.py": "c7956b89e1b72cbb6a55143ba4123855369fd0de",
    "vllm/models/hy_v4/nvidia/dfly_speculator.py": "ec30895604bdce9e761252c0ae9bd9f02799656e",
    "vllm/models/hy_v4/nvidia/dfly_inputs.py": "573842f1186a2cd0e69d06e0f1f6336ed6f5ad0b",
    "vllm/models/hy_v4/nvidia/dfly_target_reuse.py": "757dd1bf10e71444a688dde972d344475a2f701d",
    "vllm/models/hy_v4/dfly_config.py": "54dfd4d9455ca25a89a0b0eeda2406ff94c0345e",
    "vllm/models/hy_v4_flash/dfly_config.py": "68533d2225cb62427cbb1038c5bdfab197181fd8",
    "vllm/v1/worker/gpu/spec_decode/dfly/config.py": "d69aa9ea56807e44f923679170ca16f6f2b073c1",
    "vllm/models/hy_v4/nvidia/model.py": "8eebff472fbf4522752b6d9bda566c028024cb5a",
    "vllm/models/hy_v4/nvidia/attention.py": "f4b15ebdc9e3d88610986806ddd1d420e8766a55",
    "vllm/models/hy_v4/nvidia/moe.py": "145dab1a4e4ec49471f4f23dc5d5fcc98a67f480",
    "vllm/models/hy_v4_flash/nvidia/compressed_indexer.py": "c4fccee4603750cda7ce1f549f4f841693b27d0b",
    "vllm/models/hy_v4_flash/nvidia/indexpool_ops.py": "10b60302109f6cee46347e480f40b6d7dbb875d4",
    "tests/v1/spec_decode/test_dfly_prepare_inputs.py": "6bd06a9f734241b9f86ef7153457cf5d33a53967",
}

RED_ARMS = {
    "layers": "every draft layer reads the next draft layer's target layer (29, 33, 37, 41, 25)",
    "width": "the MLP at half its width (the first 4,608 channels of gate, up and down)",
    "no-correction": "hidden correction off: the head reads hidden[s + 1] itself",
    "topk": "dfly_target_index_topk 1024: the history keeps the first 1,024 slots of the target's row",
}

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("request")
ap.add_argument("target")
ap.add_argument("out")
ap.add_argument("--vllm-src", required=True)
ap.add_argument("--dtype", choices=["f64", "bf16x2a"], default="f64")
ap.add_argument("--k", type=int, default=0, help="drafts a round (default: speculative_config.num_speculative_tokens)")
ap.add_argument("--anchors", default="all",
                help="source positions: all, a comma list, or a spec 'stride:S,range:A-B,at:P'")
ap.add_argument("--force-drafts", default="")
ap.add_argument("--red", choices=sorted(RED_ARMS), default="")
a = ap.parse_args()

for rel, pin in PINS.items():
    path = os.path.join(a.vllm_src, rel)
    assert blob(path) == pin, f"{rel} is not the pinned vLLM revision"

F64 = a.dtype == "f64"
DT = torch.float64 if F64 else torch.float32
torch.set_grad_enabled(False)

# ---- the pinned test's torch reference of the sparse index rows ------------------------------------
test_src = open(os.path.join(a.vllm_src, "tests/v1/spec_decode/test_dfly_prepare_inputs.py")).read()
tree = ast.parse(test_src)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_reference_reuse_indices")
REUSE_NS = {"torch": torch}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "test_dfly_prepare_inputs.py", "exec"), REUSE_NS)


def reuse_rows(snapshot_row, anchor, local_window, k):
    """`_reference_reuse_indices` of the pinned test for one request: the key list of each block row."""
    width = snapshot_row.numel() + local_window + k + 1
    REUSE_NS["QUERY_WIDTH"] = k + 1
    rows = REUSE_NS["_reference_reuse_indices"](
        snapshot_row[None].to(torch.int32), torch.tensor([anchor]), local_window, torch.tensor([True]), width)
    return [r[r >= 0].to(torch.int64) for r in rows]


# ---- the artifact ----------------------------------------------------------------------------------
top = json.load(open(os.path.join(a.checkpoint, "config.json")))
text = top["text_config"]
index_path = os.path.join(a.checkpoint, "model.safetensors.index.json")
if os.path.exists(index_path):
    index = json.load(open(index_path))["weight_map"]
else:
    index = {}
    for f in sorted(os.listdir(a.checkpoint)):
        if f.endswith(".safetensors"):
            with open(os.path.join(a.checkpoint, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                for name in json.loads(fh.read(n)):
                    if name != "__metadata__":
                        assert name not in index, name
                        index[name] = f
NP = {"BF16": np.uint16, "F32": np.float32, "I32": np.int32, "F8_E4M3": np.uint8, "U8": np.uint8}
SHARDS = {}


def shard(file):
    if file not in SHARDS:
        path = os.path.join(a.checkpoint, file)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        SHARDS[file] = (np.memmap(path, dtype=np.uint8, mode="r"), 8 + n, header)
    return SHARDS[file]


def header(name):
    return shard(index[name])[2][name]


def stored(name):
    mm, base, hdr = shard(index[name])
    h = hdr[name]
    s, e = h["data_offsets"]
    t = torch.from_numpy(np.ascontiguousarray(mm[base + s: base + e].view(NP[h["dtype"]]).reshape(h["shape"])))
    if h["dtype"] == "BF16":
        return t.view(torch.bfloat16)
    if h["dtype"] == "F8_E4M3":
        return t.view(torch.float8_e4m3fn)
    return t


E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
                    dtype=torch.float64)


def weight(stem):
    """A drafter weight dequantized exactly in float64: BF16 as stored, FP8 E4M3 times its 128x128
    block `weight_scale_inv`, NVFP4 (ModelOpt: e2m1 pairs low nibble first, an E4M3 scale per 16
    inputs, the tensor's F32 `weight_scale_2`)."""
    name = stem + ".weight" if not stem.endswith((".gate", "learnable_sink_param")) else stem
    h = header(name)
    if h["dtype"] in ("BF16", "F32"):
        return stored(name).to(torch.float64)
    if h["dtype"] == "F8_E4M3":
        value = stored(name).to(torch.float64)
        grid = stored(stem + ".weight_scale_inv").to(torch.float64)
        full = grid.repeat_interleave(128, 0).repeat_interleave(128, 1)[: value.shape[0], : value.shape[1]]
        return value * full
    assert h["dtype"] == "U8", (name, h["dtype"])
    packed = stored(name).to(torch.int64)
    codes = torch.stack([E2M1[packed & 0x0F], E2M1[packed >> 4]], dim=-1).flatten(-2)
    scale = stored(stem + ".weight_scale").to(torch.float64).repeat_interleave(16, dim=-1)
    scale_2 = stored(stem + ".weight_scale_2").to(torch.float64)
    assert scale_2.numel() == 1
    return codes * scale * scale_2


# ---- the derived draft config (`_derive_dfly_config`, `HYV4FlashDFlySpeculatorConfig`) --------------
num_layers = top["dfly_num_hidden_layers"]
assert type(num_layers) is int and num_layers >= 1
assert top.get("dfly_mtp_fusion", "slot0") == "slot0" and top.get("dfly_enable_hidden_correction", True) is True
assert top.get("dfly_mlp_type", "dense") == "dense" and top.get("dfly_enable_ihc", False) is False
assert top.get("dfly_attn_causal", True) is True
mask_token = top["dfly_mask_token_id"]
assert 0 <= mask_token < text["vocab_size"]
target_topk = top.get("dfly_target_index_topk", text["index_topk"])
assert target_topk == text["index_topk"], "DFly target_index_topk disagrees with the target"
local_window = top.get("dfly_context_local_window") or 0
index_layers = top["dfly_target_index_layer"]
if type(index_layers) is int:
    index_layers = [index_layers] * num_layers
kv_layers = top["dfly_target_kv_layers"]
assert len(index_layers) == len(kv_layers) == num_layers and len(set(kv_layers)) == num_layers
for layer in index_layers + kv_layers:
    assert 1 <= layer < text["num_hidden_layers"]
    assert text["layer_types"][layer] == "deepseek_sparse_attention", "DFly target layers are sparse"
rope = text["rope_parameters"]["deepseek_sparse_attention"]
assert rope["rope_type"] == "default"
theta = float(rope["rope_theta"])
declared_width = top.get("dfly_intermediate_size")
hc_width = top.get("dfly_hidden_correction_intermediate_size", text["hidden_size"])
spec = top.get("speculative_config") or {}
K = a.k or spec.get("num_speculative_tokens") or 0
assert 1 <= K <= 16, "DFly requires num_speculative_tokens in [1, 16]"
H = text["hidden_size"]
HEADS = text["num_attention_heads"]
NOPE, ROPE, VDIM = text["qk_nope_head_dim"], text["qk_rope_head_dim"], text["v_head_dim"]
QL, KVL = text["q_lora_rank"], text["kv_lora_rank"]
EPS = float(np.float32(text["rms_norm_eps"])) if F64 else text["rms_norm_eps"]
if a.red == "layers":
    index_layers = index_layers[1:] + index_layers[:1]
    kv_layers = kv_layers[1:] + kv_layers[:1]
if a.red == "topk":
    target_topk = 1024

# ---- weights (`HYV4DFlyForCausalLM.load_weights` renames: `<i>.` -> `layers.<i>.`, `final_norm` -> `norm`,
# `confidence_head.*` ignored) ---------------------------------------------------------------------------
P = "model.dfly_layers."
widths = set()


def to_backend(w):
    return w if F64 else w.to(torch.float32)


layers = []
for i in range(num_layers):
    p = f"{P}{i}."
    w = {k: to_backend(weight(p + k)) for k in [
        "input_layernorm", "post_attention_layernorm", "self_attn.q_a_proj", "self_attn.q_a_layernorm",
        "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_a_layernorm", "self_attn.kv_b_proj",
        "self_attn.linear_gate", "self_attn.o_proj", "self_attn.post_proj_norm", "mlp.gate_proj", "mlp.up_proj",
        "mlp.down_proj"]}
    w["sink"] = to_backend(weight(p + "self_attn.learnable_sink_param"))
    # `materialize_ffn`: the width from the checkpoint's shapes, one width for every layer
    g, u, d = w["mlp.gate_proj"], w["mlp.up_proj"], w["mlp.down_proj"]
    assert g.shape[1] == u.shape[1] == H and d.shape[0] == H and g.shape[0] == u.shape[0] == d.shape[1]
    widths.add(g.shape[0])
    if a.red == "width":
        half = g.shape[0] // 2
        w["mlp.gate_proj"], w["mlp.up_proj"], w["mlp.down_proj"] = g[:half], u[:half], d[:, :half]
    kvb = w["self_attn.kv_b_proj"].view(HEADS, NOPE + VDIM, KVL)
    w["W_UK"], w["W_UV"] = kvb[:, :NOPE, :], kvb[:, NOPE:, :]
    layers.append(w)
assert len(widths) == 1, "Inconsistent DFly FFN widths in checkpoint"
WIDTH = widths.pop()
assert declared_width in (None, WIDTH), "dfly_intermediate_size disagrees with checkpoint"
glob = {k: to_backend(weight(P + k)) for k in [
    "final_norm", "mtp_fusion.eh_proj", "mtp_fusion.enorm", "mtp_fusion.hnorm", "mtp_fusion.gate",
    "hidden_correction.gate_proj", "hidden_correction.up_proj", "hidden_correction.down_proj",
    "hidden_correction.embed_norm", "hidden_correction.hidden_norm"]}
assert glob["hidden_correction.gate_proj"].shape == (hc_width, 2 * H)
embed_table = stored("model.embed_tokens.weight")
lm_head = to_backend(weight("lm_head"))
log(f"drafter: {num_layers} layers, MLP width {WIDTH} (declared {declared_width}), k={K}, target index "
    f"{index_layers} kv {kv_layers}, topk {target_topk}, window {local_window}, mask {mask_token}, {a.dtype}"
    + (f", red arm {a.red}: {RED_ARMS[a.red]}" if a.red else ""))


def embed(ids):
    return to_backend(embed_table[torch.tensor(ids)].to(torch.float64))


# ---- the backend's arithmetic -----------------------------------------------------------------------
def a16(x):
    if F64:
        return x
    hi = x.to(torch.bfloat16).to(torch.float32)
    return hi + (x - hi).to(torch.bfloat16).to(torch.float32)


def linear(x, w):
    return a16(x) @ w.T


def rms(x, wt):
    # vLLM RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * wt


def rope(x, positions):
    """`get_rope(is_neox_style=False)` on the trailing dims: adjacent pairs rotated by pos * theta^(-2i/d)."""
    d = x.shape[-1]
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    ang = positions.to(torch.float64)[:, None] * inv[None, :]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    while cos.dim() < x.dim():
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x2 * cos + x1 * sin
    return out


def silu(x):
    return x * torch.sigmoid(x)


# ---- the target's inputs ----------------------------------------------------------------------------
req = json.load(open(a.request))
tokens = req["token_ids"]
tgt = json.load(open(os.path.join(a.target, "target.json")))
assert tgt["request"] == req["name"] and tgt["positions"] == len(tokens)
T = len(tokens)


def target_array(meta):
    data = np.fromfile(os.path.join(a.target, meta["file"]), dtype="<f8")
    assert hashlib.sha256(data.tobytes()).hexdigest() == meta["sha256"], meta["file"]
    return torch.from_numpy(data.reshape(meta["shape"]))


hidden_rows = target_array(tgt["hidden"])
target_kv, target_hist = {}, {}
for m in tgt["layers"]:
    target_kv[m["layer"]] = target_array(m)
    flat = np.fromfile(os.path.join(a.target, m["history_file"]), dtype="<i4")
    assert hashlib.sha256(flat.tobytes()).hexdigest() == m["history_sha256"]
    offs = np.fromfile(os.path.join(a.target, m["history_offsets_file"]), dtype="<i8")
    target_hist[m["layer"]] = (torch.from_numpy(flat.astype(np.int64)), offs)
for layer in set(index_layers) | set(kv_layers):
    assert layer in target_kv, f"the target dump has no layer {layer}"
if not F64:
    hidden_rows = hidden_rows.to(torch.float32)
    target_kv = {k: v.to(torch.float32) for k, v in target_kv.items()}


def snapshot(layer, p):
    """The target index layer's top-k row of position `p` (`topk_snapshot`): its attended slots in
    ascending order (`expand_groups` compacts them), padded with -1 to `target_topk`."""
    flat, offs = target_hist[layer]
    keys = flat[offs[p]: offs[p + 1]]
    row = torch.full((max(target_topk, keys.numel()),), -1, dtype=torch.int64)
    row[: keys.numel()] = keys
    return row[:target_topk], keys


# ---- one draft round --------------------------------------------------------------------------------
def draft(p, forced):
    anchor_tok = tokens[p + 1]
    pos = torch.arange(p + 1, p + 2 + K)
    ids = [anchor_tok] + [mask_token] * K
    e = embed(ids)
    f = glob
    fused = linear(torch.cat([rms(e[:1], f["mtp_fusion.enorm"]), rms(hidden_rows[p][None], f["mtp_fusion.hnorm"])],
                             -1), f["mtp_fusion.eh_proj"])
    e[:1] = e[:1] + f["mtp_fusion.gate"] * fused
    hidden, residual = e, None
    digests = []
    for i, w in enumerate(layers):
        if residual is not None:
            hidden = hidden + residual
        residual = hidden
        x = rms(hidden, w["input_layernorm"])
        q = linear(rms(linear(x, w["self_attn.q_a_proj"]), w["self_attn.q_a_layernorm"]), w["self_attn.q_b_proj"])
        q = q.view(K + 1, HEADS, NOPE + ROPE)
        kv = linear(x, w["self_attn.kv_a_proj_with_mqa"])
        latent = rms(kv[:, :KVL], w["self_attn.kv_a_layernorm"])
        k_rope = rope(kv[:, KVL:], pos)
        q_rope = rope(q[..., NOPE:], pos)
        # the absorbed query: q_lat[t, h] = W_UK[h]^T q_nope[t, h]
        q_lat = torch.einsum("thn,hnl->thl", a16(q[..., :NOPE]), w["W_UK"])
        snap, attended = snapshot(index_layers[i], p)
        rows = reuse_rows(snap, p + 1, local_window, K)
        cache_l = target_kv[kv_layers[i]]
        out_lat = torch.empty(K + 1, HEADS, KVL, dtype=DT)
        for j, keys in enumerate(rows):
            hist = keys[keys <= p]
            blk = keys[keys > p] - (p + 1)
            assert bool((blk == torch.arange(j + 1)).all()), "the causal block is rows 0..j"
            klat = torch.cat([cache_l[hist, :KVL], latent[blk]], 0)
            krope = torch.cat([cache_l[hist, KVL:], k_rope[blk]], 0)
            s = (torch.einsum("hl,sl->hs", q_lat[j], klat) + torch.einsum("hr,sr->hs", q_rope[j], krope)) \
                * (NOPE + ROPE) ** -0.5
            sink = w["sink"][:, None]
            mx = torch.maximum(s.max(-1, keepdim=True).values, sink)
            pr = torch.exp(s - mx)
            den = pr.sum(-1, keepdim=True) + torch.exp(sink - mx)
            out_lat[j] = (pr / den) @ klat
            if j == K:
                digests.append(hashlib.sha256(hist.numpy().astype("<i4").tobytes()).hexdigest()[:16])
        o = torch.einsum("thl,hvl->thv", a16(out_lat), w["W_UV"]).reshape(K + 1, HEADS * VDIM)
        o = o * torch.sigmoid(linear(x, w["self_attn.linear_gate"]))
        hidden = rms(linear(o, w["self_attn.o_proj"]), w["self_attn.post_proj_norm"])
        hidden = hidden + residual
        residual = hidden
        x = rms(hidden, w["post_attention_layernorm"])
        hidden = linear(silu(linear(x, w["mlp.gate_proj"])) * linear(x, w["mlp.up_proj"]), w["mlp.down_proj"])
    out = rms(hidden + residual, f["final_norm"])
    prev = anchor_tok
    logits, own = [], []
    for s in range(K):
        hrow = out[s + 1][None]
        if a.red == "no-correction":
            c = hrow
        else:
            comb = torch.cat([rms(hrow, f["hidden_correction.hidden_norm"]),
                              rms(embed([prev]), f["hidden_correction.embed_norm"])], -1)
            c = hrow + linear(silu(linear(comb, f["hidden_correction.gate_proj"]))
                              * linear(comb, f["hidden_correction.up_proj"]), f["hidden_correction.down_proj"])
        row = (c @ lm_head.T)[0]  # the target's head: F32 over F32 inputs (f64 in the oracle)
        lg = row.to(torch.float32)
        logits.append(lg)
        top = int(torch.argmax(lg))  # greedy_argmax: the first of equal maxima
        own.append(top)
        prev = forced[s] if forced is not None else top
    return torch.stack(logits), own, digests


# ---- the anchors ------------------------------------------------------------------------------------
def parse_anchors(spec):
    last = T - 2  # a source position needs the token after it
    if spec == "all":
        return list(range(0, last + 1))
    out = set()
    for part in spec.split(","):
        if part.startswith("stride:"):
            out |= set(range(0, last + 1, int(part[7:])))
        elif part.startswith("range:"):
            lo, hi = part[6:].split("-")
            out |= set(range(int(lo), min(int(hi), last) + 1))
        elif part.startswith("at:"):
            out.add(int(part[3:]))
        else:
            out.add(int(part))
    return sorted(p for p in out if 0 <= p <= last)


anchors = parse_anchors(a.anchors)
forced_tape = None
if a.force_drafts:
    ft = json.load(open(os.path.join(a.force_drafts, "tape.json")))
    assert ft["request"] == req["name"] and ft["k"] == K and not ft["forced_drafts"]
    assert ft["anchors"] == anchors, "the forcing tape drafted other anchors"
    forced_tape = ft["drafts"]

os.makedirs(a.out, exist_ok=True)
lf = open(os.path.join(a.out, "logits.f32"), "wb")
lsha = hashlib.sha256()
drafts, hist_digests = [], []
for n, p in enumerate(anchors):
    logits, own, dig = draft(p, forced_tape[n] if forced_tape is not None else None)
    data = logits.numpy().astype("<f4").tobytes()
    lf.write(data)
    lsha.update(data)
    drafts.append(own)
    hist_digests.append(dig)
    if n % 25 == 0 or n == len(anchors) - 1:
        log(f"anchor {n + 1}/{len(anchors)} p={p} drafts={own} target_next={tokens[p + 2: p + 2 + K]}")
lf.close()

# acceptance of the drafts against the request's own continuation (teacher-forced: the request's
# tokens are what the target produced)
accepted = []
for n, p in enumerate(anchors):
    m = 0
    while m < K and p + 2 + m < T and drafts[n][m] == tokens[p + 2 + m]:
        m += 1
    accepted.append(m)
tape = {
    "format": "memra-hy4f-dfly-tape-v1",
    "engine": f"vllm-pr68296@8cbb80efa397-transcribed-cpu-{a.dtype}",
    "backend_dtype": a.dtype,
    "red_arm": a.red or None,
    "request": req["name"],
    "token_ids": tokens,
    "k": K,
    "vocab": int(lm_head.shape[0]),
    "mlp_width": WIDTH,
    "mask_token": mask_token,
    "target_index_layers": index_layers,
    "target_kv_layers": kv_layers,
    "target_index_topk": target_topk,
    "anchors": anchors,
    "drafts": drafts,
    "history_digests": hist_digests,
    "accepted_teacher_forced": accepted,
    "logits": {"file": "logits.f32", "sha256": lsha.hexdigest(), "shape": [len(anchors), K, int(lm_head.shape[0])]},
    "forced_drafts": bool(a.force_drafts),
    "target": {"hidden_sha256": tgt["hidden"]["sha256"], "layers": [(m["layer"], m["sha256"], m["history_sha256"])
                                                                    for m in tgt["layers"]]},
    "vllm_pins": PINS,
    "torch": torch.__version__,
    "threads": torch.get_num_threads(),
    "seconds": round(time.time() - T0, 1),
}
json.dump(tape, open(os.path.join(a.out, "tape.json"), "w"), indent=1)
mean = sum(accepted) / max(1, len(accepted))
print(json.dumps({"request": req["name"], "dtype": a.dtype, "red": a.red or None, "anchors": len(anchors), "k": K,
                  "logits_sha256": lsha.hexdigest(), "mean_accepted_teacher_forced": round(mean, 4),
                  "seconds": tape["seconds"]}))
