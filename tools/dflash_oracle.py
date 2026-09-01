#!/usr/bin/env python3
"""DFlash draft-forward oracle (bring-up step 2, DFLASH-BRINGUP-PLAN.md).

Runs the z-lab reference DFlashDraftModel on the real backbone-only checkpoint with
FIXED-SEED synthetic inputs (target_hidden + noise_embedding stand-ins), dumping every
layer's intermediates. The memra loader+forward must reproduce the final hidden states
(and intermediates, for bisecting) bit-close (f32 rel ~1e-3 class vs bf16 reference).

Synthetic inputs isolate the DRAFT math from target plumbing: target taps / embed /
lm_head are memra-native pieces gated separately.

Usage (in ~/.venvs/torch):
  python tools/dflash_oracle.py /data/ai-ml/hf-models/dspark-gemma4-31b-draft/backbone-only \
      /data/cache/dflash-oracle.npz
"""
import sys, json, numpy as np, torch

ckpt_dir, out_path = sys.argv[1], sys.argv[2]
sys.path.insert(0, "/data/projects/dflash")
from dflash.model import DFlashDraftModel  # noqa: E402
from transformers import Qwen3Config  # noqa: E402

cfg = json.load(open(f"{ckpt_dir}/config.json"))
config = Qwen3Config(**{k: v for k, v in cfg.items() if k != "architectures"})
config.num_target_layers = 62  # gemma-4-31B layer count (only used if target_layer_ids absent)
config._attn_implementation = "eager"

torch.manual_seed(0)
model = DFlashDraftModel(config)
from safetensors.torch import load_file  # noqa: E402
sd = load_file(f"{ckpt_dir}/model.safetensors")
missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", missing, "unexpected:", unexpected)
model = model.to(torch.float32).eval()   # f32 reference (memra math is f32 accumulate)

H = config.hidden_size
NT = len(model.target_layer_ids)
CTX, BLK = 8, config.block_size
g = torch.Generator().manual_seed(42)
# scale ~1.0 like real residual-stream features
target_hidden = torch.randn(1, CTX, NT * H, generator=g, dtype=torch.float32)
noise_embedding = torch.randn(1, BLK, H, generator=g, dtype=torch.float32)
position_ids = torch.arange(0, CTX + BLK).unsqueeze(0)

with torch.inference_mode():
    # replicate DFlashDraftModel.forward but keep intermediates
    ctx = model.hidden_norm(model.fc(target_hidden))
    pos_emb = model.rotary_emb(noise_embedding, position_ids)
    inter = {"ctx_features": ctx.numpy()}
    h = noise_embedding
    from transformers.cache_utils import DynamicCache
    kv = DynamicCache()
    for i, layer in enumerate(model.layers):
        h = layer(hidden_states=h, target_hidden=ctx, attention_mask=None,
                  position_ids=position_ids, past_key_value=kv, use_cache=True,
                  cache_position=None, position_embeddings=pos_emb, is_causal=False)
        inter[f"layer{i}_out"] = h.numpy()
    out = model.norm(h)
    inter["final"] = out.numpy()

# $d is needed by the manifest AND the flat dumps, so it is resolved before either.
import os
d = os.path.dirname(out_path) or "."

# ---- GEOMETRY MANIFEST (GATE-INTEGRITY-20260819 §5) ----
#
# The flat dumps are naked f32 with no shape and no provenance. dflash_parity used to RECOVER
# the context length from their byte count — `ctx = th.len() / (n_taps * hidden)` — so a
# reference regenerated under a different hidden size or tap set was indistinguishable from a
# correct one, and the value compare proceeded against a reinterpreted buffer. Worse, the
# geometry that lives only in config scalars (`head_dim`, `rope_theta`, `sliding_window`,
# per-layer sliding) never reaches the bytes at all: the dumps are byte-identical under a
# rotary width that is wrong by 4x. That is the defect an n_rot lane found surviving a
# byte-parity gate at maxdiff=0.0e0 on 2026-08-19.
#
# So the producer records what it produced UNDER, in the same flat key=value shape the Rust side
# parses without a dependency, and the gate refuses a dump whose manifest disagrees with the
# checkpoint it is testing. `layer_types` is read the way the loader reads it (substring match
# for "sliding_attention", in order) so the two derivations cannot drift.
#
# EVERY scalar below is read from the checkpoint's config.json (`cfg`), NOT from the
# transformers config OBJECT. Two reasons, one of which is a receipt:
#
#   * config.json is what the CONSUMER reads — crates/memra-engine's loader and
#     parity_geometry.rs compare against the checkpoint's own JSON. Going through
#     Qwen3Config puts transformers' defaulting and renaming between the two sides of a
#     comparison whose whole purpose is that the two sides agree.
#   * it broke, exactly there. This block was `repr(float(config.rope_theta))`, and
#     transformers 5.x moved rope config off the attribute (into `rope_parameters`), so on the
#     only torch environment on the rig the oracle wrote every flat dump, printed "flat dumps
#     written to <dir>", and THEN died with
#     `AttributeError: 'Qwen3Config' object has no attribute 'rope_theta'` — leaving a full set
#     of dumps with no manifest, which is precisely the state the gate refuses. Following the
#     regen hint the gate itself prints would have overwritten the 2026-07-13 reference in place
#     and produced that state. Measured 2026-08-20.
#
# A missing key is a REFUSAL, never a default: a manifest field guessed by the producer is a
# field the consumer cannot use to disagree with anything.
def _cfg_scalar(key):
    if key not in cfg:
        raise SystemExit(
            f"dflash_oracle: {ckpt_dir}/config.json has no '{key}'. The geometry manifest "
            "records what this run computed UNDER, so a guessed value would make the parity "
            "gate agree with a fiction. Refusing to write a manifest with a hole in it."
        )
    return cfg[key]


import transformers as _tf  # noqa: E402  (recorded as provenance, see below)

_layer_types = cfg.get("layer_types") or []
_manifest = {
    "producer": "tools/dflash_oracle.py",
    "ckpt_dir": ckpt_dir,
    "dtype": "f32",
    # THE LIBRARY VERSIONS ARE PART OF THE GEOMETRY, and that is measured rather than argued.
    # Regenerating this reference on 2026-08-20 under torch 2.13.0+cpu / transformers 5.15.0 and
    # comparing against the 2026-07-13 set: the four dumps dflash_parity actually reads
    # (target_hidden, noise_embedding, ctx_features, final) and every layer output agree to
    # rel <= 1.1e-6 — f32 accumulation noise, inside this oracle's documented tolerance class.
    # But the layer-0 BISECT dumps do not: l0_q and l0_k (POST-rope) differ at rel ~1.1-1.6
    # while l0_q0 and l0_qn (PRE-rope) are identical to 4e-7. That is a rope-convention change
    # between transformers 4.x and 5.x, and it means those four handles measure something
    # different depending on the library — invisible in every byte count and in every config
    # scalar. Corollary 3 of the parity rule ("provenance is part of the geometry") therefore
    # includes the library that implemented the rotation.
    # The Rust reader looks fields up by key and ignores ones it does not know, so these are
    # additive: they are for the human bisecting a mismatch.
    "torch_version": torch.__version__,
    "transformers_version": _tf.__version__,
    "hidden": int(_cfg_scalar("hidden_size")),
    "n_layer": int(_cfg_scalar("num_hidden_layers")),
    "block_size": BLK,
    "ctx": CTX,
    "n_taps": NT,
    "target_layer_ids": ",".join(str(int(x)) for x in model.target_layer_ids),
    "head_dim": int(_cfg_scalar("head_dim")),
    "n_head": int(_cfg_scalar("num_attention_heads")),
    "n_head_kv": int(_cfg_scalar("num_key_value_heads")),
    "rope_theta": repr(float(_cfg_scalar("rope_theta"))),
}
if cfg.get("sliding_window") is not None:
    _manifest["sliding_window"] = int(cfg["sliding_window"])
if _layer_types:
    _manifest["layer_sliding"] = ",".join(
        "1" if "sliding_attention" in str(t) else "0" for t in _layer_types
    )
# Shapes are recorded too: they are what a human reads when a length assertion fires.
for _name, _shape in [
    ("target_hidden", (CTX, NT * H)),
    ("noise_embedding", (BLK, H)),
    ("ctx_features", (CTX, H)),
    ("final", (BLK, H)),
]:
    _manifest[f"shape.{_name}"] = ",".join(str(int(x)) for x in _shape)
_manifest_path = f"{d}/dflash-geometry.txt"
with open(_manifest_path, "w") as _fh:
    _fh.write("# geometry manifest for the dflash-*.f32 dumps in this directory.\n")
    _fh.write("# Written by tools/dflash_oracle.py; asserted by dflash_parity before any\n")
    _fh.write("# value compare (crates/memra-engine/src/parity_geometry.rs).\n")
    for _k, _v in _manifest.items():
        _fh.write(f"{_k}={_v}\n")
print(f"geometry manifest -> {_manifest_path} ({len(_manifest)} fields)")

np.savez(out_path,
         target_hidden=target_hidden.numpy(),
         noise_embedding=noise_embedding.numpy(),
         position_ids=position_ids.numpy(),
         **inter)
print(f"saved {out_path}: final shape {out.shape}, |final| mean {out.abs().mean():.4f}")

# flat f32 dumps for the Rust parity bin (row-major, little-endian)
for name, arr in [("target_hidden", target_hidden), ("noise_embedding", noise_embedding),
                  ("ctx_features", torch.from_numpy(inter["ctx_features"])),
                  ("final", out)]:
    a = (arr.numpy() if hasattr(arr, "numpy") else arr).astype(np.float32)
    a.tofile(f"{d}/dflash-{name}.f32")
for i in range(len(model.layers)):
    inter[f"layer{i}_out"].astype(np.float32).tofile(f"{d}/dflash-layer{i}_out.f32")
print("flat dumps written to", d)


# ---- layer-0 sub-stage dumps (parity bisect) ----
import torch.nn.functional as F
import os
from dflash.model import apply_rotary_pos_emb
l0 = model.layers[0]
with torch.inference_mode():
    hs = noise_embedding
    ctx_t = torch.from_numpy(inter["ctx_features"])
    xn = l0.input_layernorm(hs)
    xn.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_xn.f32")
    a = l0.self_attn
    bsz, q_len = xn.shape[:-1]; ctx_len = ctx_t.shape[1]
    q0 = a.q_proj(xn)
    q0.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_q0.f32")
    q = q0.view(bsz, q_len, -1, a.head_dim)
    q = a.q_norm(q)
    q.reshape(1, q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_qn.f32")
    q = q.transpose(1, 2)
    k_ctx = a.k_proj(ctx_t); k_noise = a.k_proj(xn)
    v_ctx = a.v_proj(ctx_t); v_noise = a.v_proj(xn)
    k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, a.head_dim)
    v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, a.head_dim)
    k = a.k_norm(k).transpose(1, 2); v = v.transpose(1, 2)
    if not os.environ.get("DFLASH_NOROPE"):
        cos, sin = model.rotary_emb(hs, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    q.transpose(1,2).reshape(1, q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_q.f32")
    k.transpose(1,2).reshape(1, ctx_len+q_len, -1).numpy().astype(np.float32).tofile(f"{d}/dflash-l0_k.f32")
    # eager GQA attention, non-causal, no mask
    kk = k.repeat_interleave(a.num_key_value_groups, dim=1)
    vv = v.repeat_interleave(a.num_key_value_groups, dim=1)
    att = torch.softmax((q @ kk.transpose(-1, -2)) * a.scaling, dim=-1)
    ao = (att @ vv).transpose(1, 2).reshape(bsz, q_len, -1)
    ao.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_attn.f32")
    o = a.o_proj(ao)
    x1 = hs + o
    x1.numpy().astype(np.float32).tofile(f"{d}/dflash-l0_x1.f32")
print("layer0 sub-stages dumped")
