#!/usr/bin/env python3
"""DFlash2 drafter oracle (lane/dflash2-port-20260820).

Runs the z-lab REFERENCE `DFlash2DraftModel` (dflash/model.py — the semantic program
the checkpoint was trained against; sha-pinned in the darklanes census,
DFLASH2-EVAL-20260820.md §1) on the q38 DFlash2 export with FIXED-SEED synthetic
inputs, dumping every stage the memra port must reproduce:

  stage 1  ctx_features   = hidden_norm(fc(taps))                    [CTX, H]
  stage 2  layer{i}_out   per-layer residual block rows (conv-wrapped
           attention + conv-wrapped mlp, symmetric-window mask)       [BLK, H]
  stage 3  final          = norm(x)                                   [BLK, H]
  stage 3w final_win      = full forward at a REDUCED window (win_window,
           ctx win_ctx > window) — exercises the per-query old-side mask
           arithmetic without a 2048-deep fixture                     [BLK, H]
  stage 4  conv isolation layer-0 attention_conv prepare/finish on
           seeded rows (prepare_out, finish_out)                      [BLK, H] x2
  stage 5  selector       candidate top-k + greedy path walk on seeded
           logits/hidden (path EXACT is the bar; candidates + unary
           dumped so the gate pins the top-k too)                     [NDR], [NDR,K]

HARVEST: DFlash2 is a MASK-FILL-family checkpoint (b-1 drafts, anchor row is not a
draft — reference dflash_generate takes rows `1 - verify_size:`); the manifest
records harvest=dflash and the gate refuses anything else.

Census discipline: load_state_dict must report ZERO missing / ZERO unexpected keys
(81-tensor export; the two codebooks are stored WITHOUT `.weight` — renamed here
exactly like the reference from_pretrained key_mapping).

Requires transformers >= 5 (the checkpoint's rope_parameters config style — probed,
fail-closed). Reference model.py is passed by PATH and imported standalone.

Usage:
  python tools/dflash2_oracle.py <export_dir> <out.npz> <model_py_path>
"""

import hashlib
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

if len(sys.argv) < 4:
    sys.exit(__doc__)
ckpt_dir, out_path, model_py = sys.argv[1], sys.argv[2], sys.argv[3]

import transformers  # noqa: E402

_tv = transformers.__version__
if int(_tv.split(".")[0]) < 5:
    sys.exit(
        f"transformers {_tv} < 5: the DFlash2 export is transformers-5 config style "
        "(nested rope_parameters); a v4 Qwen3Config silently mis-reads rope_theta "
        "(the known divergence class) — refusing."
    )

# ---- import the pinned reference implementation by path ----
_spec = importlib.util.spec_from_file_location("zlab_dflash_model", model_py)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["zlab_dflash_model"] = _mod
_spec.loader.exec_module(_mod)
DFlash2DraftModel = _mod.DFlash2DraftModel
_model_py_sha = hashlib.sha256(open(model_py, "rb").read()).hexdigest()

from transformers.models.qwen3.modeling_qwen3 import Qwen3Config  # noqa: E402

raw = json.load(open(f"{ckpt_dir}/config.json"))
assert raw["architectures"] == ["DFlash2DraftModel"], raw["architectures"]
config = Qwen3Config(**{k: v for k, v in raw.items() if k not in ("architectures",)})
config._attn_implementation = "eager"

torch.manual_seed(0)
model = DFlash2DraftModel(config)
from safetensors.torch import load_file  # noqa: E402

sd = load_file(f"{ckpt_dir}/model.safetensors")
# the reference from_pretrained key_mapping, applied verbatim
_map = {
    f"candidate_selector.{n}": f"candidate_selector.{n}.weight"
    for n in ("predecessor_codebook", "successor_codebook")
}
sd = {_map.get(k, k): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(f"CENSUS FAIL missing={missing} unexpected={unexpected}")
    sys.exit(1)
print(f"census OK: {len(sd)} tensors, zero missing / zero unexpected")
model = model.to(torch.float32).eval()

H = config.hidden_size
V = config.vocab_size
NT = len(model.target_layer_ids)
BLK = model.block_size
CTX = 8
NDR = BLK - 1  # mask-fill: drafts = rows 1..b-1
d2cfg = raw["dflash_config"]
RANK = int(d2cfg["selector_rank"])
TOPK = int(d2cfg["selector_top_k"])

g = torch.Generator().manual_seed(42)
taps = torch.randn(1, CTX, NT * H, generator=g)
noise = torch.randn(1, BLK, H, generator=g)
sel_logits = torch.randn(1, NDR, V, generator=g)  # lm_head stand-in for the selector
sel_hidden = torch.randn(1, NDR, H, generator=g)  # final-hidden stand-ins
conv_x = torch.randn(1, BLK, H, generator=g)  # conv-isolation prepare input
conv_y = torch.randn(1, BLK, H, generator=g)  # conv-isolation finish input
anchor = torch.tensor([17], dtype=torch.long)
WIN_CTX = 24
WIN_WINDOW = 16
taps_win = torch.randn(1, WIN_CTX, NT * H, generator=g)

dump: dict[str, np.ndarray] = {
    "taps": taps.numpy().reshape(CTX, NT * H),
    "noise": noise.numpy().reshape(BLK, H),
    "sel_logits": sel_logits.numpy().reshape(NDR, V),
    "sel_hidden": sel_hidden.numpy().reshape(NDR, H),
    "conv_x": conv_x.numpy().reshape(BLK, H),
    "conv_y": conv_y.numpy().reshape(BLK, H),
    "anchor": anchor.numpy(),
    "taps_win": taps_win.numpy().reshape(WIN_CTX, NT * H),
}

with torch.inference_mode():
    # stage 1: ctx features
    ctx_f = model.hidden_norm(model.fc(taps))
    dump["ctx_features"] = ctx_f.numpy().reshape(CTX, H)

    # stages 2-3: block forward (conv-wrapped layers; each attention builds the
    # symmetric-window mask itself — is_causal False + sliding_window from config)
    pos = torch.arange(CTX + BLK).unsqueeze(0)
    hidden = noise
    pos_emb = model.rotary_emb(hidden, pos)
    for i, layer in enumerate(model.layers):
        hidden = layer(
            hidden_states=hidden,
            target_hidden=ctx_f,
            attention_mask=None,
            position_ids=pos,
            position_embeddings=pos_emb,
        )
        dump[f"layer{i}_out"] = hidden.numpy().reshape(BLK, H)
    final = model.norm(hidden)
    dump["final"] = final.numpy().reshape(BLK, H)

    # stage 3w: reduced-window forward — a SECOND model instance sharing weights,
    # with sliding_window shrunk so ctx 24 crosses it (per-query old-side masking).
    raw_win = dict(raw)
    raw_win["sliding_window"] = WIN_WINDOW
    cfg_win = Qwen3Config(
        **{k: v for k, v in raw_win.items() if k not in ("architectures",)}
    )
    cfg_win._attn_implementation = "eager"
    model_win = DFlash2DraftModel(cfg_win)
    mw_missing, mw_unexpected = model_win.load_state_dict(sd, strict=False)
    assert not mw_missing and not mw_unexpected
    model_win = model_win.to(torch.float32).eval()
    for lay in model_win.layers:
        assert lay.self_attn.sliding_window == WIN_WINDOW
        assert lay.self_attn.is_causal is False
    ctx_f_win = model_win.hidden_norm(model_win.fc(taps_win))
    final_win = model_win(
        position_ids=torch.arange(WIN_CTX + BLK).unsqueeze(0),
        noise_embedding=noise,
        target_hidden=taps_win,
    )
    dump["final_win"] = final_win.numpy().reshape(BLK, H)
    dump["ctx_features_win"] = ctx_f_win.numpy().reshape(WIN_CTX, H)

    # stage 4: conv isolation (layer-0 attention_conv; prepare on conv_x, finish on
    # conv_y with prepare's dynamic kernel — the exact module contract)
    conv = model.layers[0].attention_conv
    prep_out, fin_kernel = conv.prepare(conv_x)
    fin_out = conv.finish(conv_y, fin_kernel)
    dump["conv_prepare_out"] = prep_out.numpy().reshape(BLK, H)
    dump["conv_finish_out"] = fin_out.numpy().reshape(BLK, H)

    # stage 5: selector (greedy). Replicate select()'s own top-k for the dump, then
    # run select() itself for the path.
    unary, candidates = torch.topk(sel_logits, TOPK, dim=-1, sorted=False)
    path, cand_out, qrows = model.candidate_selector.select(
        sel_hidden, sel_logits, anchor, temperature=0.0
    )
    assert qrows is None
    assert torch.equal(cand_out, candidates)
    dump["sel_unary"] = unary.numpy().reshape(NDR, TOPK)
    dump["sel_candidates"] = candidates.numpy().reshape(NDR, TOPK)
    dump["sel_path"] = path.numpy().reshape(NDR)

np.savez(out_path, **dump)
flat_dir = pathlib.Path(out_path).parent
for k, v in dump.items():
    if v.dtype in (np.int64, np.int32):
        (flat_dir / f"dflash2-{k}.u32").write_bytes(v.astype(np.uint32).tobytes())
    else:
        (flat_dir / f"dflash2-{k}.f32").write_bytes(v.astype(np.float32).tobytes())
print(f"oracle dumped {len(dump)} arrays -> {out_path} (+flat twins in {flat_dir})")
for k, v in dump.items():
    print(f"  {k}: {v.shape}")

# ---- GEOMETRY MANIFEST (GATE-INTEGRITY-20260819 §5; provenance, not re-derivation) ----
_attn0 = model.layers[0].self_attn
# rope_theta recorded from what the run ACTUALLY rotated with (the transformers-5
# rope divergence class fails silently otherwise): inv_freq[j] = theta^(-2j/d).
_inv = model.rotary_emb.inv_freq.float()
_dim = _inv.shape[0] * 2
_theta_used = float(_inv[1].item() ** (-_dim / 2.0))
_rope_params = getattr(config, "rope_parameters", None) or {}
_theta_cfg = float(_rope_params.get("rope_theta", 0.0))
assert abs(_theta_used - _theta_cfg) / _theta_cfg < 1e-3, (
    f"rotary_emb built with theta {_theta_used:.6g} but config says {_theta_cfg:.6g} "
    "— the transformers-5 rope divergence class; refusing"
)
assert float(model.rotary_emb.attention_scaling) == 1.0, "unexpected rope scaling"
_manifest = {
    "producer": "tools/dflash2_oracle.py",
    "ckpt_dir": ckpt_dir,
    "model_py_sha256": _model_py_sha,
    "transformers": _tv,
    "harvest": "dflash",  # mask-fill family by construction
    "dtype": "f32",
    "hidden": int(H),
    "n_layer": int(len(model.layers)),
    "block_size": int(BLK),
    "ctx": int(CTX),
    "n_taps": int(NT),
    "target_layer_ids": ",".join(str(int(x)) for x in model.target_layer_ids),
    "head_dim": int(_attn0.head_dim),
    "n_head": int(config.num_attention_heads),
    "n_head_kv": int(config.num_key_value_heads),
    "rope_theta": repr(_theta_used),
    "sliding_window": int(_attn0.sliding_window),
    "is_causal": "1" if _attn0.is_causal else "0",
    "layer_sliding": ",".join(
        "1" if lay.self_attn.sliding_window is not None else "0" for lay in model.layers
    ),
    "vocab": int(V),
    "selector_rank": int(RANK),
    "selector_top_k": int(TOPK),
    "conv_kernel_size": int(d2cfg["conv_kernel_size"]),
    "conv_group_size": int(d2cfg["conv_group_size"]),
    "win_ctx": int(WIN_CTX),
    "win_window": int(WIN_WINDOW),
    "ndr": int(NDR),
}
for _name, _arr in dump.items():
    _manifest[f"shape.{_name}"] = ",".join(str(int(x)) for x in _arr.shape)
_manifest_path = flat_dir / "dflash2-geometry.txt"
with open(_manifest_path, "w") as _fh:
    _fh.write("# geometry manifest for the dflash2-*.f32/.u32 dumps in this directory.\n")
    _fh.write("# Written by tools/dflash2_oracle.py; asserted by dflash2_parity before\n")
    _fh.write("# any value compare (crates/memra-engine/src/parity_geometry.rs).\n")
    for _k, _v in _manifest.items():
        _fh.write(f"{_k}={_v}\n")
print(f"geometry manifest -> {_manifest_path} ({len(_manifest)} fields)")
