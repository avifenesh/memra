#!/usr/bin/env python3
"""DSpark Q38 drafter oracle (lane/dspark-q38-recover, 2026-08-16).

Runs the SpecForge reference DSparkDraftModel (the code that TRAINED arm-a) on the
arm-a HF export with FIXED-SEED synthetic inputs, dumping every stage the memra arm
must reproduce:

  stage 1  ctx_features   = hidden_norm(fc(taps))                 [CTX, H]
  stage 2  layer{i}_out   per-layer residual-stream block rows    [BLK, H]
  stage 3  final          = norm(x)                               [BLK, H]
  stage 4  markov chain   chained greedy tokens over synthetic
           base logits + biased logits per step                   [NDR], [NDR, V]
  stage 5  confidence     raw AcceptRatePredictor output (concat
           hidden + markov prev-embedding), per harvested row     [NDR]

HARVEST CONVENTION (DSPARK-POSTMORTEM-20260820.md). NDR (the harvested draft-row
count) and which `final` rows feed stages 4-5 are properties of the checkpoint's
TRAINING STRATEGY, recorded into the geometry manifest as `harvest=`:
  --harvest=dspark (DEFAULT — this is the q38 DSPARK-strategy oracle): SpecForge
      OnlineDSparkModel trains ALL BLK rows with SHIFTED labels (row k -> anchor+k+1,
      label_offsets = arange(1, block_size+1) — specforge/algorithms/common/
      dflash_family_model.py:816); sglang's DSPARK worker harvests gamma = BLK drafts
      with the anchor row's output as draft 1 (v0.5.17 dspark_draft.py:248,260).
      NDR = BLK; stages 4-5 ride final[:, :, :] (all rows).
  --harvest=dflash: the z-lab mask-fill convention (row k fills anchor+k, anchor row
      loss-excluded — dflash_family_model.py:453-472). NDR = BLK-1; stages 4-5 ride
      final[:, 1:, :]. Kept only to reproduce pre-postmortem dumps; the parity gate
      REFUSES manifest-less dumps for dspark-class exports.

Synthetic taps/noise isolate DRAFT math from target plumbing (taps/embed/lm_head are
memra-native pieces gated separately — same isolation as tools/dflash_oracle.py).
The markov stage uses a synthetic base-logits matrix (fixed seed) so the chain is
gated independently of the target lm_head.

Census discipline: load_state_dict must report ZERO missing and ZERO unexpected keys
(62-tensor export) or the oracle refuses.

Usage:
  python tools/dspark_q38_oracle.py <export_dir> <out.npz> [specforge_dir] [--harvest={dspark|dflash}]
"""

import json
import pathlib
import sys

import numpy as np
import torch

_flags = [a for a in sys.argv[1:] if a.startswith("--")]
_pos = [a for a in sys.argv[1:] if not a.startswith("--")]
HARVEST = "dspark"
for _f in _flags:
    if _f.startswith("--harvest="):
        HARVEST = _f.split("=", 1)[1]
    else:
        sys.exit(f"unknown flag {_f} (only --harvest={{dspark|dflash}})")
if HARVEST not in ("dspark", "dflash"):
    sys.exit(
        f"--harvest={HARVEST}: unknown convention (dspark|dflash) — refusing; a wrong "
        "convention gates the memra chain against rows the drafter was not trained "
        "for (DSPARK-POSTMORTEM-20260820.md)"
    )

ckpt_dir, out_path = _pos[0], _pos[1]
sys.path.insert(0, _pos[2] if len(_pos) > 2 else "/tmp/SpecForge")

import transformers  # noqa: E402  (recorded as provenance, see the manifest)
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config  # noqa: E402

from specforge.modeling.draft.dspark import DSparkDraftModel  # noqa: E402

raw = json.load(open(f"{ckpt_dir}/config.json"))
config = Qwen3Config(
    **{k: v for k, v in raw.items() if k not in ("architectures", "dflash_config")}
)
config.dflash_config = dict(raw["dflash_config"])
config.block_size = raw["block_size"]
config.num_target_layers = 64  # unused: target_layer_ids explicit in dflash_config
config._attn_implementation = "eager"

torch.manual_seed(0)
model = DSparkDraftModel(config)
from safetensors.torch import load_file  # noqa: E402

sd = load_file(f"{ckpt_dir}/model.safetensors")
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

# NDR = harvested draft rows: DSPARK-strategy = all BLK rows (shifted labels, anchor
# row included); dflash mask-fill = BLK-1 mask rows. See the docstring / postmortem.
NDR = BLK if HARVEST == "dspark" else BLK - 1
ROW0 = 0 if HARVEST == "dspark" else 1  # first `final` row feeding stages 4-5
flat_dir = pathlib.Path(out_path).parent

# ---- GEOMETRY MANIFEST (GATE-INTEGRITY-20260819 §5) ----
#
# The flat twins are naked f32/u32 with no shape and no provenance, and dspark_q38_parity used
# to RECOVER the context length from a byte count — `ctx = taps.len() / (n_taps * h)` — with no
# remainder check and no assertion that the dump came from this export. Its other structural
# checks were products (`noise.len() == b*h`, `base.len() == (b-1)*v`), and a product is blind to
# any factorisation multiplying out the same. The geometry that lives only in config scalars
# (head_dim, rope_theta, sliding_window, per-layer sliding) never reaches the bytes at all:
# byte-identical dumps under a rotary width wrong by 4x is a defect that survived a
# maxdiff=0.0e0 gate on 2026-08-19.
#
# EVERY required scalar below is read from the checkpoint's config.json (`raw`), NOT from the
# transformers config OBJECT, and the manifest is written BEFORE any dump — the same shape
# tools/dflash_oracle.py took after it broke exactly there (cd0c6594a6): its manifest block was
# `repr(float(config.rope_theta))`, transformers 5.x moved rope config off the attribute (into
# `rope_parameters`), and the oracle wrote every flat dump, THEN died with
# `AttributeError: 'Qwen3Config' object has no attribute 'rope_theta'` — a full set of dumps
# with no manifest, which is precisely the state the gate refuses. This file kept the sibling
# or-chains (`getattr(_rope_cfg, "rope_theta", None) or config.rope_theta`), which die the same
# way on 5.x and, where the two sources disagree, silently record whichever is truthy
# (hermes `de4af50ef29cbe5b`). config.json is also what the CONSUMER reads: parity_geometry.rs
# compares against the checkpoint's own JSON, so going through Qwen3Config puts transformers'
# defaulting and renaming between the two sides of a comparison whose whole purpose is that the
# two sides agree.
#
# A missing key is a REFUSAL, never a default: a manifest field guessed by the producer is a
# field the consumer cannot use to disagree with anything.


def _both(key):
    """(top-level value, dflash_config value) for `key` in the checkpoint's config.json."""
    return raw.get(key), (raw.get("dflash_config") or {}).get(key)


def _cfg_scalar(key):
    """Required config.json scalar, fail-closed on absence AND on ambiguity.

    The Rust loader greps config.json text for the FIRST occurrence of a key, so a key
    present both top-level and in dflash_config with DIFFERENT values has no single truth
    to record — refusing beats guessing (a guessed manifest row manufactures a false
    red or, worse, a false green).
    """
    top, nested = _both(key)
    if top is None and nested is None:
        raise SystemExit(
            f"dspark_q38_oracle: {ckpt_dir}/config.json has no '{key}' (top-level or "
            "dflash_config). The geometry manifest records what this run computed UNDER, "
            "so a guessed value would make the parity gate agree with a fiction. "
            "Refusing to write a manifest with a hole in it."
        )
    if top is not None and nested is not None and top != nested:
        raise SystemExit(
            f"dspark_q38_oracle: config.json carries '{key}' both top-level ({top!r}) and "
            f"in dflash_config ({nested!r}) and they disagree — the Rust loader greps the "
            "FIRST occurrence, so there is no single truth to record. Refusing rather "
            "than guessing."
        )
    return top if top is not None else nested


def _unambiguous(key):
    """Optional config.json scalar: the value if the two locations agree (or only one has
    it), else None — the omission is reported by the gate rather than read as agreement."""
    top, nested = _both(key)
    vals = [x for x in (top, nested) if x is not None]
    if not vals:
        return None
    if len(vals) == 2 and vals[0] != vals[1]:
        return None
    return vals[0]


N_LAYER = int(_cfg_scalar("num_hidden_layers"))
if N_LAYER != len(model.layers):
    raise SystemExit(
        f"dspark_q38_oracle: config.json num_hidden_layers={N_LAYER} but the constructed "
        f"reference model has {len(model.layers)} layers — the manifest would describe a "
        "different program than the one about to produce the dumps. Refusing."
    )

_manifest = {
    "producer": "tools/dspark_q38_oracle.py",
    "ckpt_dir": ckpt_dir,
    # The harvest convention this dump's stage-4/5 rows encode (asserted by the gate
    # BEFORE any value compare — DSPARK-POSTMORTEM-20260820.md).
    "harvest": HARVEST,
    "dtype": "f32",
    # THE LIBRARY VERSIONS ARE PART OF THE GEOMETRY (dflash_oracle.py, cd0c6594a6, measured):
    # regenerating the dflash reference under transformers 5.15 left the four dumps its parity
    # gate reads within f32 noise of the 4.x set, but the POST-rope bisect dumps (l0_q, l0_k)
    # moved at rel ~1.1-1.6 while the PRE-rope ones stayed at 4e-7 — a rope-convention change
    # between transformers 4.x and 5.x, invisible in every byte count and every config scalar.
    # Corollary 3 of the parity rule ("provenance is part of the geometry") therefore includes
    # the library that implemented the rotation. The Rust reader looks fields up by key and
    # ignores ones it does not know, so these are additive: for the human bisecting a mismatch.
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "hidden": int(_cfg_scalar("hidden_size")),
    "n_layer": N_LAYER,
    "block_size": int(_cfg_scalar("block_size")),
    "ctx": int(CTX),
    "n_taps": int(NT),
    "target_layer_ids": ",".join(str(int(x)) for x in model.target_layer_ids),
    "head_dim": int(_cfg_scalar("head_dim")),
    "n_head": int(_cfg_scalar("num_attention_heads")),
    "n_head_kv": int(_cfg_scalar("num_key_value_heads")),
    "rope_theta": repr(float(_cfg_scalar("rope_theta"))),
    "markov_vocab": int(V),
}

_sw = _unambiguous("sliding_window")
if _sw is not None:
    _manifest["sliding_window"] = int(_sw)
_lt = _unambiguous("layer_types")
if _lt:
    _manifest["layer_sliding"] = ",".join(
        "1" if "sliding_attention" in str(t) else "0" for t in _lt
    )
# markov rank = the width of the per-token embedding the chain gathers; recorded only when the
# export names it unambiguously (exactly one 2-D tensor with V rows under markov_head).
_ranks = {
    tuple(t.shape)[1]
    for k, t in sd.items()
    if k.startswith("markov_head") and t.ndim == 2 and tuple(t.shape)[0] == V
}
if len(_ranks) == 1:
    _manifest["markov_rank"] = int(next(iter(_ranks)))

# Shapes are PREDICTED from the config the model was built under — config is the authority,
# the dump is the claimant (corollary 1) — and every produced array is asserted against its
# prediction below, before anything is written.
_expected_shapes = {
    "taps": (CTX, NT * H),
    "noise": (BLK, H),
    "base_logits": (NDR, V),
    "anchor": (1,),
    "ctx_features": (CTX, H),
    **{f"layer{i}_out": (BLK, H) for i in range(N_LAYER)},
    "final": (BLK, H),
    "markov_tokens": (NDR,),
    "markov_logits": (NDR, V),
    "confidence": (NDR,),
}
for _name, _shape in _expected_shapes.items():
    _manifest[f"shape.{_name}"] = ",".join(str(int(x)) for x in _shape)

_manifest_path = flat_dir / "dspark-geometry.txt"
with open(_manifest_path, "w") as _fh:
    _fh.write("# geometry manifest for the dspark-*.f32/.u32 dumps in this directory.\n")
    _fh.write("# Written by tools/dspark_q38_oracle.py; asserted by dspark_q38_parity before\n")
    _fh.write("# any value compare (crates/memra-engine/src/parity_geometry.rs).\n")
    for _k, _v in _manifest.items():
        _fh.write(f"{_k}={_v}\n")
print(f"geometry manifest -> {_manifest_path} ({len(_manifest)} fields)")

g = torch.Generator().manual_seed(42)
taps = torch.randn(1, CTX, NT * H, generator=g)  # raw tapped residual rows
noise = torch.randn(1, BLK, H, generator=g)  # embed-row stand-ins
base_logits = torch.randn(1, NDR, V, generator=g)  # lm_head stand-in for markov
anchor = torch.tensor([17], dtype=torch.long)  # first_prev_token_ids

dump: dict[str, np.ndarray] = {
    "taps": taps.numpy().reshape(CTX, NT * H),
    "noise": noise.numpy().reshape(BLK, H),
    "base_logits": base_logits.numpy().reshape(NDR, V),
    "anchor": anchor.numpy(),
}

with torch.inference_mode():
    # stage 1: ctx features
    ctx_f = model.hidden_norm(model.fc(taps))
    dump["ctx_features"] = ctx_f.numpy().reshape(CTX, H)

    # stages 2-3: block forward (full non-causal, no draft KV — first-light contract)
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

    # stage 4: markov chained greedy over synthetic base logits. Rows = the HARVEST
    # convention's draft rows (dspark: all BLK rows, anchor row = draft 1 — the
    # sglang-serving/OnlineDSparkModel alignment; dflash: mask rows 1..BLK-1).
    tokens, corrected = model.markov_head.sample_block_tokens(
        base_logits,
        first_prev_token_ids=anchor,
        hidden_states=final[:, ROW0:, :],  # vanilla head ignores hidden_states
        temperature=0.0,
    )
    dump["markov_tokens"] = tokens.numpy().reshape(NDR)
    dump["markov_logits"] = corrected.numpy().reshape(NDR, V)

    # stage 5: confidence (raw, pre-sigmoid — module output; sglang's DSPARK planner
    # consumes it for verify-window sizing; gated here per harvested row)
    prev_ids = torch.cat([anchor.unsqueeze(0), tokens[:, :-1]], dim=1)
    conf = model.predict_confidence(final[:, ROW0:, :], prev_token_ids=prev_ids)
    dump["confidence"] = conf.numpy().reshape(NDR)

# The run must match its own manifest before a single byte lands next to it: a dump whose
# shape disagrees with the config-predicted one is a different program than the manifest
# describes, and leaving both on disk would hand the gate a lie with provenance attached.
for _name, _arr in dump.items():
    _want = _manifest.get(f"shape.{_name}")
    _got = ",".join(str(int(x)) for x in _arr.shape)
    if _want != _got:
        raise SystemExit(
            f"dspark_q38_oracle: dump '{_name}' has shape {_got} but the manifest "
            f"predicted {_want} from config.json — the run and its recorded provenance "
            "disagree; refusing to write dumps that contradict their manifest."
        )

np.savez(out_path, **dump)
# flat .f32/.u32 twins for the rust gate (house pattern: dflash_parity reads flat dumps)
for k, v in dump.items():
    if v.dtype in (np.int64, np.int32):
        (flat_dir / f"dspark-{k}.u32").write_bytes(
            v.astype(np.uint32).tobytes()
        )
    else:
        (flat_dir / f"dspark-{k}.f32").write_bytes(v.astype(np.float32).tobytes())
print(f"oracle dumped {len(dump)} arrays -> {out_path} (+flat twins in {flat_dir})")
for k, v in dump.items():
    print(f"  {k}: {v.shape}")
