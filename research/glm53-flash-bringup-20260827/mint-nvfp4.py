#!/usr/bin/env python3
"""GLM-5.3-Flash BF16 -> NVFP4 W4A16 (weight-only) streaming mint.

Source : zai-org/GLM-5.3-Flash-BF16 @ f12e0fe1f6b2ea274c11a569582edfd99d993c5e (656 GB bf16)
Output : modelopt/compressed-tensors NVFP4 HF safetensors checkpoint (~205 GB),
         the exact layout memra's nvfp4_repack.rs consumes:
           <stem>.weight         U8       [out, in/2]  packed e2m1, elem 2i -> low nibble
           <stem>.weight_scale   F8_E4M3  [out, in/16] per-16 block scale
           <stem>.weight_scale_2 F32      scalar       per-tensor macro-scale
         (no input_scale: weight-only mint; memra quantizes activations dynamically)

Quantization math: modelopt.torch.quantization.qtensor.NVFP4QTensor.quantize —
the SAME function nvidia-modelopt's export_hf_checkpoint/to_quantized_weight calls
(verified at the 0.46.0 tag: modelopt/torch/quantization/qtensor/nvfp4_tensor.py).
Packaging is streamed per-tensor (NVIDIA's own pattern for this lineage:
examples/deepseek/deepseek_v4/quantize_to_nvfp4.py) because the model-level
export path cannot load this checkpoint on this box — receipts in MINT-NOTES.md:
  1. 656 GB bf16 > VRAM+RAM; the only fitting modelopt path (init_quantized_weights)
     loads via accelerate.load_checkpoint_and_dispatch, which bypasses transformers'
     checkpoint-key conversion;
  2. the glm5_next modeling FUSES q/k/v_conv1d into self_attn.conv1d and nests
     f_a/f_b under forget_gate — raw key-matching silently drops those KDA weights;
  3. the modeling never instantiates the MTP layer
     (_keys_to_ignore_on_load_unexpected = [r"layers\\.45\\.", ...]) — the export
     would lose NextN entirely.
Streaming per-tensor needs no transformers, no calibration, and bounded RAM.

Precision split (owner-pinned, mirrors the vendor's own FP8 exclusions + CENSUS.md):
  QUANTIZE: MoE routed experts + shared experts + dense MLPs (layers 0-2) +
            MLA projections q_a/q_b/kv_a_proj_with_mqa/o_proj on the 12 DSA layers
            (11 full-attn + MTP layer 45).
  KEEP    : ALL KDA tensors (q/k/v_proj, q/k/v_conv1d, b_proj, f_*/g_*, o_norm,
            A_log, dt_bias, o_proj on the 34 KDA layers), kv_b_proj, indexer.*,
            mlp.gate.* (router + e_score_correction_bias), hc_* (mHC), all norms,
            embed_tokens, lm_head, model.visual.*, MTP scaffolding
            (eh_proj/enorm/hnorm/shared_head.norm).

Every tensor in the source index MUST classify as exactly one of QUANTIZE/KEEP or
the mint aborts (fail loudly, never default silently).

Run on the mint box via mint-run.sh. No calibration data, no forward passes.
"""

import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------- config

SRC_DIR = Path(os.environ.get("MINT_SRC", os.path.expanduser("~/models/glm53-bf16")))
OUT_DIR = Path(os.environ.get("MINT_OUT", os.path.expanduser("~/models/glm53-nvfp4")))
PINNED_REVISION = "f12e0fe1f6b2ea274c11a569582edfd99d993c5e"  # informational; verified at download
BLOCK = 16          # modelopt NVFP4 block size (per-16 e4m3 scales)
MEMRA_QK = 64       # memra block_nvfp4 super-block; every quantized in_features % 64 == 0
SHARD_BYTES = 10 * 1000**3          # ~10 GB shards, matches modelopt max_shard_size default
SPOT_CHECK_EVERY = int(os.environ.get("MINT_SPOT_EVERY", "500"))  # cross-impl dequant gate cadence
MIN_MODELOPT = (0, 45, 0)           # W4A16_NVFP4 landed in 0.45 (CHANGELOG.rst)

# Census-derived hard expectations (CENSUS.md: FP8 twin has 76,108 tensors of which
# 37,338 carry weight_scale_inv == the vendor's own quantize set; BF16 twin therefore
# has 76,108 - 37,338 = 38,770 tensors, and our quantize set must equal the vendor's).
EXPECTED_TOTAL_TENSORS = 38_770
EXPECTED_QUANT_TENSORS = 37_338     # 37,152 experts + 129 shared + 9 dense + 48 MLA
EXPECTED_KDA_LAYERS = 34
EXPECTED_FULL_ATTN_LAYERS = 11


class MintError(RuntimeError):
    pass


def die(msg: str) -> None:
    raise MintError(msg)


# ----------------------------------------------------------------------------- preflight

def preflight():
    import modelopt
    ver = tuple(int(x) for x in modelopt.__version__.split(".")[:3])
    if ver < MIN_MODELOPT:
        die(f"nvidia-modelopt {modelopt.__version__} < 0.45; W4A16 NVFP4 needs >= 0.45 "
            "(pin: nvidia-modelopt==0.46.0)")
    import torch
    import safetensors  # noqa: F401
    if not (SRC_DIR / "model.safetensors.index.json").is_file():
        die(f"source index missing: {SRC_DIR}/model.safetensors.index.json")
    if not (SRC_DIR / "config.json").is_file():
        die(f"source config missing: {SRC_DIR}/config.json")
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        die(f"output dir {OUT_DIR} exists and is not empty; refuse to overwrite a prior mint")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[preflight] modelopt {modelopt.__version__}  torch {torch.__version__}  device {dev}")
    return dev


# ----------------------------------------------------------------------------- census

def read_st_header(path: Path):
    """safetensors header only (no tensor bytes): name -> {dtype, shape, data_offsets}."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def load_layer_split(cfg: dict):
    tc = cfg.get("text_config", cfg)
    la = tc["linear_attn_config"]
    kda = sorted(la["kda_layers"])
    full = sorted(la["full_attn_layers"])
    n_layers = tc["num_hidden_layers"]
    mtp = n_layers  # NextN layer index (45)
    if len(kda) != EXPECTED_KDA_LAYERS:
        die(f"config kda_layers count {len(kda)} != {EXPECTED_KDA_LAYERS}")
    if len(full) != EXPECTED_FULL_ATTN_LAYERS:
        die(f"config full_attn_layers count {len(full)} != {EXPECTED_FULL_ATTN_LAYERS}")
    if set(kda) & set(full):
        die("kda_layers and full_attn_layers overlap")
    if sorted(set(kda) | set(full)) != list(range(n_layers)):
        die("kda+full layers do not cover 0..num_hidden_layers-1")
    return kda, full, mtp


LAYER_PREFIX = "model.language_model.layers."

KDA_KEEP_SUFFIXES = (
    "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
    "self_attn.q_conv1d.weight", "self_attn.k_conv1d.weight", "self_attn.v_conv1d.weight",
    "self_attn.b_proj.weight", "self_attn.f_a_proj.weight", "self_attn.f_b_proj.weight",
    "self_attn.g_a_proj.weight", "self_attn.g_b_proj.weight",
    "self_attn.o_norm.weight", "self_attn.A_log", "self_attn.dt_bias",
)
MLA_KEEP_SUFFIXES = (
    "self_attn.q_a_layernorm.weight", "self_attn.kv_a_layernorm.weight",
    "self_attn.kv_b_proj.weight",
)
MLA_QUANT_SUFFIXES = (
    "self_attn.q_a_proj.weight", "self_attn.q_b_proj.weight",
    "self_attn.kv_a_proj_with_mqa.weight",
)
ANYLAYER_KEEP_SUFFIXES = (
    "input_layernorm.weight", "post_attention_layernorm.weight",
    "hc_attn_base", "hc_attn_fn", "hc_attn_scale",
    "hc_ffn_base", "hc_ffn_fn", "hc_ffn_scale",
    "mlp.gate.weight", "mlp.gate.e_score_correction_bias",
    # MTP scaffolding (vendor keeps these; "mapping_proj" in modules_to_not_convert)
    "eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight",
)
SHARED_EXPERT_QUANT_SUFFIXES = (
    "mlp.shared_experts.gate_proj.weight", "mlp.shared_experts.up_proj.weight",
    "mlp.shared_experts.down_proj.weight",
)
DENSE_MLP_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
TOPLEVEL_KEEP = ("lm_head.weight", "model.language_model.embed_tokens.weight",
                 "model.language_model.norm.weight")


def make_classifier(kda_layers, full_layers, mtp_layer):
    kda = set(kda_layers)
    mla = set(full_layers) | {mtp_layer}
    dense = {0, 1, 2}

    def classify(name: str) -> str:
        """Return 'quant' or 'keep'; raise on anything unrecognized."""
        if name in TOPLEVEL_KEEP:
            return "keep"
        if name.startswith("model.visual."):
            return "keep"
        if not name.startswith(LAYER_PREFIX):
            die(f"unclassifiable tensor (unknown prefix): {name}")
        rest = name[len(LAYER_PREFIX):]
        layer_s, _, suffix = rest.partition(".")
        if not layer_s.isdigit():
            die(f"unclassifiable tensor (no layer index): {name}")
        layer = int(layer_s)

        # --- MoE / MLP ---
        if suffix.startswith("mlp.experts."):
            parts = suffix.split(".")
            # mlp.experts.<idx>.<proj>.weight
            if (len(parts) == 5 and parts[2].isdigit()
                    and parts[3] in ("gate_proj", "up_proj", "down_proj")
                    and parts[4] == "weight"):
                return "quant"
            die(f"unclassifiable expert tensor: {name}")
        if suffix in SHARED_EXPERT_QUANT_SUFFIXES:
            return "quant"
        if suffix in DENSE_MLP_SUFFIXES:
            if layer not in dense:
                die(f"dense-style mlp tensor outside dense layers 0-2: {name}")
            return "quant"

        # --- attention: o_proj exists on BOTH types, split by layer index ---
        if suffix == "self_attn.o_proj.weight":
            if layer in kda:
                return "keep"
            if layer in mla:
                return "quant"
            die(f"o_proj on layer {layer} which is neither KDA nor MLA: {name}")
        if suffix in MLA_QUANT_SUFFIXES:
            if layer not in mla:
                die(f"MLA tensor on non-MLA layer {layer}: {name}")
            return "quant"
        if suffix in MLA_KEEP_SUFFIXES or suffix.startswith("self_attn.indexer."):
            if layer not in mla:
                die(f"MLA/indexer tensor on non-MLA layer {layer}: {name}")
            return "keep"
        if suffix in KDA_KEEP_SUFFIXES:
            if layer not in kda:
                die(f"KDA tensor on non-KDA layer {layer}: {name}")
            return "keep"
        if suffix in ANYLAYER_KEEP_SUFFIXES:
            return "keep"

        die(f"UNCLASSIFIED tensor (extend the classifier deliberately, never default): {name}")
        return "unreachable"

    return classify


def census(classify):
    index = json.loads((SRC_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    names = sorted(weight_map)
    if len(names) != EXPECTED_TOTAL_TENSORS:
        die(f"BF16 twin tensor count {len(names)} != census-derived {EXPECTED_TOTAL_TENSORS} "
            "(76,108 FP8 tensors minus 37,338 weight_scale_inv). Investigate before minting.")

    shard_headers = {}
    for shard in sorted(set(weight_map.values())):
        shard_headers[shard] = read_st_header(SRC_DIR / shard)

    plan = {}   # name -> (cls, dtype, shape)
    n_quant = 0
    for name in names:
        meta = shard_headers[weight_map[name]].get(name)
        if meta is None:
            die(f"tensor {name} in index but not in shard {weight_map[name]}")
        cls = classify(name)
        dtype, shape = meta["dtype"], meta["shape"]
        if cls == "quant":
            n_quant += 1
            if dtype != "BF16":
                die(f"quantize-set tensor {name} has dtype {dtype}, expected BF16")
            if len(shape) != 2:
                die(f"quantize-set tensor {name} is not 2D: {shape}")
            if shape[1] % MEMRA_QK != 0:
                die(f"{name} in_features {shape[1]} not divisible by {MEMRA_QK} "
                    "(memra block_nvfp4 requires %64; modelopt requires %16)")
        plan[name] = (cls, dtype, shape)

    if n_quant != EXPECTED_QUANT_TENSORS:
        die(f"quantize-set size {n_quant} != {EXPECTED_QUANT_TENSORS} "
            "(the vendor FP8 checkpoint's own quantize set). The split drifted — stop.")
    print(f"[census] {len(names)} tensors: quant={n_quant} keep={len(names) - n_quant}")
    return weight_map, plan


# ----------------------------------------------------------------------------- dequant cross-check

# Standard (non-doubled) e2m1 codebook, index = 4-bit code. memra's KVALUES_MXFP4 is 2x
# this and its ue4m3_to_f32 returns raw*0.5 — the 2x and 0.5 cancel (nvfp4_repack.rs).
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def ue4m3_byte_to_f32(x: int) -> float:
    """UNSIGNED e4m3 scale byte -> f32, the RAW value (2x memra's GGUF-halved return).
    Mirrors nvfp4_repack.rs::ue4m3_to_f32 without the *0.5: NaN codes 0x7F -> 0,
    byte 0 -> 0. Sign bit must never be set (asserted at mint time)."""
    if x in (0x00, 0x7F):
        return 0.0
    exp = (x >> 3) & 0xF
    man = x & 0x7
    if exp == 0:
        return man * 2.0 ** -9          # subnormal: (man/8) * 2^-6
    return (1.0 + man / 8.0) * 2.0 ** (exp - 7)


def memra_style_dequant(packed, scale_u8, scale2, out_f, in_f):
    """Independent re-implementation of the CONSUMER math from nvfp4_repack.rs:
    value = std_e2m1(code) * raw_ue4m3(scale_byte) * weight_scale_2.
    Integer bit-decode of the scale byte; element 2i taken from the LOW nibble."""
    import torch
    codes = torch.empty(out_f, in_f, dtype=torch.uint8)
    codes[:, 0::2] = packed & 0x0F
    codes[:, 1::2] = packed >> 4
    e2m1 = torch.tensor(E2M1, dtype=torch.float32)
    vals = e2m1[codes.long()]
    lut = torch.tensor([ue4m3_byte_to_f32(b) for b in range(256)], dtype=torch.float32)
    sc = lut[scale_u8.long()].repeat_interleave(BLOCK, dim=1)   # [out, in]
    return vals * sc * float(scale2)


def spot_check(name, w, qt, dev_scale, dev_scale2, packed, scale, scale2):
    """Cross-implementation gate: modelopt's OWN dequantize (torch fp8 decode +
    get_e2m1_values LUT, on the quantize device) vs the memra-math dequant
    (integer bit decode, on CPU) must agree; both must be a sane e2m1
    approximation of the source weight. packed/scale are the CPU copies."""
    import torch
    out_f, in_f = w.shape
    scale_u8 = scale.view(torch.uint8)
    if int((scale_u8 & 0x80).ne(0).sum()) != 0:
        die(f"[spot-check {name}] per-block scale byte with SIGN BIT set — memra decodes "
            "scales as unsigned ue4m3; a negative modelopt scale breaks the contract")
    if int(scale_u8.eq(0x7F).sum()) != 0:
        die(f"[spot-check {name}] per-block scale byte 0x7F (NaN code) present")
    memra_deq = memra_style_dequant(packed, scale_u8, scale2, out_f, in_f)
    modelopt_deq = qt.dequantize(
        torch.float32, scale=dev_scale, double_scale=dev_scale2.float(),
        block_sizes={-1: BLOCK},
    ).reshape(out_f, in_f).cpu()
    # 1-ulp association-order noise between (code*scale)*scale2 and code*(scale*scale2)
    # is expected (~4e-7 rel); a nibble-order or scale-decode error produces O(1) rel
    # differences. rtol 1e-5 separates the two decisively.
    if not torch.allclose(memra_deq, modelopt_deq, rtol=1e-5, atol=1e-30):
        bad = int((~torch.isclose(memra_deq, modelopt_deq, rtol=1e-5, atol=1e-30)).sum())
        die(f"[spot-check {name}] memra-math dequant != modelopt dequant on {bad} elements "
            "— nibble order or scale semantics drifted; DO NOT ship this mint")
    ref = w.float().cpu()
    err = (modelopt_deq - ref).abs()
    if not torch.isfinite(modelopt_deq).all():
        die(f"[spot-check {name}] non-finite dequant values")
    if float(modelopt_deq.abs().sum()) == 0.0 and float(ref.abs().sum()) > 0.0:
        die(f"[spot-check {name}] all-zero quantized tensor from non-zero source")
    rel = float((err / ref.abs().clamp_min(1e-6)).median())
    print(f"[spot-check] {name}: median rel err {rel:.4f}, max abs err {float(err.max()):.4f}")


# ----------------------------------------------------------------------------- shard writer

class ShardWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.buf = {}
        self.buf_bytes = 0
        self.shards = []
        self.weight_map = {}
        self.total = 0

    def add(self, name, tensor):
        if name in self.weight_map or name in self.buf:
            die(f"duplicate output tensor {name}")
        t = tensor.contiguous().cpu()
        self.buf[name] = t
        nbytes = t.numel() * t.element_size()
        self.buf_bytes += nbytes
        self.total += nbytes
        if self.buf_bytes >= SHARD_BYTES:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        idx = len(self.shards) + 1
        fname = f"model-{idx:05d}.safetensors"   # renamed to -of-NNNNN at finalize
        from safetensors.torch import save_file
        save_file(self.buf, str(self.out_dir / fname))
        for n in self.buf:
            self.weight_map[n] = fname
        self.shards.append(fname)
        self.buf = {}
        self.buf_bytes = 0

    def finalize(self):
        self.flush()
        n = len(self.shards)
        renames = {}
        for i, old in enumerate(self.shards, start=1):
            new = f"model-{i:05d}-of-{n:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / new)
            renames[old] = new
        weight_map = {k: renames[v] for k, v in self.weight_map.items()}
        index = {"metadata": {"total_size": self.total}, "weight_map": weight_map}
        (self.out_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True))
        return weight_map


# ----------------------------------------------------------------------------- mint

def mint(device, weight_map, plan):
    import torch
    from safetensors import safe_open
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    writer = ShardWriter(OUT_DIR)
    consumed = set()
    n_quant_done = 0
    n_done = 0
    t0 = time.time()

    for shard in sorted(set(weight_map.values())):
        with safe_open(str(SRC_DIR / shard), framework="pt", device="cpu") as f:
            for name in sorted(k for k in f.keys() if k in plan):
                if name in consumed:
                    die(f"tensor {name} appears twice across shards")
                consumed.add(name)
                cls, _dtype, _shape = plan[name]
                w = f.get_tensor(name)
                if cls == "keep":
                    writer.add(name, w)
                else:
                    out_f, in_f = w.shape
                    if not name.endswith(".weight"):
                        die(f"quantize-set tensor without .weight suffix: {name}")
                    stem = name[: -len(".weight")]
                    wq = w.to(device)
                    # try_tensorrt stays at its default False: the trtllm fast path
                    # emits CUTLASS-swizzled scales, NOT the modelopt layout memra
                    # repacks (nvfp4_tensor.py quantize/dequantize @0.46.0).
                    qt, scale, scale2 = NVFP4QTensor.quantize(wq, BLOCK)
                    packed = qt._quantized_data
                    if packed.dtype != torch.uint8 or list(packed.shape) != [out_f, in_f // 2]:
                        die(f"{name}: packed {packed.dtype} {list(packed.shape)} != "
                            f"uint8 [{out_f},{in_f // 2}]")
                    if scale.dtype != torch.float8_e4m3fn or \
                            list(scale.shape) != [out_f, in_f // BLOCK]:
                        die(f"{name}: weight_scale {scale.dtype} {list(scale.shape)} != "
                            f"float8_e4m3fn [{out_f},{in_f // BLOCK}]")
                    if scale2.numel() != 1:
                        die(f"{name}: weight_scale_2 numel {scale2.numel()} != 1")
                    dev_scale, dev_scale2 = scale, scale2
                    packed = qt._quantized_data.cpu()
                    scale = dev_scale.cpu()
                    scale2 = dev_scale2.float().reshape(()).cpu()  # scalar f32 (modelopt shape)
                    if n_quant_done % SPOT_CHECK_EVERY == 0:
                        spot_check(name, w, qt, dev_scale, dev_scale2,
                                   packed, scale, float(scale2))
                    writer.add(f"{stem}.weight", packed)
                    writer.add(f"{stem}.weight_scale", scale)
                    writer.add(f"{stem}.weight_scale_2", scale2)
                    n_quant_done += 1
                    del wq, qt
                n_done += 1
                if n_done % 2000 == 0:
                    print(f"[mint] {n_done}/{len(plan)} tensors, "
                          f"{writer.total / 1e9:.1f} GB out, {time.time() - t0:.0f}s",
                          flush=True)

    missing = set(plan) - consumed
    if missing:
        die(f"{len(missing)} tensors in index never seen in shards, e.g. {sorted(missing)[:5]}")
    writer.finalize()
    print(f"[mint] wrote {len(writer.shards)} shards, {writer.total / 1e9:.1f} GB, "
          f"{n_quant_done} quantized tensors")


# ----------------------------------------------------------------------------- config + aux

def exclude_module_list(plan):
    """Module names of KEPT weights a deployment framework could mistake for GEMM
    weights: every kept .weight with ndim >= 2 (Linears, convs, embeddings, vision)
    plus the MoE router module. Explicit names, no wildcards — mirrors the role of
    exclude_modules in modelopt's get_quant_config (unquantized quantizer-capable
    layers + routers), enumerated rather than prefix-summarized."""
    mods = set()
    for name, (cls, _dtype, shape) in plan.items():
        if cls != "keep" or not name.endswith(".weight"):
            continue
        if len(shape) >= 2:
            mods.add(name[: -len(".weight")])
    # router weight is 2D so already included; keep set comprehension simple.
    return sorted(mods)


def write_configs(plan):
    import modelopt
    src_cfg = json.loads((SRC_DIR / "config.json").read_text())
    exclude = exclude_module_list(plan)

    # hf_quant_config.json — legacy modelopt export format
    # (quant_utils.get_quant_config + process_layer_quant_config @0.46.0)
    hf_quant_config = {
        "producer": {"name": "modelopt", "version": modelopt.__version__},
        "quantization": {
            "quant_algo": "W4A16_NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": BLOCK,
            "exclude_modules": exclude,
        },
    }
    (OUT_DIR / "hf_quant_config.json").write_text(json.dumps(hf_quant_config, indent=4))

    # config.json quantization_config — convert_hf_quant_config_format @0.46.0
    # output for W4A16_NVFP4 (weights-only config group, quant_method "modelopt")
    src_cfg["quantization_config"] = {
        "config_groups": {
            "group_0": {
                "weights": {"dynamic": False, "num_bits": 4, "type": "float",
                            "group_size": BLOCK},
                "targets": ["Linear"],
            }
        },
        "ignore": exclude,
        # The same keep list under the key memra's loader actually reads. Its matcher
        # (source.rs preserves_source_dtype) does exact-or-dotted-prefix on the HF name
        # AFTER `model.language_model.` is unwrapped, and it reads ONLY
        # modules_to_not_convert -- a compressed-tensors `ignore` list is invisible to it,
        # so the >=1M-element BF16 tensors we deliberately kept get Q8_0 re-encoded at
        # load. Writing both keys states one fact in both dialects; omitting this one
        # silently discards the precision split the mint exists to produce.
        "modules_to_not_convert": [n.replace("model.language_model.", "model.") for n in exclude],
        "quant_algo": "W4A16_NVFP4",
        "producer": {"name": "modelopt", "version": modelopt.__version__},
        "quant_method": "modelopt",
    }
    (OUT_DIR / "config.json").write_text(json.dumps(src_cfg, indent=4))

    copied = []
    for p in sorted(SRC_DIR.iterdir()):
        if not p.is_file():
            continue
        if p.suffix == ".safetensors" or p.name in ("model.safetensors.index.json",
                                                    "config.json"):
            continue
        if p.suffix in (".json", ".jinja", ".txt", ".model", ".py", ".md"):
            shutil.copy2(p, OUT_DIR / p.name)
            copied.append(p.name)
    print(f"[config] wrote config.json + hf_quant_config.json "
          f"({len(exclude)} exclude_modules); copied: {', '.join(copied)}")


# ----------------------------------------------------------------------------- final verification

def verify(plan):
    index = json.loads((OUT_DIR / "model.safetensors.index.json").read_text())
    out_map = index["weight_map"]
    headers = {}
    for shard in sorted(set(out_map.values())):
        headers.update(read_st_header(OUT_DIR / shard))

    n_quant = n_keep = 0
    for name, (cls, dtype, shape) in plan.items():
        if cls == "keep":
            meta = headers.get(name)
            if meta is None:
                die(f"[verify] kept tensor missing from output: {name}")
            if meta["dtype"] != dtype or meta["shape"] != list(shape):
                die(f"[verify] kept tensor {name} changed: {meta} vs {dtype} {shape}")
            if name.endswith(".weight"):
                stem = name[: -len(".weight")]
                for forbidden in (f"{stem}.weight_scale", f"{stem}.weight_scale_2"):
                    if forbidden in headers:
                        die(f"[verify] kept tensor has stray scale: {forbidden}")
            n_keep += 1
        else:
            stem = name[: -len(".weight")]
            out_f, in_f = shape
            w = headers.get(f"{stem}.weight")
            s = headers.get(f"{stem}.weight_scale")
            s2 = headers.get(f"{stem}.weight_scale_2")
            if w is None or s is None or s2 is None:
                die(f"[verify] incomplete NVFP4 triple for {stem}")
            if w["dtype"] != "U8" or w["shape"] != [out_f, in_f // 2]:
                die(f"[verify] {stem}.weight wrong: {w}")
            if s["dtype"] != "F8_E4M3" or s["shape"] != [out_f, in_f // BLOCK]:
                die(f"[verify] {stem}.weight_scale wrong: {s}")
            # 0-dim scalar is the only shape the mint writes (reshape(())); observed
            # in the micro end-to-end run. Fail loudly if that ever drifts.
            if s2["dtype"] != "F32" or s2["shape"] != []:
                die(f"[verify] {stem}.weight_scale_2 wrong: {s2}")
            n_quant += 1

    expected_out = n_keep + 3 * n_quant
    if len(headers) != expected_out:
        die(f"[verify] output tensor count {len(headers)} != expected {expected_out}")
    total = index["metadata"]["total_size"]
    print(f"[verify] OK: {n_quant} NVFP4 triples + {n_keep} kept tensors, "
          f"{total / 1e9:.1f} GB total")


# ----------------------------------------------------------------------------- main

def main():
    device = preflight()
    cfg = json.loads((SRC_DIR / "config.json").read_text())
    kda, full, mtp = load_layer_split(cfg)
    classify = make_classifier(kda, full, mtp)
    weight_map, plan = census(classify)

    # cross-check config layer split against checkpoint reality
    for i in kda:
        if f"{LAYER_PREFIX}{i}.self_attn.q_proj.weight" not in plan:
            die(f"config says layer {i} is KDA but checkpoint lacks its q_proj")
    for i in list(full) + [mtp]:
        if f"{LAYER_PREFIX}{i}.self_attn.q_a_proj.weight" not in plan:
            die(f"config says layer {i} is MLA/DSA but checkpoint lacks its q_a_proj")

    mint(device, weight_map, plan)
    write_configs(plan)
    verify(plan)
    print("MINT-DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 — loud single exit point
        import traceback
        traceback.print_exc()
        print(f"MINT-FAILED: {e}", file=sys.stderr)
        print("MINT-FAILED")
        sys.exit(1)
