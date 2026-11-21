#!/usr/bin/env python3
"""Lane E CPU oracle for Inkling 2 Mini (CheckpointParity / TinyParity).

Runs the transformers PR #51050 model code (head 6173a2f679ff4ed393afc499aefc569da7e68bc1) with ONE change:
the expert weights stay in their shipped quantized bytes (NVFP4 / MXFP8) and are dequantized per routed
expert. Arms (the "weight program" of the experts):

  exact  : W = fl64(code * scale * amax / denominator) (NVFP4) or e4m3 * 2^(s-127) (MXFP8, exact).
           Expert linears accumulate in f64 on the activations and round to the activation dtype.
  f32w   : the PR's own f32 dequant arithmetic (fl32(fl32(code*scale) * fl32(amax/den))), no BF16 cast.
  bf16w  : RED ARM. The PR's stock load: the f32 dequant above cast to BF16 (rounds code*scale*amax/2688).

Activation dtype (--dtype): f32 (default), f64 (source-transformed copy of the PR module, tiny fixture),
bf16 (stock PR activations; with --arm bf16w this is the unmodified PR program).

No file here is part of memra. External engines are oracles only.
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PR_HEAD = "6173a2f679ff4ed393afc499aefc569da7e68bc1"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def e4m3_lut():
    lut = np.zeros(256, dtype=np.float64)
    for b in range(256):
        e = (b >> 3) & 0xF
        mnt = b & 7
        if e == 0:
            v = (mnt / 8.0) * 2.0 ** -6
        elif e == 15 and mnt == 7:
            v = float("nan")
        else:
            v = (1 + mnt / 8.0) * 2.0 ** (e - 7)
        lut[b] = -v if b & 0x80 else v
    return lut


E4M3 = e4m3_lut()
E4M3_F32 = E4M3.astype(np.float32)
_e2 = np.array(E2M1, dtype=np.float64)
_byte = np.arange(256)
BYTE2 = np.stack((_e2[_byte & 0xF], _e2[_byte >> 4]), axis=-1)  # low nibble = even element
BYTE2_F32 = BYTE2.astype(np.float32)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class STFile:
    """Raw safetensors reader (mmap), independent of torch's float8 dtype."""

    NP = {"U8": np.uint8, "F8_E4M3": np.uint8, "F32": np.dtype("<f4"), "BF16": np.dtype("<u2"), "F64": np.dtype("<f8"),
          "I64": np.dtype("<i8"), "I32": np.dtype("<i4")}

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.h = json.loads(f.read(n))
        self.base = 8 + n
        self.h.pop("__metadata__", None)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def keys(self):
        return list(self.h)

    def info(self, k):
        return self.h[k]["dtype"], self.h[k]["shape"]

    def raw(self, k):
        d = self.h[k]
        b, e = d["data_offsets"]
        return self.mm[self.base + b:self.base + e].view(self.NP[d["dtype"]]).reshape(d["shape"])


class ExpertStore:
    """Quantized expert tensors of one physical layer, read from the checkpoint bytes."""

    def __init__(self, st, layer):
        p = f"model.llm.layers.{layer}.mlp.experts."
        self.layer = layer
        self.t = {}
        for nm in ("w13", "w2"):
            wd, _ = st.info(p + nm + "_weight")
            d = {"w": st.raw(p + nm + "_weight"), "s": st.raw(p + nm + "_weight_scale")}
            if wd == "U8":
                d["kind"] = "nvfp4"
                d["amax"] = np.array(st.raw(p + nm + "_weight_amax"))
                d["den"] = np.array(st.raw(p + nm + "_weight_outer_scale_denominator"))
                assert st.info(p + nm + "_weight_scale")[0] == "F8_E4M3"
            else:
                assert wd == "F8_E4M3" and st.info(p + nm + "_weight_scale")[0] == "U8"
                d["kind"] = "mxfp8"
            self.t[nm] = d
        self.kind = self.t["w13"]["kind"]
        self.num_experts = self.t["w13"]["w"].shape[0]

    def _codescale32(self, d, e):
        """val(code) * e4m3(block scale) in f32 (exact: 3 + 4 significand bits), [rows, blocks, 16]."""
        sb = np.asarray(d["s"][e])
        if ((sb & 0x7F) == 0x7F).any():
            raise ValueError(f"NaN scale byte in layer {self.layer} expert {e}")
        w = torch.from_numpy(np.ascontiguousarray(d["w"][e])).long()
        sc = torch.from_numpy(E4M3_F32[sb])
        return torch.from_numpy(BYTE2_F32)[w].view(w.shape[0], sc.shape[1], 16) * sc[..., None]

    def exact(self, nm, e):
        """f64 weight [rows, cols] in checkpoint row order; one correctly rounded division for NVFP4, exact for MXFP8."""
        d = self.t[nm]
        if d["kind"] == "nvfp4":
            v = self._codescale32(d, e).double() * float(d["amax"][e])  # exact in f64: 7 + 24 bits
            v = v / float(d["den"][0])
        else:
            wb = np.asarray(d["w"][e])
            if ((wb & 0x7F) == 0x7F).any():
                raise ValueError(f"NaN weight byte in layer {self.layer} expert {e}")
            ex = torch.from_numpy(np.asarray(d["s"][e]).astype(np.float64))
            w = torch.from_numpy(np.ascontiguousarray(wb)).long()
            v = torch.from_numpy(E4M3_F32)[w].view(w.shape[0], ex.shape[1], 32).double() * torch.exp2(ex - 127.0)[..., None]
        return v.reshape(v.shape[0], -1)

    def codescale(self, nm, e):
        """NVFP4 only: f64 val(code) * e4m3(block scale), without the per-expert amax / denominator (exact)."""
        d = self.t[nm]
        assert d["kind"] == "nvfp4"
        v = self._codescale32(d, e).double()
        return v.reshape(v.shape[0], -1)

    def pr_f32(self, nm, e):
        """The PR's Inkling2LoopDequantizeExperts arithmetic in f32, before its cast to the param dtype."""
        d = self.t[nm]
        w = torch.from_numpy(np.ascontiguousarray(d["w"][e]))
        rows = w.shape[0]
        if d["kind"] == "nvfp4":
            block_scale = torch.from_numpy(E4M3_F32[np.asarray(d["s"][e])])
            expert_factor = torch.from_numpy(d["amax"].astype(np.float32)) / torch.from_numpy(d["den"].astype(np.float32))
            values = torch.from_numpy(BYTE2_F32)[w.long()].view(*block_scale.shape, 16) * block_scale[..., None]
            values = values * expert_factor[e]
        else:
            block_scale = torch.from_numpy(np.asarray(d["s"][e]).astype(np.float32))
            values = torch.from_numpy(E4M3_F32)[w.long()].view(*block_scale.shape, 32)
            values = values * torch.exp2(block_scale - 127)[..., None]
        return values.reshape(rows, -1)


# ---- activation quantizers of the vLLM Blackwell program (lane C notes, src/vllm/NOTES.md section 2) ----
FP32_MAX = 3.4028234663852886e38
_E4M3_POS = torch.from_numpy(E4M3[:127].copy())  # bytes 0x00..0x7E, increasing, f64


def bf16r(t):
    return t.to(torch.bfloat16).float()


def e4m3_rne(v):
    """f32 magnitudes in [0, 448] -> nearest e4m3 value, ties to the even byte. Assumes the triton cast is RNE."""
    v64 = v.double()
    hi = torch.searchsorted(_E4M3_POS, v64.contiguous()).clamp(1, 126)
    lo = hi - 1
    a, b = _E4M3_POS[lo], _E4M3_POS[hi]
    da, db = v64 - a, b - v64
    pick_hi = (db < da) | ((db == da) & (hi % 2 == 0))
    return torch.where(pick_hi, b, a).float()


def e2m1_rne(x):
    a = x.abs()
    m = torch.full_like(a, 6.0)
    for thr, val, incl in ((5.0, 4.0, True), (3.5, 3.0, False), (2.5, 2.0, True), (1.75, 1.5, False), (1.25, 1.0, True),
                           (0.75, 0.5, False), (0.25, 0.0, True)):
        m = torch.where((a <= thr) if incl else (a < thr), torch.full_like(a, val), m)
    return torch.copysign(m, x)


def _pairsum16(e):
    e = e.reshape(*e.shape[:-1], 8, 2).sum(-1)
    e = e.reshape(*e.shape[:-1], 4, 2).sum(-1)
    e = e.reshape(*e.shape[:-1], 2, 2).sum(-1)
    return e.sum(-1)


def nvfp4_act_quant(x):
    """x f32 [R, K]. Returns (val(code) * block scale [R, K] f32, row amax [R]); four-over-six by block MAE."""
    f = torch.float32
    R, K = x.shape
    amax = x.abs().amax(dim=1)
    enc = torch.clamp(torch.tensor(1536.0, dtype=f) / amax, max=FP32_MAX)
    dec = torch.tensor(1.0, dtype=f) / enc
    outer = amax * torch.tensor(1.0 / 1536.0, dtype=f)
    xb = x.reshape(R, K // 16, 16)
    bmax = xb.abs().amax(dim=-1)
    cand = []
    for c in (1.0 / 6.0, 0.25):
        s = e4m3_rne(torch.clamp(bmax * (enc * torch.tensor(c, dtype=f))[:, None], max=448.0))
        inv = torch.clamp(torch.tensor(1.0, dtype=f) / (s * dec[:, None]), max=FP32_MAX)
        code = e2m1_rne(xb * inv[..., None])
        val = code * s[..., None]
        err = _pairsum16((val * outer[:, None, None] - xb).abs())
        cand.append((val, err))
    choose4 = cand[1][1] < cand[0][1]
    vals = torch.where(choose4[..., None], cand[1][0], cand[0][0])
    return vals.reshape(R, K), amax, float(choose4.float().mean())


def mxfp8_act_quant(x):
    """x f32 [R, K] -> dequantized e4m3 * 2^(exp-127) values [R, K] f32 (exact), per 32-element block."""
    R, K = x.shape
    xb = x.reshape(R, K // 32, 32)
    amax = xb.abs().amax(dim=-1)
    raw = (amax * torch.tensor(1.0 / 448.0, dtype=torch.float32)).numpy()
    bits = raw.view(np.uint32).astype(np.uint64)
    exponent = (((bits + 0x007FFFFF) & 0x7F800000) >> 23).astype(np.int64)
    exponent = np.where(raw <= np.float32(2.0 ** -127), 0, np.minimum(exponent, 254))
    scale = np.exp2(exponent.astype(np.float64) - 127.0)  # exponent 0 -> 2^-127
    inverse = np.where(exponent == 0, 2.0 ** 127, 1.0 / scale)
    y = (xb.double() * torch.from_numpy(inverse)[..., None]).float()  # power-of-two scaling: exact unless it overflows
    y = torch.clamp(y, -448.0, 448.0)
    val = torch.copysign(e4m3_rne(y.abs()), y)
    return (val.double() * torch.from_numpy(scale)[..., None]).float().reshape(R, K)


def load_modeling(f64):
    import transformers.models.inkling2_loop.modeling_inkling2_loop as m
    if not f64:
        return m
    src = open(m.__file__).read()
    src64 = src.replace(".float()", ".double()").replace("torch.float32", "torch.float64")
    path = m.__file__.replace("modeling_inkling2_loop.py", "modeling_inkling2_loop_f64.py")
    if not os.path.exists(path) or open(path).read() != src64:
        with open(path, "w") as f:
            f.write(src64)
    import importlib
    return importlib.import_module("transformers.models.inkling2_loop.modeling_inkling2_loop_f64")


def install_quant_experts(m, arm, acc_dtype):
    class QuantExperts(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.num_experts = config.n_routed_experts
            self.act_fn = m.Inkling2LoopSoftcappedSwiGLU(config)
            self.store = None
            self.arm = arm
            self.dequant_s = 0.0
            self.n_dequant = 0
            self.choose4 = []

        def weights(self, e, like):
            t0 = time.time()
            if self.arm == "exact":
                w13, w2 = self.store.exact("w13", e), self.store.exact("w2", e)
            else:
                w13, w2 = self.store.pr_f32("w13", e), self.store.pr_f32("w2", e)
                if self.arm == "bf16w":
                    w13, w2 = w13.to(torch.bfloat16), w2.to(torch.bfloat16)
                # the PR's Interleave(dim=1): checkpoint rows are (gate_0, up_0, gate_1, up_1, ...)
                w13 = torch.cat((w13[0::2], w13[1::2]), dim=0)
                w13, w2 = w13.to(like.dtype), w2.to(like.dtype)
            self.dequant_s += time.time() - t0
            self.n_dequant += 1
            return w13, w2

        def forward_actq(self, hidden_states, top_k_index, top_k_weights):
            """W4A4 / W8A8 expert program as lane C read it from the vLLM image (NOTES.md 2.2-2.5), simulated on CPU.
            Known gaps to the GPU program: exact-math tanh/sigmoid, GEMM sums taken in f64 and rounded once to f32,
            e4m3 casts assumed RNE, the 8 expert rows summed in ascending expert id, PR router, f32 stream outside."""
            assert hidden_states.dtype == torch.float32
            st = self.store
            nv = st.kind == "nvfp4"
            x = bf16r(hidden_states)
            if nv:
                xv, xamax, c4 = nvfp4_act_quant(x)
                self.choose4.append(c4)
                rcp = {nm: torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(1536.0, dtype=torch.float32) * torch.tensor(float(st.t[nm]["den"][0]), dtype=torch.float32)) for nm in ("w13", "w2")}
            else:
                xv = mxfp8_act_quant(x)
            final = torch.zeros_like(hidden_states)
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                e = int(expert_idx[0])
                if e == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[e])
                t0 = time.time()
                if nv:
                    w13, w2 = st.codescale("w13", e), st.codescale("w2", e)
                else:
                    w13, w2 = st.exact("w13", e), st.exact("w2", e)
                self.dequant_s += time.time() - t0
                self.n_dequant += 1
                acc = (xv[token_idx].double() @ w13.T).float()
                if nv:
                    acc = acc * ((xamax[token_idx] * torch.tensor(float(st.t["w13"]["amax"][e]), dtype=torch.float32)) * rcp["w13"])[:, None]
                gate_up = bf16r(acc)
                gamma = top_k_weights[token_idx, top_k_pos, None].float()
                y = bf16r(self.act_fn(gate_up[:, 0::2], gate_up[:, 1::2]) * gamma)
                if nv:
                    yv, yamax, _ = nvfp4_act_quant(y)
                    acc = (yv.double() @ w2.T).float()
                    acc = acc * ((yamax * torch.tensor(float(st.t["w2"]["amax"][e]), dtype=torch.float32)) * rcp["w2"])[:, None]
                else:
                    acc = (mxfp8_act_quant(y).double() @ w2.T).float()
                final.index_add_(0, token_idx, bf16r(acc))
            return bf16r(final)

        def forward(self, hidden_states, top_k_index, top_k_weights):
            if self.arm == "actq":
                return self.forward_actq(hidden_states, top_k_index, top_k_weights)
            final_hidden_states = torch.zeros_like(hidden_states, dtype=acc_dtype)
            with torch.no_grad():
                expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1).permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                w13, w2 = self.weights(int(expert_idx), current_state)
                if self.arm == "exact":
                    gate_up = (current_state.double() @ w13.T).to(current_state.dtype)
                    current = self.act_fn(gate_up[:, 0::2], gate_up[:, 1::2])
                    current = (current.double() @ w2.T).to(current_state.dtype)
                else:
                    gate, up = F.linear(current_state, w13).chunk(2, dim=-1)
                    current = F.linear(self.act_fn(gate, up), w2)
                current = current * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current.to(final_hidden_states.dtype))
            return final_hidden_states.to(hidden_states.dtype)

    m.Inkling2LoopExperts = QuantExperts
    orig_init = m.Inkling2LoopPreTrainedModel._init_weights

    def _init_weights(self, module):
        if isinstance(module, QuantExperts):
            return
        orig_init(self, module)

    m.Inkling2LoopPreTrainedModel._init_weights = _init_weights
    return QuantExperts


def build_text_config(m, ckpt_cfg, overrides):
    tc = dict(ckpt_cfg["text_config"])
    tc.pop("model_type", None)
    tc.pop("torch_dtype", None)
    local = tc.pop("local_layer_ids", None)
    if local is not None and "layer_types" not in tc:
        tc["layer_types"] = ["hybrid_sliding" if i in local else "hybrid" for i in range(tc["num_hidden_layers"])]
    cfg_cls = sys.modules[m.Inkling2LoopTextConfig.__module__].Inkling2LoopTextConfig
    cfg = cfg_cls.from_dict(tc)
    for k, v in overrides.items():
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v))
    src = ckpt_cfg["text_config"]
    chk = {
        "moe_intermediate_size": src["intermediate_size"], "conv_kernel_size": src["sconv_kernel_size"],
        "route_scale": src["route_scale"], "loop_layer_index_start": src["loop_config"]["core_start"],
        "loop_layer_index_end": src["loop_config"]["core_end"], "loop_num_passes": src["loop_config"]["num_passes"],
        "loop_gate_rank": src["loop_config"]["gate_rank"], "n_routed_experts": src["n_routed_experts"],
        "sliding_window_size": src["sliding_window_size"], "rel_extent": src["rel_extent"], "d_rel": src["d_rel"],
        "unpadded_vocab_size": src["unpadded_vocab_size"], "vocab_size": src["vocab_size"],
        "n_layers_per_attn_res_block": src["n_layers_per_attn_res_block"],
        "mlp_gate_softcap": src["mlp_gate_softcap"], "mlp_up_softcap": src["mlp_up_softcap"],
        "final_logit_softcapping": src["final_logit_softcapping"], "rms_norm_eps": src["rms_norm_eps"],
        "qk_l2_clip_threshold": src["qk_l2_clip_threshold"], "rel_l2_clip_threshold": src["rel_l2_clip_threshold"],
        "num_experts_per_tok": src["num_experts_per_tok"], "log_scaling_alpha": src["log_scaling_alpha"],
    }
    for k, v in chk.items():
        if k in overrides:
            continue
        assert getattr(cfg, k) == v, (k, getattr(cfg, k), v)
    if "log_scaling_n_floor" not in overrides:
        assert cfg.log_scaling_n_floor == src["log_scaling_n_floor"]
    return cfg


def weights_file(ckpt):
    for n in ("model-00001-of-00001.safetensors", "model.safetensors"):
        p = os.path.join(ckpt, n)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(ckpt)


def is_expert_key(k):
    return ".mlp.experts.w13_weight" in k or ".mlp.experts.w2_weight" in k


def cmd_filter(args):
    """Write the non-expert text tensors (checkpoint names, bytes untouched) for the PR loader."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    src = weights_file(args.ckpt)
    os.makedirs(args.work, exist_ok=True)
    out = {}
    with safe_open(src, framework="pt") as f:
        for k in f.keys():
            if k.startswith("model.llm.") and not is_expert_key(k):
                out[k] = f.get_tensor(k)
    save_file(out, os.path.join(args.work, "model.safetensors"), metadata={"format": "pt"})
    with open(os.path.join(args.work, "generation_config.json"), "w") as g:
        g.write("{}")
    print("filtered", len(out), "tensors ->", args.work)


def load_model(ckpt, work, arm, dtype, overrides, stock=False):
    f64 = dtype == "f64"
    m = load_modeling(f64)
    acc = torch.float64 if f64 else torch.float32
    ckpt_cfg = json.load(open(os.path.join(ckpt, "config.json")))
    if stock:
        # the unmodified PR: its loader dequantizes every expert to the param dtype
        cfg = build_text_config(m, ckpt_cfg, overrides)
        tdt = {"f64": torch.float64, "f32": torch.float32, "bf16": torch.bfloat16}[dtype]
        model, info = m.Inkling2LoopForCausalLM.from_pretrained(ckpt, config=cfg, dtype=tdt, output_loading_info=True)
        print("stock loading info", {k: (v if not isinstance(v, (list, set)) else sorted(v)[:8]) for k, v in info.items()}, flush=True)
        model.eval()
        return m, model, cfg
    QE = install_quant_experts(m, arm, acc)
    cfg = build_text_config(m, ckpt_cfg, overrides)
    model, info = m.Inkling2LoopForCausalLM.from_pretrained(work, config=cfg, dtype=torch.float32, output_loading_info=True)
    bad = {k: v for k, v in info.items() if v and k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
    assert not bad, bad
    model = {"f64": model.double, "f32": model.float, "bf16": lambda: model.to(torch.bfloat16)}[dtype]()
    if dtype == "bf16":
        # same modules the PR keeps in fp32 (_keep_in_fp32_modules_strict)
        for name, p in list(model.named_parameters()) + list(model.named_buffers()):
            if any(s in name for s in model._keep_in_fp32_modules_strict):
                p.data = p.data.float()
    model.eval()
    st = STFile(weights_file(ckpt))
    for i, layer in enumerate(model.model.layers):
        assert isinstance(layer.mlp.experts, QE)
        layer.mlp.experts.store = ExpertStore(st, i)
    return m, model, cfg


class Keep:
    def __init__(self, spec, total):
        self.all = spec == "all"
        self.ranges = []
        if not self.all:
            for part in spec.split(","):
                a, b = part.split(":")
                a, b = int(a), int(b)
                if a < 0:
                    a += total
                if b <= 0:
                    b += total
                self.ranges.append((max(a, 0), min(b, total)))

    def idx(self, pos0, n):
        if self.all:
            return list(range(n))
        return [i for i in range(n) if any(a <= pos0 + i < b for a, b in self.ranges)]

    def positions(self, total):
        return self.idx(0, total)


class Recorder:
    def __init__(self, model, cfg, total, resid_keep, logits_keep, trace, topk=16):
        self.on = True
        self.cfg = cfg
        self.n_slots = cfg.total_num_hidden_layers
        self.resid_keep = Keep(resid_keep, total)
        self.logits_keep = Keep(logits_keep, total)
        self.trace = trace
        self.topk = topk
        self.pos0 = 0
        self.slot = None
        self.call = 0
        self.order = []
        S = self.n_slots
        self.resid_attn = [[] for _ in range(S)]
        self.resid_ffn = [[] for _ in range(S)]
        self.moe_in = [[] for _ in range(S)]
        self.moe_out = [[] for _ in range(S)]
        self.route_ids = [[] for _ in range(S)]
        self.route_w = [[] for _ in range(S)]
        self.route_margin = [[] for _ in range(S)]
        self.final_hidden = []
        self.logits_pre = []
        self.logits_post = []
        self.top_ids = []
        self.top_pre = []
        self.traces = {}
        self.last_post = None
        self.last_pre = None
        self.handles = []
        lm = model.model
        for layer in lm.layers:
            self.handles.append(layer.register_forward_pre_hook(self._layer_pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(self._layer_post))
            self.handles.append(layer.mlp_input.register_forward_pre_hook(self._mlp_input_pre))
            self.handles.append(layer.mlp.gate.register_forward_hook(self._gate_post))
            self.handles.append(layer.mlp.register_forward_hook(self._moe_post))
            if trace:
                names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.relative_position_bias",
                         "self_attn.sink_proj", "self_attn", "post_attention_convolution", "mlp", "post_mlp_convolution",
                         "attention_input", "mlp_input", "q_message_gate", "k_message_gate", "v_message_gate",
                         "mlp_message_gate"]
                for nm in names:
                    try:
                        sub = layer.get_submodule(nm)
                    except AttributeError:
                        continue
                    if sub is None:
                        continue
                    self.handles.append(sub.register_forward_hook(self._make_trace(nm)))
        self.handles.append(lm.norm.register_forward_hook(self._norm_post))
        self.handles.append(model.lm_head.register_forward_hook(self._head_post))

    def _np(self, t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.cpu().numpy().copy()

    def _tr(self, name, t):
        if isinstance(t, (tuple, list)):
            t = torch.stack([x for x in t if x is not None], dim=0)
        self.traces[f"s{self.slot:02d}.{name}.call{self.call}"] = self._np(t)

    def _make_trace(self, nm):
        def hook(mod, args, out):
            if not self.on:
                return
            if nm == "self_attn":
                out = out[0]
            self._tr(nm.replace("self_attn.", "attn."), out)
        return hook

    def _layer_pre(self, mod, args, kwargs):
        if not self.on:
            return
        self.slot = mod.self_attn.layer_idx + kwargs.get("cache_offset", 0)
        if self.call == 0:
            self.order.append(self.slot)
        if self.trace:
            self._tr("layer_in", args[0])
            msg = args[2] if len(args) > 2 else kwargs.get("messages")
            if msg is not None:
                self._tr("messages", msg)

    def _mlp_input_pre(self, mod, args):
        if not self.on:
            return
        x = args[1][0]
        idx = self.resid_keep.idx(self.pos0, x.shape[0])
        if idx:
            self.resid_attn[self.slot].append(self._np(x[idx]))

    def _layer_post(self, mod, args, out):
        if not self.on:
            return
        x = out[0]
        idx = self.resid_keep.idx(self.pos0, x.shape[0])
        if idx:
            self.resid_ffn[self.slot].append(self._np(x[idx]))

    def _moe_post(self, mod, args, out):
        if not self.on:
            return
        x, y = args[0][0], out[0]
        idx = self.resid_keep.idx(self.pos0, x.shape[0])
        if idx:
            self.moe_in[self.slot].append(self._np(x[idx]))
            self.moe_out[self.slot].append(self._np(y[idx]))

    def _gate_post(self, mod, args, out):
        if not self.on:
            return
        logits, w, ids = out
        order = ids.argsort(dim=-1)
        self.route_ids[self.slot].append(ids.gather(1, order).to(torch.int32).numpy().copy())
        self.route_w[self.slot].append(self._np(w.gather(1, order)))
        sel = mod.score_fn(logits) + mod.e_score_correction_bias
        top = sel.topk(mod.top_k + 1, dim=-1).values
        self.route_margin[self.slot].append(self._np(top[:, mod.top_k - 1] - top[:, mod.top_k]))

    def _norm_post(self, mod, args, out):
        if not self.on:
            return
        self.final_hidden.append(self._np(out[0]))

    def _head_post(self, mod, args, out):
        if not self.on:
            return
        self.last_pre = out[0].detach().float() if out.dtype == torch.bfloat16 else out[0].detach()

    def end_call(self, post_logits):
        post = post_logits[0].detach()
        pre = self.last_pre
        n = post.shape[0]
        idx = self.logits_keep.idx(self.pos0, n)
        if idx:
            self.logits_pre.append(self._np(pre[idx]))
            self.logits_post.append(self._np(post[idx]))
        tv, ti = pre.topk(self.topk, dim=-1)
        self.top_ids.append(ti.to(torch.int32).numpy().copy())
        self.top_pre.append(self._np(tv))
        self.last_post = post[-1]
        self.pos0 += n
        self.call += 1

    def remove(self):
        for h in self.handles:
            h.remove()


def slot_table(cfg):
    n, a, b, p = cfg.num_hidden_layers, cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
    rows = []
    for s in range(cfg.total_num_hidden_layers):
        if s < n:
            phys, pas = s, 0
        else:
            phys = a + (s - n) % (b - a)
            pas = 1 + (s - n) // (b - a)
        rows.append({"slot": s, "physical_layer": phys, "pass": pas, "attention": cfg.layer_types[phys]})
    return rows


def save_tensors(path, tensors, manifest, relroot):
    from safetensors.numpy import save_file
    tensors = {k: np.ascontiguousarray(v) for k, v in tensors.items()}
    save_file(tensors, path)
    entry = {"file": os.path.relpath(path, relroot), "sha256": sha256_file(path), "bytes": os.path.getsize(path), "tensors": {}}
    for k, v in tensors.items():
        entry["tensors"][k] = {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": sha256_bytes(v.tobytes())}
    manifest.append(entry)


def cmd_capture(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    overrides = dict(kv.split("=") for kv in args.override)
    t_load = time.time()
    m, model, cfg = load_model(args.ckpt, args.work, args.arm, args.dtype, overrides, stock=args.stock)
    t_load = time.time() - t_load
    prompt = json.load(open(args.prompt))
    ids = prompt["input_ids"]
    T = len(ids)
    G = args.greedy
    total = T + max(G - 1, 0)
    rec = Recorder(model, cfg, total, args.resid_keep, args.logits_keep, args.trace)
    os.makedirs(args.out, exist_ok=True)
    cache = None
    timing = {"load_s": t_load, "prefill_chunks_s": [], "step_s": []}
    chunk = args.chunk if args.chunk > 0 else T
    with torch.no_grad():
        for s in range(0, T, chunk):
            t0 = time.time()
            x = torch.tensor(ids[s:s + chunk], dtype=torch.long)[None]
            out = model(input_ids=x, cache=cache, use_cache=True)
            cache = out.cache
            rec.end_call(out.logits)
            timing["prefill_chunks_s"].append(time.time() - t0)
            print(f"prefill {s}+{x.shape[1]}/{T} {timing['prefill_chunks_s'][-1]:.1f}s", flush=True)
        greedy, margins = [], []
        for i in range(G):
            top2 = rec.last_post.topk(2)
            tok = int(torch.argmax(rec.last_post))
            greedy.append(tok)
            margins.append(float(top2.values[0] - top2.values[1]))
            if i == G - 1:
                break
            t0 = time.time()
            out = model(input_ids=torch.tensor([[tok]], dtype=torch.long), cache=cache, use_cache=True)
            cache = out.cache
            rec.end_call(out.logits)
            timing["step_s"].append(time.time() - t0)
            print(f"step {i} tok {tok} {timing['step_s'][-1]:.1f}s", flush=True)
    all_ids = ids + greedy[:-1] if G > 0 else ids
    assert rec.pos0 == len(all_ids) == total
    npdt = np.float64 if args.dtype == "f64" else np.float32
    S = rec.n_slots
    manifest = []
    rpos = rec.resid_keep.positions(total)
    lpos = rec.logits_keep.positions(total)
    cat = lambda xs: np.concatenate(xs, axis=0)
    root = args.out
    if rpos:
        save_tensors(os.path.join(root, "resid_attn.safetensors"), {"resid_attn": np.stack([cat(rec.resid_attn[s]) for s in range(S)]), "positions": np.array(rpos, dtype=np.int64)}, manifest, root)
        save_tensors(os.path.join(root, "resid_ffn.safetensors"), {"resid_ffn": np.stack([cat(rec.resid_ffn[s]) for s in range(S)]), "positions": np.array(rpos, dtype=np.int64)}, manifest, root)
    if rpos:
        save_tensors(os.path.join(root, "moe_in.safetensors"), {"moe_in": np.stack([cat(rec.moe_in[s]) for s in range(S)]), "positions": np.array(rpos, dtype=np.int64)}, manifest, root)
        save_tensors(os.path.join(root, "moe_out.safetensors"), {"moe_out": np.stack([cat(rec.moe_out[s]) for s in range(S)]), "positions": np.array(rpos, dtype=np.int64)}, manifest, root)
    save_tensors(os.path.join(root, "router.safetensors"), {
        "route_ids": np.stack([cat(rec.route_ids[s]) for s in range(S)]),
        "route_weights": np.stack([cat(rec.route_w[s]) for s in range(S)]),
        "route_margin": np.stack([cat(rec.route_margin[s]) for s in range(S)]),
    }, manifest, root)
    save_tensors(os.path.join(root, "head.safetensors"), {
        "final_hidden": cat(rec.final_hidden), "top_ids": cat(rec.top_ids), "top_logits_pre": cat(rec.top_pre),
    }, manifest, root)
    if lpos:
        save_tensors(os.path.join(root, "logits_pre.safetensors"), {"logits_pre": cat(rec.logits_pre), "positions": np.array(lpos, dtype=np.int64)}, manifest, root)
        save_tensors(os.path.join(root, "logits_post.safetensors"), {"logits_post": cat(rec.logits_post), "positions": np.array(lpos, dtype=np.int64)}, manifest, root)
    save_tensors(os.path.join(root, "tokens.safetensors"), {
        "input_ids": np.array(ids, dtype=np.int64), "greedy_tokens": np.array(greedy, dtype=np.int64),
        "all_ids": np.array(all_ids, dtype=np.int64),
    }, manifest, root)
    if args.trace:
        save_tensors(os.path.join(root, "trace.safetensors"), rec.traces, manifest, root)
    checks = {}
    if args.mono_check:
        rec.on = False
        with torch.no_grad():
            mono = model(input_ids=torch.tensor(all_ids, dtype=torch.long)[None], use_cache=False).logits[0]
        if args.dtype == "bf16":
            mono = mono.float()
        mono = mono.numpy()
        if lpos:
            taped = cat(rec.logits_post)
            d = np.abs(mono[lpos].astype(np.float64) - taped.astype(np.float64))
            checks["mono_vs_cached_post_logits_max_abs"] = float(d.max())
            checks["mono_vs_cached_post_logits_mean_abs"] = float(d.mean())
            checks["mono_vs_cached_argmax_equal"] = int((mono[lpos].argmax(-1) == taped.argmax(-1)).sum())
            checks["mono_vs_cached_positions"] = len(lpos)
        # causality probe: change the last token, earlier positions must not move
        if len(all_ids) > 2:
            alt = list(all_ids)
            alt[-1] = (alt[-1] + 1) % cfg.unpadded_vocab_size
            with torch.no_grad():
                mono2 = model(input_ids=torch.tensor(alt, dtype=torch.long)[None], use_cache=False).logits[0]
            mono2 = (mono2.float() if args.dtype == "bf16" else mono2).numpy()
            checks["causality_max_abs_prefix_delta"] = float(np.abs(mono2[:-1].astype(np.float64) - mono[:-1].astype(np.float64)).max())
            checks["causality_last_pos_delta"] = float(np.abs(mono2[-1].astype(np.float64) - mono[-1].astype(np.float64)).max())
    dq = [(getattr(l.mlp.experts, "dequant_s", 0.0), getattr(l.mlp.experts, "n_dequant", 0)) for l in model.model.layers]
    timing["dequant_s"] = sum(a for a, _ in dq)
    timing["n_dequant"] = sum(b for _, b in dq)
    c4 = [v for l in model.model.layers for v in getattr(l.mlp.experts, "choose4", [])]
    if c4:
        checks["actq_fraction_blocks_choosing_scale4"] = float(np.mean(c4))
    timing["prefill_s"] = sum(timing["prefill_chunks_s"])
    timing["steps_s"] = sum(timing["step_s"])
    import transformers
    eos = json.load(open(os.path.join(args.ckpt, "config.json"))).get("eos_token_id")
    meta = {
        "what": "Inkling 2 Mini oracle tape (transformers PR #51050 code, experts kept quantized)",
        "pr_head": PR_HEAD, "transformers_version": transformers.__version__, "torch_version": torch.__version__,
        "arm": "stock-pr" if args.stock else args.arm,
        "arm_note": {
            "exact": "float-activation class: exact f64 dequant per routed expert, expert GEMMs summed in f64 and rounded to the activation dtype, routing weight applied AFTER the down projection (PR order)",
            "f32w": "float activations, PR f32 dequant arithmetic, PR order",
            "bf16w": "RED ARM: float activations against the PR's BF16-cast dequantized weights",
            "actq": "act-quant class (the default target): W4A4 per-token amax four-over-six on NVFP4 layers, W8A8 on MXFP8 layers, routing weight multiplied BEFORE the down projection and its quantization, bf16 rounding points inside the MoE as in the vLLM program; CPU simulation from lane C's NOTES.md section 2, not a capture of the vLLM binary; outside the MoE the stream is f32 and the router is the PR's",
        }.get(args.arm), "activation_dtype": args.dtype, "stored_dtype": str(np.dtype(npdt)),
        "config_overrides": overrides, "checkpoint_dir": os.path.abspath(args.ckpt),
        "weights_sha256": prompt.get("weights_sha256"),
        "prompt_name": prompt.get("name"), "prompt_note": prompt.get("note"), "rendered": prompt.get("rendered"),
        "n_prompt_tokens": T, "n_greedy": G, "greedy_tokens": greedy, "greedy_top2_margin_post": margins,
        "greedy_text": prompt.get("_decode"), "eos_token_id": eos,
        "eos_in_greedy_at": [i for i, t in enumerate(greedy) if t == eos],
        "total_positions": total, "prefill_chunk": chunk, "decode": "1-token steps on the PR DynamicCache",
        "resid_keep": args.resid_keep, "logits_keep": args.logits_keep,
        "slots": slot_table(cfg), "slot_execution_order": rec.order,
        "tensor_notes": {
            "resid_attn": "[slot, kept position, hidden] residual stream after attention: input of the FFN depth mix (attn conv output, plus the block residual except on a block's first layer)",
            "resid_ffn": "[slot, kept position, hidden] layer output (after the FFN conv and residual add)",
            "moe_in": "[slot, kept position, hidden] input of the MoE block (after the depth mix, the loop controller and mlp_norm): router and expert input",
            "moe_out": "[slot, kept position, hidden] output of the MoE block before the FFN short conv; with moe_in, route_ids and route_weights this is a teacher-forced per-slot MoE test",
            "route_ids": "[slot, position, top_k] selected experts sorted by ascending id (the PR accumulates in ascending id)",
            "route_weights": "[slot, position, top_k] sigmoid score / sum(selected) * route_scale, same order as route_ids",
            "route_margin": "[slot, position] (score+bias) of the k-th pick minus the (k+1)-th: near-zero means a tie-sensitive pick",
            "final_hidden": "[position, hidden] output of the final RMS norm (after the output depth mix)",
            "logits_pre": "[kept position, unpadded vocab] lm_head output before the softcap",
            "logits_post": "[kept position, unpadded vocab] cap*tanh(pre/cap)",
            "top_ids/top_logits_pre": "[position, 16] top-16 of logits_pre for every position",
            "greedy_tokens": "argmax(logits_post) at the last position, then each fed back; token i is predicted at position n_prompt-1+i",
        },
        "self_checks": checks, "timing": timing, "threads": args.threads, "files": manifest,
    }
    if args.tokenizer_decode:
        try:
            from tokenizers import Tokenizer
            tk = Tokenizer.from_file(os.path.join(args.ckpt, "tokenizer.json"))
            meta["greedy_text"] = tk.decode(greedy, skip_special_tokens=False)
        except Exception as ex:  # noqa
            meta["greedy_text"] = f"decode failed: {ex}"
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({"greedy": greedy, "checks": checks, "timing": {k: v for k, v in timing.items() if not isinstance(v, list)}}))
    print("CAPTURE_DONE", flush=True)


def cmd_dequant_check(args):
    """Layout check: my exact f64 dequant against the PR's Inkling2LoopDequantizeExperts on single experts."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts
    st = STFile(weights_file(args.ckpt))
    op = Inkling2LoopDequantizeExperts()

    class P:
        def __init__(self, dt):
            self.dt = dt

        def get_parameter(self, name):
            return torch.zeros(1, dtype=self.dt)

    res = []
    for layer in args.layers:
        store = ExpertStore(st, layer)
        for nm in ("w13", "w2"):
            d = store.t[nm]
            for e in args.experts:
                if e >= store.num_experts:
                    continue
                w = torch.from_numpy(np.ascontiguousarray(d["w"][e:e + 1]))
                s = torch.from_numpy(np.ascontiguousarray(d["s"][e:e + 1]))
                if d["kind"] == "nvfp4":
                    s = torch.from_numpy(E4M3_F32[s.numpy()])
                    inp = {"w": w, "s": s, "a": torch.from_numpy(d["amax"][e:e + 1].copy()), "d": torch.from_numpy(d["den"].copy())}
                else:
                    w = torch.from_numpy(E4M3_F32[w.numpy()])
                    inp = {"w": w, "s": s}
                pats = ["w", "s", "a", "d"]
                out = {}
                for dt in (torch.float32, torch.bfloat16):
                    out[dt] = op.convert(dict(inp), pats, ["t"], model=P(dt), full_layer_name="x")["t"][0].double()
                ex = store.exact(nm, e)
                mine32 = store.pr_f32(nm, e).double()
                nz = ex != 0
                rel_bf = ((out[torch.bfloat16] - ex).abs()[nz] / ex.abs()[nz])
                rel_f32 = ((out[torch.float32] - ex).abs()[nz] / ex.abs()[nz])
                r = {
                    "layer": layer, "kind": d["kind"], "tensor": nm, "expert": e, "shape": list(ex.shape),
                    "pr_f32_vs_my_pr_f32_max_abs": float((out[torch.float32] - mine32).abs().max()),
                    "pr_f32_vs_exact_max_rel": float(rel_f32.max()), "pr_f32_exact_equal_frac": float((out[torch.float32] == ex).double().mean()),
                    "pr_bf16_vs_exact_max_rel": float(rel_bf.max()), "pr_bf16_vs_exact_mean_rel": float(rel_bf.mean()),
                    "pr_bf16_exact_equal_frac": float((out[torch.bfloat16] == ex).double().mean()),
                    "abs_max": float(ex.abs().max()), "zero_frac": float((~nz).double().mean()),
                    "exact_sha256_f64": sha256_bytes(ex.numpy().tobytes()),
                }
                if d["kind"] == "nvfp4":
                    r["amax"] = float(d["amax"][e])
                    r["den"] = float(d["den"][0])
                res.append(r)
                print(json.dumps(r), flush=True)
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    print("DEQUANT_CHECK_DONE")


def cmd_compare(args):
    from safetensors.numpy import load_file
    res = {"a": args.a, "b": args.b, "tensors": {}}
    for fn in sorted(os.listdir(args.a)):
        if not fn.endswith(".safetensors") or not os.path.exists(os.path.join(args.b, fn)):
            continue
        A, B = load_file(os.path.join(args.a, fn)), load_file(os.path.join(args.b, fn))
        for k in sorted(set(A) & set(B)):
            x, y = A[k], B[k]
            r = {"shape_a": list(x.shape), "shape_b": list(y.shape)}
            if x.shape == y.shape:
                r["bitwise_equal"] = bool(x.dtype == y.dtype and x.tobytes() == y.tobytes())
                if x.dtype.kind == "f" or y.dtype.kind == "f":
                    d = np.abs(x.astype(np.float64) - y.astype(np.float64))
                    r["max_abs"] = float(d.max()) if d.size else 0.0
                    r["mean_abs"] = float(d.mean()) if d.size else 0.0
                    r["ref_rms"] = float(np.sqrt((x.astype(np.float64) ** 2).mean())) if d.size else 0.0
                    if k in ("resid_attn", "resid_ffn"):
                        r["max_abs_per_slot"] = [float(v) for v in d.reshape(d.shape[0], -1).max(axis=1)]
                    if k in ("logits_pre", "logits_post"):
                        r["argmax_equal"] = int((x.argmax(-1) == y.argmax(-1)).sum())
                        r["rows"] = int(x.shape[0])
                else:
                    r["equal_frac"] = float((x == y).mean()) if x.size else 1.0
                    if k == "route_ids":
                        rows = (x == y).all(axis=-1)
                        r["rows_equal_frac"] = float(rows.mean())
                        r["rows_equal_frac_per_slot"] = [float(v) for v in rows.mean(axis=1)]
            res["tensors"][fn + ":" + k] = r
    s = json.dumps(res, indent=1)
    if args.out:
        open(args.out, "w").write(s)
    brief = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in res["tensors"].items() if not k.startswith("trace")}
    print(json.dumps(brief, indent=1))
    if any(k.startswith("trace") for k in res["tensors"]):
        tr = [(v.get("max_abs", 0.0), k) for k, v in res["tensors"].items() if k.startswith("trace") and "max_abs" in v]
        print("trace worst:", sorted(tr, reverse=True)[:5])
    print("COMPARE_DONE")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("filter")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--work", required=True)
    a.set_defaults(fn=cmd_filter)
    a = sub.add_parser("capture")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--work", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--prompt", required=True)
    a.add_argument("--arm", choices=["exact", "f32w", "bf16w", "actq"], default="exact")
    a.add_argument("--dtype", choices=["f32", "f64", "bf16"], default="f32")
    a.add_argument("--greedy", type=int, default=32)
    a.add_argument("--chunk", type=int, default=0)
    a.add_argument("--resid-keep", default="all")
    a.add_argument("--logits-keep", default="all")
    a.add_argument("--override", action="append", default=[])
    a.add_argument("--trace", action="store_true")
    a.add_argument("--mono-check", action="store_true")
    a.add_argument("--tokenizer-decode", action="store_true")
    a.add_argument("--threads", type=int, default=16)
    a.add_argument("--stock", action="store_true", help="unmodified PR loader and experts (tiny fixture cross-check)")
    a.set_defaults(fn=cmd_capture)
    a = sub.add_parser("compare")
    a.add_argument("a")
    a.add_argument("b")
    a.add_argument("--out")
    a.set_defaults(fn=cmd_compare)
    a = sub.add_parser("dequant-check")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--layers", type=int, nargs="+", default=[0, 20])
    a.add_argument("--experts", type=int, nargs="+", default=[0, 127])
    a.add_argument("--out")
    a.set_defaults(fn=cmd_dequant_check)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
