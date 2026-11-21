#!/usr/bin/env python3
"""Module oracles for the inkling2 reference operations.

Each record is one module of the transformers PR 51050 model code (head 6173a2f679ff4ed393afc499aefc569da7e68bc1),
source-transformed to f64 by ink2_oracle.py, run alone on seeded inputs: its parameters, its inputs and its output.
The conversion ops the PR loads a checkpoint through (Interleave, the message-query merge, the vocabulary unpad and
the expert dequantizer) run on tensors in the CHECKPOINT layout, so a record also pins how the stored layout maps to
the module. memra-reference's inkling2 tests replay every record against the op of the same role.

Container: INK2AQ01 (name -> tag -> bytes), the format of the activation-quant goldens. Floats are f64 little
endian; parameters are f32-exact values stored as f64; "bytes_*" tags are raw stored bytes; "ids_*" are i64.

  python -I module_oracles.py --out crates/memra-reference/src/fixtures/inkling2-module-oracles.bin
"""
import argparse
import hashlib
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

import ink2_oracle as O

CONFIG = {
    "text_config": {
        "model_max_length": 4096, "hidden_size": 16, "num_hidden_layers": 4, "vocab_size": 24,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 4, "d_rel": 3, "rel_extent": 6,
        "q_bias": False, "o_bias": False, "log_scaling_n_floor": 7, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
        "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 2.0, "rel_l2_clip_threshold": 1.5,
        "local_layer_ids": [0, 2], "sconv_kernel_size": 4, "unpadded_vocab_size": 20, "final_logit_softcapping": 3.0,
        "swa_head_dim": 4, "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "sliding_window_size": 5,
        "n_layers_per_attn_res_block": 2, "attn_res_normalize_values": True, "n_routed_experts": 6,
        "num_experts_per_tok": 2, "n_shared_experts": 0, "intermediate_size": 8, "mlp_gate_softcap": 2.0,
        "mlp_up_softcap": 3.0, "route_scale": 2.0, "use_gate_bias": True,
        "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 3, "is_varloop": False},
    }
}
T = 12  # past the window (5), the relative extent (6) and the log-scale floor (7)


class Rng:
    def __init__(self, seed):
        self.seed = seed

    def normal(self, name, shape, std=1.0, mean=0.0):
        h = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        r = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return torch.from_numpy(r.normal(mean, std, size=shape).astype(np.float32).astype(np.float64))

    def integers(self, name, low, high, shape):
        h = hashlib.sha256(f"{self.seed}:{name}".encode()).digest()
        r = np.random.default_rng(int.from_bytes(h[:8], "little"))
        return r.integers(low, high, size=shape)


class Records:
    def __init__(self):
        self.records = {}

    def put(self, name, **blobs):
        out = {}
        for tag, value in blobs.items():
            if isinstance(value, bytes):
                out[tag] = value
            elif isinstance(value, torch.Tensor):
                assert value.dtype == torch.float64, (name, tag, value.dtype)
                out[tag] = value.detach().contiguous().numpy().astype("<f8").tobytes()
            else:
                arr = np.asarray(value)
                out[tag] = arr.astype("<i8").tobytes() if arr.dtype.kind in "iu" else arr.astype("<f8").tobytes()
        assert name not in self.records, name
        self.records[name] = out

    def write(self, path):
        b = bytearray(b"INK2AQ01")
        b += struct.pack("<I", len(self.records))
        for name in sorted(self.records):
            nb = name.encode()
            b += struct.pack("<H", len(nb)) + nb + struct.pack("<I", len(self.records[name]))
            for tag in sorted(self.records[name]):
                tb = tag.encode()
                blob = self.records[name][tag]
                b += struct.pack("<H", len(tb)) + tb + struct.pack("<I", len(blob)) + blob
        with open(path, "wb") as f:
            f.write(bytes(b))
        return hashlib.sha256(bytes(b)).hexdigest(), len(b)


def setp(param, value):
    assert tuple(param.shape) == tuple(value.shape), (param.shape, value.shape)
    with torch.no_grad():
        param.copy_(value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    torch.set_num_threads(1)
    import transformers.core_model_loading as cml

    M = O.load_modeling(True)
    cfg = O.build_text_config(M, CONFIG, {})
    cfg._attn_implementation = "eager"
    tc = CONFIG["text_config"]
    H, heads, kvh, hd, rank = tc["hidden_size"], tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"], tc["d_rel"]
    K, E, F, k = tc["sconv_kernel_size"], tc["n_routed_experts"], tc["intermediate_size"], tc["num_experts_per_tok"]
    g, R = Rng(args.seed), Records()
    pos = torch.arange(T).unsqueeze(0)

    with torch.no_grad():
        # ---- Q projection: per-channel scale, L2 clip, log scale (full layers only)
        for name, log_scaling in (("query_full", True), ("query_sliding", False)):
            mod = M.Inkling2LoopQueryProjection(cfg, heads, hd, log_scaling=log_scaling).double()
            w, scale, x = g.normal(name + ".w", (heads * hd, H), H ** -0.5), g.normal(name + ".scale", (heads, hd), 0.3, 1.0), g.normal(name + ".x", (1, T, H))
            setp(mod.linear.weight, w), setp(mod.norm.scale, scale)
            R.put(name, w=w, scale=scale, x=x[0], out=mod(x, pos)[0].transpose(0, 1))  # [T, heads, hd]

        # ---- K and V projections: short convolution (static, dynamic), then the K clip or the V norm
        for dynamic in (False, True):
            for is_key in (True, False):
                name = f"{'key' if is_key else 'value'}_{'dynamic' if dynamic else 'static'}"
                mod = M.Inkling2LoopKeyValueProjection(cfg, kvh, hd, 1 if dynamic else 0, 0 if is_key else 1, is_key, dynamic=dynamic).double()
                w, x = g.normal(name + ".w", (kvh * hd, H), 2.0 * H ** -0.5), g.normal(name + ".x", (1, T, H))
                setp(mod.linear.weight, w)
                blobs = {"w": w, "x": x[0]}
                if dynamic:
                    blobs["static_w"] = g.normal(name + ".static_w", (K, kvh * hd), 0.5)
                    blobs["weight_proj"] = g.normal(name + ".weight_proj", (kvh * K, H), 0.5 * H ** -0.5)
                    setp(mod.sconv.static_weight, blobs["static_w"]), setp(mod.sconv.weight_proj.weight, blobs["weight_proj"])
                else:
                    blobs["taps"] = g.normal(name + ".taps", (kvh * hd, 1, K), 0.5)
                    setp(mod.sconv.conv1d.weight, blobs["taps"])
                R.put(name, out=mod(x)[0].transpose(0, 1), **blobs)  # [T, kvh, hd]

        # ---- output short convolution
        mod = M.Inkling2LoopShortConvolution(H, K, 0, 2).double()
        taps, x = g.normal("conv.taps", (H, 1, K), 0.5), g.normal("conv.x", (1, T, H))
        setp(mod.conv1d.weight, taps)
        R.put("conv_static", taps=taps, x=x[0], out=mod(x)[0])

        # ---- relative position bias
        for name, sliding in (("relative_full", False), ("relative_sliding", True)):
            extent = tc["sliding_window_size"] if sliding else tc["rel_extent"]
            mod = M.Inkling2LoopRelativePositionBias(cfg, heads, extent, log_scaling=not sliding).double()
            w, scale, proj, x = g.normal(name + ".w", (heads * rank, H), H ** -0.5), g.normal(name + ".scale", (heads, rank), 0.3, 1.0), g.normal(name + ".proj", (rank, extent), 0.5), g.normal(name + ".x", (1, T, H))
            setp(mod.linear.weight, w), setp(mod.norm.scale, scale), setp(mod.proj, proj)
            distance = M.get_relative_distance(x, None, is_sliding=sliding)
            R.put(name, w=w, scale=scale, proj=proj, x=x[0], out=mod(x, distance, pos)[0].transpose(0, 1))  # [T(query), heads, T(key)]

        # ---- the whole attention module with the masks the model builds
        for name, layer in (("attention_sliding", 0), ("attention_full", 1)):
            sliding = layer == 0
            mod = (M.Inkling2LoopSlidingAttention if sliding else M.Inkling2LoopFullAttention)(cfg, layer).double()
            extent = tc["sliding_window_size"] if sliding else tc["rel_extent"]
            blobs = {
                "wq": g.normal(name + ".wq", (heads * hd, H), H ** -0.5), "q_scale": g.normal(name + ".q_scale", (heads, hd), 0.3, 1.0),
                "wk": g.normal(name + ".wk", (kvh * hd, H), 2.0 * H ** -0.5), "wv": g.normal(name + ".wv", (kvh * hd, H), H ** -0.5),
                "wr": g.normal(name + ".wr", (heads * rank, H), H ** -0.5), "r_scale": g.normal(name + ".r_scale", (heads, rank), 0.3, 1.0),
                "proj": g.normal(name + ".proj", (rank, extent), 0.5), "sink": g.normal(name + ".sink", (heads, H), 1.5 * H ** -0.5),
                "gate": g.normal(name + ".gate", (heads * hd, H), H ** -0.5), "wo": g.normal(name + ".wo", (H, heads * hd), (heads * hd) ** -0.5),
                "xq": g.normal(name + ".xq", (1, T, H)), "xk": g.normal(name + ".xk", (1, T, H)), "xv": g.normal(name + ".xv", (1, T, H)),
            }
            setp(mod.q_proj.linear.weight, blobs["wq"]), setp(mod.q_proj.norm.scale, blobs["q_scale"])
            setp(mod.k_proj.linear.weight, blobs["wk"]), setp(mod.v_proj.linear.weight, blobs["wv"])
            setp(mod.relative_position_bias.linear.weight, blobs["wr"]), setp(mod.relative_position_bias.norm.scale, blobs["r_scale"])
            setp(mod.relative_position_bias.proj, blobs["proj"]), setp(mod.sink_proj.weight, blobs["sink"])
            setp(mod.o_proj.gate_proj.weight, blobs["gate"]), setp(mod.o_proj.linear.weight, blobs["wo"])
            for which, proj in (("k", mod.k_proj), ("v", mod.v_proj)):
                if sliding:
                    blobs[which + "_taps"] = g.normal(f"{name}.{which}_taps", (kvh * hd, 1, K), 0.5)
                    setp(proj.sconv.conv1d.weight, blobs[which + "_taps"])
                else:
                    blobs[which + "_static_w"] = g.normal(f"{name}.{which}_static_w", (K, kvh * hd), 0.5)
                    blobs[which + "_weight_proj"] = g.normal(f"{name}.{which}_weight_proj", (kvh * K, H), 0.5 * H ** -0.5)
                    setp(proj.sconv.static_weight, blobs[which + "_static_w"]), setp(proj.sconv.weight_proj.weight, blobs[which + "_weight_proj"])
            mask_kwargs = {"config": cfg, "inputs_embeds": blobs["xq"], "attention_mask": None, "cache": None, "position_ids": pos}
            mask = (M.create_sliding_window_causal_mask if sliding else M.create_causal_mask)(**mask_kwargs)
            assert mask is not None
            out, _ = mod(hidden_states=blobs["xq"], key_hidden_states=blobs["xk"], value_hidden_states=blobs["xv"], attention_mask=mask,
                         relative_distance=M.get_relative_distance(blobs["xq"], None, is_sliding=sliding), position_ids=pos)
            for key in ("xq", "xk", "xv"):
                blobs[key] = blobs[key][0]
            R.put(name, out=out[0], **blobs)

        # ---- block attention residual: one and three queries, with and without earlier blocks
        for name, queries, blocks in (("mix_one", 1, 2), ("mix_three", 3, 2), ("mix_first", 1, 0)):
            mod = M.Inkling2LoopBlockAttentionResidual(cfg, num_queries=queries)
            q = g.normal(name + ".q", (H,) if queries == 1 else (queries, H))
            setp(mod.query, q)
            states, partial = g.normal(name + ".states", (1, T, blocks, H), 1.5), g.normal(name + ".partial", (1, T, H))
            out = torch.stack(mod(states, partial), dim=1)[0]  # [queries, T, H]
            R.put(name, q=q, states=states[0], partial=partial[0], out=out.transpose(0, 1))  # [T, queries, H]

        # ---- loop message: the checkpoint's two query tensors through the PR's merge, then the mix
        loop_layers = tc["loop_config"]["core_end"] - tc["loop_config"]["core_start"]
        queries, kv_queries = g.normal("message.queries", (2 * loop_layers, H)), g.normal("message.kv_queries", (2 * loop_layers, H))
        names = ["middle_cycle_cross_pass_queries", "middle_cycle_cross_pass_kv_queries"]
        merged = cml.Inkling2LoopMergeMessageQueries().convert({names[0]: queries, names[1]: kv_queries}, source_patterns=names, target_patterns=["message_input.query"])["message_input.query"]
        mod = M.Inkling2LoopMessageAttentionResidual(cfg)
        setp(mod.query, merged)
        states, partial = g.normal("message.states", (1, T, 2, H), 1.5), g.normal("message.partial", (1, T, H))
        R.put("message", queries=queries, kv_queries=kv_queries, states=states[0], partial=partial[0],
              out=torch.stack([mod(states, partial, i)[0] for i in range(loop_layers)], dim=0))  # [loop layer, T, 4, H]

        # ---- loop controller
        mod = M.Inkling2LoopMessageGate(cfg).double()
        rank_gate = tc["loop_config"]["gate_rank"]
        down, forget, gain = g.normal("controller.down", (rank_gate, 2 * H), (2 * H) ** -0.5), g.normal("controller.forget", (H, rank_gate), rank_gate ** -0.5), g.normal("controller.gain", (H, rank_gate), rank_gate ** -0.5)
        setp(mod.down_proj.weight, down), setp(mod.forget_gate_proj.weight, forget), setp(mod.message_gate_proj.weight, gain)
        x, message = g.normal("controller.x", (T, H)), g.normal("controller.message", (T, H))
        R.put("controller", down=down, forget=forget, gain=gain, x=x, message=message, out=mod(x, message))

        # ---- router
        mod = M.Inkling2LoopTopKRouter(cfg).double()
        w, bias, x = g.normal("router.w", (E, H), 2.0 * H ** -0.5), g.normal("router.bias", (E,), 0.3), g.normal("router.x", (T, H))
        setp(mod.weight, w)
        mod.e_score_correction_bias.copy_(bias)
        _, weights, indices = mod(x)
        order = indices.argsort(dim=-1)
        R.put("router", w=w, bias=bias, x=x, ids_selected=indices.gather(1, order).numpy(), weights=weights.gather(1, order))

        # ---- experts: checkpoint-layout w13 through the PR's Interleave, then the PR's expert module
        w13, w2 = g.normal("experts.w13", (E, 2 * F, H), H ** -0.5), g.normal("experts.w2", (E, H, F), F ** -0.5)
        mod = M.Inkling2LoopExperts(cfg).double()
        setp(mod.gate_up_proj, cml.Interleave(dim=1).convert({"w13": w13}, source_patterns=["w13"], target_patterns=["gate_up_proj"])["gate_up_proj"])
        setp(mod.down_proj, w2)
        x = g.normal("experts.x", (T, H), 2.0)
        selected = torch.from_numpy(np.stack([np.sort(g.integers(f"experts.sel{t}", 0, E, (E,)).argsort()[:k]) for t in range(T)]))
        weights = g.normal("experts.weights", (T, k), 0.3, 1.0)
        R.put("experts", w13=w13, w2=w2, x=x, ids_selected=selected.numpy(), weights=weights, out=mod(x, selected, weights))
        gate, up = g.normal("swiglu.gate", (64,), 3.0), g.normal("swiglu.up", (64,), 4.0)
        R.put("swiglu", gate=gate, up=up, out=M.Inkling2LoopSoftcappedSwiGLU(cfg)(gate, up))

        # ---- expert dequantization: the PR's op on stored bytes, param dtype f32
        class F32Param:
            def get_parameter(self, _name):
                return torch.zeros(1, dtype=torch.float32)

        op = cml.Inkling2LoopDequantizeExperts()
        rows, cols, experts = 6, 64, 3
        codes = g.integers("nvfp4.codes", 0, 256, (experts, rows, cols // 2)).astype(np.uint8)
        scales = g.integers("nvfp4.scales", 0x60, 0x7F, (experts, rows, cols // 16)).astype(np.uint8)
        scales[0, 0, 0], scales[1, 1, 1], scales[2, 2, 2] = 0x00, 0x03, 0x2A  # zero, subnormal, small
        amax = g.normal("nvfp4.amax", (experts,), 0.2, 0.7).float().abs()
        den = torch.tensor([2688.0])
        names = ["w", "w_scale", "w_amax", "w_outer_scale_denominator"]
        out = op.convert({names[0]: torch.from_numpy(codes), names[1]: torch.frombuffer(bytearray(scales.tobytes()), dtype=torch.float8_e4m3fn).reshape(scales.shape),
                          names[2]: amax, names[3]: den}, source_patterns=names, target_patterns=["t"], model=F32Param(), full_layer_name="t")["t"]
        assert out.dtype == torch.float32
        R.put("dequant_nvfp4", bytes_codes=codes.tobytes(), bytes_scales=scales.tobytes(), amax=amax.double(), denominator=den.double(), out=out.double())
        values = g.integers("mxfp8.values", 0, 256, (experts, rows, cols)).astype(np.uint8)
        values[(values & 0x7F) == 0x7F] = 0x80  # no NaN codes; -0 instead
        exponents = g.integers("mxfp8.exponents", 120, 131, (experts, rows, cols // 32)).astype(np.uint8)
        names = ["w", "w_scale", "w_amax", "w_outer_scale_denominator"]
        out = op.convert({names[0]: torch.frombuffer(bytearray(values.tobytes()), dtype=torch.float8_e4m3fn).reshape(values.shape), names[1]: torch.from_numpy(exponents)},
                         source_patterns=names, target_patterns=["t"], model=F32Param(), full_layer_name="t")["t"]
        assert out.dtype == torch.float32
        R.put("dequant_mxfp8", bytes_values=values.tobytes(), bytes_exponents=exponents.tobytes(), out=out.double())

        # ---- embedding norm, and the head: norm, unpadded output projection, softcap
        vocab, keep = tc["vocab_size"], tc["unpadded_vocab_size"]
        mod = M.Inkling2LoopNormedEmbedding(vocab, H, None, tc["rms_norm_eps"]).double()
        table, weight = g.normal("embed.table", (vocab, H)), g.normal("embed.norm", (H,), 0.2, 1.0)
        setp(mod.weight, table), setp(mod.embed_norm.weight, weight)
        ids = torch.from_numpy(g.integers("embed.ids", 0, vocab, (T,)))
        R.put("embedding", table=table, norm=weight, ids_tokens=ids.numpy(), out=mod(ids))

        class Holder:
            def get_text_config(self):
                return cfg

        unembed, weight, x = g.normal("head.unembed", (vocab, H), 0.5), g.normal("head.norm", (H,), 0.2, 1.0), g.normal("head.x", (T, H), 2.0)
        kept = cml.Inkling2UnpadVocab().convert({"lm_head.weight": unembed}, source_patterns=["lm_head.weight"], target_patterns=["lm_head.weight"], config=Holder())["lm_head.weight"]
        norm = M.Inkling2LoopRMSNorm(H, eps=tc["rms_norm_eps"]).double()
        setp(norm.weight, weight)
        logits = M.Inkling2LoopSoftcap(tc["final_logit_softcapping"])(torch.nn.functional.linear(norm(x), kept))
        assert logits.shape == (T, keep)
        R.put("head", unembed=unembed, norm=weight, x=x, out=logits)

    sha, size = R.write(args.out)
    print(f"{len(R.records)} records, {size} bytes, sha256 {sha}")


if __name__ == "__main__":
    main()
