#!/usr/bin/env python3
"""Patch the v2-trained MTP head into the official BF16 GGUF, in place.

Name/transform map = memra `hf_mapping.rs resolve_mtp_block` (the map the engine
uses to READ these tensors, so writing through the same map is round-trip safe):
norms are stored +1 in GGUF (HF RMSNorm computes (1+w)·x̂, GGUF kernels w·x̂),
matrices are byte-identical row-major, per-expert HF tensors stack into the
fused 3D `ffn_{gate,up,down}_exps` GGUF layout. Fails closed: every blk.40
tensor in the file must be covered (shared_head excluded — head reuses lm_head),
every write must match shape and dtype byte length.
"""
import argparse
import sys

import numpy as np
import torch
from safetensors import safe_open

MTP_IL = 40  # 40 trunk layers; GGUF block_count 41 includes the NextN block


def trained_tensors(path):
    out = {}
    with safe_open(path, framework="pt") as f:
        for name in f.keys():
            out[name] = f.get_tensor(name)
    return out


def to_bytes(t: torch.Tensor, ggml_type: str) -> bytes:
    if ggml_type == "F32":
        return t.to(torch.float32).contiguous().numpy().tobytes()
    if ggml_type == "BF16":
        return t.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    if ggml_type == "F16":
        return t.to(torch.float16).contiguous().numpy().tobytes()
    raise SystemExit(f"FATAL: unsupported ggml dtype {ggml_type} for a patched tensor")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--mtp", required=True)
    args = ap.parse_args()

    sys.path.insert(0, "/home/ubuntu/llama-nvfp4/gguf-py")
    from gguf import GGUFReader

    tr = trained_tensors(args.mtp)
    p = f"blk.{MTP_IL}."
    L = "mtp.layers.0."
    plain = {
        p + "nextn.eh_proj.weight": "mtp.fc.weight",
        p + "attn_q.weight": L + "self_attn.q_proj.weight",
        p + "attn_k.weight": L + "self_attn.k_proj.weight",
        p + "attn_v.weight": L + "self_attn.v_proj.weight",
        p + "attn_output.weight": L + "self_attn.o_proj.weight",
        p + "ffn_gate_inp.weight": L + "mlp.gate.weight",
        p + "ffn_gate_shexp.weight": L + "mlp.shared_expert.gate_proj.weight",
        p + "ffn_up_shexp.weight": L + "mlp.shared_expert.up_proj.weight",
        p + "ffn_down_shexp.weight": L + "mlp.shared_expert.down_proj.weight",
        p + "ffn_gate_inp_shexp.weight": L + "mlp.shared_expert_gate.weight",
    }
    plus_one = {
        p + "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
        p + "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
        p + "nextn.shared_head_norm.weight": "mtp.norm.weight",
        p + "attn_norm.weight": L + "input_layernorm.weight",
        p + "post_attention_norm.weight": L + "post_attention_layernorm.weight",
        p + "ffn_norm.weight": L + "post_attention_layernorm.weight",
        p + "attn_q_norm.weight": L + "self_attn.q_norm.weight",
        p + "attn_k_norm.weight": L + "self_attn.k_norm.weight",
    }
    exps = {
        p + "ffn_gate_exps.weight": "gate_proj",
        p + "ffn_up_exps.weight": "up_proj",
        p + "ffn_down_exps.weight": "down_proj",
    }
    skip = {p + "nextn.shared_head.weight"}  # head reuses lm_head; leave verbatim if present

    r = GGUFReader(args.gguf, "r+")
    n_exp = sum(1 for k in tr if ".mlp.experts." in k and k.endswith("gate_proj.weight"))
    patched, seen = 0, []
    for t in r.tensors:
        if not t.name.startswith(p):
            continue
        seen.append(t.name)
        if t.name in skip:
            continue
        ggml_type = t.tensor_type.name
        if t.name in exps:
            proj = exps[t.name]
            src = torch.stack(
                [tr[f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"] for e in range(n_exp)]
            )
        elif t.name in plain:
            src = tr[plain[t.name]]
        elif t.name in plus_one:
            src = tr[plus_one[t.name]].to(torch.float32) + 1.0
        else:
            raise SystemExit(f"FATAL: unmapped MTP tensor in GGUF: {t.name} ({ggml_type})")
        raw = to_bytes(src, ggml_type)
        buf = t.data.view(np.uint8).reshape(-1)
        if len(raw) != buf.nbytes:
            raise SystemExit(
                f"FATAL: byte mismatch {t.name}: trained {len(raw)} vs gguf {buf.nbytes}"
            )
        buf[:] = np.frombuffer(raw, dtype=np.uint8)
        patched += 1
    del r  # flush memmap
    print(f"patched {patched} tensors ({len(seen)} blk.{MTP_IL} tensors seen, n_exp={n_exp})")
    assert patched >= 20, "suspiciously few tensors patched"
    print("GGUF PATCH DONE")


if __name__ == "__main__":
    main()
