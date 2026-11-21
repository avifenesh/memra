#!/usr/bin/env python3
"""NVFP4 weight-quantizer goldens for the nvfp4-rig mint (lane M).

Runs lane F's quantizer, klh.nvfp4_weight_quant (~/.cache/inkling2mini/klsens/klh.py, unmodified),
on synthetic edge cases and on slices of the shipped checkpoint, and writes its codes, scale bytes,
per-tensor (or per-expert) scale and dequantized values to one safetensors file. The Rust mint
(memra_gguf::model_packs::inkling2::nvfp4_rig) must reproduce every byte.

usage: KLH_DEV=cpu|cuda [GOLDEN_SMALL=1] venv/bin/python golden_quant.py OUT
The repository fixture (crates/memra-gguf/src/model_packs/inkling2/fixtures/nvfp4-rig-golden.bin, safetensors
bytes) is the GOLDEN_SMALL=1 run on the CPU.
The device is part of the receipt: CPU and CUDA runs are both kept and compared.
"""
import os
import sys

import numpy as np
import torch
from safetensors.numpy import save_file

sys.path.insert(0, os.path.expanduser("~/.cache/inkling2mini/klsens"))
import klh  # noqa: E402

out_path = sys.argv[1]
g = torch.Generator().manual_seed(20261121)
cases = {}


SMALL = os.environ.get("GOLDEN_SMALL") == "1"   # the repository fixture: short slices, BF16 sources as BF16, no deq


def rows(n):
    return max(1, n // 8) if SMALL else n


def add(name, w, per_expert=False, bf16_source=False):
    w = w.float().contiguous()
    q, codes, sb, t = klh.nvfp4_weight_quant(w.to(klh.DEV), per_expert_dim0=per_expert, want_codes=True)
    d = dict(codes=codes.cpu().numpy(), scales=sb.cpu().numpy(), t=t.reshape(-1).float().cpu().numpy(),
             per_expert=np.array([1 if per_expert else 0], dtype=np.uint8))
    if SMALL and bf16_source:
        d["w_bf16"] = (w.cpu().numpy().view(np.uint32) >> 16).astype(np.uint16)
    else:
        d["w"] = w.cpu().numpy()
    if not (SMALL and bf16_source):
        d["deq"] = q.cpu().numpy()
    cases[name] = d


add("gauss", torch.randn(64, 256, generator=g))
add("heavy", torch.randn(64, 256, generator=g) ** 3 * torch.rand(64, 256, generator=g))
z = torch.randn(32, 128, generator=g)
z[:, :16] = 0.0
z[:, 16:32] = -0.0
z[:, 32:48] = 0.0
z[:, 32] = 3.25
z[:, 48:64] = 1.0
z[3, 64:80] = -2.5
add("edges", z)
ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0])
add("ties", (ties.repeat(8, 4) * torch.tensor([1.0, 0.5, 2.0, 0.125, 1.5, 3.0, 0.75, 1e-3])[:, None]))
add("tiny", torch.randn(16, 64, generator=g) * 1e-30)
add("mixed_scale", torch.randn(32, 512, generator=g) * torch.logspace(-6, 2, 32)[:, None])
add("experts3d", torch.randn(3, 32, 64, generator=g) * torch.tensor([1.0, 1e-3, 40.0])[:, None, None], per_expert=True)

O = klh.O
st = O.STFile(O.weights_file(klh.CKPT))


def bf16(name, rows=None):
    a = np.ascontiguousarray(st.raw(name)).view(np.uint16)
    shape = st.info(name)[1]
    a = a.reshape(shape)
    if rows is not None:
        a = a[:rows]
    return torch.from_numpy((a.astype(np.uint32) << 16).view(np.float32).copy())


add("L0_wq_du", bf16("model.llm.layers.0.attn.wq_du.weight", rows(128)), bf16_source=True)
add("L5_wo_ud", bf16("model.llm.layers.5.attn.wo_ud.weight", rows(64)), bf16_source=True)
add("L23_wr_du", bf16("model.llm.layers.23.attn.wr_du.weight", rows(384)), bf16_source=True)
add("unembed", bf16("model.llm.unembed.weight", rows(256)), bf16_source=True)
store = klh.GpuStore(st, 20)
for nm in ("w13", "w2"):
    W = store.deq(nm, torch.arange(0, 2, device=klh.DEV))[:, :rows(96)]
    add(f"L20_{nm}_e01", W.cpu(), per_expert=True)
    del W

flat = {}
for name, d in cases.items():
    for k, v in d.items():
        flat[f"{name}.{k}"] = v
save_file(flat, out_path, metadata={"device": str(klh.DEV), "torch": torch.__version__, "klh": "klsens/klh.py nvfp4_weight_quant"})
print("wrote", out_path, len(cases), "cases on", klh.DEV)
