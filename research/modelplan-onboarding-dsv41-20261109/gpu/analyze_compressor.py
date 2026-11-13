"""Where does the ratio-2 compressor latent difference come from? Recompute layer L's compressor from the capture's
input in float64 (the truth), in fp32 the vendor way (torch GPU, no TF32), and in fp32 with the GEMM accumulated in
fp32 on tensor-core-free hardware order (torch CPU), then compare each, and the rewrite's dumped latent, to the truth
and to the capture.

usage: analyze_compressor.py CKPT CAPTURE_DIR MINE_DIR LAYER"""
import json
import os
import sys

import torch
from safetensors import safe_open

ckpt, cap, mine, L = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
torch.backends.cuda.matmul.allow_tf32 = False
DT = {"bf16": torch.bfloat16, "f32": torch.float32}


def load(root, k):
    idx = json.load(open(os.path.join(root, "index.json")))["tensors"][k]
    f = os.path.join(root, idx["file"])
    return torch.from_file(f, size=os.path.getsize(f), dtype=torch.uint8).view(DT[idx["dtype"]]).reshape(idx["shape"]).clone()


wm = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]


def w(name):
    with safe_open(os.path.join(ckpt, wm[name]), "pt") as fh:
        return fh.get_tensor(name)


p = f"layers.{L}.attn.compressor"
wkv, wg, nw = w(f"{p}.wkv.weight"), w(f"{p}.wgate.weight"), w(f"{p}.norm.weight")
x = load(cap, f"s0/L{L}/comp/x")[0]
ref = load(cap, f"s0/L{L}/comp/out")[0]
got = load(mine, f"s0/L{L}/comp/out")[0]
s = x.size(0) - x.size(0) % 2


def comp(x, wkv, wg, nw, dt, dev):
    xf = x[:s].to(dev, dt)
    kv = xf @ wkv.to(dev, dt).t()
    sc = xf @ wg.to(dev, dt).t()
    kv, sc = kv.unflatten(0, (-1, 2)), sc.unflatten(0, (-1, 2))
    lat = (kv * sc.softmax(dim=1)).sum(dim=1).to(torch.bfloat16)
    lf = lat.to(dt)
    y = lf * torch.rsqrt(lf.square().mean(-1, keepdim=True) + 1e-20)
    return (nw.to(dev, dt) * y).to(torch.bfloat16).cpu(), lat.cpu()


truth, tlat = comp(x, wkv, wg, nw, torch.float64, "cuda")
gpu32, glat = comp(x, wkv, wg, nw, torch.float32, "cuda")
cpu32, clat = comp(x, wkv, wg, nw, torch.float32, "cpu")


def rel(a, b):
    a, b = a.double(), b.double()
    return f"{((a - b).norm() / b.norm()).item():.3e} exact={(a == b).double().mean().item() * 100:.3f}%"


print("layer", L, "rows", s // 2)
print("capture vs torch-gpu-fp32 (sanity)   ", rel(ref, gpu32))
print("capture vs float64 truth             ", rel(ref, truth))
print("torch-cpu-fp32 vs truth              ", rel(cpu32, truth))
print("rewrite vs truth                     ", rel(got, truth))
print("rewrite vs capture                   ", rel(got, ref))
print("pooled (pre-norm) gpu32 vs truth     ", rel(glat, tlat))
