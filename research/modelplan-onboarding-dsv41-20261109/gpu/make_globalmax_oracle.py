"""A copy of the oracle whose sparse_attn rounds P to bf16 against one global max per row (the lane's first stand-in)
instead of the tilelang schedule's running max over key blocks of 64. Both are valid realizations of the vendor
program; the pair gives a GPU-only floor ("same program, another valid attention schedule").

usage: make_globalmax_oracle.py ORACLE_DIR OUT_DIR"""
import os
import re
import shutil
import sys

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f.endswith(".py") or f.endswith(".json") or f.endswith(".i32"):
        shutil.copy(os.path.join(src, f), os.path.join(dst, f))
k = open(os.path.join(dst, "kernel.py")).read()
a = k.index("def sparse_attn(")
b = k.index("\ndef ", a + 1)
new = '''def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Global-max variant: P is rounded to bf16 against the row's single max over every selected entry."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    qchunk = QCHUNK  # 256 rows on every device: the whole-row gather is larger than the blocked schedule
    for bi in range(b):
        for m0 in range(0, m, qchunk):
            idx = topk_idxs[bi, m0:m0 + qchunk].long()
            valid = idx >= 0
            g = kv[bi][idx.clamp_min(0)]
            g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), dtype=g.dtype))
            s = torch.einsum("chd,ckd->chk", q[bi, m0:m0 + qchunk].float(), g.float()) * softmax_scale
            s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
            mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
            p = torch.exp(s - mx)
            den = p.sum(-1) + torch.exp(attn_sink.float().view(1, h) - mx.squeeze(-1))
            o = torch.einsum("chk,ckd->chd", p.to(torch.bfloat16).float(), g.float()) / den.unsqueeze(-1)
            out[bi, m0:m0 + qchunk] = o.to(q.dtype)
    return out

'''
k = k[:a] + new + k[b + 1:]
open(os.path.join(dst, "kernel.py"), "w").write(k)
print("wrote", dst)
