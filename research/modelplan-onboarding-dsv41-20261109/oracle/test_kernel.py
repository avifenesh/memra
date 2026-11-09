"""Self-checks for the oracle kernels: quantization grids, scale rounding, GEMM equivalence, attention with sinks."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import kernel as K

torch.manual_seed(0)
# pow2 ceil: exact powers stay, others round up
v = torch.tensor([1.0, 1.5, 0.25, 3.0, 2.0**-20, 7.9])
assert torch.equal(K._pow2_ceil(v), torch.tensor([1.0, 2.0, 0.25, 4.0, 2.0**-20, 8.0])), K._pow2_ceil(v)
# fp4 rounding: grid points fixed, ties to even mantissa
x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.3, -2.6, 0.1])
assert torch.equal(K._round_fp4(x), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.5, -3.0, 0.0])), K._round_fp4(x)
# fp8 act quant: in-place result equals dequant of the returned pair; scales are powers of two
a = torch.randn(4, 64, dtype=torch.bfloat16) * 3
q, s = K.act_quant(a.clone(), 32, "ue8m0", torch.float8_e8m0fnu)
ip = K.act_quant(a.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True)
deq = (q.float().unflatten(-1, (2, 32)) * s.float().unsqueeze(-1)).flatten(-2).bfloat16()
assert torch.equal(ip, deq)
assert torch.equal(torch.log2(s.float()), torch.log2(s.float()).round())
assert (ip.float() - a.float()).abs().max() <= a.float().abs().max() / 8, "fp8 error too large"
# fp8 gemm vs dense dequant matmul
w = (torch.randn(64, 64) * 0.1).to(torch.float8_e4m3fn)
ws = torch.full((2, 2), 0.5).to(torch.float8_e8m0fnu)
y = K.fp8_gemm(q, s, w, ws, torch.float8_e8m0fnu, 32)
ref = deq.float() @ (w.float() * 0.5).t()
assert torch.allclose(y.float(), ref, rtol=1e-2, atol=1e-2)
# fp4 unpack: low nibble first
b = torch.tensor([[0x21, 0xF7] * 16], dtype=torch.uint8).view(torch.int8).view(torch.float4_e2m1fn_x2)
bs = torch.ones(1, 2).to(torch.float8_e8m0fnu)
u = K.unpack_fp4(b, bs)
assert u[0, :4].tolist() == [0.5, 1.0, 6.0, -6.0], u[0, :4]
# sparse attention with a sink equals dense softmax over [scores, sink] with the sink row dropped
q4 = torch.randn(1, 3, 2, 16, dtype=torch.bfloat16)
kv = torch.randn(1, 5, 16, dtype=torch.bfloat16)
sink = torch.tensor([0.3, -1.0])
idx = torch.tensor([[[0, 1, 2, -1], [1, 3, 4, -1], [-1, -1, -1, -1]]], dtype=torch.int32)
o = K.sparse_attn(q4, kv, sink, idx, 0.25)
for m in range(2):
    sel = idx[0, m][idx[0, m] >= 0].long()
    sc = (q4[0, m].float() @ kv[0, sel].float().t()) * 0.25
    full = torch.cat([sc, sink.view(2, 1)], 1).softmax(-1)[:, :-1]
    assert torch.allclose(o[0, m].float(), full @ kv[0, sel].float(), atol=2e-2), m
assert torch.equal(o[0, 2].float(), torch.zeros(2, 16))
# more than one 64-entry block: the online form stays within bf16 rounding of the direct sink softmax, and the
# single-block form (one block of 64) equals the direct bf16-probability form, with the same correctly rounded
# primitives, exactly
q5 = torch.randn(1, 2, 2, 16, dtype=torch.bfloat16)
kv5 = torch.randn(1, 150, 16, dtype=torch.bfloat16)
idx5 = torch.stack([torch.randperm(150)[:130] for _ in range(2)]).int().unsqueeze(0)
o5 = K.sparse_attn(q5, kv5, sink, idx5, 0.25)
for m in range(2):
    sel = idx5[0, m].long()
    sc = (q5[0, m].float() @ kv5[0, sel].float().t()) * 0.25
    full = torch.cat([sc, sink.view(2, 1)], 1).softmax(-1)[:, :-1]
    assert torch.allclose(o5[0, m].float(), full @ kv5[0, sel].float(), atol=2e-2), m
idx6 = idx5[:, :, :64].contiguous()
o6 = K.sparse_attn(q5, kv5, sink, idx6, 0.25)
for m in range(2):
    sel = idx6[0, m].long()
    sc = (q5[0, m].double() @ kv5[0, sel].double().t()).float() * 0.25
    mx = sc.amax(-1, keepdim=True)
    p = torch.exp((sc - mx).double()).float()
    den = p.double().sum(-1).float() + torch.exp((sink - mx.squeeze(-1)).double()).float()
    direct = ((p.bfloat16().double() @ kv5[0, sel].double()).float() / den.unsqueeze(-1)).bfloat16()
    assert torch.equal(o6[0, m], direct), m
# sinkhorn: comb doubly stochastic
mixes = torch.randn(1, 2, 24)
pre, post, comb = K.hc_split_sinkhorn(mixes, torch.ones(3), torch.zeros(24), 4, 20, 1e-6)
assert torch.allclose(comb.sum(-1), torch.ones(1, 2, 4), atol=1e-3) and torch.allclose(comb.sum(-2), torch.ones(1, 2, 4), atol=1e-3)
# row-chunked quantizers equal the one-shot ones (blocks are per row)
big = torch.randn(64, 8, 96, dtype=torch.bfloat16) * 5
one_q, one_s = K._act_quant(big.clone(), 32, "ue8m0", torch.float8_e8m0fnu)
one_ip, one_f4 = K._act_quant(big.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True), K._fp4_act_quant(big.clone(), 32, True)
saved, K._CHUNK_ELEMS = K._CHUNK_ELEMS, 1000
q2, s2 = K.act_quant(big.clone(), 32, "ue8m0", torch.float8_e8m0fnu)
assert torch.equal(q2.view(torch.uint8), one_q.view(torch.uint8)) and torch.equal(s2.view(torch.uint8), one_s.view(torch.uint8))
assert torch.equal(K.act_quant(big.clone(), 32, "ue8m0", torch.float8_e8m0fnu, True), one_ip)
assert torch.equal(K.fp4_act_quant(big.clone(), 32, True), one_f4)
K._CHUNK_ELEMS = saved
print("KERNEL-SELFTEST-OK")
