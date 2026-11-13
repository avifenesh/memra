"""Random-input fixtures for the dsv41 GPU kernel gates, computed by the vendor program itself.

The reference is the vendor inference/model.py (pinned rev 1bae0e10) with the oracle's torch stand-ins for its tilelang
kernels (oracle/kernel.py: the exact quantization grids, scale rounding and the sparse-attention precision program).
Every case calls the vendor module or function on real V4.1 sizes (dim 5120, 64 x 512 heads, 32 x 128 index heads,
2304 expert width) with random weights; a few cases shrink only a count (experts, top-k, candidate blocks, Engram rows)
so the selection logic is exercised at a small width. CPU only.

usage: make_fixtures.py ORACLE_DIR REF_DIR OUT_DIR [--seed 0]"""
import argparse
import json
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("oracle")
ap.add_argument("ref")
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, a.oracle, os.path.join(a.ref, "inference")]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(a.threads)
torch.set_default_dtype(torch.bfloat16)
import kernel as K  # noqa: E402  (oracle stand-ins shadow the vendor tilelang module)

assert os.path.dirname(os.path.abspath(K.__file__)) == os.path.abspath(a.oracle)
import model as M  # noqa: E402
from engram import EngramLayout  # noqa: E402
from fixio import Fix  # noqa: E402

E8 = torch.float8_e8m0fnu
cfg = json.load(open(os.path.join(a.ref, "inference", "config.json")))
cfg.update(max_batch_size=1, max_seq_len=1024, temperature=0.0, vision_n_layers=0)
g = torch.Generator().manual_seed(a.seed)


def args(**kw):
    c = dict(cfg)
    c.update(kw)
    return M.ModelArgs(**c)


def rn(*shape, s=1.0, dtype=torch.bfloat16):
    return (torch.randn(*shape, generator=g, dtype=torch.float32) * s).to(dtype)


def fp8w(n, k, s=0.5):
    return (torch.randn(n, k, generator=g, dtype=torch.float32) * s).to(torch.float8_e4m3fn)


def e8(shape, lo=-14, hi=-6):
    return torch.pow(2.0, torch.randint(lo, hi, shape, generator=g).float()).to(E8)


def fp4w(n, k):
    return torch.randint(-128, 128, (n, k // 2), generator=g, dtype=torch.int8).view(torch.float4_e2m1fn_x2)


def set_linear(lin, scale=0.02):
    w = lin.weight
    if w.dtype == torch.float8_e4m3fn:
        n, k = w.shape
        w.data = fp8w(n, k)
        lin.scale.data = e8(tuple(lin.scale.shape))
        w.scale = lin.scale
    elif w.dtype == torch.float4_e2m1fn_x2:
        n, k2 = w.shape
        w.data = fp4w(n, k2 * 2)
        lin.scale.data = e8(tuple(lin.scale.shape), -10, -4)
        w.scale = lin.scale
    else:
        w.data = rn(*w.shape, s=scale).to(w.dtype)


def set_norm(n):
    n.weight.data = (1.0 + rn(n.dim, s=0.1).float()).to(n.weight.dtype)


def rows_with_spread(m, k):
    """bf16 rows whose magnitudes span ~1e-3..1e3, one all-zero 32-block and one outlier block."""
    x = torch.randn(m, k, generator=g, dtype=torch.float32) * torch.pow(10.0, torch.linspace(-3, 3, m, dtype=torch.float32)).unsqueeze(1)
    x[0, :32] = 0
    if m > 1:
        x[1, 32:64] *= 1000
    return x.to(torch.bfloat16)


def freqs(ratio, n):
    if ratio:
        return M.precompute_freqs_cis(64, n, cfg["original_seq_len"], cfg["compress_rope_theta"], cfg["rope_factor"],
                                      cfg["beta_fast"], cfg["beta_slow"])
    return M.precompute_freqs_cis(64, n, 0, cfg["rope_theta"], cfg["rope_factor"], cfg["beta_fast"], cfg["beta_slow"])


def put_freqs(fx, name, fc):
    fx.put(name + "/cos", fc.real.float().contiguous())
    fx.put(name + "/sin", fc.imag.float().contiguous())


def case(name):
    # the oracle caches dequantized fp8 weights by data_ptr: a freed weight's address can come back for a new one
    K._DEQ.clear()
    fx = Fix(os.path.join(a.out, name))
    print("case", name, flush=True)
    return fx


# ---------------------------------------------------------------- act_quant fp8 per-32 pow2 (+ in-place qdq)
fx = case("actq")
x = rows_with_spread(37, 5120)
q, s = K.act_quant(x.clone(), 32, "ue8m0", E8)
ip = K.act_quant(x.clone(), 32, "ue8m0", E8, True)
fx.put("x", x); fx.put("q", q); fx.put("s", s); fx.put("qdq", ip)
fx.close()

# ---------------------------------------------------------------- fp8 32x32 block GEMM (linear(): act_quant + fp8_gemm)
for (m, n, k) in [(1, 1280, 5120), (3, 512, 5120), (64, 5120, 8192), (200, 4096, 1280), (33, 2304, 5120), (130, 25600, 6144)]:
    fx = case(f"fp8gemm_{m}_{n}_{k}")
    x = rn(m, k)
    w = fp8w(n, k)
    ws = e8(((n + 31) // 32, (k + 31) // 32))
    w.scale = ws
    y = M.linear(x, w)
    fx.put("x", x); fx.put("w", w); fx.put("ws", ws); fx.put("y", y)
    fx.close()

# ---------------------------------------------------------------- fp4 quant-dequant: indexer (g32 e8m0) and compressed KV (g16 e4m3)
fx = case("fp4qdq")
x = rows_with_spread(64, 4096)
# exact e2m1 ties after a power-of-two scale: the group amax 6 pins s = 1, then 0.25, 0.75, ... sit on the midpoints
ties = torch.tensor([6.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, 0.1] * 2, dtype=torch.float32)
x[2, :32] = ties.to(torch.bfloat16)
fx.put("x32", x)
fx.put("y32", K.fp4_act_quant(x.clone(), 32, True))
x16 = rows_with_spread(64, 512)
x16[3, :16] = ties[:16].to(torch.bfloat16)
fx.put("x16", x16)
fx.put("y16", K.fp4_act_quant(x16.clone(), 16, True, scale_dtype=torch.float8_e4m3fn))
fx.close()

# ---------------------------------------------------------------- RMSNorm (eps 1e-20) and RoPE tables + rotation
fx = case("norm_rope")
for d in (5120, 1280, 512, 128):
    nm = M.RMSNorm(d, cfg["norm_eps"])
    set_norm(nm)
    x = rn(29, d, s=3.0)
    fx.put(f"norm{d}/x", x); fx.put(f"norm{d}/w", nm.weight.data); fx.put(f"norm{d}/y", nm(x))
for ratio in (0, 1):
    fc = freqs(ratio, 1024)
    put_freqs(fx, f"freqs{ratio}", fc)
    x = rn(1, 23, 64, 64, s=2.0)
    y = x.clone()
    M.apply_rotary_emb(y, fc[100:123])
    yi = y.clone()
    M.apply_rotary_emb(yi, fc[100:123], True)
    fx.put(f"rope{ratio}/x", x); fx.put(f"rope{ratio}/y", y); fx.put(f"rope{ratio}/yinv", yi)
fx.meta("rope_start", 100)
fx.close()

def sattn_schedule(q, kv, sink, idx, scale, block=64):
    """The vendor tilelang sparse_attn schedule in fp32 torch: key blocks of 64 in index order, online max (floor
    -1e30), fp32 exp and denominator, P cast to bf16 for PV, the sink in the denominator only. The torch stand-in
    (`o`) takes one global max instead; the two differ only by where P is rounded to bf16."""
    _, m, h, d = q.shape
    k = idx.size(2)
    out = torch.empty(1, m, h, d, dtype=torch.bfloat16)
    for i in range(m):
        qf = q[0, i].float()
        mx = torch.full((h,), -1e30)
        se = torch.zeros(h)
        acc = torch.zeros(h, d)
        ids = idx[0, i].long()
        for b0 in range(0, k, block):
            bi = ids[b0:b0 + block]
            g = torch.where((bi >= 0).unsqueeze(-1), kv[0][bi.clamp_min(0)].float(), torch.zeros(()))
            sc = (qf @ g.t()) * scale
            sc = sc.masked_fill((bi < 0).unsqueeze(0), float("-inf"))
            mnew = torch.maximum(mx, sc.amax(-1))
            scp = torch.exp(mx - mnew)
            pexp = torch.exp(sc - mnew.unsqueeze(-1))
            se = se * scp + pexp.sum(-1)
            acc = acc * scp.unsqueeze(-1) + pexp.to(torch.bfloat16).float() @ g
            mx = mnew
        se = se + torch.exp(sink.float() - mx)
        out[0, i] = (acc / se.unsqueeze(-1)).to(torch.bfloat16)
    return out


# ---------------------------------------------------------------- sink sparse attention (V4.1: 64 heads x 512, scale 512^-0.5)
for (m, n, k, qs) in [(50, 700, 640, 1.0), (7, 300, 200, 4.0)]:
    fx = case(f"sattn_{m}_{n}_{k}")
    q = rn(1, m, 64, 512, s=qs)
    kv = rn(1, n, 512)
    K.act_quant(kv, 32, "ue8m0", E8, True)
    idx = torch.randint(0, n, (1, m, k), generator=g, dtype=torch.int32)
    drop = torch.rand(1, m, k, generator=g, dtype=torch.float32) < 0.2
    idx = torch.where(drop, -1, idx)
    idx[0, 0] = -1  # a row with nothing to attend to
    idx[0, 1, : k // 2] = -1
    sink = torch.randn(64, generator=g, dtype=torch.float32) * 2
    o = K.sparse_attn(q, kv, sink, idx, 512 ** -0.5)
    fx.put("q", q); fx.put("kv", kv); fx.put("idx", idx); fx.put("sink", sink); fx.put("o", o)
    fx.put("o_sched", sattn_schedule(q, kv, sink, idx, 512 ** -0.5))
    fx.meta("scale", 512 ** -0.5)
    fx.close()

# ---------------------------------------------------------------- single-pass mHC pieces (generic in d)
fx = case("hc")
ns = types.SimpleNamespace(norm_eps=cfg["norm_eps"], hc_mult=4, hc_sinkhorn_iters=cfg["hc_sinkhorn_iters"], hc_eps=cfg["hc_eps"])
s_ = 21
x = rn(1, s_, 4, 5120, s=2.0)
fn = torch.randn(24, 4 * 5120, generator=g, dtype=torch.float32) * 0.01
sc = torch.rand(3, generator=g, dtype=torch.float32) + 0.5
base = torch.randn(24, generator=g, dtype=torch.float32) * 0.5
pre, post, comb = M.Block.hc_mixes(ns, x, fn, sc, base)
xf = x.flatten(2).float()
mixes = F.linear(xf, fn) * torch.rsqrt(xf.square().mean(-1, keepdim=True) + ns.norm_eps)
carried = torch.softmax(torch.randn(1, s_, 4, generator=g, dtype=torch.float32), -1)
y = M.Block.hc_pre(ns, x, carried)
sub = rn(1, s_, 5120, s=1.5)
z = M.Block.hc_post(ns, sub, x, post, comb)
for k_, v_ in dict(x=x, fn=fn, scale=sc, base=base, mixes=mixes, pre=pre, post=post, comb=comb, carried=carried, y=y,
                   sub=sub, z=z).items():
    fx.put(k_, v_)
fx.close()

# ---------------------------------------------------------------- CSA2 compressor, ratio 1 (projection) and 2 (softmax gate)
for lid, ratio in ((20, 1), (2, 2)):
    fx = case(f"comp_r{ratio}")
    ar = args()
    c = M.Compressor(ar, lid)
    set_linear(c.wkv)
    if ratio > 1:
        set_linear(c.wgate, 0.05)
        c.wkv.weight.data = c.wkv.weight.data.to(torch.bfloat16).float()  # checkpoint bf16, promoted to fp32
        c.wgate.weight.data = c.wgate.weight.data.to(torch.bfloat16).float()
    set_norm(c.norm)
    s_ = 37
    x = rn(1, s_ + 4, 5120)
    fx.put("wkv", c.wkv.weight.data.to(torch.bfloat16))
    if ratio > 1:
        fx.put("wgate", c.wgate.weight.data.to(torch.bfloat16))
    fx.put("norm", c.norm.weight.data)
    fx.put("x", x)
    lat = c(x[:, :s_].clone(), 0)
    fx.put("prefill", lat)
    for i in range(4):
        o = c(x[:, s_ + i: s_ + i + 1].clone(), s_ + i)
        fx.put(f"decode{i}", o if o is not None else torch.empty(0))
    # KV-source tail: RoPE at the group's first position, then fp4 g16 e4m3 quant-dequant
    fc = freqs(ratio, 1024)
    r = lat.clone()
    M.apply_rotary_emb(r[..., -64:], fc[: s_ - s_ % ratio: ratio])
    K.fp4_act_quant(r, 16, True, scale_dtype=torch.float8_e4m3fn)
    fx.put("cache", r)
    fx.meta("seqlen", s_)
    fx.meta("ratio", ratio)
    fx.close()


# ---------------------------------------------------------------- indexer (owns k at 20 / 2, candidate source 20, rescore 24)
def indexer_replica(idx_mod, x, qr, latent, start_pos, offset):
    """Indexer.forward line by line, returning its intermediates; checked against the module's own output."""
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = idx_mod.compress_ratio, idx_mod.rope_head_dim, start_pos + seqlen
    out = {}
    if idx_mod.owns_k and latent is not None:
        fr = idx_mod.freqs_cis[: seqlen - seqlen % ratio: ratio] if start_pos == 0 else idx_mod.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        k = idx_mod.k_norm(idx_mod.wk(latent))
        out["k_pre"] = k.clone()
        M.apply_rotary_emb(k[..., -rd:], fr)
        K.fp4_act_quant(k, 32, True)
        out["k"] = k
    q = idx_mod.wq_b(qr).unflatten(-1, (idx_mod.n_local_heads, idx_mod.index_head_dim))
    out["q_lin"] = q.clone()
    M.apply_rotary_emb(q[..., -rd:], idx_mod.freqs_cis[start_pos:end_pos])
    K.fp4_act_quant(q, 32, True)
    out["q"] = q
    w = idx_mod.weights_proj(x) * (idx_mod.softmax_scale * idx_mod.n_heads ** -0.5)
    out["w"] = w
    return out


fx = case("indexer")
ar = args(index_topk=24, candidate_topk_blocks=5, candidate_block_size=8)
s_ = 120
x = rn(1, s_ + 2, 5120)
qr = rn(1, s_ + 2, 1280)
lat1 = rn(1, s_ + 2, 512)
mods = {}
for lid in (20, 24):
    im = M.Indexer(ar, lid)
    set_linear(im.wq_b)
    set_linear(im.weights_proj, 0.05)
    if im.owns_k:
        set_linear(im.wk, 0.05)
        set_norm(im.k_norm)
    im.freqs_cis = freqs(1, ar.max_seq_len)
    mods[lid] = im
    fx.put(f"L{lid}/wq_b", im.wq_b.weight.data); fx.put(f"L{lid}/wq_b_s", im.wq_b.scale.data)
    fx.put(f"L{lid}/wproj", im.weights_proj.weight.data)
    if im.owns_k:
        fx.put(f"L{lid}/wk", im.wk.weight.data); fx.put(f"L{lid}/knorm", im.k_norm.weight.data)
fx.put("x", x); fx.put("qr", qr); fx.put("latent", lat1)
for step, (sp, ln) in enumerate([(0, s_), (s_, 1), (s_ + 1, 1)]):
    xs, qs_, ls = x[:, sp: sp + ln], qr[:, sp: sp + ln], lat1[:, sp: sp + ln]
    for lid in (20, 24):
        im = mods[lid]
        rep = indexer_replica(im, xs, qs_, ls if im.owns_k else None, sp, ln if sp == 0 else 128)
        # score with the vendor expression, then mask/candidates/top-k by the module itself
        out = im(xs, qs_, ls.clone() if im.owns_k else None, sp, ln if sp == 0 else 128)
        index_k = M.shared_attn.index_k[:1, : (sp + ln) // 1]
        sc = torch.einsum("bshd,btd->bsht", rep["q"], index_k)
        sc = (sc.relu_() * rep["w"].unsqueeze(-1)).sum(dim=2)
        for k_, v_ in rep.items():
            fx.put(f"s{step}/L{lid}/{k_}", v_)
        fx.put(f"s{step}/L{lid}/score_raw", sc)
        fx.put(f"s{step}/L{lid}/idx", out)
        if lid == 20:
            fx.put(f"s{step}/cand", M.shared_attn.candidates)
    fx.put(f"s{step}/index_k", M.shared_attn.index_k[:1, : sp + ln].clone())
fx.meta("seqlen", s_)
fx.meta("index_topk", 24); fx.meta("candidate_topk_blocks", 5); fx.meta("candidate_block_size", 8)
fx.close()

# ratio-2 index source (layer 2): owns k, no candidates, top-24 of 60 compressed
fx = case("indexer_r2")
ar = args(index_topk=24)
s_ = 121
im = M.Indexer(ar, 2)
set_linear(im.wq_b); set_linear(im.weights_proj, 0.05); set_linear(im.wk, 0.05); set_norm(im.k_norm)
im.freqs_cis = freqs(2, ar.max_seq_len)
x = rn(1, s_, 5120); qr = rn(1, s_, 1280); lat = rn(1, s_ // 2, 512)
rep = indexer_replica(im, x, qr, lat, 0, s_)
out = im(x, qr, lat.clone(), 0, s_)
index_k = M.shared_attn.index_k[:1, : s_ // 2]
sc = (torch.einsum("bshd,btd->bsht", rep["q"], index_k).relu_() * rep["w"].unsqueeze(-1)).sum(dim=2)
for k_, v_ in dict(wq_b=im.wq_b.weight.data, wq_b_s=im.wq_b.scale.data, wproj=im.weights_proj.weight.data, wk=im.wk.weight.data,
                   knorm=im.k_norm.weight.data, x=x, qr=qr, latent=lat, score_raw=sc, idx=out, **rep).items():
    fx.put(k_, v_)
fx.meta("seqlen", s_); fx.meta("index_topk", 24)
fx.close()

# ---------------------------------------------------------------- MoE: sqrtsoftplus gate with bias, fp4 experts, fp8 shared, clamps
fx = case("moe")
ar = args(n_routed_experts=16)
moe = M.MoE(3, ar)
set_linear_scale = 0.02
moe.gate.weight.data = rn(16, 5120, s=0.02)
moe.gate.bias.data = torch.randn(16, generator=g, dtype=torch.float32) * 0.05
for e in list(moe.experts) + [moe.shared_experts]:
    for lin in (e.w1, e.w2, e.w3):
        set_linear(lin)
x = rn(1, 45, 5120, s=1.0)
wts, ids = moe.gate(x.view(-1, 5120))
y = moe(x)
fx.put("x", x); fx.put("gate_w", moe.gate.weight.data); fx.put("gate_b", moe.gate.bias.data)
fx.put("route_w", wts); fx.put("route_i", ids.to(torch.int32)); fx.put("y", y)
for i, e in enumerate(list(moe.experts) + [moe.shared_experts]):
    tag = f"e{i}" if i < 16 else "shared"
    for nm in ("w1", "w2", "w3"):
        lin = getattr(e, nm)
        fx.put(f"{tag}/{nm}", lin.weight.data); fx.put(f"{tag}/{nm}_s", lin.scale.data)
# one expert alone with large activations, so the SwiGLU clamps engage
xe = rn(9, 5120, s=40.0)
we = torch.rand(9, 1, generator=g, dtype=torch.float32)
fx.put("solo/x", xe); fx.put("solo/wt", we); fx.put("solo/y", moe.experts[0](xe, we))
fx.put("solo_shared/y", moe.shared_experts(xe))
fx.meta("n_experts", 16); fx.meta("topk", 6); fx.meta("route_scale", cfg["route_scale"]); fx.meta("swiglu_limit", cfg["swiglu_limit"])
fx.close()

# ---------------------------------------------------------------- Engram (GPU side: dequant gather, wkv, normalized gate)
fx = case("engram")
ar = args()
lay = EngramLayout(max_ngram_size=4, layer_ids=(1, 14), num_embeddings=(5000, 5000), primes=((), ()), n_heads=8, head_dim=256)
en = M.Engram(ar, 1, lay)
en.embed.weight.data = (torch.randn(5000, 256, generator=g, dtype=torch.float32) * 2).to(torch.float8_e4m3fn)
en.embed.scale.data = e8((5000, 8), -8, -2)
set_linear(en.wkv)
en.q_weight.data = (1.0 + rn(4, 5120, s=0.2).float()).to(torch.bfloat16)
en.k_weight.data = (1.0 + rn(4, 5120, s=0.2).float()).to(torch.bfloat16)


def engram_embed_forward(self, indices):  # the oracle's host gather (F.embedding has no fp8 CPU kernel)
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    li = (indices - self.vocab_start_idx).masked_fill(mask, 0)
    flat = li.reshape(-1)
    values = self.weight.view(torch.uint8).index_select(0, flat).view(torch.float8_e4m3fn).view(*li.shape, -1)
    scales = self.scale.view(torch.uint8).index_select(0, flat).view(torch.float8_e8m0fnu).view(*li.shape, -1)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0)


M.ParallelEngramEmbedding.forward = engram_embed_forward
s_ = 19
h = rn(1, s_, 4, 5120, s=2.0)
hid = torch.randint(0, 5000, (1, s_, 24), generator=g)
emb = en.embed(hid)
kv = en.wkv(emb.flatten(-2))
y = en(h, hid)
for k_, v_ in dict(table=en.embed.weight.data, table_s=en.embed.scale.data, wkv=en.wkv.weight.data, wkv_s=en.wkv.scale.data,
                   qw=en.q_weight.data, kw=en.k_weight.data, h=h, hash=hid.to(torch.int32), emb=emb, kv=kv, y=y).items():
    fx.put(k_, v_)
fx.close()
print("FIXTURES-DONE", flush=True)
