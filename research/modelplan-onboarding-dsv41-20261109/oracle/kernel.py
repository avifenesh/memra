"""Torch stand-ins for DeepSeek-V4.1's reference tilelang kernels (inference/kernel.py at revision
1bae0e1088993b9b82de08f59de5be5a34e42170), so the vendor model.py runs unchanged on CPU (or any GPU) as a correctness
oracle. Same entry points, same quantization grids and scale rounding, the same online-softmax blocking in
sparse_attn. The free choices the tilelang kernels leave to the hardware (fp32 accumulation order, the exp
approximation) are pinned to the canonical one: GEMMs and sums accumulate in float64 and round once to float32,
exp and sigmoid are evaluated in float64 and rounded once (see exact.py, which applies the same rule to the torch ops
model.py calls). memra's reference executor uses the same primitive semantics."""
import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
QCHUNK = 256  # queries per sparse_attn slice (bounds the gathered-KV working set)


def _cr(fn, x: torch.Tensor) -> torch.Tensor:
    """fn evaluated in float64, rounded once to float32."""
    return fn(x.double()).float()


def _sum32(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """A float64-accumulated sum rounded once to float32."""
    return x.double().sum(dim, keepdim=keepdim).float()


def _pow2_ceil(v: torch.Tensor) -> torch.Tensor:
    """2**ceil(log2(v)) for positive normal fp32 v, bit-exact with the kernel's fast_log2_ceil/fast_pow2."""
    bits = v.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((e + 127) << 23).to(torch.int32).view(torch.float32)


def _round_fp4(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 already clamped to [-6, 6] onto the e2m1 grid, nearest with ties to even mantissa."""
    a = x.abs()
    q = torch.where(a <= 0.25, 0.0, torch.where(a < 0.75, 0.5, torch.where(a <= 1.25, 1.0, torch.where(a < 1.75, 1.5,
        torch.where(a <= 2.5, 2.0, torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)))))))
    return torch.copysign(q, x)


def _scale_to(s: torch.Tensor, scale_dtype):
    return s.to(scale_dtype) if scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn) else s


_CHUNK_ELEMS = 1 << 26  # quantize long-context activations this many elements at a time (blocks are per row, so exact)


def _u8(t):
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _row_chunks(x):
    rows = x.view(-1, x.size(-1))
    step = max(1, _CHUNK_ELEMS // x.size(-1))
    return [rows[r0:r0 + step] for r0 in range(0, rows.size(0), step)]


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    if x.numel() <= _CHUNK_ELEMS or not x.is_contiguous():
        return _act_quant(x, block_size, scale_fmt, scale_dtype, inplace)
    parts = [_act_quant(r, block_size, scale_fmt, scale_dtype, inplace) for r in _row_chunks(x)]
    if inplace:
        return x
    q = torch.cat([_u8(p[0]) for p in parts]).view(parts[0][0].dtype).view(*x.shape[:-1], -1)
    sc = torch.cat([_u8(p[1]) for p in parts]).view(parts[0][1].dtype).view(*x.shape[:-1], -1)
    return q, sc


def _act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    N = x.size(-1)
    assert N % block_size == 0
    g = x.float().unflatten(-1, (N // block_size, block_size))
    amax = g.abs().amax(-1).clamp_min(1e-4)
    s = _pow2_ceil(amax * (1 / FP8_MAX)) if scale_fmt is not None else amax * (1 / FP8_MAX)
    q = (g / s.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    return q.flatten(-2), _scale_to(s, scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    if inplace and x.numel() > _CHUNK_ELEMS and x.is_contiguous():
        for r in _row_chunks(x):
            _fp4_act_quant(r, block_size, True, scale_dtype)
        return x
    return _fp4_act_quant(x, block_size, inplace, scale_dtype)


def _fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    g = x.float().unflatten(-1, (N // block_size, block_size))
    amax = g.abs().amax(-1)
    if scale_dtype == torch.float8_e4m3fn:
        s = (amax.clamp_min(6 * 2.0**-9) / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        s = _pow2_ceil(amax.clamp_min(6 * 2.0**-126) * (1 / FP4_MAX))
    q = _round_fp4((g / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX))
    if inplace:
        x.copy_((q * s.unsqueeze(-1)).flatten(-2).to(x.dtype))
        return x
    raise NotImplementedError("the oracle only needs the in-place fp4 quant-dequant")


def _deq_rows(w: torch.Tensor, s: torch.Tensor, block: int) -> torch.Tensor:
    """fp8 weight [N, K] with per-(block x block) scales [ceil(N/b), K/b] -> fp32 [N, K]."""
    N, K = w.shape
    sf = s.float().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)[:, :K]
    return w.float() * sf


def unpack_fp4(b: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """fp4 weight stored [N, K/2] (two e2m1 per byte, low nibble first) with per-32 scales [N, K/32] -> fp32 [N, K]."""
    u = b.contiguous().view(torch.uint8)
    t = FP4_TABLE.to(u.device)
    v = torch.stack([t[(u & 0x0F).long()], t[(u >> 4).long()]], -1).flatten(-2)
    return v * b_s.float().repeat_interleave(32, -1)


_DEQ = {}  # fp8 dense weights are a few GB: dequantize each once (fp4 experts are re-unpacked per call)


def _deq_cached(b, b_s, block_size):
    """Dequantized fp8 weight, cached as bf16: an e4m3 value times a power-of-two scale has at most 4 significant bits,
    so bf16 holds it exactly (asserted once per weight); the GEMM still runs in fp32."""
    key = (b.data_ptr(), tuple(b.shape), block_size)
    bd = _DEQ.get(key)
    if bd is None:
        f = _deq_rows(b, b_s, block_size)
        bd = f.to(torch.bfloat16)
        assert torch.equal(bd.float(), f), "fp8 dequant not exact in bf16"
        _DEQ[key] = bd
    return bd


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    K = a.size(-1)
    ad = a.float() * a_s.float().repeat_interleave(block_size, -1)
    bd = _deq_cached(b, b_s, block_size).float()
    y = (ad.reshape(-1, K).double() @ bd.double().t()).float()
    return y.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    K = a.size(-1)
    ad = a.float() * a_s.float().repeat_interleave(act_block_size, -1)
    bd = unpack_fp4(b, b_s)
    y = (ad.reshape(-1, K).double() @ bd.double().t()).float()
    return y.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


SPARSE_BLOCK = 64  # selected entries per online-softmax block in the tilelang sparse_attn_kernel


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d] bf16, kv [b,n,d] bf16, idxs [b,m,k] (-1 = empty slot). Mirrors sparse_attn_kernel: the selected
    entries are walked in blocks of 64 with a running max that starts at -1e30; each block's probabilities are
    exp(s - running_max) in fp32, cast to bf16 for the PV product, while the fp32 denominator sums the unrounded
    values; both rescale by exp(old_max - new_max) at every block, and the sink joins the denominator once at the end
    as exp(sink - final_max). A row with no valid entry returns zeros."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    for bi in range(b):
        for m0 in range(0, m, QCHUNK):
            idx = topk_idxs[bi, m0:m0 + QCHUNK].long()  # [c, k]
            qf = q[bi, m0:m0 + QCHUNK].float()  # [c, h, d]
            c = idx.size(0)
            run_max = torch.full((c, h), -1e30)
            den = torch.zeros(c, h)
            acc = torch.zeros(c, h, d)
            for k0 in range(0, idx.size(1), SPARSE_BLOCK):
                bidx = idx[:, k0:k0 + SPARSE_BLOCK]
                valid = bidx >= 0
                g = kv[bi][bidx.clamp_min(0)]  # [c, kb, d]
                g = torch.where(valid.unsqueeze(-1), g, torch.zeros((), dtype=g.dtype))
                s = torch.einsum("chd,ckd->chk", qf.double(), g.double()).float() * softmax_scale
                s = s.masked_fill(~valid.unsqueeze(1), float("-inf"))
                new_max = torch.maximum(run_max, s.amax(-1))
                rescale = _cr(torch.exp, run_max - new_max)
                p = _cr(torch.exp, s - new_max.unsqueeze(-1))
                den = den * rescale + _sum32(p, -1)
                pv = torch.einsum("chk,ckd->chd", p.to(torch.bfloat16).double(), g.double()).float()
                acc = acc * rescale.unsqueeze(-1) + pv
                run_max = new_max
            den = den + _cr(torch.exp, attn_sink.float().view(1, h) - run_max)
            out[bi, m0:m0 + QCHUNK] = (acc / den.unsqueeze(-1)).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = _cr(torch.sigmoid, m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * _cr(torch.sigmoid, m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = _cr(lambda t: t.softmax(-1), comb) + eps
    comb = comb / (_sum32(comb, -2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (_sum32(comb, -1, keepdim=True) + eps)
        comb = comb / (_sum32(comb, -2, keepdim=True) + eps)
    return pre, post, comb
