#!/usr/bin/env python3
"""Golden vectors for the Inkling 2 Mini act-quant numeric program.

A line-for-line PORT to torch CPU float32 of the triton ops in the vLLM image
(vllm/models/inkling2/nvidia/ops/nvfp4.py, mxfp8.py, activation.py, moe.py,
moe_sm120.py; image digest in NOTES below). The triton kernels cannot be imported
on CPU, so nothing here imports them: each function names the source lines it ports.

What the port fixes and what it does not:
- quantizers (NVFP4 four-over-six, MXFP8), row scales, e2m1 and e4m3 rounding: every
  operation is one IEEE fp32 operation in the source order (enable_fp_fusion=False),
  so the port is the program.
- GEMM: the source accumulates in the tensor core, whose addition order is hardware
  defined. The goldens carry the fp32 INDEX-ORDER sum (a software model, labelled as
  such) plus the float64 sum and sum of magnitudes, for an error bound.
- SiTU: the source uses PTX approx instructions. The goldens evaluate exp2 and tanh in
  float64 and round once to fp32 (labelled 'f64 elementary functions').

Run: uv run --with torch --with numpy python3 gen_goldens.py OUT.bin
"""
import struct
import sys

import numpy as np
import torch

F32 = torch.float32
FP32_MAX = torch.tensor(3.4028234663852886e38, dtype=F32)


def f32(v):
    return torch.tensor(v, dtype=F32)


# ---------------------------------------------------------------- formats
E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=F32)


def e2m1(x):
    """nvfp4.py:28-63 _e2m1: code (sign in bit 3) and value."""
    a = x.abs()
    code = torch.where(
        a <= 0.25, 0,
        torch.where(a < 0.75, 1,
        torch.where(a <= 1.25, 2,
        torch.where(a < 1.75, 3,
        torch.where(a <= 2.5, 4,
        torch.where(a < 3.5, 5,
        torch.where(a <= 5.0, 6, 7))))))).to(torch.int64)
    value = E2M1_VALUES[code]
    sign = (x.view(torch.int32) >> 31) & 1
    return (code | (sign << 3)).to(torch.uint8), torch.where(sign != 0, -value, value)


def e4m3_encode(x):
    """fp32 -> float8_e4m3fn byte: round to nearest, ties to even, subnormals kept,
    saturating at 448 (cvt.rn.satfinite.e4m3x2.f32). Software, in float64 (exact)."""
    v = x.to(torch.float64)
    sign = ((x.view(torch.int32) >> 31) & 1).to(torch.int64)
    a = v.abs().clamp(max=448.0)
    # subnormal step 2**-9 below 2**-6
    sub = torch.round(a * 512.0)  # torch.round is half-to-even
    m, e = torch.frexp(a)  # a = m * 2**e, m in [0.5, 1)
    e1 = (e - 1).to(torch.int64)
    q = torch.round(m * 16.0).to(torch.int64)  # 8..16
    carry = q == 16
    e1 = torch.where(carry, e1 + 1, e1)
    q = torch.where(carry, torch.full_like(q, 8), q)
    normal = ((e1 + 7) << 3) | (q - 8)
    byte = torch.where(a < 2.0**-6, sub.to(torch.int64), normal)
    byte = byte.clamp(max=0x7E)
    return (byte | (sign << 7)).to(torch.uint8)


def e4m3_decode(b):
    b = b.to(torch.int64)
    sign = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = (b & 7).to(torch.float64)
    mag = torch.where(e == 0, m * 2.0**-9, (1.0 + m / 8.0) * torch.pow(2.0, (e - 7).to(torch.float64)))
    return torch.where(sign != 0, -mag, mag).to(F32)


def bf16_round(x):
    """fp32 -> bf16 -> fp32, RNE (torch's own cast)."""
    return x.to(torch.bfloat16).to(F32)


def bf16_bits(x):
    return (x.contiguous().view(torch.int32) >> 16).to(torch.int32).numpy().astype(np.uint16)


# ---------------------------------------------------------------- NVFP4 act quant
def block_error(x, reconstructed, variant=None):
    """nvfp4.py:88-94: pairwise tree over 16."""
    error = (reconstructed - x).abs()
    if variant == "seq_err":
        acc = error[..., 0]
        for i in range(1, 16):
            acc = acc + error[..., i]
        return acc
    for _ in range(4):
        error = error[..., 0::2] + error[..., 1::2]
    return error[..., 0]


def nvfp4_row_scalars(amax, variant=None):
    """nvfp4.py:231-240."""
    encoder = torch.minimum(f32(1536.0) / amax, FP32_MAX)
    decoder = f32(1.0) / encoder
    outer = amax * f32(1.0 / 1536.0)
    if variant == "outer_div":
        outer = amax / f32(1536.0)
    return encoder, decoder, outer


def nvfp4_quantize(x, variant=None):
    """nvfp4.py nvfp4_quantize: x fp32 [M, K] (bf16 values). Returns packed codes
    [M, K/2] u8, e4m3 scale bytes [M, K/16] u8, amax [M] fp32."""
    m, k = x.shape
    amax = x.abs().max(dim=1).values
    encoder, decoder, outer = nvfp4_row_scalars(amax, variant)
    xb = x.reshape(m, k // 16, 16)
    block_max = xb.abs().max(dim=2).values
    enc = encoder[:, None]
    dec = decoder[:, None]
    # nvfp4.py:98-108 _block_scales
    scale6 = torch.minimum(block_max * (enc * f32(1.0 / 6.0)), f32(448.0))
    scale4 = torch.minimum(block_max * (enc * f32(0.25)), f32(448.0))
    if variant == "div6":
        scale6 = torch.minimum(block_max * (enc / f32(6.0)), f32(448.0))
    if variant == "mul_order":
        scale6 = torch.minimum((block_max * enc) * f32(1.0 / 6.0), f32(448.0))
        scale4 = torch.minimum((block_max * enc) * f32(0.25), f32(448.0))
    byte6 = e4m3_encode(scale6)
    byte4 = e4m3_encode(scale4)
    scale6 = e4m3_decode(byte6)
    scale4 = e4m3_decode(byte4)
    inverse6 = torch.minimum(f32(1.0) / (scale6 * dec), FP32_MAX)
    inverse4 = torch.minimum(f32(1.0) / (scale4 * dec), FP32_MAX)
    # nvfp4.py:112-129 _quantize_block
    code6, rounded6 = e2m1(xb * inverse6[:, :, None])
    code4, rounded4 = e2m1(xb * inverse4[:, :, None])
    recon6 = rounded6 * scale6[:, :, None]
    recon4 = rounded4 * scale4[:, :, None]
    o = outer[:, None, None]
    error6 = block_error(xb, recon6 * o, variant)
    error4 = block_error(xb, recon4 * o, variant)
    if variant == "recon_assoc":
        error6 = block_error(xb, rounded6 * (scale6[:, :, None] * o))
        error4 = block_error(xb, rounded4 * (scale4[:, :, None] * o))
    choose4 = error4 < error6
    if variant == "tie_to_4":
        choose4 = error4 <= error6
    codes = torch.where(choose4[:, :, None], code4, code6).reshape(m, k)
    scales = torch.where(choose4, byte4, byte6)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed, scales, amax, choose4


# ---------------------------------------------------------------- MXFP8 act quant
def mxfp8_quantize(x):
    """mxfp8.py:14-30 _mxfp8_block on [M, K] in blocks of 32. Returns e4m3 bytes
    [M, K] and ue8m0 exponent bytes [M, K/32]."""
    m, k = x.shape
    g = x.reshape(m, k // 32, 32)
    amax = g.abs().max(dim=2).values
    raw = amax * f32(1.0 / 448.0)
    min_scale = f32(2.0**-127)
    exponent = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    exponent = torch.where(raw <= min_scale, 0, torch.clamp(exponent, max=254))
    exponent = torch.where(raw != raw, 255, exponent).to(torch.int32)
    scale = torch.maximum((exponent << 23).view(F32), min_scale)
    inverse = torch.where(exponent == 0, f32(2.0**127), f32(1.0) / scale)
    values = g * inverse[:, :, None]
    values = torch.clamp(values, -448.0, 448.0)
    return e4m3_encode(values).reshape(m, k), exponent.to(torch.uint8)


def mxfp8_dequant(values, exponents):
    """moe_sm120.py:77-82 _dequant_e4m3 (before the bf16 store, which is exact here)."""
    e = exponents.to(torch.int32)
    bits = torch.where(e == 0, 0x00400000, e << 23).to(torch.int32)
    s = bits.view(F32)
    k = values.shape[-1]
    return e4m3_decode(values) * s.repeat_interleave(32, dim=-1)[..., :k]


# ---------------------------------------------------------------- GEMM models
def unpack_nvfp4(packed, scale_bytes):
    """codes [.., K/2] + e4m3 scales [.., K/16] -> fp32 val(code) * scale."""
    lo = packed & 0xF
    hi = packed >> 4
    codes = torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1).to(torch.int64)
    val = E2M1_VALUES[codes & 7]
    val = torch.where((codes & 8) != 0, -val, val)
    return val * e4m3_decode(scale_bytes).repeat_interleave(16, dim=-1)


def dot_index_order(a, w):
    """a [R, K], w [R, N, K] fp32 -> fp32 [R, N], products then adds in ascending k.
    Also the float64 sum and sum of magnitudes."""
    r, n, k = w.shape
    acc = torch.zeros((r, n), dtype=F32)
    for i in range(k):
        acc = acc + a[:, None, i] * w[:, :, i]
    p = a[:, None, :].double() * w.double()
    return acc, p.sum(dim=2), p.abs().sum(dim=2)


def rcp_rn(x):
    return f32(1.0) / x


def row_scale(act_amax, w_amax, denominator):
    """moe.py:346-353 / activation.py:153-170: (amax * w_amax) * rcp_rn(1536 * denom)."""
    return (act_amax * w_amax) * rcp_rn(f32(1536.0) * denominator)


def situ(gate_up, gamma, gate_cap=4.0, up_cap=25.0):
    """activation.py:64-78 with float64 exp2/tanh rounded once to fp32.
    gate_up [R, 2F] interleaved (gate, up); gamma [R]. Returns bf16-rounded fp32 [R, F]."""
    gate = gate_up[:, 0::2]
    up = gate_up[:, 1::2]
    t = gate * f32(-1.4426950408889634)
    ex = torch.pow(torch.tensor(2.0, dtype=torch.float64), t.double()).to(F32)
    sigmoid = f32(1.0) / (ex + f32(1.0))
    gate_c = f32(gate_cap) * torch.tanh((gate / f32(gate_cap)).double()).to(F32)
    up_c = f32(up_cap) * torch.tanh((up / f32(up_cap)).double()).to(F32)
    return bf16_round(((gate_c * sigmoid) * up_c) * gamma[:, None])


# ---------------------------------------------------------------- container
class Out:
    def __init__(self):
        self.records = []

    def add(self, name, **blobs):
        self.records.append((name, blobs))

    def write(self, path):
        with open(path, "wb") as f:
            f.write(b"INK2AQ01")
            f.write(struct.pack("<I", len(self.records)))
            for name, blobs in self.records:
                nb = name.encode()
                f.write(struct.pack("<H", len(nb)) + nb)
                f.write(struct.pack("<I", len(blobs)))
                for tag, arr in blobs.items():
                    tb = tag.encode()
                    data = np.ascontiguousarray(arr).tobytes()
                    f.write(struct.pack("<H", len(tb)) + tb)
                    f.write(struct.pack("<I", len(data)) + data)


def u32s(*v):
    return np.array(v, dtype=np.uint32)


def to_bf16(x):
    return bf16_round(x.to(F32))


def from_bits16(bits):
    return (torch.tensor(np.asarray(bits, dtype=np.uint16).astype(np.int32)) << 16).to(torch.int32).view(F32)


# ---------------------------------------------------------------- cases
def nvfp4_adversarial(k, gen):
    rows = []
    names = []

    def add(name, row):
        names.append(name)
        rows.append(to_bf16(row))

    z = torch.zeros(k)
    add("zero", z)
    add("neg_zero", -z)
    r = z.clone(); r[1::2] = -0.0
    add("mixed_zero", r)
    for idx, val in ((0, 1.0), (k - 1, -3.5), (17, 2.0**-120), (33, -(2.0**120))):
        r = z.clone(); r[idx] = val
        add(f"single_max_{idx}", r)
    # ties at every e2m1 threshold: amax 6 gives encoder 256 and a 6-based scale whose
    # inverse is 1, so the listed values land exactly on the thresholds.
    ties = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
                         -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0])
    for shift in (0, -40, 60):
        add(f"ties_6_2^{shift}", (ties * 2.0**shift).repeat(k // 16))
    # the same thresholds for the 4-based candidate (block max 4, inverse 1 at scale 384)
    ties4 = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 4.0, 0.0,
                          -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -4.0, -0.0])
    r = ties4.repeat(k // 16).clone(); r[0] = 6.0
    add("ties_4", r)
    # one ulp (bf16) either side of the thresholds
    up = from_bits16(bf16_bits(to_bf16(ties)) + 1)
    dn = from_bits16(bf16_bits(to_bf16(ties)) - 1)
    r = torch.cat([up, dn]).repeat(k // 32); r[7] = 6.0; r[15] = -6.0
    add("ties_nextafter", r)
    # denormal inputs: encoder clamps to FP32_MAX, block scales are e4m3 subnormals
    den = from_bits16(torch.randint(1, 0x80, (k,), generator=gen).numpy())
    add("denormal", den)
    r = den.clone(); r[::16] = 0.0; r[5] = from_bits16([0x0001])[0]
    add("denormal_min", r)
    # largest finite bf16 and a row mixing it with small blocks (inverse overflows to FP32_MAX)
    big = from_bits16([0x7F7F])[0]
    r = torch.randn(k, generator=gen); r[3] = big; r[100] = -big
    add("huge_mixed", r)
    add("huge_all", torch.full((k,), float(big)) * torch.sign(torch.randn(k, generator=gen)))
    # blocks spanning 2**-20..2**20 in one row
    e = torch.linspace(-20, 20, k // 16).repeat_interleave(16)
    add("block_ladder", torch.randn(k, generator=gen) * torch.pow(2.0, e))
    add("constant", torch.full((k,), 0.8125))
    add("alternating", torch.tensor([1.0, -1.0]).repeat(k // 2) * 3.0)
    # e4m3 scale halfway points: block max chosen so block_max * (encoder / 6) is halfway
    r = torch.randn(k, generator=gen)
    r[0] = 6.0
    halves = torch.tensor([1.0625, 1.1875, 1.3125, 1.4375, 1.5625, 1.6875, 1.8125, 1.9375])
    for j, h in enumerate(halves.tolist()):
        blk = 16 * (j + 1)
        r[blk:blk + 16] = r[blk:blk + 16].clamp(-0.01, 0.01)
        r[blk] = 6.0 * h / 256.0 * 2.0 ** (j - 3)
    add("scale_halfway", r)
    return names, torch.stack(rows)


def mxfp8_adversarial(k, gen):
    rows, names = [], []

    def add(name, row):
        names.append(name)
        rows.append(to_bf16(row))

    z = torch.zeros(k)
    add("zero", z)
    add("neg_zero", -z)
    for idx, val in ((0, 1.0), (k - 1, -448.0), (40, 2.0**-126), (70, -(2.0**126))):
        r = z.clone(); r[idx] = val
        add(f"single_max_{idx}", r)
    # amax exactly 448 * 2**j: raw is a power of two and the round-up keeps it
    r = torch.randn(k, generator=gen)
    for b in range(k // 32):
        r[32 * b:32 * b + 32] *= 2.0 ** (b % 23 - 11)
        r[32 * b] = 448.0 * 2.0 ** (b % 23 - 11)
    add("pow2_amax", r)
    r2 = r.clone(); r2[::32] = from_bits16(bf16_bits(to_bf16(r[::32])) + 1)
    add("pow2_amax_up", r2)
    r3 = r.clone(); r3[::32] = from_bits16(bf16_bits(to_bf16(r[::32])) - 1)
    add("pow2_amax_down", r3)
    # e4m3 halfway values at scale 1 (amax 448): normal and subnormal ties
    h = [448.0]
    for e in range(-6, 8):
        for mm in range(8):
            h.append((1 + mm / 8 + 1 / 16) * 2.0**e)
    for s in range(8):
        h.append((s + 0.5) * 2.0**-9)
    hv = torch.tensor(h)
    hv = hv[to_bf16(hv) == hv]
    rr = torch.zeros(k)
    reps = k // 32
    for b in range(reps):
        seg = hv[(b * 30) % len(hv):][:30]
        rr[32 * b] = 448.0
        rr[32 * b + 1:32 * b + 1 + len(seg)] = seg * (1 if b % 2 == 0 else -1)
    add("e4m3_halfway", rr)
    den = from_bits16(torch.randint(1, 0x80, (k,), generator=gen).numpy())
    add("denormal", den)
    r = torch.zeros(k); r[0] = from_bits16([0x0001])[0]; r[32] = from_bits16([0x0041])[0]; r[64] = 2.0**-118
    add("denormal_edges", r)
    big = from_bits16([0x7F7F])[0]
    r = torch.randn(k, generator=gen); r[3] = big; r[100] = -big
    add("huge_mixed", r)
    add("block_ladder", torch.randn(k, generator=gen) * torch.pow(2.0, torch.linspace(-30, 30, k // 32).repeat_interleave(32)))
    return names, torch.stack(rows)


def random_rows(m, k, gen):
    x = torch.randn(m, k, generator=gen)
    x = x * torch.pow(2.0, torch.linspace(-6, 6, m))[:, None]
    x[m // 2:] = x[m // 2:] * torch.exp(torch.randn(m - m // 2, k, generator=gen))  # heavy tail
    return to_bf16(x)


def random_w4(e, n, k, gen):
    codes = torch.randint(0, 256, (e, n, k // 2), generator=gen, dtype=torch.int64).to(torch.uint8)
    scales = torch.randint(0x20, 0x70, (e, n, k // 16), generator=gen, dtype=torch.int64).to(torch.uint8)
    amax = (torch.rand(e, generator=gen) + 0.5).to(F32)
    denom = f32(6.0 * 448.0 / 2.0)
    return codes, scales, amax, denom


def random_w8(e, n, k, gen):
    w = torch.randint(0, 256, (e, n, k), generator=gen, dtype=torch.int64)
    w = torch.where((w & 0x7F) == 0x7F, w & 0x80, w).to(torch.uint8)  # no NaN codes
    ex = torch.randint(110, 124, (e, n, k // 32), generator=gen, dtype=torch.int64).to(torch.uint8)
    return w, ex


def main(path):
    gen = torch.Generator().manual_seed(20261121)
    out = Out()

    # ---- e2m1 / e4m3 unit grids
    th = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 7.0, 1e30, 3.4028234663852886e38]
    t = torch.tensor(th, dtype=F32)
    bits = t.view(torch.int32)
    grid = torch.cat([t, (bits + 1).view(F32), (bits - 1).clamp(min=0).view(F32), torch.tensor([float("inf")], dtype=F32)])
    grid = torch.cat([grid, -grid])
    code, _ = e2m1(grid)
    out.add("e2m1/grid", x=grid.numpy(), code=code.numpy())

    allb = torch.arange(0, 0x7F, dtype=torch.int64)
    vals = e4m3_decode(allb.to(torch.uint8)).double()
    mids = (vals[:-1] + vals[1:]) / 2
    m32 = mids.to(F32)
    assert torch.equal(m32.double(), mids)
    mb = m32.view(torch.int32)
    g4 = torch.cat([vals.to(F32), m32, (mb + 1).view(F32), (mb - 1).view(F32), torch.tensor([448.0, 2.0**-10, 2.0**-11, 2.0**-30, 1e-45])])
    g4 = torch.cat([g4, -g4])
    by = e4m3_encode(g4)
    # torch's own cast agrees wherever it is defined (|x| <= 448)
    torch_cast = None
    try:
        torch_cast = g4.to(torch.float8_e4m3fn).view(torch.uint8)
    except Exception as exc:  # noqa: BLE001
        print("torch float8_e4m3fn cast unavailable:", exc)
    if torch_cast is not None:
        same = int((torch_cast == by).sum())
        print(f"e4m3: software RNE vs torch cast: {same}/{by.numel()} equal")
        assert same == by.numel(), (g4[torch_cast != by], by[torch_cast != by], torch_cast[torch_cast != by])
    out.add("e4m3/grid", x=g4.numpy(), byte=by.numpy())
    sat = torch.tensor([449.0, 464.0, 480.0, 1e10, 3.4028234663852886e38, -449.0, -1e10], dtype=F32)
    out.add("e4m3/satfinite", x=sat.numpy(), byte=e4m3_encode(sat).numpy())

    # ---- NVFP4 quantizer
    stats = {}
    for name, x in (
        ("rand_k2304", random_rows(8, 2304, gen)),
        ("rand_k1152", random_rows(8, 1152, gen)),
        ("rand_k64", random_rows(64, 64, gen)),
        ("adv_k2304", nvfp4_adversarial(2304, gen)[1]),
        ("adv_k1152", nvfp4_adversarial(1152, gen)[1]),
        ("adv_k256", nvfp4_adversarial(256, gen)[1]),
    ):
        packed, scales, amax, choose4 = nvfp4_quantize(x)
        out.add("nvfp4q/" + name, dims=u32s(*x.shape), x=bf16_bits(x), codes=packed.numpy(), scales=scales.numpy(), amax=amax.numpy())
        stats[name] = (int(choose4.sum()), choose4.numel())
    print("nvfp4 blocks choosing the 4-based scale:", stats)

    # ---- discriminators: rows on which a plausible wrong reading of the source changes
    # the output bytes. Searched at random; each kept row differs under the named variant.
    variants = ("seq_err", "outer_div", "div6", "mul_order", "recon_assoc", "tie_to_4")
    found = {v: [] for v in variants}
    tried = 0
    for batch in range(40):
        kk = 32
        xs = torch.randn(200000, kk, generator=gen) * torch.exp(torch.randn(200000, 1, generator=gen) * 3.0)
        if batch % 2 == 1:
            xs = torch.round(xs * 16.0) / 16.0  # coarse grid: many exact error ties
        xs = to_bf16(xs)
        tried += xs.shape[0]
        base = nvfp4_quantize(xs)
        for v in variants:
            if len(found[v]) >= 48:
                continue
            alt = nvfp4_quantize(xs, v)
            diff = (alt[0] != base[0]).any(dim=1) | (alt[1] != base[1]).any(dim=1)
            for i in torch.nonzero(diff)[:48 - len(found[v]), 0].tolist():
                found[v].append(xs[i])
        if all(len(found[v]) >= 48 for v in variants):
            break
    print("discriminator rows per variant (of", tried, "rows tried):", {v: len(r) for v, r in found.items()})
    for v in variants:
        if not found[v]:
            continue
        x = torch.stack(found[v])
        packed, scales, amax, _ = nvfp4_quantize(x)
        alt = nvfp4_quantize(x, v)
        assert ((alt[0] != packed).any(dim=1) | (alt[1] != scales).any(dim=1)).all()
        out.add("nvfp4q/disc_" + v, dims=u32s(*x.shape), x=bf16_bits(x), codes=packed.numpy(), scales=scales.numpy(), amax=amax.numpy())

    # ---- MXFP8 quantizer
    for name, x in (
        ("rand_k2304", random_rows(8, 2304, gen)),
        ("rand_k1152", random_rows(8, 1152, gen)),
        ("adv_k2304", mxfp8_adversarial(2304, gen)[1]),
        ("adv_k1152", mxfp8_adversarial(1152, gen)[1]),
    ):
        values, exps = mxfp8_quantize(x)
        out.add("mxfp8q/" + name, dims=u32s(*x.shape), x=bf16_bits(x), values=values.numpy(), exps=exps.numpy())

    # ---- W4A4 GEMM (index-order model) and row scales
    for name, (e, n, k, r) in (("small", (3, 32, 128, 12)), ("k1152", (2, 16, 1152, 8)), ("k2304", (2, 16, 2304, 8))):
        wc, ws, wamax, denom = random_w4(e, n, k, gen)
        x = random_rows(r, k, gen)
        expert = torch.randint(0, e, (r,), generator=gen)
        packed, scales, amax, _ = nvfp4_quantize(x)
        a = unpack_nvfp4(packed, scales)
        w = unpack_nvfp4(wc, ws)[expert]
        acc, exact, mag = dot_index_order(a, w)
        rs = row_scale(amax, wamax[expert], denom)
        y = bf16_round((acc * f32(1.0)) * rs[:, None])
        out.add("w4a4/" + name, dims=u32s(e, n, k, r), wcodes=wc.numpy(), wscales=ws.numpy(), wamax=wamax.numpy(),
                denom=np.array([float(denom)], dtype=np.float32), expert=expert.numpy().astype(np.uint32), x=bf16_bits(x),
                acc=acc.numpy(), rowscale=rs.numpy(), out=bf16_bits(y), exact=exact.numpy(), mag=mag.numpy())

    # ---- W8A8 GEMM (index-order model)
    for name, (e, n, k, r) in (("small", (3, 32, 128, 12)), ("k1152", (2, 16, 1152, 8)), ("k2304", (2, 16, 2304, 8))):
        w8, wex = random_w8(e, n, k, gen)
        x = random_rows(r, k, gen)
        expert = torch.randint(0, e, (r,), generator=gen)
        values, exps = mxfp8_quantize(x)
        a = mxfp8_dequant(values, exps)
        w = mxfp8_dequant(w8, wex)[expert]
        acc, exact, mag = dot_index_order(a, w)
        y = bf16_round(acc * f32(1.0))
        out.add("w8a8/" + name, dims=u32s(e, n, k, r), w=w8.numpy(), wexps=wex.numpy(), expert=expert.numpy().astype(np.uint32),
                x=bf16_bits(x), acc=acc.numpy(), out=bf16_bits(y), exact=exact.numpy(), mag=mag.numpy())

    # ---- whole expert block: quant x once per token, w13, SiTU * gamma, quant, w2, combine
    for kind in ("nvfp4", "mxfp8"):
        e, h, f, tokens, topk = 6, 128, 64, 5, 3
        x = to_bf16(torch.randn(tokens, h, generator=gen) * 2.0)
        sel = torch.stack([torch.randperm(e, generator=gen)[:topk] for _ in range(tokens)])
        gamma = (torch.rand(tokens, topk, generator=gen) * 2.0 + 0.1).to(F32)
        rows_e = sel.reshape(-1)
        rows_g = gamma.reshape(-1)
        xr = x.repeat_interleave(topk, dim=0)
        if kind == "nvfp4":
            w13c, w13s, w13a, d13 = random_w4(e, 2 * f, h, gen)
            w2c, w2s, w2a, d2 = random_w4(e, h, f, gen)
            packed, scales, amax, _ = nvfp4_quantize(x)
            a = unpack_nvfp4(packed, scales).repeat_interleave(topk, dim=0)
            acc, _, _ = dot_index_order(a, unpack_nvfp4(w13c, w13s)[rows_e])
            rs13 = row_scale(amax.repeat_interleave(topk), w13a[rows_e], d13)
            gate_up = bf16_round((acc * f32(1.0)) * rs13[:, None])
            y = situ(gate_up, rows_g)
            yp, ys, yamax, _ = nvfp4_quantize(y)
            acc2, _, _ = dot_index_order(unpack_nvfp4(yp, ys), unpack_nvfp4(w2c, w2s)[rows_e])
            rs2 = row_scale(yamax, w2a[rows_e], d2)
            down = bf16_round((acc2 * f32(1.0)) * rs2[:, None])
            blobs = dict(w13codes=w13c.numpy(), w13scales=w13s.numpy(), w13amax=w13a.numpy(), w13denom=np.array([float(d13)], dtype=np.float32),
                         w2codes=w2c.numpy(), w2scales=w2s.numpy(), w2amax=w2a.numpy(), w2denom=np.array([float(d2)], dtype=np.float32))
        else:
            w13, w13e = random_w8(e, 2 * f, h, gen)
            w2, w2e = random_w8(e, h, f, gen)
            w13e = (w13e - 8).to(torch.uint8)
            values, exps = mxfp8_quantize(x)
            a = mxfp8_dequant(values, exps).repeat_interleave(topk, dim=0)
            acc, _, _ = dot_index_order(a, mxfp8_dequant(w13, w13e)[rows_e])
            gate_up = bf16_round(acc * f32(1.0))
            y = situ(gate_up, rows_g)
            yv, ye = mxfp8_quantize(y)
            acc2, _, _ = dot_index_order(mxfp8_dequant(yv, ye), mxfp8_dequant(w2, w2e)[rows_e])
            down = bf16_round(acc2 * f32(1.0))
            blobs = dict(w13=w13.numpy(), w13exps=w13e.numpy(), w2=w2.numpy(), w2exps=w2e.numpy())
        comb = torch.zeros(tokens, h, dtype=F32)
        dr = down.reshape(tokens, topk, h)
        for s in range(topk):
            comb = comb + dr[:, s]
        comb = bf16_round(comb)
        assert torch.isfinite(down).all() and torch.isfinite(y).all()
        out.add("chain/" + kind, dims=u32s(e, h, f, tokens, topk), x=bf16_bits(x), sel=sel.numpy().astype(np.uint32), gamma=gamma.numpy(),
                gate_up=bf16_bits(gate_up), y=bf16_bits(y), down=bf16_bits(down), out=bf16_bits(comb), **blobs)
        del xr

    out.write(path)
    print("records:", len(out.records))


if __name__ == "__main__":
    main(sys.argv[1])
