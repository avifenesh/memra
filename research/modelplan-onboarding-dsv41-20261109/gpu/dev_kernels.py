"""Development loop for cu/dsv41_gpu.cu: load the kernels as a standalone shared object (nvcc -shared, the same
flags as build.rs) through ctypes and run them on the random fixtures with torch on the GPU. The receipts come from
the Rust gate (dsv41-gate kernels); this only finds bugs fast without the 16-minute engine build.

usage: dev_kernels.py LIB.so FIXTURE_ROOT [case-prefix ...]"""
import ctypes as C
import json
import os
import sys

import torch

lib = C.CDLL(sys.argv[1])
root = sys.argv[2]
only = sys.argv[3:]
dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False
DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64, "u8": torch.uint8,
      "i8": torch.int8, "f8e4m3": torch.uint8, "e8m0": torch.uint8}
VP, L, I, F = C.c_void_p, C.c_long, C.c_int, C.c_float


def fx(case):
    d = os.path.join(root, case)
    idx = json.load(open(os.path.join(d, "index.json")))
    out = {}
    for k, e in idx["tensors"].items():
        raw = torch.from_file(os.path.join(d, e["file"]), size=os.path.getsize(os.path.join(d, e["file"])), dtype=torch.uint8)
        t = raw.view(DT[e["dtype"]]) if raw.numel() else torch.empty(0, dtype=DT[e["dtype"]])
        out[k] = t.reshape(e["shape"]).to(dev)
    return out, idx["meta"]


def p(t):
    return VP(t.data_ptr()) if t is not None else VP(0)


S = VP(0)  # default stream


def rc(name, r):
    if r != 0:
        raise RuntimeError(f"{name}: rc {r}")
    torch.cuda.synchronize()


def ulp_stats(got, want):
    g = got.contiguous().view(torch.int16).to(torch.int32)
    w = want.contiguous().view(torch.int16).to(torch.int32)
    # order map for bf16 bits: negative values descending
    go = torch.where(g < 0, -(g & 0x7FFF), g)
    wo = torch.where(w < 0, -(w & 0x7FFF), w)
    d = (go - wo).abs()
    gf, wf = got.float(), want.float()
    fin = torch.isfinite(wf) & torch.isfinite(gf)
    rel = ((gf - wf)[fin].norm() / wf[fin].norm().clamp_min(1e-30)).item()
    return dict(n=g.numel(), exact=int((d == 0).sum()), max_ulp=int(d[fin].max()) if fin.any() else 0,
                over1=int((d[fin] > 1).sum()), rel_l2=rel, nonfin_mismatch=int((~fin & (g != w)).sum()))


def show(name, st):
    print(f"{name:45s} n={st['n']} exact={st['exact'] / st['n'] * 100:.3f}% max_ulp={st['max_ulp']} >1ulp={st['over1']} "
          f"rel_l2={st['rel_l2']:.3e} nonfin_mismatch={st['nonfin_mismatch']}", flush=True)


def want(case):
    return not only or any(case.startswith(o) for o in only)


def f32rel(name, got, want_):
    rel = ((got - want_).norm() / want_.norm()).item()
    print(f"{name:45s} rel_l2={rel:.3e} max_abs={(got - want_).abs().max().item():.3e}", flush=True)


lib.memra_v41_act_quant.argtypes = [VP, L, VP, VP, L, VP, L, VP, L, I, I, VP]
lib.memra_v41_gemm_f8.argtypes = [VP, L, VP, L, VP, L, VP, L, F, I, VP, L, I, I, I, I, VP]
lib.memra_v41_gemm_bf16.argtypes = [VP, L, L, VP, L, L, VP, L, L, I, I, I, I, I, VP]
lib.memra_v41_fp4_qdq.argtypes = [VP, L, I, I, I, VP]
lib.memra_v41_rmsnorm.argtypes = [VP, L, VP, VP, L, I, I, F, VP]
lib.memra_v41_rope.argtypes = [VP, L, I, I, I, VP, VP, I, I, VP, I, VP]
lib.memra_v41_sparse_attn.argtypes = [VP, L, VP, VP, L, I, VP, F, VP, L, I, I, I, VP]
lib.memra_v41_hc_mixes.argtypes = [VP, VP, VP, VP, I, I, I, I, F, F, VP, VP, VP, VP, VP]
lib.memra_v41_hc_pre.argtypes = [VP, VP, I, I, I, VP, L, VP]
lib.memra_v41_hc_post.argtypes = [VP, L, VP, VP, VP, I, I, I, VP, VP]
lib.memra_v41_comp_pool2.argtypes = [VP, VP, L, VP, L, I, I, VP]
lib.memra_v41_index_score.argtypes = [VP, L, VP, I, VP, L, F, I, I, VP, L, I, VP, L, I, VP]
lib.memra_v41_topk.argtypes = [VP, L, I, I, I, I, VP, L, I, VP]
lib.memra_v41_candidates.argtypes = [VP, L, I, I, I, I, I, VP, L, I, VP]
lib.memra_v41_route.argtypes = [VP, L, VP, I, I, I, F, VP, VP, I, VP]
lib.memra_v41_swiglu.argtypes = [VP, VP, L, VP, F, VP, L, I, I, VP]
lib.memra_v41_engram_dequant.argtypes = [VP, VP, L, I, VP, VP]
lib.memra_v41_engram_gate.argtypes = [VP, VP, VP, VP, I, I, I, F, F, VP, VP]


def actq(x2):
    m, k = x2.shape
    q = torch.empty(m, k, dtype=torch.uint8, device=dev)
    s = torch.empty(m, k // 32, dtype=torch.uint8, device=dev)
    rc("act_quant", lib.memra_v41_act_quant(p(x2), k, None, p(q), k, p(s), k // 32, None, 0, m, k, S))
    return q, s


def linear_f8(x2, w, ws, fmt=0, out_f32=False, s2=1.0):
    m, k = x2.shape
    n = w.shape[0]
    q, s = actq(x2)
    y = torch.empty(m, n, dtype=torch.float32 if out_f32 else torch.bfloat16, device=dev)
    ldb = k if fmt == 0 else k // 2
    ldbs = ws.shape[1]
    rc("gemm_f8", lib.memra_v41_gemm_f8(p(q), k, p(s), k // 32, p(w), ldb, p(ws), ldbs, s2, fmt, p(y), n, int(out_f32), m, n, k, S))
    return y


if want("actq"):
    t, _ = fx("actq")
    q, s = actq(t["x"])
    print("act_quant codes exact", bool((q == t["q"]).all()), "scales exact", bool((s == t["s"]).all()))
    y = t["x"].clone()
    rc("qdq", lib.memra_v41_act_quant(p(y), y.shape[1], None, None, 0, None, 0, p(y), y.shape[1], y.shape[0], y.shape[1], S))
    show("act_quant qdq", ulp_stats(y, t["qdq"]))

for case in sorted(os.listdir(root)):
    if case.startswith("fp8gemm_") and want(case):
        t, _ = fx(case)
        show(case, ulp_stats(linear_f8(t["x"], t["w"], t["ws"]), t["y"]))

if want("fp4qdq"):
    t, _ = fx("fp4qdq")
    for g, xn, yn in ((32, "x32", "y32"), (16, "x16", "y16")):
        x = t[xn].clone()
        rc("fp4", lib.memra_v41_fp4_qdq(p(x), x.shape[1], x.shape[0], x.shape[1], g, S))
        show(f"fp4_qdq g{g}", ulp_stats(x, t[yn]))

if want("norm_rope"):
    t, meta = fx("norm_rope")
    for d in (5120, 1280, 512, 128):
        x, w = t[f"norm{d}/x"], t[f"norm{d}/w"]
        y = torch.empty_like(x)
        rc("rms", lib.memra_v41_rmsnorm(p(x), d, p(w), p(y), d, x.shape[0], d, 1e-20, S))
        show(f"rmsnorm {d}", ulp_stats(y, t[f"norm{d}/y"]))
    for r in (0, 1):
        x = t[f"rope{r}/x"].clone()
        cs, sn = t[f"freqs{r}/cos"].contiguous(), t[f"freqs{r}/sin"].contiguous()
        _, s_, h, d = x.shape
        rc("rope", lib.memra_v41_rope(p(x), h * d, s_, h, d, p(cs), p(sn), meta["rope_start"], 1, None, 0, S))
        show(f"rope{r} fwd (vendor tables)", ulp_stats(x, t[f"rope{r}/y"]))
        rc("rope", lib.memra_v41_rope(p(x), h * d, s_, h, d, p(cs), p(sn), meta["rope_start"], 1, None, 1, S))
        show(f"rope{r} inv (vendor tables)", ulp_stats(x, t[f"rope{r}/yinv"]))

for case in sorted(os.listdir(root)):
    if case.startswith("sattn_") and want(case):
        t, meta = fx(case)
        q, kv, idx, sink = t["q"], t["kv"], t["idx"], t["sink"]
        _, m, h, d = q.shape
        k = idx.shape[2]
        o = torch.empty_like(q)
        rc("sattn", lib.memra_v41_sparse_attn(p(q), h * d, p(kv), p(idx), k, k, p(sink), meta["scale"], p(o), h * d, m, h, d, S))
        show(case, ulp_stats(o, t["o"]))

if want("hc"):
    t, _ = fx("hc")
    x = t["x"]
    _, s_, hc, d = x.shape
    mixes = torch.empty(s_, 24, device=dev)
    pre = torch.empty(s_, 4, device=dev)
    post = torch.empty(s_, 4, device=dev)
    comb = torch.empty(s_, 16, device=dev)
    rc("hc_mixes", lib.memra_v41_hc_mixes(p(x), p(t["fn"]), p(t["scale"]), p(t["base"]), s_, hc, d, 20, 1e-20, 1e-6, p(mixes), p(pre), p(post), p(comb), S))
    for nm, g_ in (("mixes", mixes), ("pre", pre), ("post", post), ("comb", comb)):
        f32rel(f"hc {nm}", g_.flatten(), t[nm].flatten())
    y = torch.empty(s_, d, dtype=torch.bfloat16, device=dev)
    rc("hc_pre", lib.memra_v41_hc_pre(p(x), p(t["carried"].contiguous()), s_, hc, d, p(y), d, S))
    show("hc_pre", ulp_stats(y, t["y"][0]))
    z = torch.empty_like(x)
    rc("hc_post", lib.memra_v41_hc_post(p(t["sub"]), d, p(x), p(t["post"].contiguous()), p(t["comb"].contiguous()), s_, hc, d, p(z), S))
    show("hc_post", ulp_stats(z, t["z"]))

for case, ratio in (("comp_r1", 1), ("comp_r2", 2)):
    if not want(case):
        continue
    t, meta = fx(case)
    s_ = meta["seqlen"]
    x = t["x"][0, :s_].contiguous()
    n_c = s_ // ratio
    lat = torch.empty(n_c, 512, dtype=torch.bfloat16, device=dev)
    if ratio == 1:
        rc("gemm", lib.memra_v41_gemm_bf16(p(x), 5120, 0, p(t["wkv"]), 5120, 0, p(lat), 512, 0, 0, s_, 512, 5120, 1, S))
    else:
        kvf = torch.empty(s_, 512, device=dev)
        scf = torch.empty(s_, 512, device=dev)
        rc("gemm", lib.memra_v41_gemm_bf16(p(x), 5120, 0, p(t["wkv"]), 5120, 0, p(kvf), 512, 0, 1, s_, 512, 5120, 1, S))
        rc("gemm", lib.memra_v41_gemm_bf16(p(x), 5120, 0, p(t["wgate"]), 5120, 0, p(scf), 512, 0, 1, s_, 512, 5120, 1, S))
        rc("pool", lib.memra_v41_comp_pool2(p(kvf), p(scf), 512, p(lat), 512, n_c, 512, S))
    rc("norm", lib.memra_v41_rmsnorm(p(lat), 512, p(t["norm"]), p(lat), 512, n_c, 512, 1e-20, S))
    show(f"{case} latent", ulp_stats(lat, t["prefill"][0]))

if want("indexer"):
    t, meta = fx("indexer")
    s_ = meta["seqlen"]
    cb, cbk, topk = meta["candidate_block_size"], meta["candidate_topk_blocks"], meta["index_topk"]
    for step, sp, ln in ((0, 0, s_), (1, s_, 1), (2, s_ + 1, 1)):
        n_c = sp + ln
        for lid in (20, 24):
            pf = f"s{step}/L{lid}"
            q, k, w = t[f"{pf}/q"].contiguous(), t[f"s{step}/index_k"][0].contiguous(), t[f"{pf}/w"].contiguous()
            sc = torch.empty(ln, n_c, dtype=torch.bfloat16, device=dev)
            rc("score", lib.memra_v41_index_score(p(q), 4096, p(k), n_c, p(w), 32, 1.0, sp, 1, None, 0, cb, p(sc), n_c, ln, S))
            ref = t[f"{pf}/score_raw"][0].clone()
            if sp == 0:
                ref = ref.masked_fill(torch.arange(n_c, device=dev) >= torch.arange(1, ln + 1, device=dev).unsqueeze(1), float("-inf"))
            show(f"index score {pf}", ulp_stats(sc, ref))
            score = ref
            if lid == 24:
                score = score.masked_fill(~t[f"s{step}/cand"][0].bool(), float("-inf"))
            else:
                nb = (n_c + cb - 1) // cb
                keep = torch.zeros(ln, nb, dtype=torch.uint8, device=dev)
                rc("cand", lib.memra_v41_candidates(p(score.contiguous()), n_c, n_c, cb, cbk, sp, 1, p(keep), nb, ln, S))
                wantb = t[f"s{step}/cand"][0][:, ::cb].to(torch.uint8)
                print(f"candidates {pf}: equal rows {int((keep == wantb).all(1).sum())}/{ln}", flush=True)
            tw = min(topk, n_c)
            off = ln if sp == 0 else 128
            out = torch.empty(ln, tw, dtype=torch.int32, device=dev)
            sc2 = score.contiguous()
            rc("topk", lib.memra_v41_topk(p(sc2), n_c, n_c, tw, tw, off, p(out), tw, ln, S))
            wi = t[f"{pf}/idx"][0]
            same = sum(int(torch.equal(torch.sort(out[r][out[r] >= 0]).values, torch.sort(wi[r][wi[r] >= 0]).values)) for r in range(ln))
            print(f"topk {pf}: identical rows {same}/{ln}", flush=True)

if want("moe"):
    t, meta = fx("moe")
    x = t["x"][0]
    s_ = x.shape[0]
    ne, tk = meta["n_experts"], meta["topk"]
    raw = torch.empty(s_, ne, device=dev)
    rc("gemm", lib.memra_v41_gemm_bf16(p(x), 5120, 0, p(t["gate_w"]), 5120, 0, p(raw), ne, 0, 1, s_, ne, 5120, 1, S))
    sel = torch.empty(s_, tk, dtype=torch.int32, device=dev)
    selw = torch.empty(s_, tk, device=dev)
    rc("route", lib.memra_v41_route(p(raw), ne, p(t["gate_b"]), ne, tk, 1, meta["route_scale"], p(sel), p(selw), s_, S))
    print("route ids identical", bool((sel == t["route_i"]).all()), flush=True)
    f32rel("route weights", selw, t["route_w"])

    def expert(tag, xin, wrow):
        g_ = linear_f8(xin, t[f"{tag}/w1"], t[f"{tag}/w1_s"], 0 if t[f"{tag}/w1"].dtype == torch.uint8 else 1)
        u_ = linear_f8(xin, t[f"{tag}/w3"], t[f"{tag}/w3_s"], 0 if t[f"{tag}/w3"].dtype == torch.uint8 else 1)
        h_ = torch.empty_like(g_)
        rc("swiglu", lib.memra_v41_swiglu(p(g_), p(u_), 2304, p(wrow), 10.0, p(h_), 2304, xin.shape[0], 2304, S))
        return linear_f8(h_, t[f"{tag}/w2"], t[f"{tag}/w2_s"], 0 if t[f"{tag}/w2"].dtype == torch.uint8 else 1)

    show("expert0 mxfp4 clamped", ulp_stats(expert("e0", t["solo/x"], t["solo/wt"].contiguous()), t["solo/y"]))
    show("shared fp8 clamped", ulp_stats(expert("shared", t["solo/x"], None), t["solo_shared/y"]))

if want("engram"):
    t, _ = fx("engram")
    h = t["h"].clone()
    _, s_, hc, d = h.shape
    hid = t["hash"][0].long()
    rows = t["table"][hid.flatten()].contiguous()
    scs = t["table_s"][hid.flatten()].contiguous()
    emb = torch.empty(s_ * 24, 256, dtype=torch.bfloat16, device=dev)
    rc("deq", lib.memra_v41_engram_dequant(p(rows), p(scs), s_ * 24, 256, p(emb), S))
    show("engram dequant", ulp_stats(emb.view(s_, -1), t["emb"][0].reshape(s_, -1)))
    kv = linear_f8(emb.view(s_, -1), t["wkv"], t["wkv_s"])
    show("engram wkv", ulp_stats(kv, t["kv"][0]))
    rc("gate", lib.memra_v41_engram_gate(p(h), p(t["kv"].contiguous()), p(t["qw"]), p(t["kw"]), s_, hc, d, 1e-20, 5120 ** -0.5, p(h), S))
    show("engram gate", ulp_stats(h, t["y"]))
print("DEV-DONE", flush=True)


def sattn_emul(q, kv, sink, idx, scale, block=64, bf16_p=True, online=True):
    """the tilelang schedule in torch: key blocks of 64 in index order, online max, bf16 P into PV, fp32 denominator"""
    _, m, h, d = q.shape
    k = idx.shape[2]
    out = torch.empty(m, h, d, device=dev)
    qf = q[0].float()
    for i in range(m):
        mx = torch.full((h,), -1e30, device=dev)
        se = torch.zeros(h, device=dev)
        acc = torch.zeros(h, d, device=dev)
        ids = idx[0, i].long()
        if not online:
            g = kv[0][ids.clamp_min(0)].float()
            sc = (qf[i] @ g.t()) * scale
            sc = sc.masked_fill((ids < 0).unsqueeze(0), float("-inf"))
            mx = torch.maximum(mx, sc.amax(-1))
        for b0 in range(0, k, block):
            bi = ids[b0:b0 + block]
            g = torch.where((bi >= 0).unsqueeze(-1), kv[0][bi.clamp_min(0)].float(), torch.zeros((), device=dev))
            sc = (qf[i] @ g.t()) * scale
            sc = sc.masked_fill((bi < 0).unsqueeze(0), float("-inf"))
            mnew = torch.maximum(mx, sc.amax(-1)) if online else mx
            scp = torch.exp(mx - mnew)
            pexp = torch.exp(sc - mnew.unsqueeze(-1))
            se = se * scp + pexp.sum(-1)
            pp = pexp.to(torch.bfloat16).float() if bf16_p else pexp
            acc = acc * scp.unsqueeze(-1) + pp @ g
            mx = mnew
        se = se + torch.exp(sink - mx)
        out[i] = acc / se.unsqueeze(-1)
    return out.to(torch.bfloat16).unsqueeze(0)


if want("sattn_check"):
    for case in sorted(os.listdir(root)):
        if not case.startswith("sattn_"):
            continue
        t, meta = fx(case)
        q, kv, idx, sink = t["q"], t["kv"], t["idx"], t["sink"]
        _, m, h, d = q.shape
        k = idx.shape[2]
        o = torch.empty_like(q)
        rc("sattn", lib.memra_v41_sparse_attn(p(q), h * d, p(kv), p(idx), k, k, p(sink), meta["scale"], p(o), h * d, m, h, d, S))
        emu = sattn_emul(q, kv, sink, idx, meta["scale"])
        exact = sattn_emul(q, kv, sink, idx, meta["scale"], bf16_p=False)
        glob = sattn_emul(q, kv, sink, idx, meta["scale"], online=False)
        show(f"{case} kernel vs tilelang-schedule emul", ulp_stats(o, emu))
        show(f"{case} kernel vs fp32-P exact", ulp_stats(o, exact))
        show(f"{case} oracle(global max) vs fp32-P exact", ulp_stats(t["o"], exact))
        show(f"{case} oracle vs global-max emul", ulp_stats(t["o"], glob))

if want("topk_check"):
    t, meta = fx("indexer")
    s_ = meta["seqlen"]
    cb, cbk, topk = meta["candidate_block_size"], meta["candidate_topk_blocks"], meta["index_topk"]
    ref = t["s0/L20/score_raw"][0].clone()
    n_c = s_
    ref = ref.masked_fill(torch.arange(n_c, device=dev) >= torch.arange(1, s_ + 1, device=dev).unsqueeze(1), float("-inf"))
    nb = (n_c + cb - 1) // cb
    keep = torch.zeros(s_, nb, dtype=torch.uint8, device=dev)
    rc("cand", lib.memra_v41_candidates(p(ref.contiguous()), n_c, n_c, cb, cbk, 0, 1, p(keep), nb, s_, S))
    wantb = t["s0/cand"][0][:, ::cb].to(torch.uint8)
    for r in range(s_):
        if not torch.equal(keep[r], wantb[r]):
            bm = ref[r].float().view(-1, cb).amax(-1) if n_c % cb == 0 else torch.nn.functional.pad(ref[r].float(), (0, -n_c % cb), value=float("-inf")).view(-1, cb).amax(-1)
            last = r // cb
            bm[last] = float("inf")
            print("cand row", r, "kernel", keep[r].nonzero().flatten().tolist(), "vendor", wantb[r].nonzero().flatten().tolist(),
                  "block maxima", [round(v, 5) for v in bm.tolist()])
    tw = min(topk, n_c)
    out = torch.empty(s_, tw, dtype=torch.int32, device=dev)
    rc("topk", lib.memra_v41_topk(p(ref.contiguous()), n_c, n_c, tw, tw, s_, p(out), tw, s_, S))
    wi = t["s0/L20/idx"][0]
    for r in range(s_):
        a = torch.sort(out[r][out[r] >= 0]).values - s_
        b = torch.sort(wi[r][wi[r] >= 0]).values - s_
        if not torch.equal(a, b):
            sa = set(a.tolist()); sb = set(b.tolist())
            print("topk row", r, "only kernel", {i: ref[r, i].item() for i in sa - sb}, "only vendor", {i: ref[r, i].item() for i in sb - sa},
                  "kth", torch.sort(ref[r].float(), descending=True).values[tw - 1].item())
print("CHECKS-DONE", flush=True)
