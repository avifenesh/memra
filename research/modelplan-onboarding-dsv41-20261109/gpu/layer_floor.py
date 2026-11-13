"""Single-layer noise floor of the vendor program: rerun chosen trunk layers on the CPU (the oracle's torch stand-ins,
torch CPU GEMMs) from a GPU capture's inputs, and write their outputs in the capture's key layout. Comparing this CPU
output with the captured GPU output is "same program, different backend" for one layer with identical inputs: the band
a native GPU rewrite of that layer is held to.

usage: layer_floor.py ORACLE_DIR CKPT REF CAPTURE_DIR OUT_DIR --layers 1,2,14,20,21,24 [--threads 16]"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("oracle"); ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("capture"); ap.add_argument("out")
ap.add_argument("--layers", required=True); ap.add_argument("--threads", type=int, default=16)
ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--index-qchunk", type=int, default=0, help="the oracle's row-sliced indexer (long prompts)")
ap.add_argument("--attn-schedule", choices=["standin", "online"], default="standin",
                help="online: replace the oracle's global-max sparse_attn stand-in by the tilelang schedule "
                     "(key blocks of 64 in index order, online max, bf16 P)")
c = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import json  # noqa: E402

from fixio import Fix  # noqa: E402

ro = os.path.join(c.oracle, "run_oracle.py")
src = open(ro).read()
cut = src.index("\nhidden = []\n")
sys.argv = [ro, c.ckpt, c.ref, os.devnull, c.out, "--device", "cpu", "--threads", str(c.threads), "--max-seq", str(c.max_seq),
            "--index-qchunk", str(c.index_qchunk)]
G = {"__name__": "__oracle_loader__", "__file__": ro}
exec(compile(src[:cut], ro, "exec"), G)
torch, M, model = G["torch"], G["M"], G["model"]
if c.index_qchunk and "long_ctx" not in G:
    # the oracle installs its row-sliced indexer only for a GPU device; a long CPU floor needs it too
    import long_ctx  # noqa: E402  (the oracle dir is on sys.path through its loader)

    long_ctx.install(M, c.index_qchunk)
    G["long_ctx"] = long_ctx

if c.attn_schedule == "online":
    def sparse_attn_online(q, kv, attn_sink, topk_idxs, softmax_scale, qchunk=256, block=64):
        b, m, h, d = q.shape
        out = torch.empty_like(q)
        k = topk_idxs.size(-1)
        for bi_ in range(b):
            for m0 in range(0, m, qchunk):
                idx_ = topk_idxs[bi_, m0:m0 + qchunk].long()
                qf = q[bi_, m0:m0 + qchunk].float()
                c_ = qf.size(0)
                mx = torch.full((c_, h), -1e30)
                se = torch.zeros(c_, h)
                acc = torch.zeros(c_, h, d)
                for b0 in range(0, k, block):
                    ib = idx_[:, b0:b0 + block]
                    valid = ib >= 0
                    g = kv[bi_][ib.clamp_min(0)].float()
                    g = torch.where(valid.unsqueeze(-1), g, torch.zeros(()))
                    sc = torch.einsum("chd,ckd->chk", qf, g) * softmax_scale
                    sc = sc.masked_fill(~valid.unsqueeze(1), float("-inf"))
                    mnew = torch.maximum(mx, sc.amax(-1))
                    scp = torch.exp(mx - mnew)
                    pexp = torch.exp(sc - mnew.unsqueeze(-1))
                    se = se * scp + pexp.sum(-1)
                    acc = acc * scp.unsqueeze(-1) + torch.einsum("chk,ckd->chd", pexp.to(torch.bfloat16).float(), g)
                    mx = mnew
                se = se + torch.exp(attn_sink.float().view(1, h) - mx)
                out[bi_, m0:m0 + qchunk] = (acc / se.unsqueeze(-1)).to(q.dtype)
        return out

    M.sparse_attn = sparse_attn_online
    print("sparse_attn: tilelang schedule (online max, key blocks of 64)", flush=True)

idx = json.load(open(os.path.join(c.capture, "index.json")))
DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64, "u8": torch.uint8}


def load(k):
    e = idx["tensors"][k]
    t = torch.from_file(os.path.join(c.capture, e["file"]), size=os.path.getsize(os.path.join(c.capture, e["file"])), dtype=torch.uint8)
    return t.view(DT[e["dtype"]]).reshape(e["shape"]).clone()


def has(k):
    return k in idx["tensors"]


fx = Fix(os.path.join(c.out))
sa = M.shared_attn
cur = {"p": None, "post": 0}


def keep(name):
    def f(mod, args, out):
        if cur["p"] is None:
            return
        if isinstance(out, tuple):
            for j, t in enumerate(out):
                fx.put(f"{cur['p']}/{name}/out{j}", t)
        else:
            fx.put(f"{cur['p']}/{name}/out", out)
    return f


_hc_post = M.Block.hc_post


def hc_post(self, x, res, post, comb):
    r = _hc_post(self, x, res, post, comb)
    if cur["p"] is not None:
        fx.put(f"{cur['p']}/hcpost{cur['post']}", r)
        cur["post"] += 1
    return r


M.Block.hc_post = hc_post
for li in [int(v) for v in c.layers.split(",")]:
    p = f"s0/L{li}"
    layer = model.layers[li]
    at = layer.attn
    hooks = [at.register_forward_hook(keep("attn")), layer.ffn.register_forward_hook(keep("moe")),
             layer.ffn.gate.register_forward_hook(keep("gate"))]
    ratio = at.compress_ratio
    # the CSA2 state this layer reads, from the capture, padded into full-size caches
    if ratio and not at.is_kv_source and has(f"{p}/in/compress_kv"):
        ck = load(f"{p}/in/compress_kv")
        buf = torch.zeros(1, c.max_seq // ratio, ck.size(-1), dtype=ck.dtype)
        buf[:, : ck.size(1)] = ck
        sa.compress_kv = buf
    if ratio and has(f"{p}/in/index_k"):
        ik = load(f"{p}/in/index_k")
        buf = torch.zeros(1, c.max_seq // ratio, ik.size(-1), dtype=ik.dtype)
        buf[:, : ik.size(1)] = ik
        sa.index_k = buf
    if ratio and not at.is_index_source and has(f"{p}/in/topk_idxs"):
        sa.topk_idxs = load(f"{p}/in/topk_idxs")
    if ratio and has(f"{p}/in/candidates"):
        sa.candidates = load(f"{p}/in/candidates").bool()
    if ratio and has(f"{p}/in/candidates_blocks"):
        blocks = load(f"{p}/in/candidates_blocks").bool()
        width = (x_len := idx["tensors"][f"{p}/in/x"]["shape"][1]) // ratio
        bs = G["args"].candidate_block_size
        if c.index_qchunk:
            sa.candidates = G["long_ctx"].BlockCandidates(blocks, bs, width)
        else:
            sa.candidates = blocks.repeat_interleave(bs, dim=-1)[..., :width]
    t0 = time.time()
    with torch.inference_mode():
        x = load(f"{p}/in/x")
        if layer.engram is not None and has(f"{p}/engram/x"):
            hx = load(f"{p}/engram/x")
            hid = load(f"{p}/engram/hash")
            eo = layer.engram(hx, hid, None)
            fx.put(f"{p}/engram/out", eo)
            x = eo
        cur["p"], cur["post"] = p, 0
        h, pre = layer(x, 0, load(f"{p}/in/pre_mix"), None)
        cur["p"] = None
    for hk in hooks:
        hk.remove()
    fx.put(f"{p}/out/x", h)
    fx.put(f"{p}/out/pre_mix", pre)
    print(f"layer {li}: {time.time() - t0:.1f}s", flush=True)
fx.meta("layers", c.layers)
fx.meta("attn_schedule", c.attn_schedule)
fx.close()
print("FLOOR-DONE", flush=True)
