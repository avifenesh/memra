"""Per-layer capture of the vendor DSv4.1 program on the real checkpoint, for the memra GPU layer gates.

Loads the model exactly as oracle/run_oracle.py does (its loader is executed from that file, unchanged), then runs the
prefill (and optional decode steps) of tape prompts with hooks that record, for the chosen layers: the block input
(stream, carried pre-mix, the shared CSA2 state the layer reads), the block output (stream, next pre-mix), and with
--ops the per-operation intermediates (mHC mixes, collapsed sublayer inputs, attention input/output, compressor latent,
indexer inputs and selection, sparse-attention operands and output, MoE input/routing/output, Engram input/hashes/output).

usage: capture_layers.py ORACLE_DIR CKPT REF TAPE_DIR OUT_DIR --ids capital,needle --layers 0,1,2 [--ops 1,2]
       [--decode 2] [--device cuda] [--threads 8] [--max-seq 8192]"""
import argparse
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("oracle"); ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("tapes"); ap.add_argument("out")
ap.add_argument("--ids", required=True); ap.add_argument("--layers", required=True); ap.add_argument("--ops", default="")
ap.add_argument("--decode", type=int, default=0); ap.add_argument("--device", default="cuda")
ap.add_argument("--threads", type=int, default=8); ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--expert-cache-gb", type=float, default=24); ap.add_argument("--index-qchunk", type=int, default=0)
c = ap.parse_args()
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fixio import Fix  # noqa: E402

LAYERS = {int(v) for v in c.layers.split(",") if v}
OPS = {int(v) for v in c.ops.split(",") if v}

# ---- the oracle's loader, executed unchanged from its own file (everything before its hidden-state hooks)
ro = os.path.join(c.oracle, "run_oracle.py")
src = open(ro).read()
cut = src.index("\nhidden = []\n")
sys.argv = [ro, c.ckpt, c.ref, os.devnull, c.out, "--device", c.device, "--threads", str(c.threads), "--max-seq",
            str(c.max_seq), "--expert-cache-gb", str(c.expert_cache_gb), "--index-qchunk", str(c.index_qchunk)]
G = {"__name__": "__oracle_loader__", "__file__": ro}
exec(compile(src[:cut], ro, "exec"), G)
torch, M, model = G["torch"], G["M"], G["model"]
dev = G["a"].device

cap = {}
state = {"layer": None, "step": 0, "on": False, "hc": 0, "pre": 0, "post": 0}


def put(k, t):
    if t is None:
        return
    cap[f"s{state['step']}/L{state['layer']}/{k}"] = t.detach().to("cpu", copy=True)


def ops_on():
    return state["on"] and state["layer"] in OPS


def block_pre(mod, args):
    li = mod.layer_id
    state["layer"], state["hc"], state["pre"], state["post"] = li, 0, 0, 0
    state["on"] = li in LAYERS
    if not state["on"]:
        return
    x, start_pos, pre_mix = args[0], args[1], args[2]
    put("in/x", x); put("in/pre_mix", pre_mix)
    ratio = mod.attn.compress_ratio
    sa = M.shared_attn
    if ratio:
        n = (start_pos + x.size(1)) // ratio
        if sa.compress_kv is not None and not mod.attn.is_kv_source:
            put("in/compress_kv", sa.compress_kv[:1, :n])
        if sa.index_k is not None:
            put("in/index_k", sa.index_k[:1, :n])
        if sa.topk_idxs is not None and not mod.attn.is_index_source:
            put("in/topk_idxs", sa.topk_idxs)
        if sa.candidates is not None:
            put("in/candidates", sa.candidates)


def block_post(mod, args, out):
    if state["on"]:
        put("out/x", out[0]); put("out/pre_mix", out[1])
        sa = M.shared_attn
        if mod.attn.compress_ratio:
            n = (args[1] + args[0].size(1)) // mod.attn.compress_ratio
            if mod.attn.is_kv_source:
                put("out/compress_kv", sa.compress_kv[:1, :n])
                put("out/index_k", sa.index_k[:1, :n])
            if mod.attn.is_index_source:
                put("out/topk_idxs", sa.topk_idxs)
            if mod.attn.indexer is not None and mod.attn.indexer.is_candidate_source:
                put("out/candidates", sa.candidates)
    state["on"] = False


_hc_mixes, _hc_pre, _hc_post = M.Block.hc_mixes, M.Block.hc_pre, M.Block.hc_post


def hc_mixes(self, x, fn, scale, base):
    r = _hc_mixes(self, x, fn, scale, base)
    if ops_on():
        k = state["hc"]; state["hc"] += 1
        put(f"hc{k}/pre", r[0]); put(f"hc{k}/post", r[1]); put(f"hc{k}/comb", r[2])
    return r


def hc_pre(self, x, mix):
    r = _hc_pre(self, x, mix)
    if ops_on():
        k = state["pre"]; state["pre"] += 1
        put(f"hcpre{k}", r)
    return r


def hc_post(self, x, res, post, comb):
    r = _hc_post(self, x, res, post, comb)
    if ops_on():
        k = state["post"]; state["post"] += 1
        put(f"sub{k}", x); put(f"hcpost{k}", r)
    return r


M.Block.hc_mixes, M.Block.hc_pre, M.Block.hc_post = hc_mixes, hc_pre, hc_post

_sparse = M.sparse_attn


def sparse_attn(q, kv, sink, idx, scale):
    o = _sparse(q, kv, sink, idx, scale)
    if ops_on():
        put("sattn/q", q); put("sattn/kv", kv); put("sattn/idx", idx); put("sattn/o", o)
    return o


M.sparse_attn = sparse_attn


def hook(name, ins):
    def f(mod, args, out):
        if not ops_on():
            return
        for i, nm in ins:
            if i < len(args) and isinstance(args[i], torch.Tensor):
                put(f"{name}/{nm}", args[i])
        if isinstance(out, tuple):
            for j, t in enumerate(out):
                put(f"{name}/out{j}", t)
        elif isinstance(out, torch.Tensor):
            put(f"{name}/out", out)
    return f


def engram_hook(mod, args, out):
    # Engram runs before its block's forward, so it keys itself by its own layer id
    if mod.layer_id not in OPS:
        return
    keep = state["layer"]
    state["layer"] = mod.layer_id
    put("engram/x", args[0]); put("engram/hash", args[1]); put("engram/out", out)
    state["layer"] = keep


for li, layer in enumerate(model.layers):
    layer.register_forward_pre_hook(block_pre)
    layer.register_forward_hook(block_post)
    if li not in OPS:
        continue
    at = layer.attn
    at.register_forward_hook(hook("attn", [(0, "x")]))
    at.q_norm.register_forward_hook(hook("qr", []))
    at.kv_norm.register_forward_hook(hook("kvn", []))
    if at.compressor is not None:
        at.compressor.register_forward_hook(hook("comp", [(0, "x")]))
    if at.indexer is not None:
        at.indexer.register_forward_hook(hook("indexer", [(0, "x"), (1, "qr"), (2, "latent")]))
    layer.ffn.register_forward_hook(hook("moe", [(0, "x")]))
    layer.ffn.gate.register_forward_hook(hook("gate", [(0, "x")]))
    layer.ffn.shared_experts.register_forward_hook(hook("shared", []))
    if layer.engram is not None:
        layer.engram.register_forward_hook(engram_hook)

for cid in c.ids.split(","):
    tape = torch.load(os.path.join(c.tapes, f"{cid}.pt"), weights_only=False)
    ids = list(tape["prompt_ids"])
    gen = list(tape["gen_ids"])
    cap.clear()
    t0 = time.time()
    with torch.inference_mode():
        state["step"] = 0
        _, logits, _ = model(torch.tensor([ids], device=dev), 0)
        lg = [logits[0].float().cpu()]
        pos = len(ids)
        for st in range(c.decode):
            state["step"] = st + 1
            _, logits, _ = model(torch.tensor([[gen[st]]], device=dev), pos)
            lg.append(logits[0].float().cpu())
            pos += 1
    fx = Fix(os.path.join(c.out, cid))
    for k, v in cap.items():
        fx.put(k, v)
    fx.put("prompt_ids", torch.tensor(ids, dtype=torch.int32))
    fx.put("gen_ids", torch.tensor(gen[: max(c.decode, 1)], dtype=torch.int32))
    fx.put("logits", torch.stack(lg))
    diffs = {}
    if tape.get("hidden") is not None:
        hid = tape["hidden"]
        for li in sorted(LAYERS):
            k = f"s0/L{li}/out/x"
            if k in cap and li < hid.size(0):
                a_, b_ = cap[k][0].float(), hid[li].float()
                diffs[li] = float((a_ - b_).abs().max())
    fx.meta("tape_hidden_maxabs_vs_capture", diffs)
    fx.meta("layers", sorted(LAYERS)); fx.meta("ops", sorted(OPS)); fx.meta("decode", c.decode)
    fx.meta("seconds", round(time.time() - t0, 1)); fx.meta("device", str(dev))
    fx.close()
    print(cid, len(ids), "tokens", f"{time.time() - t0:.1f}s", "tape diffs", diffs, flush=True)
print("CAPTURE-DONE", flush=True)
