"""DeepSeek-V4.1-Flash-1017 correctness oracle: the vendor's reference model.py, unchanged, on CPU with torch stand-ins
for its tilelang kernels (oracle/kernel.py) and the checkpoint memory-mapped (no conversion, nothing resident until read).
Primitives are correctly rounded (oracle/exact.py) and top-k follows the CUDA tie rule (lowest index wins among equal
scores at the boundary), the semantics memra's reference executor uses, so the two compare bit for bit; the manifest
records it as primitives=correctly-rounded, the numeric class oracle_to_checkpoint_tape.py writes by default.

usage: run_oracle.py CKPT_DIR REF_DIR CASES_JSON OUT_DIR [--max-new 32] [--max-seq 8192] [--full-logits 8] [--hidden]
CASES_JSON: [{"id": ..., "messages": [...]} | {"id": ..., "prompt_ids": [...]}]. Writes OUT_DIR/<id>.pt with the prompt
ids, greedy completion, per-step top-32 logits, the first --full-logits steps' full fp32 logits and (with --hidden) the
residual stream after every backbone layer at the prefill, plus OUT_DIR/manifest.json with the load census.

--device cuda keeps every weight but the routed experts and the Engram tables on one GPU; experts are copied in when the
router picks them (LRU, --expert-cache-gb) and the Engram gather stays on the host. --index-qchunk N runs the
prefill indexer N query rows at a time (long_ctx.py: the vendor ops per row slice; the one-shot score matrices do not
fit at long context). fp32 GEMMs run without TF32 and bf16 GEMMs without reduced-precision reduction."""
import argparse, json, mmap, os, struct, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("ref"); ap.add_argument("cases"); ap.add_argument("out")
ap.add_argument("--max-new", type=int, default=32); ap.add_argument("--max-seq", type=int, default=8192)
ap.add_argument("--full-logits", type=int, default=8); ap.add_argument("--hidden", action="store_true")
ap.add_argument("--threads", type=int, default=16); ap.add_argument("--only", default="")
ap.add_argument("--tf-len", type=int, default=2048)  # teacher-forced cases ({"id", "text"}): tokens scored per text
ap.add_argument("--device", default="cpu"); ap.add_argument("--expert-cache-gb", type=float, default=24)
ap.add_argument("--index-qchunk", type=int, default=0); ap.add_argument("--hidden-max-len", type=int, default=1 << 30)
a = ap.parse_args()
sys.path[:0] = [HERE, os.path.join(a.ref, "inference"), os.path.join(a.ref, "encoding")]

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_num_threads(a.threads)
torch.set_default_dtype(torch.bfloat16)
import exact  # noqa: E402
import kernel  # noqa: E402,F401  (oracle stand-ins; must shadow the vendor kernel module)
assert os.path.dirname(os.path.abspath(kernel.__file__)) == HERE
exact.install()


def cuda_rule_topk(self, k, dim=-1, largest=True, sorted=True):
    """torch.topk with the CUDA tie rule: the k largest, equal values taken in index order."""
    assert dim in (-1, self.dim() - 1) and largest
    indices = torch.sort(self, dim=-1, descending=True, stable=True).indices[..., :k]
    if not sorted:
        indices = indices.sort(dim=-1).values
    return torch.return_types.topk((self.gather(-1, indices), indices))


torch.Tensor.topk = cuda_rule_topk
import model as M  # noqa: E402
from encoding import encode_messages  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

DT = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu, "BF16": torch.bfloat16, "F32": torch.float32,
      "I8": torch.int8, "I32": torch.int32, "I64": torch.int64, "F16": torch.float16, "U8": torch.uint8}


def mmap_safetensors(path):
    """name -> zero-copy CPU tensor over a read-only mmap of the shard (pages fault in on use, evictable)."""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    base = 8 + n
    out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        s, e = v["data_offsets"]
        buf = memoryview(mm)[base + s: base + e]
        t = torch.frombuffer(buf, dtype=torch.uint8) if e > s else torch.empty(0, dtype=torch.uint8)
        out[k] = t.view(DT[v["dtype"]]).view(v["shape"]) if e > s else torch.empty(v["shape"], dtype=DT[v["dtype"]])
    return out, (f, mm)


def engram_embed_forward(self, indices):
    """ParallelEngramEmbedding.forward without F.embedding on fp8 (no CPU kernel): gather raw bytes, then the same
    dequant arithmetic as the vendor code. The tables stay on the host; the result moves to the indices' device."""
    dev = indices.device
    indices = indices.cpu()
    mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
    li = (indices - self.vocab_start_idx).masked_fill(mask, 0)
    flat = li.reshape(-1)
    values = self.weight.view(torch.uint8).index_select(0, flat).view(torch.float8_e4m3fn).view(*li.shape, -1)
    scales = self.scale.view(torch.uint8).index_select(0, flat).view(torch.float8_e8m0fnu).view(*li.shape, -1)
    values = values.float().unflatten(-1, (-1, self.block_size)) * scales.float().unsqueeze(-1)
    values = values.flatten(-2).to(torch.bfloat16)
    return values.masked_fill(mask.unsqueeze(-1), 0).to(dev)


M.ParallelEngramEmbedding.forward = engram_embed_forward

cfg = json.load(open(os.path.join(a.ref, "inference", "config.json")))
cfg.update(max_batch_size=1, max_seq_len=a.max_seq, temperature=0.0, vision_n_layers=0)
args = M.ModelArgs(**cfg)
tok = AutoTokenizer.from_pretrained(a.ckpt)
t0 = time.time()
with torch.device("cpu"):
    model = M.Transformer(args, tok)
print(f"built in {time.time() - t0:.0f}s", flush=True)

# ---- weights: assign mmap tensors in place of the uninitialized parameters (identity kept, so weight.scale holds)
idx = json.load(open(os.path.join(a.ckpt, "model.safetensors.index.json")))["weight_map"]
ck, keep = {}, []
for shard in sorted(set(idx.values())):
    d, h = mmap_safetensors(os.path.join(a.ckpt, shard)); ck.update(d); keep.append(h)
params = dict(model.named_parameters(remove_duplicate=False))
used, converted, report = set(), [], {}
for name, p in params.items():
    if name in used:
        continue
    src = name
    if name.startswith("mtp.") and name.split(".", 2)[-1] in ("embed.weight", "head.weight"):
        continue  # tied to the backbone (same Parameter object, already assigned)
    if src not in ck:
        report.setdefault("missing", []).append(name); continue
    t = ck[src]
    used.add(src)
    if name.endswith("attn.wo_a.weight") and p.dtype == torch.bfloat16 and t.dtype == torch.float8_e4m3fn:
        s = ck[src.replace("weight", "scale")]; used.add(src.replace("weight", "scale"))
        bo, bi = t.size(0) // s.size(0), t.size(1) // s.size(1)
        t = (t.float().unflatten(0, (-1, bo)).unflatten(-1, (-1, bi)) * s.float()[:, None, :, None]).flatten(2, 3).flatten(0, 1)
        converted.append(name)
    if p.dtype == torch.float4_e2m1fn_x2 and t.dtype == torch.int8:
        t = t.view(torch.float4_e2m1fn_x2)
    elif p.dtype != t.dtype:
        assert t.numel() * t.element_size() < 4 << 30, (name, p.dtype, t.dtype)
        t = t.to(p.dtype); converted.append(name)
    assert tuple(t.shape) == tuple(p.shape), (name, tuple(t.shape), tuple(p.shape))
    p.data = t
ignored_prefix = ("vision.", "aligner.", "image_")
unused = sorted(k for k in ck if k not in used and not k.startswith(ignored_prefix) and not k.endswith(".bias_vl"))
report.update(n_params=len(params), n_ckpt=len(ck), unused=unused, converted=len(converted),
              primitives="correctly-rounded", topk="cuda-tie-rule")
os.makedirs(a.out, exist_ok=True)
json.dump(report, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
print("load report:", {k: (v if not isinstance(v, list) else len(v)) for k, v in report.items()}, flush=True)
assert not report.get("missing") and not unused, "checkpoint and model disagree (see manifest.json)"
model.eval()

if a.device != "cpu":
    import collections, inspect, textwrap
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    def to_dev(t):
        if t.dtype == torch.float4_e2m1fn_x2:
            return t.view(torch.uint8).to(dev).view(torch.float4_e2m1fn_x2)
        return t.to(dev)

    streamed, experts = set(), []
    for mod in model.modules():
        if isinstance(mod, M.MoE):
            for e in mod.experts:
                if e is not None:
                    experts.append(e); streamed.update(id(p) for p in e.parameters())
        if isinstance(mod, M.ParallelEngramEmbedding):
            streamed.update(id(p) for p in mod.parameters())
    resident = 0
    for p in model.parameters():
        if id(p) not in streamed:
            p.data = to_dev(p.data); resident += p.numel() * p.element_size()
    for mod in model.modules():
        for k, b in list(mod._buffers.items()):
            if b is not None:
                mod._buffers[k] = to_dev(b)
    cache, held = collections.OrderedDict(), [0]
    budget = int(a.expert_cache_gb * (1 << 30))

    def expert_in(mod, inp):
        key = id(mod)
        if key in cache:
            cache.move_to_end(key); return
        cpu = [(p, p.data) for p in mod.parameters()]
        for p, t in cpu:
            p.data = to_dev(t)
        nb = sum(t.numel() * t.element_size() for _, t in cpu)
        cache[key] = (cpu, nb); held[0] += nb
        while held[0] > budget and len(cache) > 1:
            _, (cp, n) = cache.popitem(last=False)
            for p, t in cp:
                p.data = t
            held[0] -= n

    for e in experts:
        e.register_forward_pre_hook(expert_in)
    torch.set_default_device(dev)
    if a.index_qchunk:
        import long_ctx
        long_ctx.install(M, a.index_qchunk)
    print(f"device {dev}: resident {resident / 2**30:.1f} GiB, {len(experts)} routed experts streamed "
          f"(cache {a.expert_cache_gb} GiB), index_qchunk {a.index_qchunk}", flush=True)

hidden = []
if a.hidden:
    for li, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda m, i, o, li=li: hidden.append(o[0].detach().to("cpu", copy=True))
                                    if i[1] == 0 and o[0].size(1) <= a.hidden_max_len else None)

TF = {"targets": None, "out": None}  # teacher-forced scoring inside the head, 4096 positions at a time
_head_forward = M.ParallelHead.forward


def _head(self, x, full_logits=False):
    """With TF["targets"] set, reduce each position's fp32 logits (vendor F.linear on the head weight) to its
    logsumexp, next-token NLL and top-32 slice by slice, and hand back only the last position's logits."""
    tgt = TF["targets"]
    if tgt is None or full_logits:
        return _head_forward(self, x, full_logits)
    lse, nll, tv, ti = [], [], [], []
    for a0 in range(0, x.size(1), 4096):
        lg = _head_forward(self, x[:, a0:a0 + 4096], True)[0]
        ls = torch.logsumexp(lg, -1)
        t = tgt[a0:a0 + lg.size(0)]
        nll.append(ls[: t.numel()] - lg[: t.numel()].gather(-1, t.unsqueeze(-1)).squeeze(-1))
        v, i = lg.topk(32, dim=-1)
        lse.append(ls); tv.append(v); ti.append(i)
    TF["out"] = {k: torch.cat(v).cpu() for k, v in (("lse", lse), ("nll", nll), ("top_vals", tv), ("top_ids", ti))}
    return _head_forward(self, x[:, -1:], False)


M.ParallelHead.forward = _head
cases = json.load(open(a.cases))
for c in cases:
    if a.only and c["id"] not in a.only.split(","):
        continue
    if "text" in c:  # teacher-forced: one prefill with logits at every position; next-token NLL, top-32, logsumexp
        ids = tok.encode(c["text"])[: a.tf_len]
        hidden.clear()
        t1 = time.time()
        TF["targets"] = torch.tensor(ids[1:], device=a.device)
        with torch.inference_mode():
            model(torch.tensor([ids], device=a.device), 0)
        TF["targets"] = None
        r = TF["out"]
        rec = {"id": c["id"], "prompt_ids": ids, **r, "hidden": torch.stack([h[0] for h in hidden]) if hidden else None,
               "prefill_s": time.time() - t1}
        torch.save(rec, os.path.join(a.out, f"{c['id']}.pt"))
        print(json.dumps({"id": c["id"], "tokens": len(ids), "mean_nll": round(float(r["nll"].mean()), 4), "prefill_s": round(rec["prefill_s"], 1)}), flush=True)
        continue
    ids = c.get("prompt_ids") or tok.encode(encode_messages(c["messages"], thinking_mode=c.get("thinking_mode", "chat")))
    hidden.clear()
    t1 = time.time()
    gen, tops, full = [], [], []
    with torch.inference_mode():
        x = torch.tensor([ids], device=a.device)
        nxt, logits, _ = model(x, 0)
        t_pre = time.time() - t1
        pos = len(ids)
        for step in range(a.max_new):
            lg = logits[0].float().cpu()
            v, i = lg.topk(32)
            tops.append((v.clone(), i.clone()))
            if step < a.full_logits:
                full.append(lg.clone())
            t_id = int(lg.argmax())
            gen.append(t_id)
            if t_id == tok.eos_token_id or pos + 1 >= a.max_seq:
                break
            nxt, logits, _ = model(torch.tensor([[t_id]], device=a.device), pos)
            pos += 1
    dt = time.time() - t1
    rec = {"id": c["id"], "prompt_ids": ids, "gen_ids": gen, "text": tok.decode(gen), "top_vals": torch.stack([t[0] for t in tops]),
           "top_ids": torch.stack([t[1] for t in tops]), "full_logits": torch.stack(full) if full else None,
           "hidden": torch.stack([h[0] for h in hidden]) if hidden else None, "prefill_s": t_pre, "total_s": dt}
    torch.save(rec, os.path.join(a.out, f"{c['id']}.pt"))
    print(json.dumps({"id": c["id"], "prompt_len": len(ids), "gen": len(gen), "prefill_s": round(t_pre, 1), "total_s": round(dt, 1),
                      "text": rec["text"][:200]}), flush=True)
print("ORACLE-DONE", flush=True)
