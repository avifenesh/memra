#!/usr/bin/env python3
"""Lane S: NVFP4 per component of the DSpark drafter, measured by acceptance and drafter-logit KL.

The drafter runs on CPU (accept_eval.Eval, f32 without rounding points: the GPU drafter's
class) on the route's own taps and greedy sequences (ink2-spec --per-position dumps). A
component's weights are replaced by their NVFP4 values from the class's quantizer (lane F's
klh.nvfp4_weight_quant, which lane M's ink2-mint ports byte for byte): e2m1 codes, an e4m3 scale
per 16 inputs, one f32 tensor scale; everything else stays BF16.

  base  CKPT PP_DIR OUT      BF16 drafter: proposals per position, logits along its own chain (f16)
  comp  CKPT PP_DIR OUT C... per component C: proposals (own chain: acceptance) and the logits along
                             the BF16 chain (forced), with KL(BF16 || C), top-1 agreement
Components: L{0,1,2}.{q,k,v,o,kctx,vctx,gate,up,down}, w1, rnn, hout, fold (the derived
head @ hidden_out), head (the shared output projection as nvfp4-rig stores it; the fold follows it).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, os.path.expanduser("~/.cache/inkling2mini/oracle"))
os.environ.setdefault("KLH_DEV", "cpu")
import accept_eval as A  # noqa: E402
import klh  # noqa: E402

LAYER = {"q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
         "o": "self_attn.o_proj.weight", "kctx": "self_attn.k_proj_ctx.weight", "vctx": "self_attn.v_proj_ctx.weight",
         "gate": "mlp.gate_proj.weight", "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight"}
COMPONENTS = [f"L{l}.{c}" for l in range(3) for c in LAYER] + ["w1", "rnn", "hout", "fold", "head"]


def q(w):
    return klh.nvfp4_weight_quant(w.float().contiguous())


def load(ckpt):
    w = A.load_drafter(Path(ckpt) / "mtp.safetensors")
    main = A.St(str(Path(ckpt) / "model-00001-of-00001.safetensors"))
    shared = {"embed": main.t("model.llm.embed.weight"), "embed_norm": main.t("model.llm.embed_norm.weight"),
              "head": main.t("model.llm.unembed.weight")[:200058]}
    return w, shared


def make(w, shared, comp):
    w = dict(w)
    shared = dict(shared)
    if comp is None:
        return A.Eval(w, shared, A.REAL, "f32")
    if comp.startswith("L"):
        l, c = comp[1:].split(".")
        name = f"layers.{l}.{LAYER[c]}"
        w[name] = q(w[name])
    elif comp == "w1":
        w["markov_head.markov_w1.weight"] = q(w["markov_head.markov_w1.weight"])
    elif comp == "rnn":
        w["markov_head.rnn_proj.weight"] = q(w["markov_head.rnn_proj.weight"])
    elif comp == "hout":
        w["markov_head.hidden_out.weight"] = q(w["markov_head.hidden_out.weight"])
    elif comp == "head":
        shared["head"] = q(shared["head"])
    e = A.Eval(w, shared, A.REAL, "f32")
    if comp == "fold":
        e.fold_w = q(e.fold_w)
    return e


def prompts(pp):
    for jf in sorted(Path(pp).glob("*.json")):
        if jf.name.endswith(".oracle.json"):
            continue
        m = json.load(open(jf))
        taps = torch.from_numpy(np.fromfile(Path(pp) / f"{m['id']}.taps.f32", dtype=np.float32)
                                .reshape(len(m["seq"]), 4, 2304).copy())
        yield m, taps


def accept(props, seq, P):
    a = 0
    for j, d in enumerate(props):
        if P + 1 + j >= len(seq) or d != seq[P + 1 + j]:
            break
        a += 1
    return a


def run_base(ckpt, pp, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    w, shared = load(ckpt)
    for m, taps in prompts(pp):
        t0 = time.time()
        e = make(w, shared, None)
        T0, seq = m["first_anchor"], m["seq"]
        e.commit(taps[:T0])
        props, logits = [], []
        for P in range(T0, len(seq)):
            toks, lg = e.propose(seq[P], taps[P - 1], keep=True)
            props.append(toks)
            logits.append(lg.to(torch.float16).numpy())
            e.commit(taps[P:P + 1])
        np.save(out / f"{m['id']}.base_logits.npy", np.stack(logits))
        json.dump({"id": m["id"], "kind": m["kind"], "proposals": props}, open(out / f"{m['id']}.base.json", "w"))
        print(m["id"], f"{time.time() - t0:.0f}s", flush=True)


def kl_rows(base, got):
    pb = torch.log_softmax(base.float(), -1)
    pg = torch.log_softmax(got.float(), -1)
    kl = (pb.exp() * (pb - pg)).sum(-1)
    top = (base.argmax(-1) == got.argmax(-1)).float()
    return kl, top


def run_comp(ckpt, pp, out, comps):
    out = Path(out)
    w, shared = load(ckpt)
    with open(out / "components.jsonl", "a") as fo:
        for comp in comps:
            t0 = time.time()
            e0 = make(w, shared, comp)
            for m, taps in prompts(pp):
                base = json.load(open(out / f"{m['id']}.base.json"))
                blog = torch.from_numpy(np.load(out / f"{m['id']}.base_logits.npy").astype(np.float32))
                e = A.Eval.__new__(A.Eval)
                e.__dict__.update(e0.__dict__)
                e.ctx_k = [[] for _ in range(A.REAL["layers"])]
                e.ctx_v = [[] for _ in range(A.REAL["layers"])]
                T0, seq = m["first_anchor"], m["seq"]
                e.commit(taps[:T0])
                acc, kls, tops = [], [], []
                for i, P in enumerate(range(T0, len(seq))):
                    toks, _ = e.propose(seq[P], taps[P - 1])
                    acc.append(accept(toks, seq, P))
                    _, lg = e.propose(seq[P], taps[P - 1], forced=base["proposals"][i], keep=True)
                    kl, top = kl_rows(blog[i], lg)
                    kls.append(kl.mean().item())
                    tops.append(top.mean().item())
                    e.commit(taps[P:P + 1])
                base_acc = [accept(p, seq, T0 + i) for i, p in enumerate(base["proposals"])]
                fo.write(json.dumps({"comp": comp, "id": m["id"], "kind": m["kind"], "first_anchor": T0,
                                     "accept": acc, "base_accept": base_acc, "kl": kls, "top1": tops,
                                     "proposals_equal": None}) + "\n")
                fo.flush()
            print(comp, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("THREADS", "16")))
    cmd = sys.argv[1]
    if cmd == "base":
        run_base(*sys.argv[2:5])
    elif cmd == "comp":
        run_comp(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:] or COMPONENTS)
    else:
        raise SystemExit(__doc__)
