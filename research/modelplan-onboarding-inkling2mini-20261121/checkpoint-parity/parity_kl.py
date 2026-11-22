#!/usr/bin/env python3
"""Inkling 2 Mini, declared class: the checkpoint-parity oracle without an inference engine.

Two backends of the declared program's source simulation exist: lane E's CPU oracle
(reference/ink2_oracle.py, arm actq) and lane F's GPU harness (klsens klh.py, arm actq, the
transformers PR #51050 model with the act-quant experts). This script drives the second one.

  --probe-oracle IDS OUT_DIR   last-token logits of the actq arm and of its twin with the expert
                               rows summed in descending id (actq_rev) as memra-checkpoint-oracle-v1
                               tapes; the spread between backends on the probe sets the band.
  --set NAME CORPUS ROUTE_DIR  per token KL(actq || route) against KL(actq || actq_rev), the
                               harness's own self-noise, on every document of CORPUS
                               ([{"name", "ids"}]); ROUTE_DIR holds the route's head-normalized
                               rows (ink2-gpu --dump-hidden). Both sides go through the harness's
                               head, so the comparison is of the trunk and experts; the route's
                               own head is held by the probe band.

Needs lane F's harness (KLSENS, default ~/.cache/inkling2mini/klsens) and its venv, on a GPU lease.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

KLSENS = os.path.expanduser(os.environ.get("KLSENS", "~/.cache/inkling2mini/klsens"))
sys.path.insert(0, KLSENS)
if os.environ.get("KLH_DEV", "cuda") == "cpu":
    # the harness's CUDA bookkeeping calls, as no-ops for a CPU run of the same program
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.memory_allocated = lambda *a, **k: 0
import klh  # noqa: E402
import klrun  # noqa: E402

CLASS = "inkling2-declared-activation-quant"
CLASS_NOW = CLASS
CLASSES = {"shipped": ("inkling2-declared-activation-quant", None),
           # lane M's nvfp4-rig in lane F's arm vocabulary (research/.../nvfp4-rig/lanef-jobs.json)
           "nvfp4-rig": ("inkling2-nvfp4-rig", "actq:all+kv:all+mx2nv:mx+w:unembed+w:attn:all")}


SPEC = None  # a lane F arm spec (klrun.Arms) for a class other than the shipped one


def set_arm(H, mode, rev):
    if SPEC is None:
        H.set_expert_mode(mode)
    for layer in H.layers:
        layer.mlp.experts.rev = rev


def hidden(H, ids, mode, rev):
    set_arm(H, mode, rev)
    try:
        return H.hidden_safe(torch.tensor(ids, dtype=torch.long)[None])[0]
    finally:
        set_arm(H, "exact", False)


def write_oracle(path, engine, ids, logits):
    bits = np.ascontiguousarray(logits, dtype="<f4").view("<u4")
    with open(path, "w") as f:
        f.write(f"format\tmemra-checkpoint-oracle-v1\nengine\t{engine}\nnumeric_class\t{CLASS_NOW}\n")
        f.write("tokens\t" + ",".join(str(i) for i in ids) + "\n")
        f.write(f"vocab\t{len(bits)}\n")
        for i, b in enumerate(bits):
            f.write(f"logit\t{i}\t{int(b):08x}\n")


def dist(x):
    x = np.asarray(x, dtype=np.float64)
    return {"n": int(x.size), "mean": float(x.mean()), "p50": float(np.percentile(x, 50)), "p90": float(np.percentile(x, 90)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max())}


def boot_ci(per_doc_a, per_doc_b, n=4000, seed=0):
    """95% interval of mean(a) - mean(b) over documents resampled with replacement (paired)."""
    a, b = np.asarray(per_doc_a), np.asarray(per_doc_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-oracle", nargs=2, metavar=("IDS", "OUT_DIR"))
    ap.add_argument("--set", nargs=3, action="append", default=[], metavar=("NAME", "CORPUS", "ROUTE_DIR"))
    ap.add_argument("--tape-cross", nargs=3, action="append", default=[], metavar=("TAPE_DIR", "ROUTE_DIR", "DOC_INDEX"),
                    help="lane E's CPU tape of the same program against this harness and the route at the tape's kept positions")
    ap.add_argument("--save-hidden", help="write this harness's actq (and actq_rev) rows of every document under DIR/<set>/")
    ap.add_argument("--load-hidden", help="read them back instead of running the arms")
    ap.add_argument("--peer", help="another backend's saved actq rows (DIR/<set>/<i>.actq.f32): KL(this || peer), KL(peer || route)")
    ap.add_argument("--no-rev", action="store_true", help="skip the actq_rev arm")
    ap.add_argument("--save-only", action="store_true", help="only run and save the arms")
    ap.add_argument("--class", dest="cls", default="shipped", choices=sorted(CLASSES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    H = klh.Harness()
    global SPEC, CLASS_NOW
    CLASS_NOW, SPEC = CLASSES[a.cls]
    if SPEC:
        klrun.Arms(H).apply(SPEC)
        # both sides of every comparison are this class: its head is the minted unembed
        W = H.unembed_override.to(klh.DEV)
        H.unembed_override = W
        H.head_weight = lambda ref=False: W
    vocab = json.load(open(os.path.join(klh.CKPT, "config.json"))).get("unpadded_vocab_size")
    res = {"harness": "lane F klh.py (transformers PR #51050 text model, act-quant experts)", "class": CLASS_NOW, "arm_spec": SPEC,
           "load_s": time.time() - t0, "sets": {}}
    if a.probe_oracle:
        ids = [int(t) for t in a.probe_oracle[0].split(",")]
        os.makedirs(a.probe_oracle[1], exist_ok=True)
        for name, rev in (("actq", False), ("actq_rev", True)):
            h = hidden(H, ids, "actq", rev)
            lg = H.logits(h[-1:]).float().cpu().numpy()[0]
            if vocab:
                lg = lg[:vocab]
            write_oracle(os.path.join(a.probe_oracle[1], f"klh-{name}.tsv"), f"lane-f-klh-gpu-{name}", ids, lg)
            res.setdefault("probe", {})[name] = {"argmax": int(lg.argmax()), "vocab": int(lg.size)}
    for tape_dir, rdir, di in a.tape_cross:
        from safetensors.numpy import load_file
        tok = load_file(os.path.join(tape_dir, "tokens.safetensors"))
        lp = load_file(os.path.join(tape_dir, "logits_post.safetensors"))
        ids = [int(t) for t in tok["all_ids"]]
        pos = lp["positions"].astype(np.int64)
        meta = json.load(open(os.path.join(rdir, "meta.json")))
        T, hid = len(ids), meta["hidden"]
        hn = torch.from_numpy(np.fromfile(os.path.join(rdir, f"{di}.f32"), dtype="<f4").reshape(T, hid))[pos].to(klh.DEV)
        sel = torch.from_numpy(pos).to(klh.DEV)
        ha = hidden(H, ids, "actq", False)[sel]
        hr = hidden(H, ids, "actq", True)[sel]
        lE = torch.from_numpy(lp["logits_post"].astype(np.float64)).log_softmax(-1)
        def lsm(h):
            lg = H.logits(h).double()
            if vocab:
                lg = lg[:, :vocab]
            return lg.log_softmax(-1).cpu()
        L = {"lane_e_cpu": lE, "klh_actq": lsm(ha), "klh_actq_rev": lsm(hr), "route": lsm(hn)}
        pairs = {}
        for x, y in (("lane_e_cpu", "klh_actq"), ("lane_e_cpu", "klh_actq_rev"), ("klh_actq", "klh_actq_rev"),
                     ("lane_e_cpu", "route"), ("klh_actq", "route"), ("klh_actq_rev", "route")):
            k = (L[x].exp() * (L[x] - L[y])).sum(-1).numpy()
            t1 = (L[x].argmax(-1) == L[y].argmax(-1)).double().mean().item()
            pairs[f"{x}||{y}"] = dict(dist(k), top1=t1)
        name = os.path.basename(tape_dir.rstrip("/"))
        res.setdefault("tape_cross", {})[name] = {"positions": int(pos.size), "pairs": pairs}
        print("tape_cross", name, json.dumps({k: round(v["mean"], 6) for k, v in pairs.items()}), flush=True)
    keys = ("route", "self", "route_rev", "peer", "peer_route")
    pooled = {k: [] for k in keys + ("top1_route", "top1_self", "top1_peer")}
    for name, corpus, rdir in a.set:
        docs = json.load(open(corpus))
        mp = os.path.join(rdir, "meta.json")
        hid = json.load(open(mp))["hidden"] if os.path.exists(mp) else int(H.cfg.hidden_size)
        per = {k: [] for k in pooled}
        doc_rows = []
        for i, d in enumerate(docs):
            ids = d["ids"]
            T = len(ids)

            def saved(base, arm):
                p = os.path.join(base, name, f"{i}.{arm}.f32")
                return torch.from_numpy(np.fromfile(p, dtype="<f4").reshape(T, hid)).to(klh.DEV) if os.path.exists(p) else None

            ha = saved(a.load_hidden, "actq") if a.load_hidden else None
            if ha is None:
                ha = hidden(H, ids, "actq", False)
            hr = None if a.no_rev else (saved(a.load_hidden, "rev") if a.load_hidden else None)
            if hr is None and not a.no_rev:
                hr = hidden(H, ids, "actq", True)
            if a.save_hidden:
                os.makedirs(os.path.join(a.save_hidden, name), exist_ok=True)
                ha.float().cpu().numpy().astype("<f4").tofile(os.path.join(a.save_hidden, name, f"{i}.actq.f32"))
                if hr is not None:
                    hr.float().cpu().numpy().astype("<f4").tofile(os.path.join(a.save_hidden, name, f"{i}.rev.f32"))
            if a.save_only:
                print(name, "saved", i, T, flush=True)
                continue
            hn = torch.from_numpy(np.fromfile(os.path.join(rdir, f"{i}.f32"), dtype="<f4").reshape(T, hid)).to(klh.DEV)
            labels = torch.tensor(ids[1:] + [-1], device=klh.DEV)
            row = {"name": d.get("name", str(i)), "tokens": T}
            s_route = klrun.head_stats(H, hn, ha, labels)
            per["route"].append(s_route["kl"].astype(np.float64))
            per["top1_route"].append(s_route["top1"])
            if hr is not None:
                s_self = klrun.head_stats(H, hr, ha, labels)
                per["self"].append(s_self["kl"].astype(np.float64))
                per["top1_self"].append(s_self["top1"])
                per["route_rev"].append(klrun.head_stats(H, hn, hr, labels)["kl"].astype(np.float64))
                row.update(kl_self_mean=float(per["self"][-1].mean()), kl_route_vs_rev_mean=float(per["route_rev"][-1].mean()))
            hp = saved(a.peer, "actq") if a.peer else None
            if hp is not None:
                s_peer = klrun.head_stats(H, hp, ha, labels)          # KL(this backend || the peer backend)
                per["peer"].append(s_peer["kl"].astype(np.float64))
                per["top1_peer"].append(s_peer["top1"])
                per["peer_route"].append(klrun.head_stats(H, hn, hp, labels)["kl"].astype(np.float64))  # KL(peer || route)
                row.update(kl_peer_mean=float(per["peer"][-1].mean()), kl_peer_route_mean=float(per["peer_route"][-1].mean()))
            row.update(kl_route_mean=float(per["route"][-1].mean()), top1_route=float(s_route["top1"].mean()))
            doc_rows.append(row)
            print(name, json.dumps(row), flush=True)
            del hn, ha, hr, hp
            torch.cuda.empty_cache()
        if a.save_only:
            continue
        out = {"docs": doc_rows}
        for k in keys + ("top1_route", "top1_self", "top1_peer"):
            if per[k]:
                v = np.concatenate(per[k])
                out[("kl_" + k) if k in keys else k] = dist(v) if k in keys else float(v.mean())
        for k in ("self", "peer"):
            col = "kl_%s_mean" % k
            both = [r for r in doc_rows if col in r]
            if both:
                # the documents that have the comparison (a peer may cover only part of a set)
                out["mean_route_minus_%s_ci95_by_doc" % k] = boot_ci([r["kl_route_mean"] for r in both], [r[col] for r in both])
                out["docs_with_%s" % k] = len(both)
        res["sets"][name] = out
        for k in pooled:
            pooled[k] += per[k]
    if a.set and not a.save_only:
        res["pooled"] = {"kl_" + k: dist(np.concatenate(pooled[k])) for k in keys if pooled[k]}
        for k in ("top1_route", "top1_self", "top1_peer"):
            if pooled[k]:
                res["pooled"][k] = float(np.concatenate(pooled[k]).mean())
    res["wall_s"] = time.time() - t0
    json.dump(res, open(a.out, "w"), indent=1)
    print("PARITY_KL_DONE", flush=True)


if __name__ == "__main__":
    main()
