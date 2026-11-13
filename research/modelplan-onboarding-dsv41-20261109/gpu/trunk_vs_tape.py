"""Whole-trunk comparison of a dsv41-run against a vendor oracle tape.

Generation tapes: the last prompt token's logits (`--last-logits` raw f32) against the tape's step-0 full logits:
argmax, top-5 / top-32 overlap, KL(vendor || rewrite), max |d|.
Teacher-forced tapes: the run's `--tf` directory (lse, argmax, nll) against the tape's per-position `nll`, `lse` and
`top_ids`: mean NLL of both, |delta mean NLL|, top-1 agreement, the share of positions whose NLL moves more than 0.1
nats (the vendor's own 2048-vs-2422 length effect: delta mean NLL 0.0006, top-1 98.2%, 11% of positions > 0.1).

usage: trunk_vs_tape.py TAPE.pt RUN (last-logits file, or --tf dir) [--json OUT]"""
import argparse
import json
import os

import torch

ap = argparse.ArgumentParser()
ap.add_argument("tape"); ap.add_argument("run"); ap.add_argument("--json")
a = ap.parse_args()
T = torch.load(a.tape, weights_only=False)
out = {"tape": os.path.basename(a.tape), "tokens": len(T["prompt_ids"])}


def raw(path, dt):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt)


O = torch.load(a.run, weights_only=False) if a.run.endswith(".pt") else None
if O is not None and O.get("full_logits") is not None:
    # another vendor realization of the same generation prompt: its last-token logits are the floor
    ref = T["full_logits"][0].double()
    oth = O["full_logits"][0].double()
    lp, lq = ref.log_softmax(-1), oth.log_softmax(-1)
    out.update(other=os.path.basename(a.run), argmax_vendor=int(ref.argmax()), argmax_other=int(oth.argmax()),
               argmax_equal=int(ref.argmax()) == int(oth.argmax()),
               top5_overlap=len(set(ref.topk(5).indices.tolist()) & set(oth.topk(5).indices.tolist())),
               kl=(lp.exp() * (lp - lq)).sum().item(), max_abs=(oth - ref).abs().max().item())
elif O is not None:
    # another vendor tape of the same text (another length or another valid realization): the floor
    nll, top, lse = O["nll"].double(), O["top_ids"][:, 0].long(), O["lse"].double()
    tn, tl, tt = T["nll"].double(), T["lse"].double(), T["top_ids"][:, 0].long()
    n = min(len(nll), len(tn))
    d = (nll[:n] - tn[:n]).abs()
    out.update(scored=n, other=os.path.basename(a.run), mean_nll_vendor=tn[:n].mean().item(), mean_nll_other=nll[:n].mean().item(),
               delta_mean_nll=(nll[:n].mean() - tn[:n].mean()).item(), top1_agree=(top[:n] == tt[:n]).double().mean().item(),
               share_nll_moved_gt_0p1=(d > 0.1).double().mean().item(), median_abs_dnll=d.median().item(),
               max_abs_dlse=(lse[:n] - tl[:n]).abs().max().item())
elif os.path.isdir(a.run):
    nll = raw(os.path.join(a.run, "nll.f32"), torch.float32).double()
    top = raw(os.path.join(a.run, "argmax.i32"), torch.int32).long()
    lse = raw(os.path.join(a.run, "lse.f32"), torch.float32).double()
    tn, tl, tt = T["nll"].double(), T["lse"].double(), T["top_ids"][:, 0].long()
    n = min(len(nll), len(tn))
    d = (nll[:n] - tn[:n]).abs()
    out.update(scored=n, mean_nll_vendor=tn[:n].mean().item(), mean_nll_rewrite=nll[:n].mean().item(),
               delta_mean_nll=(nll[:n].mean() - tn[:n].mean()).item(), top1_agree=(top[:n] == tt[:n]).double().mean().item(),
               share_nll_moved_gt_0p1=(d > 0.1).double().mean().item(), median_abs_dnll=d.median().item(),
               max_abs_dlse=(lse[:n] - tl[:n]).abs().max().item())
else:
    ref = T["full_logits"][0].double()
    oth = raw(a.run, torch.float32).double()
    lp, lq = ref.log_softmax(-1), oth.log_softmax(-1)
    out.update(argmax_vendor=int(ref.argmax()), argmax_rewrite=int(oth.argmax()), argmax_equal=int(ref.argmax()) == int(oth.argmax()),
               top5_overlap=len(set(ref.topk(5).indices.tolist()) & set(oth.topk(5).indices.tolist())),
               top32_overlap=len(set(ref.topk(32).indices.tolist()) & set(oth.topk(32).indices.tolist())),
               kl=(lp.exp() * (lp - lq)).sum().item(), max_abs=(oth - ref).abs().max().item(),
               p_top1_vendor=lp.exp().max().item())
print(json.dumps(out, indent=1))
if a.json:
    json.dump(out, open(a.json, "w"), indent=1)
