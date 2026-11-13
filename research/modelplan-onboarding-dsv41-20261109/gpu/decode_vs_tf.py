"""Long-context decode steps of a vendor tape against a teacher-forced prefill of the same tokens.

A decode tape holds the prompt, the greedy tokens and every step's top-32 logits (the first steps' full logits too).
The prompt plus the generated tokens, scored in one teacher-forced prefill, puts decode step k at position P-1+k. Two
prefills are compared with the decode steps: the native `dsv41-run --tf` (top_ids.i32, top_vals.f32, lse.f32) and the
vendor's own (`make_tfids_oracle.py`), the vendor's prefill-vs-decode floor. Top-1 agreement over every step; KL on
the steps with full logits, from the shared top-32 plus one rest bucket as in tf_kl.py.

usage: decode_vs_tf.py prep OUT_DIR TAPE.pt...      (OUT_DIR/decode_tf_cases.json, OUT_DIR/<name>.ids.i32)
       decode_vs_tf.py cmp TAPE.pt OTHER [--json OUT]  (OTHER: a native --tf dir or a vendor teacher-forced tape)"""
import json
import os
import sys

import torch


def name(p):
    return os.path.basename(p)[: -len(".pt")]


if sys.argv[1] == "prep":
    out, cases = sys.argv[2], []
    for p in sys.argv[3:]:
        T = torch.load(p, weights_only=False)
        ids = list(T["prompt_ids"]) + list(T["gen_ids"])[:-1]
        cases.append({"id": name(p) + "_tf", "tf_ids": ids})
        open(os.path.join(out, name(p) + ".ids.i32"), "wb").write(torch.tensor(ids, dtype=torch.int32).numpy().tobytes())
        print(name(p), len(T["prompt_ids"]), "prompt +", len(T["gen_ids"]), "generated:", len(ids), "positions")
    json.dump(cases, open(os.path.join(out, "decode_tf_cases.json"), "w"))
    sys.exit(0)

tape, other = sys.argv[2], sys.argv[3]
js = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
T = torch.load(tape, weights_only=False)
P, G = len(T["prompt_ids"]), len(T["gen_ids"])
di, dv, full = T["top_ids"].long(), T["top_vals"].double(), T["full_logits"].double()
gen = torch.tensor(list(T["gen_ids"])[:G]).long()  # the vendor's greedy argmax at each step (lowest id among ties)


def raw(path, dt):
    return torch.frombuffer(bytearray(open(path, "rb").read()), dtype=dt)


if other.endswith(".pt"):
    O = torch.load(other, weights_only=False)
    oi, ov, ol = O["top_ids"].long(), O["top_vals"].double(), O["lse"].double()
else:
    ol = raw(os.path.join(other, "lse.f32"), torch.float32).double()
    oi = raw(os.path.join(other, "top_ids.i32"), torch.int32).long().view(-1, 32)
    ov = raw(os.path.join(other, "top_vals.f32"), torch.float32).double().view(-1, 32)
pos = torch.arange(G) + P - 1
assert len(ol) >= P + G - 1, "the prefill is shorter than prompt + generated"
top1 = (oi[pos, 0] == gen).double()
kls = []
for k in range(len(full)):
    lse = torch.logsumexp(full[k], 0)
    p = (dv[k] - lse).exp()
    q = (ov[pos[k]] - ol[pos[k]]).exp()
    eq = di[k][:, None] == oi[pos[k]][None, :]
    both = eq.any(1)
    qs = (eq * q[None, :]).sum(1)
    ps = p * both
    pr = (1 - ps.sum()).clamp_min(1e-12)
    qr = (1 - qs.sum()).clamp_min(1e-12)
    term = torch.where(both, ps * (ps.clamp_min(1e-300).log() - qs.clamp_min(1e-300).log()), torch.zeros_like(ps))
    kls.append((term.sum() + pr * (pr.log() - qr.log())).item())
out = {"tape": name(tape), "other": os.path.basename(other.rstrip("/")), "prompt": P, "steps": G,
       "top1_agree": top1.mean().item(), "top1_agree_steps": int(top1.sum().item()),
       "first_disagreement": int((top1 == 0).nonzero()[0]) if (top1 == 0).any() else None,
       "full_logit_steps": len(kls), "mean_kl": sum(kls) / len(kls), "max_kl": max(kls), "kl": kls}
print(json.dumps(out, indent=1))
if js:
    json.dump(out, open(js, "w"), indent=1)
