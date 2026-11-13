"""The early decode steps against the vendor's CPU-vs-GPU floor.

Below 64 keys per attention row the two GPU attention schedules (tilelang blocks of 64, global max) are the same
arithmetic, so the schedule floor is zero at a short prompt's prefill and small on the next steps. The vendor program on
the CPU (torch CPU GEMMs, the same stand-ins; make_forced_oracle.py's copy with --device cpu), fed the same tokens, is
the backend floor there, the one the layer gates and the checkpoint band use. Exact KL against the GPU tape's full
logits on the first 8 steps, step by step, for memra, the CPU vendor and the global-max GPU vendor.

usage: cpu_floor_kl.py TAPES CPU_FLOOR GPU_FLOOR NATIVE CASE[,CASE...] [--json OUT]"""
import json
import os
import sys

import torch


def raw(p, dt):
    return torch.frombuffer(bytearray(open(p, "rb").read()), dtype=dt)


def kl(p, q):
    lp, lq = p.log_softmax(-1), q.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


tapes, cpu, gpu, nat, cases = sys.argv[1:6]
out = {"cases": {}}
allk = {"memra": [], "cpu": [], "schedule": []}
for c in cases.split(","):
    T = torch.load(os.path.join(tapes, f"{c}.pt"), weights_only=False, map_location="cpu")
    C = torch.load(os.path.join(cpu, f"{c}.pt"), weights_only=False, map_location="cpu")
    F = torch.load(os.path.join(gpu, f"{c}.pt"), weights_only=False, map_location="cpu")
    N = raw(os.path.join(nat, c, "full.f32"), torch.float32).double().view(-1, 129280)
    t = T["full_logits"].double()
    n = min(len(t), len(C["full_logits"]), len(N))
    k = {"memra": kl(t[:n], N[:n]), "cpu": kl(t[:n], C["full_logits"][:n].double()),
         "schedule": kl(t[:n], F["full_logits"][:n].double())}
    top1 = {"memra": int((N[:n].argmax(-1) == t[:n].argmax(-1)).sum()),
            "cpu": int((C["full_logits"][:n].argmax(-1) == t[:n].argmax(-1)).sum())}
    out["cases"][c] = {"steps": n, "kl": {a: v.tolist() for a, v in k.items()},
                       "mean": {a: v.mean().item() for a, v in k.items()}, "top1_on_full_steps": top1,
                       "cpu_gen_matches_tape": [int(x) for x in C["gen_ids"]] == [int(x) for x in T["gen_ids"][:n]]}
    for a in allk:
        allk[a].append(k[a])
    print(c, json.dumps(out["cases"][c]["mean"]), "top-1", top1)
cat = {a: torch.cat(v) for a, v in allk.items()}
out["pooled"] = {"steps": len(cat["memra"]), **{f"mean_kl_{a}": v.mean().item() for a, v in cat.items()},
                 "memra_over_cpu": cat["memra"].mean().item() / cat["cpu"].mean().item()}
out["pooled"]["pass_2x_cpu_floor"] = out["pooled"]["memra_over_cpu"] <= 2.0
print(json.dumps(out["pooled"], indent=1))
if "--json" in sys.argv:
    json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
