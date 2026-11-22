import sys, json, os
import numpy as np
from safetensors.numpy import load_file
a, b, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
out = {"a": a, "b": b, "common_prefix_positions": n}
la, lb = load_file(os.path.join(a, "logits_post.safetensors")), load_file(os.path.join(b, "logits_post.safetensors"))
pa, pb = list(la["positions"]), list(lb["positions"])
pos = [p for p in pa if p in set(pb) and p < n]
ia, ib = [pa.index(p) for p in pos], [pb.index(p) for p in pos]
x, y = la["logits_post"][ia].astype(np.float64), lb["logits_post"][ib].astype(np.float64)
def lsm(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))
lx, ly = lsm(x), lsm(y)
kl = (np.exp(lx) * (lx - ly)).sum(-1)
out["logits"] = {"positions": len(pos), "max_abs": float(np.abs(x - y).max()), "mean_abs": float(np.abs(x - y).mean()),
                 "kl_a_b_mean": float(kl.mean()), "kl_a_b_max": float(kl.max()), "argmax_equal": int((x.argmax(-1) == y.argmax(-1)).sum())}
for nm in ("resid_attn", "resid_ffn"):
    ra, rb = load_file(os.path.join(a, nm + ".safetensors")), load_file(os.path.join(b, nm + ".safetensors"))
    qa, qb = list(ra["positions"]), list(rb["positions"])
    ps = [p for p in qa if p in set(qb) and p < n]
    u = ra[nm][:, [qa.index(p) for p in ps]].astype(np.float64)
    v = rb[nm][:, [qb.index(p) for p in ps]].astype(np.float64)
    rel = np.sqrt(((u - v) ** 2).sum((1, 2)) / (u ** 2).sum((1, 2)))
    out[nm + "_rel_l2_per_slot"] = [round(float(r), 6) for r in rel]
ra, rb = load_file(os.path.join(a, "router.safetensors")), load_file(os.path.join(b, "router.safetensors"))
same = (ra["route_ids"][:, :n] == rb["route_ids"][:, :n]).all(-1)
out["route_rows_equal_frac_per_slot"] = [round(float(s), 4) for s in same.mean(1)]
out["route_rows_equal_frac"] = float(same.mean())
out["min_route_margin_a"] = float(ra["route_margin"][:, :n].min())
ta, tb = load_file(os.path.join(a, "tokens.safetensors")), load_file(os.path.join(b, "tokens.safetensors"))
g = (ta["greedy_tokens"] == tb["greedy_tokens"])
out["greedy_equal_prefix"] = int(np.argmin(g)) if not g.all() else len(g)
print(json.dumps(out))
