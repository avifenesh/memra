"""Per-token distribution of a GPU rewrite's error against the vendor GPU capture, next to the vendor CPU floor's.

For every tensor the rewrite dumped (dsv41-gate layer --save) and the floor also has: global rel L2, per-token rel L2
median / p90 / p99 / max, and for routings the share of tokens whose expert set matches. Writes a JSON next to the
printout.

usage: compare_dumps.py CAPTURE_DIR FLOOR_DIR MINE_DIR OUT.json"""
import json
import os
import sys

import torch

cap, flo, mine, out = sys.argv[1:5]
DT = {"bf16": torch.bfloat16, "f32": torch.float32, "i32": torch.int32, "i64": torch.int64, "u8": torch.uint8}


def index(root):
    return json.load(open(os.path.join(root, "index.json")))["tensors"]


def load(root, idx, k):
    e = idx[k]
    f = os.path.join(root, e["file"])
    return torch.from_file(f, size=os.path.getsize(f), dtype=torch.uint8).view(DT[e["dtype"]]).reshape(e["shape"])


ci, fi, mi = index(cap), index(flo), index(mine)
rows = []
for k in sorted(mi):
    if k not in ci:
        continue
    want = load(cap, ci, k)
    got = load(mine, mi, k)
    fl = load(flo, fi, k) if k in fi else None
    if k.endswith("gate/out1"):
        w = want.reshape(-1, 6).long().sort(-1).values
        g = got.reshape(-1, 6).long().sort(-1).values
        r = {"key": k, "tokens": w.size(0), "route_agree": (w == g).all(-1).double().mean().item()}
        if fl is not None:
            r["floor_route_agree"] = (w == fl.reshape(-1, 6).long().sort(-1).values).all(-1).double().mean().item()
        rows.append(r)
        print(f"{k:28s} route agree mine {r['route_agree']:.4f}" + (f" floor {r['floor_route_agree']:.4f}" if fl is not None else ""))
        continue
    w = want.float()
    n_tok = w.shape[1] if w.dim() >= 3 else w.shape[0]
    w2 = w.reshape(n_tok, -1)

    def stats(x):
        x2 = x.float().reshape(n_tok, -1)
        per = ((x2 - w2).norm(dim=1) / w2.norm(dim=1).clamp_min(1e-30)).double()
        q = torch.quantile(per, torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)).tolist()
        return {"rel_l2": ((x2 - w2).norm() / w2.norm()).item(), "tok_median": q[0], "tok_p90": q[1], "tok_p99": q[2],
                "tok_max": per.max().item()}

    r = {"key": k, "tokens": n_tok, "mine": stats(got)}
    if fl is not None:
        r["floor"] = stats(fl)
    rows.append(r)
    m = r["mine"]
    line = f"{k:28s} mine rel {m['rel_l2']:.2e} tok med {m['tok_median']:.2e} p99 {m['tok_p99']:.2e}"
    if fl is not None:
        f_ = r["floor"]
        line += f" | floor rel {f_['rel_l2']:.2e} tok med {f_['tok_median']:.2e} p99 {f_['tok_p99']:.2e}"
    print(line, flush=True)
json.dump(rows, open(out, "w"), indent=1)
