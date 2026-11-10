"""Compare two raw DSpark draft dumps of the same inputs against each other: the vendor GPU tape (export_tape.py) as
the reference, and either the vendor CPU replay (the floor) or memra's `dsv41-spec` draft-tape dump.

usage: draft_cmp.py REF_ROOT OTHER_ROOT OUT.json ID [ID ...]
Per case and pooled: for every tensor the per-step relative L2 against the reference (median, p95, max); the router's
expert-set agreement per row; the draft tokens' agreement; over the rows whose earlier picks agree (the Markov bias of a
row depends on the pick before it), the KL of the reference's distribution from the other's (exact on the steps with
full logits, and from the top-32 plus one rest bucket on every step, the lane's estimator); the confidence scores'
absolute difference; the rings' byte equality and relative L2."""
import json, os, sys
import numpy as np

ref_root, oth_root, out_path, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
DIM, HC, RANK = 5120, 4, 256


def bf16(path):
    u = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)


def f32(path):
    return np.fromfile(path, dtype="<f4")


def i32(path):
    return np.fromfile(path, dtype="<i4")


def rel_l2(a, b, rows):
    a = a.reshape(rows, -1).astype(np.float64)
    b = b.reshape(rows, -1).astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.maximum(np.linalg.norm(b, axis=1), 1e-30)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return None
    return {"n": int(x.size), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max()),
            "mean": float(x.mean())}


def kl_top(rv, ri, rl, ov, oi, ol):
    """KL(ref || other) lower bound from both sides' top-32 and one rest bucket (per row)."""
    out = np.zeros(rv.shape[0])
    for r in range(rv.shape[0]):
        p = np.exp(rv[r].astype(np.float64) - rl[r])
        od = dict(zip(oi[r].tolist(), ov[r].tolist()))
        both = [(pi, od[i]) for pi, i in zip(p, ri[r].tolist()) if i in od]
        pp = np.array([x[0] for x in both])
        qq = np.exp(np.array([x[1] for x in both], dtype=np.float64) - ol[r])
        kl = float(np.sum(pp * (np.log(pp) - np.log(qq)))) if len(both) else 0.0
        pr, qr = max(1.0 - pp.sum(), 1e-12), max(1.0 - qq.sum(), 1e-12)
        out[r] = kl + pr * (np.log(pr) - np.log(qr))
    return out


pool = {}
cases = {}
for cid in ids:
    R, O = os.path.join(ref_root, cid), os.path.join(oth_root, cid)
    meta = json.load(open(os.path.join(R, "meta.json")))
    n, stages, block = meta["steps"], meta["stages"], meta["config"]["dspark_block_size"]
    res = {"steps": n}
    # The vendor keeps each stage's ring across the cases of one process, so a slot this case never wrote holds
    # another case's row (no draft of this case reads it). Compare the slots this case wrote: the prompt's last
    # `window` positions after the seed, and the steps' positions too at the end.
    win = meta["config"]["window_size"]
    s_len = meta["prompt_len"]
    seeded = {p % win for p in range(max(0, s_len - win), s_len)}
    stepped = set(seeded)
    if n:
        stepped |= {int(p) % win for p in i32(os.path.join(R, "pos.i32"))}
    for ring, slots in (("ring0.bf16", seeded), ("ring1.bf16", stepped)):
        a, b = np.fromfile(os.path.join(R, ring), dtype="<u2"), np.fromfile(os.path.join(O, ring), dtype="<u2")
        fa, fb = (a.astype(np.uint32) << 16).view(np.float32), (b.astype(np.uint32) << 16).view(np.float32)
        used = np.array([(r % win) in slots for r in range(a.size // 512)])  # [stages][window] rows
        rl = rel_l2(fb.reshape(-1, 512)[used], fa.reshape(-1, 512)[used], int(used.sum())) if used.any() else []
        res[ring] = {"rows": int(used.sum()), "rows_byte_equal": int((a.reshape(-1, 512)[used] == b.reshape(-1, 512)[used]).all(1).sum()),
                     "rel_l2": stats(rl)}
        pool.setdefault("ring", []).extend(list(rl))
    if n == 0:
        cases[cid] = res
        continue
    names = ["main_x"] + [f"{t}{k}" for k in range(stages) for t in ("attn_in", "attn_out", "ffn_out", "h")] + ["head_x"]
    for name in names:
        a, b = bf16(os.path.join(R, name + ".bf16")), bf16(os.path.join(O, name + ".bf16"))
        rows = n if name == "main_x" else n * block
        rl = rel_l2(b, a, rows)
        res[name] = stats(rl)
        pool.setdefault(name, []).extend(list(rl))
    for k in range(stages):
        a, b = f32(os.path.join(R, f"pre{k}.f32")), f32(os.path.join(O, f"pre{k}.f32"))
        d = np.abs(a - b).reshape(n * block, -1).max(1)
        res[f"pre{k}_absmax"] = stats(d)
        pool.setdefault(f"pre{k}_absmax", []).extend(list(d))
        ra = i32(os.path.join(R, f"route_i{k}.i32")).reshape(n * block, -1)
        rb = i32(os.path.join(O, f"route_i{k}.i32")).reshape(n * block, -1)
        same = np.array([len(set(x.tolist()) & set(y.tolist())) / ra.shape[1] for x, y in zip(ra, rb)])
        res[f"route{k}_agree"] = float(same.mean())
        pool.setdefault(f"route{k}_agree", []).extend(list(same))
    da = i32(os.path.join(R, "out_ids.i32")).reshape(n, block)
    db = i32(os.path.join(O, "out_ids.i32")).reshape(n, block)
    eq = da == db
    res["draft_token_agree"] = float(eq.mean())
    res["draft_block_agree"] = float(eq.all(1).mean())
    pool.setdefault("draft_token_agree", []).extend(list(eq.reshape(-1).astype(float)))
    # rows whose earlier picks agree: the same Markov context on both sides
    ctx = np.ones_like(eq)
    ctx[:, 1:] = np.cumprod(eq[:, :-1], axis=1).astype(bool)
    pool.setdefault("draft_token_agree_same_ctx", []).extend(list(eq[ctx].astype(float)))
    res["draft_token_agree_same_ctx"] = float(eq[ctx].mean())
    ca, cb = f32(os.path.join(R, "conf.f32")).reshape(n, block), f32(os.path.join(O, "conf.f32")).reshape(n, block)
    cd = np.abs(ca - cb)[ctx]
    res["conf_absdiff"] = stats(cd)
    pool.setdefault("conf_absdiff", []).extend(list(cd))
    sa, sb = 1 / (1 + np.exp(-ca)), 1 / (1 + np.exp(-cb))
    pool.setdefault("conf_sigmoid_absdiff", []).extend(list(np.abs(sa - sb)[ctx]))
    rv, ri = f32(os.path.join(R, "top_vals.f32")).reshape(n * block, 32), i32(os.path.join(R, "top_ids.i32")).reshape(n * block, 32)
    ov, oi = f32(os.path.join(O, "top_vals.f32")).reshape(n * block, 32), i32(os.path.join(O, "top_ids.i32")).reshape(n * block, 32)
    rl_, ol_ = f32(os.path.join(R, "lse.f32")), f32(os.path.join(O, "lse.f32"))
    kl = kl_top(rv, ri, rl_, ov, oi, ol_)[ctx.reshape(-1)]
    res["kl_top32"] = stats(kl)
    pool.setdefault("kl_top32", []).extend(list(kl))
    fs = meta["full_steps"]
    if fs:
        V = 129280
        fa = f32(os.path.join(R, "logits_full.f32")).reshape(fs * block, V).astype(np.float64)
        fb = f32(os.path.join(O, "logits_full.f32")).reshape(fs * block, V).astype(np.float64)
        la = fa - np.log(np.exp(fa - fa.max(1, keepdims=True)).sum(1, keepdims=True)) - fa.max(1, keepdims=True)
        lb = fb - np.log(np.exp(fb - fb.max(1, keepdims=True)).sum(1, keepdims=True)) - fb.max(1, keepdims=True)
        klf = (np.exp(la) * (la - lb)).sum(1)[ctx[:fs].reshape(-1)]
        res["kl_full"] = stats(klf)
        pool.setdefault("kl_full", []).extend(list(klf))
        ba = f32(os.path.join(R, "bias_full.f32")).reshape(fs * block, V)
        bb = f32(os.path.join(O, "bias_full.f32")).reshape(fs * block, V)
        m = ctx[:fs].reshape(-1)
        bl = rel_l2(bb[m], ba[m], int(m.sum()))
        res["markov_bias_rel_l2"] = stats(bl)
        pool.setdefault("markov_bias_rel_l2", []).extend(list(bl))
    cases[cid] = res

summary = {}
for k, v in pool.items():
    if k.endswith("_agree") or k.endswith("_same_ctx"):
        summary[k] = {"n": len(v), "mean": float(np.mean(v))}
    else:
        summary[k] = stats(v)
json.dump({"ref": ref_root, "other": oth_root, "ids": ids, "pooled": summary, "cases": cases}, open(out_path, "w"), indent=1)
for k, v in summary.items():
    print(k, json.dumps(v))
