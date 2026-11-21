"""lane F's analyze.py stats() (klsens/analyze.py, copied unchanged in the KL / top-1 / dNLL part) and its
sequence bootstrap (B = 4000, seed 20261121)."""
import numpy as np

rng = np.random.default_rng(20261121)
B = 4000


def boot(per_sum, per_cnt):
    n = len(per_sum)
    idx = rng.integers(0, n, size=(B, n))
    m = per_sum[idx].sum(1) / np.maximum(per_cnt[idx].sum(1), 1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def stats(a):
    seq = a["seq"]; kl = a["kl"].astype(np.float64); hl = a["has_label"]; sm = a["sampled"]
    d = a["nll_arm"].astype(np.float64) - a["nll_ref"].astype(np.float64)
    real = hl & ~sm
    us = np.unique(seq)
    out = {}
    for nm, mask, val in (("kl_all", np.ones_like(hl), kl), ("kl_real", real, kl), ("kl_model", hl & sm, kl), ("dnll_real", real, d)):
        s = np.array([val[(seq == u) & mask].sum() for u in us]); c = np.array([((seq == u) & mask).sum() for u in us], dtype=np.float64)
        keep = c > 0
        out[nm] = {"mean": float(s.sum() / max(c.sum(), 1)), "ci95": boot(s[keep], c[keep]) if keep.sum() > 1 else None, "n_tok": int(c.sum()), "n_seq": int(keep.sum())}
    k = kl[real]
    out["kl_real"].update({"p50": float(np.median(k)), "p99": float(np.percentile(k, 99)), "max": float(k.max())})
    out["top1_real"] = float(a["top1"][real].mean())
    out["nll_ref_real"] = float(a["nll_ref"][real].astype(np.float64).mean())
    return out


if __name__ == "__main__":
    import json, sys
    a = np.load(sys.argv[1])
    s = stats(a)
    json.dump({"npz": sys.argv[1], "stats": s}, open(sys.argv[2], "w"), indent=1)
    print(f"{sys.argv[1]}: KLall {s['kl_all']['mean']:.4f} KLreal {s['kl_real']['mean']:.4f} {s['kl_real']['ci95']} p99 {s['kl_real']['p99']:.3f} "
          f"top1 {s['top1_real']:.3f} dNLLreal {s['dnll_real']['mean']:+.4f} {s['dnll_real']['ci95']}")
