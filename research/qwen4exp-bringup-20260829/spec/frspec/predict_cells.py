#!/usr/bin/env python3
"""Per-cell FR-Spec prediction vs measurement, on the chain each cell actually ran.

rank_ranks.py predicts per SHAPE, pooling every banked chain of that shape. That is the
right instrument for CHOOSING a width and a class before any boot exists, and the wrong one
for checking a single cell: `--trim-ab` measures ONE chain (row 0 of its prompts file), and
pooled text is not that chain. Measured consequence, kept as the correction: pooled-q put
sxc-raw at 0.820 and ogblend-raw at 1.015, while the same estimator on the cell's own chain
puts them at 0.994 and 1.012 — the cells landed at 0.991 and 1.003.

    tok/s_trim / tok/s_full = (1 - A*q) / (1 - H*(1 - N/V))

Here H is taken from the CELL's own two draft_ms_share values instead of imported, so `A`
(from the mtp10 N=11,854 receipts) is the only fitted constant left in the check.

usage: predict_cells.py <ranks_dir> <receipt.tsv>...
"""
import os
import re
import sys

V = 248320
# A per shape, fitted on mtp10's N=11,854 rows (see FRSPEC.md §2). thinkon/efflow share the
# ship-policy class (short chains, window at the k_lo floor); thinkoff and raw are the
# long-chain class where one out-of-set pick also derails the rest of the carrier chain.
A_BY_SHAPE = {"thinkon": 0.491, "efflow": 0.491, "thinkoff": 1.744, "raw": 1.402}
CLASS_FILE = {"og": "q4e-ranks-ogblend-32768.txt", "sxc": "q4e-ranks-sxc32768.txt"}


def parse(path):
    txt = open(path).read()
    out = {"file": os.path.basename(path)}
    m = re.search(r"label=(\S+)", txt)
    out["label"] = m.group(1) if m else "?"
    m = re.search(r"trim_n=(\d+)", txt)
    out["n"] = int(m.group(1))
    for arm in ("full", "trim"):
        m = re.search(rf"# arm {arm}\t.*?tok_per_s=([\d.]+)\tmean_accept_rate=([\d.]+)\tmean_accept_len=([\d.]+)", txt)
        out[f"{arm}_tps"], out[f"{arm}_acc"], out[f"{arm}_len"] = (
            float(m.group(1)), float(m.group(2)), float(m.group(3)))
        m = re.search(rf"# arm {arm}\t.*?spread_pct=([\d.]+)", txt)
        out[f"{arm}_spread"] = float(m.group(1)) if m else float("nan")
        # draft_ms_share is column 10 of the per-rep rows (rep, arm, order, rows, tokens,
        # ms, tok/s, accept, len, draft_share, hist)
        rows = [ln.split("\t") for ln in txt.splitlines()
                if re.match(r"^\d+\t" + arm + r"\t", ln)]
        out[f"{arm}_dshare"] = sum(float(r[9]) for r in rows) / len(rows)
        out[f"{arm}_reps"] = len(rows)
    out["measured"] = float(re.search(r"# speedup_trim_over_full\t([\d.]+)", txt).group(1))
    out["divergence"] = int(re.search(r"# rep0_full_vs_trim_first_divergence\t(-?\d+)", txt).group(1))
    out["chain"] = [int(x) for x in re.search(r"# rep0_full\t([0-9,]+)", txt).group(1).split(",")]
    return out


def main():
    ranks_dir, receipts = sys.argv[1], sys.argv[2:]
    sets = {k: set(int(l) for l in open(os.path.join(ranks_dir, f)))
            for k, f in CLASS_FILE.items()}
    print("cell\tclass\tshape\tN\tq_own_chain\tH_cell\tA\tpredicted\tmeasured\tfull_tps\ttrim_tps"
          "\taccept_full\taccept_trim\tlen_full\tlen_trim\tspread_full\tspread_trim\treps\tchain_divergence")
    for path in receipts:
        c = parse(path)
        parts = c["label"].split("-")
        cls = "og" if "og" in parts else "sxc"
        shape = parts[-1] if parts[-1] in A_BY_SHAPE else next(
            (p for p in parts if p in A_BY_SHAPE), "raw")
        s = sets[cls]
        q = sum(1 for t in c["chain"] if t not in s) / len(c["chain"])
        H = (c["full_dshare"] - c["trim_dshare"]) / (1.0 - c["n"] / V)
        A = A_BY_SHAPE[shape]
        pred = (1.0 - A * q) / (1.0 - H * (1.0 - c["n"] / V))
        print(f"{c['file']}\t{cls}\t{shape}\t{c['n']}\t{q:.4f}\t{H:.4f}\t{A}\t{pred:.4f}"
              f"\t{c['measured']:.4f}\t{c['full_tps']:.2f}\t{c['trim_tps']:.2f}"
              f"\t{c['full_acc']:.3f}\t{c['trim_acc']:.3f}\t{c['full_len']:.2f}\t{c['trim_len']:.2f}"
              f"\t{c['full_spread']:.3f}\t{c['trim_spread']:.3f}\t{c['full_reps']}\t{c['divergence']}")


if __name__ == "__main__":
    sys.exit(main())
