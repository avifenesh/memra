#!/usr/bin/env python3
"""Expert co-activation analysis over a MEMRA_MOE_SEL_DUMP file (numpy only).

Owner question (2026-09-02): could GLM-5.3-Flash's 288 routed experts per MoE
layer be split across two cards by co-activation (experts that fire together on
one card, always-active experts replicated on both) so that a token's 8 selected
experts rarely cross cards?

Input: the binary dump written by crates/memra-engine/src/moe_sel_dump.rs
(`MEMRA_MOE_SEL_DUMP=<path>`), format memra-moe-sel-v1, little-endian, no header:

    u8 layer, u8 n_sel, n_sel x (u16 expert_id, f32 routing_weight)

one record per (routed token, MoE layer), prime and decode alike.

Per layer and pooled (token-weighted over layers) this reports:

  * per-expert activation frequency: min / median / max (fraction of tokens that
    pick the expert), Gini of the pick counts, share of picks in the top 16;
  * the E x E co-activation matrix (counts of tokens picking both i and j),
    optionally saved as .npz;
  * the best 2-way partition under a replication budget R in {0, 16, 32, 64}:
    spectral bisection start, Kernighan-Lin refinement, replicated experts
    (counted on BOTH cards) chosen either by activation frequency or by their
    cut contribution, whichever scores better on the fit tokens;
  * for each (layer, R): P(all n_sel experts of a token on one card), the mean
    number of cards a token touches, the per-card share of token-expert pairs,
    and the same numbers for the random-halves baseline (mean over seeds) and
    the engine's contiguous even split (rank = expert // (E / 2)).

The partition is FIT on the first (1 - holdout) fraction of each layer's records
in file order and SCORED on the last `holdout` fraction (default 0.3); in-sample
numbers are reported next to it so optimism is visible rather than hidden.

Usage:
    python3 tools/moe_coact.py sel.bin --experts 288 --out-md report.md \
        --out-json report.json [--save-coact coact.npz] [--layers 3,4,5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

BUDGETS_DEFAULT = (0, 16, 32, 64)


# --------------------------------------------------------------------------- parsing


def parse_dump(path: Path):
    """Return (layer[u8][N], sel[u16][N, n], w[f32][N, n]) for a constant-n_sel dump,
    or the variable-n_sel fallback grouped by n_sel: list of such triples."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 2:
        raise SystemExit(f"{path}: empty dump (no records)")
    n0 = int(raw[1])
    if n0 == 0:
        raise SystemExit(f"{path}: first record has n_sel=0; not a memra-moe-sel-v1 dump")
    stride = 2 + 6 * n0
    fixed = raw.size % stride == 0 and bool(np.all(raw[1::stride] == n0))
    if fixed:
        dt = np.dtype(
            [("layer", "u1"), ("n", "u1"), ("slots", [("e", "<u2"), ("w", "<f4")], (n0,))]
        )
        assert dt.itemsize == stride
        rec = raw.view(dt)
        return [(rec["layer"].copy(), rec["slots"]["e"].copy(), rec["slots"]["w"].copy())]
    # Variable n_sel: sequential walk, grouped by n_sel.
    groups: dict[int, list[int]] = {}
    off = 0
    while off + 2 <= raw.size:
        n = int(raw[off + 1])
        sz = 2 + 6 * n
        if n == 0 or off + sz > raw.size:
            raise SystemExit(f"{path}: truncated or corrupt record at byte {off}")
        groups.setdefault(n, []).append(off)
        off += sz
    out = []
    for n, offs in sorted(groups.items()):
        offs_a = np.asarray(offs, dtype=np.int64)
        dt = np.dtype(
            [("layer", "u1"), ("n", "u1"), ("slots", [("e", "<u2"), ("w", "<f4")], (n,))]
        )
        idx = offs_a[:, None] + np.arange(dt.itemsize)[None, :]
        rec = raw[idx].reshape(-1).view(dt)
        out.append((rec["layer"].copy(), rec["slots"]["e"].copy(), rec["slots"]["w"].copy()))
    return out


# --------------------------------------------------------------------------- statistics


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def coactivation(sel: np.ndarray, E: int) -> np.ndarray:
    """Symmetric E x E count matrix: C[i, j] = tokens picking both i and j (i != j);
    the diagonal carries the pick count of each expert. Slot value E is the pad
    sentinel of a variable-n_sel dump and is dropped."""
    T, n = sel.shape
    B = E + 1
    C = np.zeros(B * B, dtype=np.int64)
    s = sel.astype(np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            C += np.bincount(s[:, i] * B + s[:, j], minlength=B * B)
    C = C.reshape(B, B)[:E, :E]
    C = C + C.T
    np.fill_diagonal(C, np.bincount(s.reshape(-1), minlength=B)[:E])
    return C


# --------------------------------------------------------------------------- partitioning


def spectral_start(C: np.ndarray, nodes: np.ndarray, freq: np.ndarray) -> np.ndarray:
    """Initial balanced split of `nodes` (expert ids): 0/1 per node. Fiedler vector of
    the co-activation Laplacian when the subgraph is connected enough; otherwise
    alternate by frequency (which also balances load)."""
    m = nodes.size
    side = np.zeros(m, dtype=np.int8)
    if m < 4:
        side[m // 2 :] = 1
        return side
    W = C[np.ix_(nodes, nodes)].astype(np.float64)
    np.fill_diagonal(W, 0.0)
    if W.sum() > 0:
        L = np.diag(W.sum(1)) - W
        vals, vecs = np.linalg.eigh(L)
        # Second-smallest eigenvalue's vector; a disconnected graph has vals[1] ~ 0,
        # in which case the vector still orders components, which is fine as a start.
        order = np.argsort(vecs[:, 1], kind="stable")
    else:
        order = np.argsort(-freq[nodes], kind="stable")
        side[order[1::2]] = 1
        return side
    side[order[m // 2 :]] = 1
    return side


def kernighan_lin(C: np.ndarray, nodes: np.ndarray, side: np.ndarray, max_passes: int = 20):
    """Classic KL 2-way refinement minimising the cut weight of C over `nodes`,
    keeping the two sides' sizes fixed. Returns the refined side vector."""
    W = C[np.ix_(nodes, nodes)].astype(np.float64)
    np.fill_diagonal(W, 0.0)
    side = side.copy()
    m = nodes.size
    for _ in range(max_passes):
        A = np.flatnonzero(side == 0)
        B = np.flatnonzero(side == 1)
        k = min(A.size, B.size)
        if k == 0:
            break
        # D[v] = external - internal co-activation weight of v.
        same = side[:, None] == side[None, :]
        D = (W * (~same)).sum(1) - (W * same).sum(1)
        lockedA = np.zeros(A.size, dtype=bool)
        lockedB = np.zeros(B.size, dtype=bool)
        gains = []
        swaps = []
        WAB = W[np.ix_(A, B)]
        for _step in range(k):
            G = D[A][:, None] + D[B][None, :] - 2.0 * WAB
            G[lockedA, :] = -np.inf
            G[:, lockedB] = -np.inf
            ia, ib = np.unravel_index(int(np.argmax(G)), G.shape)
            g = G[ia, ib]
            if not np.isfinite(g):
                break
            gains.append(g)
            swaps.append((ia, ib))
            lockedA[ia] = True
            lockedB[ib] = True
            a, b = A[ia], B[ib]
            # Update D for the unlocked nodes as if a and b were swapped.
            D[A] += 2.0 * W[A, a] - 2.0 * W[A, b]
            D[B] += 2.0 * W[B, b] - 2.0 * W[B, a]
        if not gains:
            break
        cum = np.cumsum(gains)
        best = int(np.argmax(cum))
        if cum[best] <= 1e-9:
            break
        for ia, ib in swaps[: best + 1]:
            side[A[ia]] = 1
            side[B[ib]] = 0
    del m
    return side


def partition(C: np.ndarray, freq: np.ndarray, E: int, replicated: np.ndarray) -> np.ndarray:
    """Full E-vector: 0 / 1 = card, 2 = replicated (both cards)."""
    assign = np.full(E, 2, dtype=np.int8)
    rep = np.zeros(E, dtype=bool)
    rep[replicated] = True
    nodes = np.flatnonzero(~rep)
    side = spectral_start(C, nodes, freq)
    side = kernighan_lin(C, nodes, side)
    assign[nodes] = side
    return assign


def cut_contribution(C: np.ndarray, assign: np.ndarray) -> np.ndarray:
    """Per expert: co-activation weight with experts on the OTHER card (replicated
    experts contribute no cut and receive none)."""
    E = assign.size
    W = C.astype(np.float64).copy()
    np.fill_diagonal(W, 0.0)
    on0 = assign == 0
    on1 = assign == 1
    ext = np.zeros(E, dtype=np.float64)
    ext[on0] = W[np.ix_(on0, on1)].sum(1)
    ext[on1] = W[np.ix_(on1, on0)].sum(1)
    return ext


def choose_replicated(
    C: np.ndarray, freq: np.ndarray, E: int, R: int, strategy: str, rounds: int = 3
) -> np.ndarray:
    if R == 0:
        return partition(C, freq, E, np.zeros(0, dtype=np.int64))
    if strategy == "freq":
        rep = np.argsort(-freq, kind="stable")[:R]
        return partition(C, freq, E, rep)
    # "cut": alternate partition <-> replicate the R experts carrying the most cut.
    rep = np.argsort(-freq, kind="stable")[:R]
    assign = partition(C, freq, E, rep)
    for _ in range(rounds):
        ext = cut_contribution(C, assign)
        ext[assign == 2] = -1.0  # keep already-replicated experts eligible last
        # Score every expert by its cut if it were NOT replicated: recompute with the
        # current replicated set folded back in as ordinary nodes is expensive; the
        # greedy proxy is ext over the current split plus the frequency of the current
        # replicated set (so a hot expert is not evicted for a cold one).
        score = ext.copy()
        score[assign == 2] = freq[assign == 2].astype(np.float64) * 0.5
        new_rep = np.argsort(-score, kind="stable")[:R]
        if set(new_rep.tolist()) == set(rep.tolist()):
            break
        rep = new_rep
        assign = partition(C, freq, E, rep)
    return assign


# --------------------------------------------------------------------------- scoring


def score(sel: np.ndarray, assign: np.ndarray) -> dict:
    """Token metrics for a partition: P(single card), mean cards touched, per-card
    share of token-expert pairs (replicated pairs go to the card the token already
    touches; when it touches both or none, to the less loaded side of that token,
    ties broken by token parity)."""
    a = np.append(assign, np.int8(3))[sel]  # [T, n] in {0, 1, 2}; 3 = pad sentinel
    need0 = (a == 0).any(1)
    need1 = (a == 1).any(1)
    cards = need0.astype(np.int64) + need1.astype(np.int64)
    p_single = float(np.mean(cards <= 1))
    mean_cards = float(np.mean(np.maximum(cards, 1)))
    n0 = (a == 0).sum(1)
    n1 = (a == 1).sum(1)
    nrep = (a == 2).sum(1)
    T = sel.shape[0]
    parity = (np.arange(T) & 1).astype(np.int64)
    # Which card takes this token's replicated pairs.
    rep_to_1 = np.where(
        need0 & ~need1,
        0,
        np.where(~need0 & need1, 1, np.where(n0 < n1, 0, np.where(n1 < n0, 1, parity))),
    )
    pairs0 = n0.sum() + (nrep * (rep_to_1 == 0)).sum()
    pairs1 = n1.sum() + (nrep * (rep_to_1 == 1)).sum()
    tot = pairs0 + pairs1
    return {
        "p_single": p_single,
        "mean_cards": mean_cards,
        "load0": float(pairs0 / tot) if tot else 0.0,
        "load1": float(pairs1 / tot) if tot else 0.0,
        "tokens": int(T),
    }


def random_assign(E: int, replicated: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assign = np.full(E, 2, dtype=np.int8)
    rep = np.zeros(E, dtype=bool)
    rep[replicated] = True
    nodes = np.flatnonzero(~rep)
    perm = rng.permutation(nodes)
    assign[perm[: nodes.size // 2]] = 0
    assign[perm[nodes.size // 2 :]] = 1
    return assign


def even_assign(E: int, replicated: np.ndarray) -> np.ndarray:
    assign = (np.arange(E) // (E // 2)).astype(np.int8)
    assign[assign > 1] = 1
    assign[replicated] = 2
    return assign


# --------------------------------------------------------------------------- driver


def analyze_layer(
    sel: np.ndarray, E: int, budgets, holdout: float, seeds: int, strategies
) -> dict:
    T, n = sel.shape
    valid = (sel != E).sum(1)
    n_sel = f"{int(valid.min())}" if valid.min() == valid.max() else f"{int(valid.min())}-{int(valid.max())}"
    cut = int(round(T * (1.0 - holdout)))
    cut = max(1, min(T - 1, cut)) if T > 1 else T
    fit, ev = sel[:cut], sel[cut:] if cut < T else sel[:0]
    if ev.shape[0] == 0:
        ev = fit
    C_all = coactivation(sel, E)
    C_fit = coactivation(fit, E) if cut < T else C_all
    counts = np.diag(C_all).astype(np.float64)
    freq_fit = np.diag(C_fit).astype(np.float64)
    frac = counts / max(T, 1)
    top16 = float(np.sort(counts)[::-1][:16].sum() / max(counts.sum(), 1))
    out = {
        "tokens": int(T),
        "n_sel": n_sel,
        "slots": int(n),
        "fit_tokens": int(fit.shape[0]),
        "eval_tokens": int(ev.shape[0]),
        "freq_min": float(frac.min()),
        "freq_median": float(np.median(frac)),
        "freq_max": float(frac.max()),
        "gini": gini(counts),
        "top16_share": top16,
        "never_picked": int((counts == 0).sum()),
        "budgets": {},
        "coact": C_all,
    }
    for R in budgets:
        best = None
        for strat in strategies:
            assign = choose_replicated(C_fit, freq_fit, E, R, strat)
            s_fit = score(fit, assign)
            if best is None or s_fit["p_single"] > best[1]["p_single"]:
                best = (strat, s_fit, assign)
        strat, s_fit, assign = best
        s_ev = score(ev, assign)
        rep = np.flatnonzero(assign == 2)
        rnd = [score(ev, random_assign(E, rep, seed)) for seed in range(seeds)]
        rnd_mean = {k: float(np.mean([r[k] for r in rnd])) for k in ("p_single", "mean_cards", "load0")}
        evn = score(ev, even_assign(E, rep))
        out["budgets"][R] = {
            "strategy": strat,
            "held_out": s_ev,
            "in_sample": s_fit,
            "random": rnd_mean,
            "even": evn,
            "assignment": assign.tolist(),
            "replicated": rep.tolist(),
            "card0": int((assign == 0).sum()),
            "card1": int((assign == 1).sum()),
        }
    return out


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:5.1f}%"


def render_md(layers: dict, pooled: dict, budgets, args) -> str:
    L = []
    L.append("# MoE expert co-activation report\n")
    L.append(f"dump: `{args.dump}`  experts: {args.experts}  holdout: {args.holdout}  "
             f"random seeds: {args.seeds}  strategies: {','.join(args.strategies)}\n")
    L.append("## Pooled (token-weighted over layers)\n")
    L.append("| R | P(single card) held-out | in-sample | cards/token | load c0/c1 | "
             "random P1 | random cards/token | even P1 | layers |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for R in budgets:
        p = pooled[R]
        L.append(
            f"| {R} | {fmt_pct(p['p_single'])} | {fmt_pct(p['p_single_in'])} | "
            f"{p['mean_cards']:.3f} | {fmt_pct(p['load0'])}/{fmt_pct(1 - p['load0'])} | "
            f"{fmt_pct(p['random_p_single'])} | {p['random_mean_cards']:.3f} | "
            f"{fmt_pct(p['even_p_single'])} | {p['layers']} |"
        )
    L.append("")
    L.append(f"Pooled activation frequency: min {fmt_pct(pooled['freq_min'])}, median "
             f"{fmt_pct(pooled['freq_median'])}, max {fmt_pct(pooled['freq_max'])}, "
             f"Gini {pooled['gini']:.3f}, top-16 share {fmt_pct(pooled['top16_share'])}, "
             f"never-picked experts (sum over layers) {pooled['never_picked']}, "
             f"records {pooled['tokens']}.\n")
    L.append("## Per layer: activation frequency\n")
    L.append("| layer | records | n_sel | freq min | median | max | Gini | top-16 share | never picked |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for il in sorted(layers):
        d = layers[il]
        L.append(
            f"| {il} | {d['tokens']} | {d['n_sel']} | {fmt_pct(d['freq_min'])} | "
            f"{fmt_pct(d['freq_median'])} | {fmt_pct(d['freq_max'])} | {d['gini']:.3f} | "
            f"{fmt_pct(d['top16_share'])} | {d['never_picked']} |"
        )
    L.append("")
    L.append("## Per layer: partition quality (held-out tokens)\n")
    L.append("| layer | R | strategy | P(single) held-out | in-sample | cards/token | "
             "load c0/c1 | random P1 | random cards | even P1 | c0/c1/rep |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for il in sorted(layers):
        for R in budgets:
            b = layers[il]["budgets"][R]
            h, i, r, e = b["held_out"], b["in_sample"], b["random"], b["even"]
            L.append(
                f"| {il} | {R} | {b['strategy']} | {fmt_pct(h['p_single'])} | "
                f"{fmt_pct(i['p_single'])} | {h['mean_cards']:.3f} | "
                f"{fmt_pct(h['load0'])}/{fmt_pct(h['load1'])} | {fmt_pct(r['p_single'])} | "
                f"{r['mean_cards']:.3f} | {fmt_pct(e['p_single'])} | "
                f"{b['card0']}/{b['card1']}/{len(b['replicated'])} |"
            )
    L.append("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path, help="MEMRA_MOE_SEL_DUMP file (memra-moe-sel-v1)")
    ap.add_argument("--experts", type=int, default=288, help="routed experts per layer (E)")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS_DEFAULT)),
                    help="replication budgets R, comma separated")
    ap.add_argument("--holdout", type=float, default=0.3,
                    help="fraction of each layer's records (file order, the tail) scored, not fit")
    ap.add_argument("--seeds", type=int, default=5, help="random-halves baseline seeds")
    ap.add_argument("--strategies", default="freq,cut",
                    help="replicated-set strategies to try: freq (top activation), cut (top cut contribution)")
    ap.add_argument("--layers", default=None, help="comma-separated layer ids to keep (default all)")
    ap.add_argument("--min-records", type=int, default=64,
                    help="skip layers with fewer records than this")
    ap.add_argument("--out-md", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--save-coact", type=Path, default=None,
                    help="write the per-layer co-activation matrices as an .npz (key = layer id)")
    args = ap.parse_args()
    args.strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    keep = None if args.layers is None else {int(x) for x in args.layers.split(",")}

    groups = parse_dump(args.dump)
    E = args.experts
    for _layer_col, sel, _w in groups:
        if sel.max(initial=0) >= E:
            raise SystemExit(
                f"expert id {int(sel.max())} >= --experts {E}; pass the right bank size"
            )
    # A variable-n_sel dump is padded to the widest record with the sentinel E, which
    # every consumer below drops; file order is preserved within each layer only when the
    # dump has one n_sel (the common case), otherwise groups are concatenated by n_sel.
    # Weights ride along for a future weighted variant; counts answer this question.
    nmax = max(sel.shape[1] for _l, sel, _w in groups)
    layer_col = np.concatenate([g[0] for g in groups])
    sel_all = np.concatenate(
        [np.pad(g[1].astype(np.int64), ((0, 0), (0, nmax - g[1].shape[1])), constant_values=E)
         for g in groups]
    )
    layers: dict[int, dict] = {}
    for il in np.unique(layer_col):
        il = int(il)
        if keep is not None and il not in keep:
            continue
        s = sel_all[layer_col == il]
        if s.shape[0] < args.min_records:
            print(f"layer {il}: {s.shape[0]} records < --min-records, skipped", file=sys.stderr)
            continue
        layers[il] = analyze_layer(s, E, budgets, args.holdout, args.seeds, args.strategies)
        print(f"layer {il}: {s.shape[0]} records, n_sel {layers[il]['n_sel']}, "
              f"R=0 P1 held-out {layers[il]['budgets'][budgets[0]]['held_out']['p_single']:.3f}",
              file=sys.stderr)
    if not layers:
        raise SystemExit("no layer met --min-records; nothing to report")

    # Pooled: token-weighted over layers (each layer keeps its own partition).
    tot = sum(d["tokens"] for d in layers.values())
    pooled: dict = {
        "tokens": tot,
        "freq_min": min(d["freq_min"] for d in layers.values()),
        "freq_max": max(d["freq_max"] for d in layers.values()),
        "freq_median": float(np.median([d["freq_median"] for d in layers.values()])),
        "gini": float(sum(d["gini"] * d["tokens"] for d in layers.values()) / tot),
        "top16_share": float(sum(d["top16_share"] * d["tokens"] for d in layers.values()) / tot),
        "never_picked": int(sum(d["never_picked"] for d in layers.values())),
    }
    for R in budgets:
        ev_tot = sum(d["budgets"][R]["held_out"]["tokens"] for d in layers.values())
        fit_tot = sum(d["budgets"][R]["in_sample"]["tokens"] for d in layers.values())

        def wmean(key_path):
            acc = 0.0
            for d in layers.values():
                b = d["budgets"][R]
                v = b
                for k in key_path:
                    v = v[k]
                acc += v * b["held_out"]["tokens"]
            return acc / ev_tot

        pooled[R] = {
            "p_single": wmean(("held_out", "p_single")),
            "mean_cards": wmean(("held_out", "mean_cards")),
            "load0": wmean(("held_out", "load0")),
            "p_single_in": sum(d["budgets"][R]["in_sample"]["p_single"] * d["budgets"][R]["in_sample"]["tokens"]
                               for d in layers.values()) / fit_tot,
            "random_p_single": wmean(("random", "p_single")),
            "random_mean_cards": wmean(("random", "mean_cards")),
            "even_p_single": wmean(("even", "p_single")),
            "layers": len(layers),
        }

    md = render_md(layers, pooled, budgets, args)
    print(md)
    if args.out_md:
        args.out_md.write_text(md)
    if args.save_coact:
        np.savez_compressed(args.save_coact, **{str(il): d["coact"] for il, d in layers.items()})
    if args.out_json:
        slim = {
            "dump": str(args.dump),
            "experts": args.experts,
            "holdout": args.holdout,
            "seeds": args.seeds,
            "budgets": budgets,
            "pooled": {str(k): v for k, v in pooled.items()},
            "layers": {
                str(il): {k: v for k, v in d.items() if k != "coact"} for il, d in layers.items()
            },
        }
        args.out_json.write_text(json.dumps(slim, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
