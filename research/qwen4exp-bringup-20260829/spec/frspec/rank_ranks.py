#!/usr/bin/env python3
"""qwen4_exp FR-Spec ranks: the independent oracle ranker, the blends, and the free
accept/tok-s estimator that decides whether a width can win before any GPU time is spent.

Runs on the rig with HF `tokenizers` 0.23.1 (the version the bringup pretokenizer work used
as oracle) against the VENDOR tokenizer.json, so the house tool's counts (memra's own
tokenizer machine, `frspec-rank <artifact_dir>`) have something outside memra to be equal
to. Read the module docstring of extract_corpus.py for the corpus.

Rank law, replicated from memra `memra_gguf::d2t::rank_top_n`: sort EVERY id in the
tokenizer id space by (count desc, id asc) and take the first N — so unseen ids pad in
ascending id order behind the seen ones.

Classes written:
  q4e-ranks-sxc32768.txt      agentic emission class (the serving-default candidate)
  q4e-ranks-prose-32768.txt   prose/thinkoff emission class
  q4e-ranks-mixed-32768.txt   0.5*normfreq(agentic) + 0.5*normfreq(prose), same rank law
  q4e-ranks-ogblend-32768.txt 0.5*normfreq(mtp10 OWN-GEN) + 0.5*normfreq(mixed corpus):
                              this model's own 404,851-token emission distribution for the
                              head, the corpus for the discovery the own-gen run never had.

The estimator (why this lane is not another 28-GPU-hour own-gen run):

  tok/s_trim / tok/s_full = (1 - A*q) / (1 - H*(1 - N/V))

  q  out-of-set share of the target's own emitted tokens (per shape, measured on the
     banked chains of THIS model);
  A  per-shape amplification of q into mean-accept-length loss, CALIBRATED on the two
     banked negative receipts (mtp10 N=11,854, where q and the measured accept-len loss
     are both known: one out-of-set pick also derails the rest of that carrier chain, so
     A > 1 on long fixed-K chains and ~0.5 at the ship policy where chains are short);
  H  full-vocab draft-head share of the round, solved from each shape's own
     draft_ms_share full-vs-trim pair (head cost is linear in rows);
  V  248,320 lm_head rows.

Calibration is printed with the prediction: a receipt that cannot reproduce the two
measurements it was fitted on is not evidence about a third.
"""
import glob
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.dirname(HERE)
CORPUS = os.path.join(HERE, "corpus")
TOPN = 32768
VOCAB_ROWS = 248320  # lm_head rows (config.json); the id space is smaller, see below
CLASSES = ("agentic", "prose")


# --------------------------------------------------------------------------- counting
def count_class(tok, name):
    """One encode per FILE, whole-file text — the house tool's read law
    (`frspec-rank`: `read_to_string` then `tok.encode(&text, false)`).

    Encoding line-by-line instead costs 8.6% more tokens on this corpus (BPE cannot merge
    across the newline that joins two turns), which would make the oracle and the house
    tool disagree for a reason that has nothing to do with either tokenizer. Measured, not
    assumed: same 5 MB slice, memra-artifact / memra-vendor / HF-artifact / HF-vendor all
    returned exactly 705,552 tokens whole-file.
    """
    from collections import Counter
    counts = Counter()
    total = 0
    files = sorted(glob.glob(os.path.join(CORPUS, name, "*.txt")))
    if not files:
        sys.exit(f"no corpus files under {os.path.join(CORPUS, name)} — run extract_corpus.py")
    per_file = {}
    for path in files:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        ids = tok.encode(text, add_special_tokens=False).ids
        counts.update(ids)
        total += len(ids)
        per_file[os.path.basename(path)] = len(ids)
    return dict(counts), total, per_file


# ------------------------------------------------------------------------- rank law
def rank_top_n(counts, id_space, n):
    """memra_gguf::d2t::rank_top_n — (count desc, id asc) over the WHOLE id space."""
    order = sorted(range(id_space), key=lambda i: (-counts.get(i, 0), i))
    return order[:n]


def norm(counts, total):
    return {i: c / total for i, c in counts.items()}


def blend(a, b, id_space, n):
    keys = set(a) | set(b)
    mixed = {i: 0.5 * a.get(i, 0.0) + 0.5 * b.get(i, 0.0) for i in keys}
    # Same law, on a float score: (score desc, id asc), unseen ids pad ascending.
    order = sorted(range(id_space), key=lambda i: (-mixed.get(i, 0.0), i))
    return order[:n], mixed


# ------------------------------------------------------- banked chains (held-out text)
def banked_chains():
    """Every real chain THIS model emitted that is banked in the lane, by shape.

    `# rep0_full` / `# rep0_*` lines of the interleaved A/B receipts and `# ids` lines of
    the sampled probes. These are the target's own committed tokens, which is exactly the
    text a trimmed draft has to be able to propose.
    """
    shapes = {}
    pats = [
        (os.path.join(SPEC, "mtp10", "ship", "ab-spec-k5-ship-*.tsv"), r"ship-(\w+)\.tsv"),
        (os.path.join(SPEC, "mtp10", "trim", "ab-trim-k5-*.tsv"), r"trim-(\w+?)(?:-ship|-k5)?\.tsv"),
        (os.path.join(SPEC, "mtp10", "shapes", "ab-spec-k5-rc-*.tsv"), r"rc-(\w+)\.tsv"),
        (os.path.join(SPEC, "mtp11", "ab-defer-k5-m11-*.tsv"), r"m11-(\w+)\.tsv"),
        (os.path.join(SPEC, "mtp10", "ship", "spec-sampled-k5-ship-*.tsv"), r"ship-(\w+)\.tsv"),
    ]
    for pat, shape_re in pats:
        for path in sorted(glob.glob(pat)):
            m = re.search(shape_re, os.path.basename(path))
            if not m:
                continue
            shape = m.group(1)
            if shape.endswith("_ship"):
                shape = shape[:-5]
            txt = open(path).read()
            # The sampled receipt now tags its chain with the arm (this lane's instrument
            # change), so accept BOTH forms: a regex that silently stops matching is how a
            # whole input class disappears from an estimator without failing anything.
            for cm in re.finditer(r"# (?:rep0_\w+|ids)\t(?:arm=\w+\t)?([0-9,]{40,})", txt):
                ids = [int(x) for x in cm.group(1).split(",")]
                shapes.setdefault(shape, []).append((os.path.basename(path), ids))
    return shapes


def out_of_set(chains, idset):
    tot = miss = 0
    for _, ids in chains:
        tot += len(ids)
        miss += sum(1 for t in ids if t not in idset)
    return miss / tot if tot else 0.0, tot


# --------------------------------------------------------------------------- estimator
# mtp10's measured N=11,854 rows: (accept_len full, accept_len trim, draft_share full,
# draft_share trim) per shape — research/qwen4exp-bringup-20260829/spec/mtp10/trim/.
MTP10 = {
    "raw": dict(len_full=5.12, len_trim=4.06, d_full=0.18, d_trim=0.09, measured=0.8824),
    "thinkon": dict(len_full=1.92, len_trim=1.83, d_full=0.12, d_trim=0.06, measured=1.0144),
    "thinkoff": dict(len_full=3.56, len_trim=2.59, d_full=0.16, d_trim=0.07, measured=0.905),
}
N_MTP10 = 11854


def head_share(d_full, d_trim, n=N_MTP10):
    """Draft-head share of the round, solved from the full/trim draft-share pair.
    d_full - d_trim = H*(1 - n/V)  =>  H = (d_full - d_trim) / (1 - n/V)."""
    return (d_full - d_trim) / (1.0 - n / VOCAB_ROWS)


def predict(H, A, q, n):
    return (1.0 - A * q) / (1.0 - H * (1.0 - n / VOCAB_ROWS))


def main():
    from tokenizers import Tokenizer

    tok_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    id_space = tok.get_vocab_size(with_added_tokens=True)
    print(f"[oracle] tokenizer {tok_path} id_space={id_space}", flush=True)

    # The pools are LIVE (agent sessions land continuously), so "reproducible from the
    # pools by the extractor" is only true against a pinned pool state: measured drift over
    # 32 minutes moved 1,766/32,768 rank positions and swapped exactly one id in and out of
    # the blend class's top-32,768. The artifact is therefore pinned by its OWN sha256, and
    # the corpus that produced it is pinned here by file hash.
    import hashlib
    corpus_sha = {}
    for cname in CLASSES:
        for path in sorted(glob.glob(os.path.join(CORPUS, cname, "*.txt"))):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            corpus_sha[f"{cname}/{os.path.basename(path)}"] = h.hexdigest()
    report = {"tokenizer": tok_path, "id_space": id_space, "lm_head_rows": VOCAB_ROWS,
              "topN": TOPN, "classes": {}, "corpus_sha256": corpus_sha}
    counts = {}
    freqs = {}
    for name in CLASSES:
        c, total, per_file = count_class(tok, name)
        counts[name] = c
        freqs[name] = norm(c, total)
        report["classes"][name] = {"tokens": total, "distinct": len(c), "per_file": per_file}
        print(f"[oracle] {name}: {total} tokens, {len(c)} distinct ids", flush=True)

    # ---- own-gen distribution (this model's OWN 404,851 emitted tokens, mtp10) --------
    og_path = os.path.join(SPEC, "mtp10", "ranks-owngen-big.txt.gz")
    og_counts = {}
    with gzip.open(og_path, "rt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            i, c = line.split("\t")[:2]
            og_counts[int(i)] = int(c)
    og_total = sum(og_counts.values())
    og_freq = norm(og_counts, og_total)
    print(f"[owngen] {og_total} tokens, {len(og_counts)} distinct ids ({og_path})", flush=True)
    report["owngen"] = {"tokens": og_total, "distinct": len(og_counts), "src": os.path.basename(og_path)}

    # ---- the four classes ------------------------------------------------------------
    sets = {}
    sets["sxc32768"] = rank_top_n(counts["agentic"], id_space, TOPN)
    sets["prose-32768"] = rank_top_n(counts["prose"], id_space, TOPN)
    sets["mixed-32768"], mixed_score = blend(freqs["agentic"], freqs["prose"], id_space, TOPN)
    sets["ogblend-32768"], _ = blend(og_freq, mixed_score, id_space, TOPN)

    for name, ids in sets.items():
        assert len(ids) == TOPN == len(set(ids)), name
        assert max(ids) < VOCAB_ROWS, (name, max(ids))
        out = os.path.join(HERE, f"q4e-ranks-{name}.txt")
        with open(out, "w") as f:
            f.write("".join(f"{i}\n" for i in ids))
        print(f"[write] {out}", flush=True)

    # ---- discovery curve + coverage of the model's OWN emission mass -----------------
    disc = []
    for n in (1024, 2048, 4096, 8192, 16384, 32768, 65536):
        row = {"topN": n}
        for cname, ids in sets.items():
            s = set(ids[:n])
            row[f"owngen_mass_{cname}"] = round(
                sum(c for i, c in og_counts.items() if i in s) / og_total, 5)
        disc.append(row)
    report["discovery"] = disc
    report["distinct_ids"] = {k: report["classes"][k]["distinct"] for k in CLASSES}

    # ---- held-out out-of-set share on banked chains ----------------------------------
    chains = banked_chains()
    print(f"[chains] shapes: " + ", ".join(
        f"{k}={sum(len(i) for _, i in v)}tok/{len(v)}chains" for k, v in sorted(chains.items())),
        flush=True)
    mtp10_ids = set(int(l.split("\t")[0]) for l in gzip.open(og_path, "rt")
                    if not l.startswith("#") and l.strip())
    q_tab = {}
    for shape, ch in sorted(chains.items()):
        row = {}
        q11854, tot = out_of_set(ch, mtp10_ids)
        row["tokens"] = tot
        row["q_mtp10_n11854"] = round(q11854, 5)
        for cname, ids in sets.items():
            for n in (8192, 16384, 32768):
                row[f"q_{cname}_n{n}"] = round(out_of_set(ch, set(ids[:n]))[0], 5)
        q_tab[shape] = row
    report["out_of_set"] = q_tab

    # ---- calibrate A, then predict ---------------------------------------------------
    cal = {}
    for shape in ("raw", "thinkon", "thinkoff"):
        m = MTP10[shape]
        if shape not in q_tab:
            continue
        q = q_tab[shape]["q_mtp10_n11854"]
        loss = 1.0 - m["len_trim"] / m["len_full"]
        A = loss / q if q else 0.0
        H = head_share(m["d_full"], m["d_trim"])
        refit = predict(H, A, q, N_MTP10)
        cal[shape] = {"q_at_n11854": q, "accept_len_loss": round(loss, 5),
                      "A": round(A, 4), "H": round(H, 5),
                      "refit_ratio": round(refit, 4),
                      "measured_ratio": m["measured"],
                      "refit_error": round(refit - m["measured"], 4)}
    report["calibration"] = cal

    # Fine width grid: the head saving SHRINKS with N while coverage rises, so the knee is
    # interior and has to be solved, not inherited from whatever the corpus discovered.
    grid = (4096, 8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304)
    pred = {}
    for shape, c in cal.items():
        ch = chains[shape]
        for cname, ids in sets.items():
            for n in grid:
                q = out_of_set(ch, set(ids[:n]))[0]
                pred.setdefault(shape, {})[f"{cname}_n{n}"] = round(
                    predict(c["H"], c["A"], q, n), 4)
    report["prediction"] = pred
    report["prediction_grid"] = list(grid)
    report["prediction_note"] = (
        "predict() holds round cost fixed apart from the head, so it is EXACT where the "
        "draft window is fixed (raw, fixed K=5) or already at the adaptive floor (thinkon, "
        "len 1.92 at k_lo=1) and CONSERVATIVE where the window sits above the floor "
        "(thinkoff refits 0.80 against a measured 0.905: a lower-accept arm also drafts a "
        "smaller window, so part of the accept loss comes back as a cheaper round)."
    )

    with gzip.open(os.path.join(HERE, "counts-cache.json.gz"), "wt") as f:
        json.dump({k: {str(i): c for i, c in v.items()} for k, v in counts.items()}, f)
    with open(os.path.join(HERE, "oracle-report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({"calibration": cal, "prediction": pred}, indent=1))
    print(json.dumps(disc, indent=1))


if __name__ == "__main__":
    sys.exit(main())
