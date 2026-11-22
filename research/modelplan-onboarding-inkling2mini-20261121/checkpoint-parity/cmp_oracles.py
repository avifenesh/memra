#!/usr/bin/env python3
"""Pairwise spread of memra-checkpoint-oracle-v1 tapes on the probe: max |a-b| over the vocab, the
top-1 and the KL of their softmaxes (f64). usage: cmp_oracles.py A.tsv B.tsv [...]"""
import itertools
import sys

import numpy as np


def read(p):
    lg = {}
    head = {}
    for line in open(p):
        f = line.rstrip("\n").split("\t")
        if f[0] == "logit":
            lg[int(f[1])] = int(f[2], 16)
        else:
            head[f[0]] = f[1]
    v = np.array([lg[i] for i in range(len(lg))], dtype=np.uint32).view(np.float32).astype(np.float64)
    return head, v


def kl(p_logits, q_logits):
    lp = p_logits - np.logaddexp.reduce(p_logits)
    lq = q_logits - np.logaddexp.reduce(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum())


tapes = {p: read(p) for p in sys.argv[1:]}
for p, (h, v) in tapes.items():
    top = np.argsort(-v)[:5]
    print(f"{h['engine']}: class {h['numeric_class']} vocab {v.size} argmax {top[0]} top5 {top.tolist()} margin {v[top[0]] - v[top[1]]:.4f}")
for a, b in itertools.combinations(tapes, 2):
    (ha, va), (hb, vb) = tapes[a], tapes[b]
    d = np.abs(va - vb)
    print(f"{ha['engine']} vs {hb['engine']}: max_abs {d.max():.6g} at {int(d.argmax())} (ref {va[d.argmax()]:.4f}), mean_abs {d.mean():.4g}, "
          f"argmax {'same' if va.argmax() == vb.argmax() else 'DIFFERENT'}, KL(a||b) {kl(va, vb):.4g}")
