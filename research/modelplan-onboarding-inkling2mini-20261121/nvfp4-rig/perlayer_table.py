#!/usr/bin/env python3
"""Per-layer table of the w:attn:L arms (lane F's harness, unpinned) with lane F's stats; writes gpu-run/perlayer-table.md."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanef_stats import stats
K = os.path.expanduser("~/.cache/inkling2mini/klsens/results/arms")
O = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu-run")
rows = ["| L | rawdoc KL real [95% CI] | rawdoc dNLL [95% CI] | chatwrap KL real [95% CI] | chatwrap dNLL [95% CI] |", "| --- | --- | --- | --- | --- |"]
out = {}
f = lambda c: "[%+.4f, %+.4f]" % tuple(c) if c else ""
for L in range(24):
    cells = [str(L)]
    for s in ("rawdoc", "chatwrap"):
        p = os.path.join(K, s, f"{s}_lanem_w_attn_L{L}.npz")
        if not os.path.exists(p):
            cells += ["pending", "pending"]
            continue
        st = stats(np.load(p))
        out[f"{s}_L{L}"] = st
        cells += ["%.5f %s" % (st["kl_real"]["mean"], f(st["kl_real"]["ci95"])), "%+.5f %s" % (st["dnll_real"]["mean"], f(st["dnll_real"]["ci95"]))]
    rows.append("| " + " | ".join(cells) + " |")
open(os.path.join(O, "perlayer-table.md"), "w").write("\n".join(rows) + "\n")
json.dump(out, open(os.path.join(O, "perlayer-stats.json"), "w"), indent=1)
print("\n".join(rows))
