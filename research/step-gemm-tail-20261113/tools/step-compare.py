#!/usr/bin/env python3
"""memra#1275 Step-3.7 check: judge base vs head per length (rules in step-check.sh's header).
usage: step-compare.py OUT_DIR [NAME...]   (prompt names, default 4100 8200; 4100b = 4,100 tokens;
exit 0 = PASS for every prompt)"""
import re
import struct
import sys
from pathlib import Path

BAND = 2.4  # argmax_margin_probe.rs header: maxdiff is "routinely 0.3-2.4" on MATCH runs
out = Path(sys.argv[1])


def logits(p: Path) -> list[float]:
    b = p.read_bytes()
    return list(struct.unpack(f"<{len(b) // 4}f", b))


def top2(v: list[float]) -> tuple[int, float, int, float]:
    i1 = max(range(len(v)), key=v.__getitem__)
    i2 = max((i for i in range(len(v)) if i != i1), key=v.__getitem__)
    return i1, v[i1], i2, v[i2]


def tokens(p: Path) -> list[int] | None:
    m = re.search(r"tokens: \[([0-9, ]*)\]", p.read_text(errors="replace").split("prompt tokens:")[-1])
    return [int(x) for x in m.group(1).split(",") if x.strip()] if m else None


def engaged(p: Path, n: int) -> bool:
    return f"[gemm-prime] ENGAGED t={n} base=0" in p.read_text(errors="replace")


ok_all = True
for n in sys.argv[2:] or ["4100", "8200"]:
    t = int(re.match(r"\d+", n).group())
    fails = []
    try:
        lb, lh = logits(out / f"{n}-base.logits"), logits(out / f"{n}-head.logits")
    except OSError as e:
        print(f"{n}: FAIL (logits missing: {e})")
        ok_all = False
        continue
    if len(lb) != len(lh) or not lb:
        fails.append(f"logit rows differ in length ({len(lb)} vs {len(lh)})")
    maxdiff = max(abs(a - b) for a, b in zip(lb, lh))
    bit_equal = lb == lh and all(struct.pack("<f", a) == struct.pack("<f", b) for a, b in zip(lb, lh))
    b1, bv1, b2, bv2 = top2(lb)
    h1, hv1, h2, hv2 = top2(lh)
    if b1 != h1:
        fails.append(f"argmax differs: base {b1} head {h1}")
    if maxdiff > BAND:
        fails.append(f"maxdiff {maxdiff:.4g} above the band {BAND}")
    tb, th = tokens(out / f"{n}-base-gen.log"), tokens(out / f"{n}-head-gen.log")
    if tb is None or th is None:
        fails.append("greedy tokens missing")
    elif tb[:16] != th[:16] or len(tb) < 16:
        k = next((i for i, (a, b) in enumerate(zip(tb, th)) if a != b), min(len(tb), len(th)))
        fails.append(f"greedy tokens differ at step {k}")
    for arm in ("base", "head"):
        for kind in ("pp", "gen"):
            if not engaged(out / f"{n}-{arm}-{kind}.log", t):
                fails.append(f"{arm} {kind}: no [gemm-prime] ENGAGED t={t}")
    delta = max(abs(lh[b1] - lb[b1]), abs(lh[b2] - lb[b2]))
    print(f"{n}: base argmax {b1} (margin {bv1 - bv2:.4g} over {b2}), head argmax {h1} (margin {hv1 - hv2:.4g} "
          f"over {h2}); last-row maxdiff {maxdiff:.4g} (band {BAND}){' BIT-EQUAL' if bit_equal else ''}; "
          f"cross-arm delta at base's contenders {delta:.4g}; greedy base {tb[:16] if tb else None} "
          f"head {th[:16] if th else None}")
    print(f"{n}: {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")
    ok_all &= not fails
print("VERDICT", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
