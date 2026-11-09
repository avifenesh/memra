"""Paired top-1 check on a subset: positions where the native agrees with the vendor and a floor does not, and the
reverse, with the exact two-sided sign-test p-value (McNemar).

usage: paired_top1.py TAPE.pt NATIVE_DIR FLOOR.pt [--positions N]"""
import math
import sys

import torch

tape, nat, flo = sys.argv[1:4]
N = int(sys.argv[sys.argv.index("--positions") + 1]) if "--positions" in sys.argv else None
T = torch.load(tape, weights_only=False)
F = torch.load(flo, weights_only=False)
vi, vv, vl = T["top_ids"].long(), T["top_vals"].double(), T["lse"].double()
ni = torch.frombuffer(bytearray(open(nat + "/top_ids.i32", "rb").read()), dtype=torch.int32).long().view(-1, 32)
n = min(len(vl), len(F["lse"]), len(ni), N or len(vl))
p = (vv[:n] - vl[:n, None]).exp()
rest = (1 - p.sum(1)).clamp_min(1e-12)
ent = -(p * p.clamp_min(1e-300).log()).sum(1) - rest * rest.log()
for label, m in (("all", torch.ones(n, dtype=torch.bool)), ("entropy>1", ent > 1.0), ("top decile", ent >= ent.quantile(0.9))):
    a = ni[:n, 0][m] == vi[:n, 0][m]
    b = F["top_ids"][:n, 0].long()[m] == vi[:n, 0][m]
    only_n, only_f = int((a & ~b).sum()), int((~a & b).sum())
    k, t = min(only_n, only_f), only_n + only_f
    pv = min(1.0, 2 * sum(math.comb(t, i) for i in range(k + 1)) / 2 ** t) if t else 1.0
    print(f"{label}: {int(m.sum())} positions, native agrees {a.double().mean() * 100:.1f}%, floor {b.double().mean() * 100:.1f}%, "
          f"native-only {only_n}, floor-only {only_f}, sign test p={pv:.3f}")
