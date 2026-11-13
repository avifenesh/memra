"""The draft head's parity verdict: memra against the vendor GPU tape, banded by the vendor's own CPU-vs-GPU floor on the
same inputs (two draft_cmp.py outputs, the same cases).

usage: draft_gate.py MEMRA_CMP.json FLOOR_CMP.json OUT.json [OUT.md]
The lane's rule (layer gates, prefill and decode gates): a tensor passes when memra's median relative L2 against the
vendor GPU run is at most twice the floor's plus 1e-4; routing passes when its expert-set agreement is within one point
of the floor's; the distribution passes when memra's mean KL is at most twice the floor's and its draft-token agreement
at most one point under the floor's (two binomial standard errors when the group is small); the confidence score passes
on twice the floor's mean absolute difference plus 1e-4."""
import json, math, sys

mem, flo = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
assert mem["ids"] == flo["ids"], "the two comparisons cover different cases"
M, F = mem["pooled"], flo["pooled"]
rows, ok_all = [], True


def add(name, kind, m, f, band, ok):
    global ok_all
    ok_all &= ok
    rows.append({"metric": name, "kind": kind, "memra": m, "floor": f, "band": band, "pass": ok})


for k in M:
    m, f = M[k], F.get(k)
    if f is None:
        continue
    if k.startswith("route"):
        band = f["mean"] - 0.01
        add(k, "expert-set agreement, mean", m["mean"], f["mean"], f">= {band:.4f}", m["mean"] >= band)
    elif k in ("draft_token_agree", "draft_token_agree_same_ctx"):
        n = m["n"]
        se = math.sqrt(max(f["mean"] * (1 - f["mean"]), 1e-9) / n)
        band = f["mean"] - max(0.01, 2 * se)
        add(k, f"agreement with the vendor GPU draft, n={n}", m["mean"], f["mean"], f">= {band:.4f}", m["mean"] >= band)
    elif k.startswith("kl"):
        band = 2 * f["mean"]
        add(k, f"mean KL(vendor GPU || x), n={m['n']}", m["mean"], f["mean"], f"<= {band:.3e}", m["mean"] <= band)
    elif k.startswith("conf"):
        band = 2 * f["mean"] + 1e-4
        add(k, "mean absolute difference", m["mean"], f["mean"], f"<= {band:.3e}", m["mean"] <= band)
    elif k.endswith("absmax"):
        band = 2 * f["median"] + 1e-4
        add(k, "median of the per-row max abs difference", m["median"], f["median"], f"<= {band:.3e}", m["median"] <= band)
    else:
        band = 2 * f["median"] + 1e-4
        add(k, f"median rel L2, n={m['n']}", m["median"], f["median"], f"<= {band:.3e}", m["median"] <= band)

json.dump({"pass": ok_all, "cases": mem["ids"], "rows": rows}, open(sys.argv[3], "w"), indent=1)
lines = ["| metric | what | memra | vendor CPU floor | band | verdict |", "|---|---|---|---|---|---|"]
for r in rows:
    fm = lambda x: f"{x:.3e}" if abs(x) < 1e-2 else f"{x:.4f}"
    lines.append(f"| {r['metric']} | {r['kind']} | {fm(r['memra'])} | {fm(r['floor'])} | {r['band']} | {'PASS' if r['pass'] else 'FAIL'} |")
md = "\n".join(lines) + "\n"
if len(sys.argv) > 4:
    open(sys.argv[4], "w").write(md)
print(md)
print("DRAFT-GATE", "PASS" if ok_all else "FAIL", f"{sum(r['pass'] for r in rows)}/{len(rows)}")
sys.exit(0 if ok_all else 1)
