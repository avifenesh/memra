# Unblind + verdict: joins blind/scores-t{T}.json (judge output, keyed by anon id)
# with blind/mapping-t{T}.json and raw/rows.jsonl, applies the PRE-REGISTERED
# comparison rule from RUBRIC.md, and prints per-arm tables + TTFT/accept banks.
import json, os, statistics as st

BANK = os.path.dirname(os.path.abspath(__file__))


def med(v):
    return st.median(v) if v else None


for T in (4, 8):
    sp = os.path.join(BANK, "blind", "scores-t%d.json" % T)
    mp = os.path.join(BANK, "blind", "mapping-t%d.json" % T)
    if not (os.path.exists(sp) and os.path.exists(mp)):
        print("t%d: scores or mapping missing, skipped" % T)
        continue
    scores = json.load(open(sp))
    mapping = json.load(open(mp))
    per = {}
    for anon, sc in scores.items():
        m = mapping[anon]
        per.setdefault(m["arm"], []).append(
            dict(anon=anon, sample=m["sample"], total=sc["total"], dq=sc.get("dq"),
                 items=sc.get("items"), valid=m["valid"],
                 invalid_reason=m["invalid_reason"]))
    print("\n===== TURN %d =====" % T)
    stats = {}
    for arm in ("cold", "gemm", "walk"):
        rows = sorted(per.get(arm, []), key=lambda r: r["sample"])
        sc = [r["total"] for r in rows if r["valid"]]
        dqs = [r for r in rows if r["dq"]]
        inv = [r for r in rows if not r["valid"]]
        stats[arm] = dict(n=len(sc), median=med(sc), lo=min(sc) if sc else None,
                          hi=max(sc) if sc else None, dq=len(dqs), invalid=len(inv))
        print("%-5s n=%d median=%s min=%s max=%s dq=%d invalid=%d | %s"
              % (arm, len(sc), med(sc), min(sc) if sc else "-", max(sc) if sc else "-",
                 len(dqs), len(inv),
                 " ".join("s%d=%.1f%s" % (r["sample"], r["total"],
                                          "*" if r["dq"] else "") for r in rows)))
    c, g, w = stats["cold"], stats["gemm"], stats["walk"]
    if c["median"] is not None:
        spread = c["hi"] - c["lo"]
        print("COLD self-spread = %.1f" % spread)
        for name, a in (("gemm", g), ("walk", w)):
            if a["median"] is None:
                print("%s: no valid rows" % name)
                continue
            indist = (abs(a["median"] - c["median"]) <= spread
                      and a["median"] >= c["median"] - 1.0)
            degrade = (a["median"] < c["median"] - spread) or (a["dq"] - c["dq"] >= 3)
            print("%s vs cold: median diff %+0.1f -> %s"
                  % (name, a["median"] - c["median"],
                     "INDISTINGUISHABLE" if indist else
                     ("DEGRADES" if degrade else "GRAY ZONE")))

# TTFT / accept bank (never part of the quality verdict)
rowsf = os.path.join(BANK, "raw", "rows.jsonl")
if os.path.exists(rowsf):
    agg = {}
    for line in open(rowsf):
        r = json.loads(line)
        if not r.get("valid"):
            continue
        k = (r["arm"], r["turn"])
        sp = r.get("spec") or {}
        agg.setdefault(k, dict(ttft=[], acc=[])).setdefault("ttft", [])
        agg[k]["ttft"].append(r.get("ttft"))
        a = sp.get("acceptance_rate") or sp.get("acceptance")
        if a is not None:
            agg[k]["acc"].append(a)
    print("\n===== TTFT / accept bank (not a verdict input) =====")
    for k in sorted(agg):
        t = [x for x in agg[k]["ttft"] if x is not None]
        a = agg[k]["acc"]
        print("%-5s t%d ttft median=%.3f min=%.3f max=%.3f | accept median=%s n=%d"
              % (k[0], k[1], med(t), min(t), max(t),
                 ("%.3f" % med(a)) if a else "-", len(t)))
