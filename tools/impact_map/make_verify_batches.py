#!/usr/bin/env python3
"""Build the skeptic batches: every narrowing claim (region or file default that excludes a family,
route or device, or every default launch) and every reported defect. Claims are grouped by file so
one skeptic reads one area; dynamic-trace misses are attached to the claims they implicate.

Usage: make_verify_batches.py RESULTS.json OBSERVED_MISSES.json OUT.json [narrow_per_batch] [defects_per_batch]
"""
import collections
import json
import sys

DEFECT = {"error", "broken_gate", "dead_code", "unreachable", "stale_flag", "stale_doc", "correctness", "policy", "build"}


def narrowing(x):
    return any(x.get(k) and "*" not in x.get(k) for k in ("families", "routes", "devices")) or bool(
        x.get("door") and x["door"].get("default_reached") is False)


def main():
    res = json.load(open(sys.argv[1]))
    misses = json.load(open(sys.argv[2])) if sys.argv[2] != "-" else {}
    n_per = int(sys.argv[4]) if len(sys.argv) > 4 else 45
    d_per = int(sys.argv[5]) if len(sys.argv) > 5 else 22
    claims = collections.defaultdict(list)
    for a in res["annotated"]:
        if not a or a.get("failed"):
            continue
        sid = a["slice"]
        for i, rg in enumerate(a.get("regions", [])):
            if narrowing(rg):
                c = {"claim": f"{sid}:R{i}", "file": rg["file"], "lines": [rg["line_start"], rg["line_end"]],
                     "label": rg.get("label", "")[:160], "predicate": rg.get("predicate", "")[:200],
                     "families": rg.get("families"), "routes": rg.get("routes"), "devices": rg.get("devices"),
                     "door": rg.get("door"), "basis": (rg.get("narrowing_basis") or "")[:300]}
                hint = misses.get(f"{sid}:R{i}")
                if hint:
                    c["trace_hint"] = hint
                claims[rg["file"]].append(c)
        for k, x in enumerate(a.get("exact_items", [])):
            claims[x.get("item", "").split("::")[0]].append({
                "claim": f"{sid}:X{k}", "item": x.get("item"), "lines": "whole item",
                "kind": "output-neutral", "basis": (x.get("why") or "")[:300],
                "check": "claim: changing this item can never change output bytes, speed, memory or serving policy (logging/metrics only); refute if any token of it feeds a numeric, timing, memory or scheduling decision"})
        for j, fd in enumerate(a.get("file_defaults", [])):
            if narrowing(fd):
                claims[fd["file"]].append({"claim": f"{sid}:F{j}", "file": fd["file"], "lines": "whole file (file default)",
                                           "families": fd.get("families"), "routes": fd.get("routes"), "devices": fd.get("devices"),
                                           "basis": (fd.get("narrowing_basis") or fd.get("note") or "")[:300]})
    # pack files into batches, keeping a file's claims together
    batches, cur = [], []
    for f in sorted(claims, key=lambda f: -len(claims[f])):
        cs = claims[f]
        while cs:
            room = n_per - len(cur)
            if room <= 0:
                batches.append(cur)
                cur, room = [], n_per
            cur += cs[:room]
            cs = cs[room:]
    if cur:
        batches.append(cur)
    defects = []
    for a in res["annotated"]:
        if not a or a.get("failed"):
            continue
        for i, f in enumerate(a.get("findings", [])):
            if (f.get("category") or "").lower() in DEFECT:
                defects.append({"id": f"{a['slice']}:F{i}", **{k: f.get(k) for k in ("category", "severity", "file", "line", "quote", "title", "detail", "fix")}})
    for key in ("packs", "routes", "devices"):
        for i, f in enumerate((res["scouts"].get(key) or {}).get("findings", [])):
            defects.append({"id": f"scout-{key}:F{i}", **{k: f.get(k) for k in ("category", "severity", "file", "line", "quote", "title", "fix")}})
    dbatches = [defects[i:i + d_per] for i in range(0, len(defects), d_per)]
    json.dump({"narrow": batches, "defects": dbatches}, open(sys.argv[3], "w"))
    print(f"narrowing claims {sum(map(len, batches))} in {len(batches)} batches; defects {len(defects)} in {len(dbatches)} batches")


if __name__ == "__main__":
    main()
