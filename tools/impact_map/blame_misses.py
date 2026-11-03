#!/usr/bin/env python3
"""For every trace miss (a kernel that ran for a cell its map reach lacks), find the annotated
regions that cut the cell on the way: regions whose lines hold a call edge into the kernel or
into one of its callers (up to 3 levels) and whose sets exclude the cell. -> {claim: hint}.

Usage: blame_misses.py MAP.json ANNOTATIONS.json OBSERVED.json > misses.json
"""
import collections
import json
import os
import sys

WORK = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
S0 = os.path.join(WORK, "stage0")


def main():
    m = json.load(open(sys.argv[1]))
    ann = json.load(open(sys.argv[2]))
    obs = json.load(open(sys.argv[3]))
    nodes = {}
    for l in open(os.path.join(S0, "graph/nodes.jsonl")):
        r = json.loads(l)
        nodes[r["id"]] = r
    callers = collections.defaultdict(list)  # dst -> [(src, lines)]
    for l in open(os.path.join(S0, "graph/edges.tsv")):
        s, d, k, n, lines = l.rstrip("\n").split("\t")
        callers[d].append((s, [int(x) for x in lines.split(",") if x]))
    regs = collections.defaultdict(list)
    for rg in ann["regions"]:
        regs[rg["file"]].append(rg)
    sets = m["reach_sets"]

    def has(item, cell):
        it = m["items"].get(item)
        if not it:
            return True
        s = sets[it["r"]]
        if s == "*":
            return True
        f, r, d = cell
        return d in s.get(f, {}).get(r, [])

    def excludes(rg, cell):
        f, r, d = cell
        if rg.get("door") and rg["door"].get("default_reached") is False:
            return True
        for key, v in (("families", f), ("routes", r), ("devices", d)):
            vals = rg.get(key) or ["*"]
            if "*" not in vals and v not in vals:
                return True
        return False

    by_name = collections.defaultdict(list)
    for i, r in nodes.items():
        if r["lang"] == "cuda" and r["kind"] == "fn":
            by_name[r["name"]].append(i)
    implied = {"plain": ("prime", "chunked_prefill", "graph"), "mtp": ("plain", "prime", "chunked_prefill", "graph")}
    out = collections.defaultdict(list)
    for ob in obs["observed"]:
        cells = [(ob["family"], r, ob["device"]) for r in (ob["route"], *implied.get(ob["route"], ()))]
        for k in by_name.get(ob["kernel"], ()):
            if any(has(k, c) for c in cells):
                continue
            cell = cells[0]
            frontier, seen = [k], {k}
            for depth in range(3):
                nxt = []
                for x in frontier:
                    for src, lines in callers.get(x, ()):
                        if src in seen or src not in nodes:
                            continue
                        seen.add(src)
                        nxt.append(src)
                        f = nodes[src]["file"]
                        for rg in regs.get(f, ()):
                            if any(rg["a"] < ln <= rg["b"] for ln in lines) and excludes(rg, cell):
                                out[rg["claim"].split(".")[0]].append(
                                    f"kernel {ob['kernel']} RAN in a {cell[0]} {ob['route']} run on {cell[2]} "
                                    f"(nsys trace), reached through {nodes[src]['qual'].split('::')[-1]} line {lines[0]}")
                frontier = nxt
    json.dump({c: sorted(set(v))[:4] for c, v in out.items()}, sys.stdout, indent=1)
    print(f"{len(out)} claims implicated", file=sys.stderr)


if __name__ == "__main__":
    main()
