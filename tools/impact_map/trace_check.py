#!/usr/bin/env python3
"""Soundness check of the map against real kernel traces.

For each traced run (nsys `cuda_gpu_kern_sum` CSV + its family, route and device), every executed
memra kernel must have that (family, route, device) cell in its map reach. A kernel that ran but
whose reach lacks the cell is a SOUNDNESS miss: the map would have skipped a test that could see a
regression. Library kernels (cuBLAS/cuBLASLt/CUTLASS internals) have no memra item and are listed
separately.

Usage: trace_check.py --map map.json --index traces/index.tsv --device rtx5090 [--json]
index.tsv rows: name<TAB>family<TAB>route ; the CSV is <dir>/<name>_cuda_gpu_kern_sum.csv
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delta  # noqa: E402


# a request on a route also runs these phases (a run-gen "plain" run primes the prompt first;
# speculative routes verify through the plain trunk)
IMPLIED = {
    "plain": ("prime", "chunked_prefill", "graph"),
    "mtp": ("plain", "prime", "chunked_prefill", "graph"),
    "dflash": ("plain", "prime", "chunked_prefill", "graph"),
    "eagle": ("plain", "prime", "chunked_prefill", "graph"),
    "batch": ("plain", "prime", "chunked_prefill", "graph"),
}


def base_name(kname):
    """'void foo<int, 4>(float*, ...)' -> 'foo'; 'ns::bar(...)' -> 'bar'; extern C names pass through."""
    k = kname.strip()
    k = re.sub(r"^(void|static|__global__)\s+", "", k)
    k = k.split("(")[0]
    depth, out = 0, []
    for ch in k:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).split("::")[-1].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--device", default="rtx5090")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = delta.Map(json.loads(Path(a.map).read_text()))
    kern = {}
    for i, it in m.items.items():
        if it["f"].startswith(delta.CU_PREFIX + "cu/") and it["k"] == "fn":
            kern.setdefault(i.split("::")[-1].split("#")[0], []).append(i)
    rows = []
    idx = Path(a.index)
    for line in idx.read_text().splitlines():
        if not line.strip():
            continue
        name, fam, route = line.split("\t")
        p = idx.parent / f"{name}_cuda_gpu_kern_sum.csv"
        if not p.exists():
            rows.append({"run": name, "error": "no kernel summary"})
            continue
        executed, library, misses, ok = set(), set(), [], 0
        with open(p) as f:
            for r in csv.DictReader(f):
                kn = r.get("Name") or r.get("name") or ""
                b = base_name(kn)
                if b in executed or b in library:
                    continue
                if b not in kern:
                    library.add(b)
                    continue
                executed.add(b)
                cells = {(fam, r, a.device) for r in (route, *IMPLIED.get(route, ()))}
                if any(cells & (m.item_cells(i) or set()) for i in kern[b]):
                    ok += 1
                else:
                    misses.append(b)
        rows.append({"run": name, "family": fam, "route": route, "memra_kernels": len(executed), "in_reach": ok,
                     "misses": sorted(misses), "library_kernels": len(library)})
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    for r in rows:
        if "error" in r:
            print(f"{r['run']}: {r['error']}")
            continue
        print(f"{r['run']:<16} {r['family']}/{r['route']}: {r['in_reach']}/{r['memra_kernels']} executed memra kernels have the cell in reach; "
              f"{len(r['misses'])} miss(es){': ' + ', '.join(r['misses'][:12]) if r['misses'] else ''}; {r['library_kernels']} library kernels")


if __name__ == "__main__":
    main()
