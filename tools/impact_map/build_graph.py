#!/usr/bin/env python3
"""Assemble the stage-0 fact base: one node per item (Rust + CUDA), typed edges, per-arch liveness.

Inputs (stage0/): items.jsonl, cu_items.jsonl, scip/{defs.jsonl,refs.tsv}, liveness.json,
fatbin120a/entries.tsv. Output: stage0/graph/{nodes.jsonl,edges.tsv,stats.json}

Edge kinds:
  ref      SCIP reference from inside src item to the item defining the symbol (precise)
  trait    reference to a trait method -> every impl of that method (conservative fan-out)
  name     name-based fallback for items SCIP did not index (cfg'd-out code)
  kernel   Rust item names a kernel by string literal -> CUDA __global__ item(s)
  ffi      Rust extern "C" decl -> CUDA extern "C" host function of the same name
  launch   CUDA host function launches a kernel (<<<>>>, cudaLaunchKernel, FuncSetAttribute)
  cucall   CUDA function body names another CUDA function (name-based, conservative)
  include  CUDA file includes another (file-level)
"""
import bisect
import collections
import json
import os
import re

ROOT = os.path.join(os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build")), "stage0")
CRATE_CU = "crates/memra-engine/"
OUT = os.path.join(ROOT, "graph")
os.makedirs(OUT, exist_ok=True)

nodes = {}
by_file = collections.defaultdict(list)
for l in open(os.path.join(ROOT, "items.jsonl")):
    r = json.loads(l)
    r["lang"] = "rust"
    nodes[r["id"]] = r
    by_file[r["file"]].append(r)
cu_macros = {}
for l in open(os.path.join(ROOT, "cu_items.jsonl")):
    r = json.loads(l)
    if r["kind"] == "pp":
        if r.get("macro"):
            cu_macros.setdefault(r["macro"], set()).update(r.get("body_idents", ()))
        continue
    r["file"] = CRATE_CU + r["file"]
    r["id"] = CRATE_CU + r["id"]
    r["lang"] = "cuda"
    nodes[r["id"]] = r
    by_file[r["file"]].append(r)

# innermost item lookup per file
index = {}
for f, items in by_file.items():
    items.sort(key=lambda r: (r["line_start"], -(r["line_end"])))
    index[f] = items


def innermost(f, line):
    best = None
    for r in index.get(f, ()):
        if r["line_start"] > line:
            break
        if r["line_start"] <= line <= r["line_end"] and r["kind"] not in ("impl", "trait"):
            if best is None or (r["line_end"] - r["line_start"]) <= (best["line_end"] - best["line_start"]):
                best = r
    return best


# faster innermost via per-file sorted starts + scan back; files are small enough in item count
def make_lookup():
    cache = {}

    def look(f, line):
        key = (f, line)
        if key in cache:
            return cache[key]
        r = innermost(f, line)
        cache[key] = r
        return r

    return look


look = make_lookup()

edges = collections.Counter()  # (src, dst, kind) -> count
edge_line = {}


def add(src, dst, kind, line=None):
    if src == dst or src is None or dst is None:
        return
    k = (src, dst, kind)
    edges[k] += 1
    if line is not None:
        lst = edge_line.setdefault(k, [])
        if len(lst) < 64 and line not in lst:
            lst.append(line)


# SCIP defs -> item
sym_item = {}
trait_methods = {}
for l in open(os.path.join(ROOT, "scip/defs.jsonl")):
    d = json.loads(l)
    if not d["f"]:
        continue
    if d["s"].endswith("/"):
        continue  # module / crate namespace symbols are not items
    it = look(d["f"], d["l"] + 1)
    if it is not None and it["kind"] != "use":
        sym_item.setdefault(d["s"], it["id"])

# trait method impls: quals like a::b::Trait::m (in trait) and a::<Trait for T>::m (impls)
impls_by_trait_method = collections.defaultdict(list)
trait_decl = {}
for r in nodes.values():
    if r["lang"] != "rust" or r["kind"] != "fn":
        continue
    m = re.search(r"<([A-Za-z0-9_]+) for [^>]+>::([A-Za-z0-9_]+)(#\d+)?$", r["qual"])
    if m:
        impls_by_trait_method[(m.group(1), m.group(2))].append(r["id"])
trait_names = {r["name"] for r in nodes.values() if r["lang"] == "rust" and r["kind"] == "trait"}
for r in nodes.values():
    if r["lang"] == "rust" and r["kind"] == "fn":
        parts = r["qual"].split("::")
        if len(parts) >= 2 and parts[-2] in trait_names:
            trait_decl[r["id"]] = (parts[-2], r["name"])

nref = 0
indexed_files = set()
for l in open(os.path.join(ROOT, "scip/refs.tsv")):
    f, line, sym = l.rstrip("\n").split("\t")
    indexed_files.add(f)
    dst = sym_item.get(sym)
    if dst is None:
        continue
    src = look(f, int(line) + 1)
    if src is None or src["kind"] == "use":
        continue
    nref += 1
    add(src["id"], dst, "ref", int(line) + 1)
    if dst in trait_decl:
        for impl in impls_by_trait_method.get(trait_decl[dst], ()):
            add(src["id"], impl, "trait", int(line) + 1)

# name-based fallback for rust items under cfgs SCIP did not see (memra_cutlass etc.) and for
# files SCIP did not index. Target = rust fns with that unique-ish name in the same crate.
fn_by_name = collections.defaultdict(list)
for r in nodes.values():
    if r["lang"] == "rust" and r["kind"] in ("fn", "struct", "enum", "const", "static", "type", "macro"):
        fn_by_name[r["name"]].append(r)


def crate_of(f):
    p = f.split("/")
    return p[1] if p[0] == "crates" else p[0]


scip_seen_src = {s for (s, d, k) in edges if k == "ref"}
nname = 0
for r in nodes.values():
    if r["lang"] != "rust" or r["test"]:
        continue
    inactive = any(c for c in r["cfg"] if re.search(r"memra_(cutlass|hopper_mma|sm100_tcgen05|portable_cuda)", c))
    if not inactive and r["file"] in indexed_files:
        continue
    for ident in r.get("idents", ()):
        for cand in fn_by_name.get(ident, ()):
            if cand["id"] != r["id"] and crate_of(cand["file"]) == crate_of(r["file"]) and len(fn_by_name[ident]) <= 6:
                add(r["id"], cand["id"], "name")
                nname += 1

# CUDA: kernels, extern C, launches, calls
cu_fns = [r for r in nodes.values() if r["lang"] == "cuda" and r["kind"] == "fn"]
cu_by_name = collections.defaultdict(list)
for r in cu_fns:
    cu_by_name[r["name"].split("::")[-1]].append(r)
globals_by_name = collections.defaultdict(list)
for r in cu_fns:
    if "__global__" in r.get("quals", []):
        globals_by_name[r["name"]].append(r)

# fatbin macro-generated kernels: names in the 120a entries not found by the scanner
entries = set()
p = os.path.join(ROOT, "fatbin120a/entries.tsv")
if os.path.exists(p):
    for l in open(p):
        entries.add(l.rstrip("\n").split("\t")[1])

nk = 0
unmatched_strings = collections.Counter()
for r in nodes.values():
    if r["lang"] != "rust":
        continue
    for s in r.get("strings", ()):
        if not s.startswith('"'):
            continue
        name = s.strip('"')
        if name in globals_by_name:
            for k in globals_by_name[name]:
                add(r["id"], k["id"], "kernel")
                nk += 1
        elif name in entries:
            unmatched_strings[name] += 1
# computed kernel names: format! templates ("fa_prefill_f32{}{}{hd_sfx}") and literal bases that
# get a suffix appended at runtime ("qmatvec_nvfp4_mmvq_b2" + "_pf"). Conservative fan-out, capped.
kernel_names = sorted(globals_by_name) + sorted(entries)
npat = 0
for r in nodes.values():
    if r["lang"] != "rust":
        continue
    for s_ in r.get("strings", ()):
        if not s_.startswith('"'):
            continue
        lit = s_.strip('"')
        hits = []
        if "{" in lit and re.fullmatch(r"[A-Za-z0-9_{}:.]+", lit):
            parts = re.split(r"\{[^}]*\}", lit)
            if sum(map(len, parts)) >= 8 and len(parts[0]) >= 4:
                rx = re.compile("^" + "[A-Za-z0-9_]*".join(map(re.escape, parts)) + "$")
                hits = [k for k in kernel_names if rx.match(k)]
        elif len(lit) >= 12 and re.fullmatch(r"[A-Za-z0-9_]+", lit):
            hits = [k for k in kernel_names if k.startswith(lit + "_")]
        cap = 150 if ("{" in lit and len(lit.split("{")[0]) >= 12) else 40
        if 0 < len(hits) <= cap:
            for k in hits:
                for g in globals_by_name.get(k, ()):
                    add(r["id"], g["id"], "kernel")
                    npat += 1
# FFI
nffi = 0
for r in nodes.values():
    if r["lang"] == "rust" and r["kind"] == "fn" and r.get("container") == "extern":
        for c in cu_by_name.get(r["name"], ()):
            if c.get("extern_c"):
                add(r["id"], c["id"], "ffi")
                nffi += 1
# launches and cu calls
for r in cu_fns:
    for k in r.get("launches", ()):
        for g in globals_by_name.get(k, ()) or cu_by_name.get(k, ()):
            add(r["id"], g["id"], "launch")
    idents = set(r.get("idents", ()))
    # expand macros used in the body (nested once): launches written inside #define bodies
    for ident in list(idents):
        for b in cu_macros.get(ident, ()):
            idents.add(b)
            idents.update(cu_macros.get(b, ()))
    for ident in idents:
        if ident == r["name"]:
            continue
        for c in cu_by_name.get(ident, ()):
            if c["id"] != r["id"] and len(cu_by_name[ident]) <= 12:
                add(r["id"], c["id"], "cucall")

with open(os.path.join(OUT, "edges.tsv"), "w") as f:
    for (s, d, k), n in edges.items():
        f.write(f"{s}\t{d}\t{k}\t{n}\t{','.join(map(str, edge_line.get((s, d, k), [])))}\n")
with open(os.path.join(OUT, "nodes.jsonl"), "w") as f:
    for r in nodes.values():
        f.write(json.dumps(r) + "\n")
kinds = collections.Counter(k for (_, _, k) in edges)
stats = dict(nodes=len(nodes), edges=len(edges), by_kind=kinds, scip_refs_mapped=nref, kernel_pattern_edges=npat,
             name_edges=nname, kernel_string_edges=nk, ffi_edges=nffi,
             fatbin_names_without_scanned_def=dict(unmatched_strings))
json.dump(stats, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=dict)
print(json.dumps(stats, indent=1, default=dict))
