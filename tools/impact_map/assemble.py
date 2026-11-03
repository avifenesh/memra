#!/usr/bin/env python3
"""Assemble the impact map from the stage-0 graph and docs/impact-map/annotations.json.

Usage: assemble.py ANNOTATIONS.json OUT_MAP.json [OBSERVED.json]
Env:   MEMRA_IMPACT_WORK (stage-0 parent dir), MEMRA_IMPACT_REPO (the tree being mapped)

Model. A CELL is (family, route, device). Each item gets REACH = the cells that can execute it.
Every served root (an item no served item calls: mains, callbacks, entry points) starts with every
cell; reach flows along call edges to callees. An edge whose call line sits strictly inside an
annotated region carries that region's family/route/device restriction (a region reachable only
behind a default-OFF door carries none: no default cell). A trait-method edge to an impl carries
the families of the impl's self type (from the loader table). Rust cfg flags and CUDA per-arch
line liveness restrict devices. Diagnostic and gate binaries are not served roots. An item no
root reaches takes its file default, or every cell: unknown is always wide, never empty.

Annotations were made on an older tree; each region is re-anchored through its item: when the
item's token hash is unchanged the region shifts with the item; otherwise its first evidence
quote is searched inside the item; a region that cannot be re-anchored is dropped (wide).
"""
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys

WORK = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
REPO = os.environ.get("MEMRA_IMPACT_REPO", str(Path(__file__).resolve().parents[2]))
S0 = os.path.join(WORK, "stage0")

FAMILIES = ["deepseek_v4", "gemma4_dense", "gemma4_moe", "glm5_next", "glm_dsa", "hy3", "llama_dense", "mimo_v2",
            "minimax_m3", "nemotron_rnnt", "olmoe", "qwen3", "qwen35", "qwen35_moe", "qwen3_moe", "qwen3_tts",
            "qwen4_exp", "step35", "whisper"]
ROUTES = ["plain", "batch", "graph", "prime", "chunked_prefill", "mtp", "dflash", "eagle", "ngram", "glm5_spec", "tp",
          "pp", "ep", "spill", "vision", "audio_asr", "audio_tts", "embed", "rerank", "constrained", "prefix_cache",
          "tools", "http_api"]
DEVICES = ["rtx5090", "rtxpro6000", "gb10", "b200", "h100", "sm89", "cpu"]
# GB10 runs the 120a build today (detect_arch maps compute capability 12.1 to 120a).
DEVICE_OF_VARIANT = {"120a": ["rtx5090", "rtxpro6000", "gb10"], "100a": ["b200"], "90a": ["h100"], "89": ["sm89"]}
CFG_DEVICES = {"memra_hopper_mma": {"h100"}, "memra_sm100_tcgen05": {"b200"}, "memra_portable_cuda": {"h100", "sm89"}}

_lines = {}
# a static's use that mutates it: atomics, cells, locks, maps, or assignment to a static mut
WRITE = re.compile(r"^\s*(\.(store|set|get_or_init|get_or_try_init|swap|fetch_\w+|compare_exchange\w*|lock|write|borrow_mut|insert|push|replace|take|clear|with)\b|\s*[-+*/|&^]?=[^=])")


def file_lines(f):
    if f not in _lines:
        try:
            _lines[f] = open(os.path.join(REPO, f), errors="replace").read().split("\n")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            _lines[f] = None
    return _lines[f]


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().strip("`")


class Vocab:
    def __init__(self, families, routes, devices):
        self.families, self.routes, self.devices = list(families), list(routes), list(devices)
        self.fi = {f: i for i, f in enumerate(self.families)}
        self.ri = {r: i for i, r in enumerate(self.routes)}
        self.di = {d: i for i, d in enumerate(self.devices)}
        self.NR, self.ND = len(self.routes), len(self.devices)
        self.ALL = (1 << (len(self.families) * self.NR * self.ND)) - 1

    def _sel(self, vals, index):
        if not vals or "*" in vals:
            return list(index.values())
        out = []
        for v in vals:
            if v not in index:
                return list(index.values())  # an unknown label widens
            out.append(index[v])
        return out

    def mask(self, fams=None, routes=None, devs=None):
        m = 0
        D = self._sel(devs, self.di)
        dm = 0
        for d in D:
            dm |= 1 << d
        for f in self._sel(fams, self.fi):
            for r in self._sel(routes, self.ri):
                m |= dm << ((f * self.NR + r) * self.ND)
        return m

    def summarize(self, m):
        if m == self.ALL:
            return "*"
        out = {}
        for f, fname in enumerate(self.families):
            for r, rname in enumerate(self.routes):
                chunk = (m >> ((f * self.NR + r) * self.ND)) & ((1 << self.ND) - 1)
                if chunk:
                    out.setdefault(fname, {})[rname] = [d for i, d in enumerate(self.devices) if chunk >> i & 1]
        return out


def main():
    ann = json.load(open(sys.argv[1]))
    out_path = sys.argv[2]
    verified_types = ann.get("verified_types")  # None = not verified yet (trial builds)
    tables = ann.get("tables", {})
    routes = list(ROUTES) + [r["id"] for r in tables.get("routes_vocab", []) if r["id"] not in ROUTES]
    families = list(FAMILIES) + [x["id"] for x in tables.get("extra_families", []) if x["id"] not in FAMILIES]
    V = Vocab(families, routes, DEVICES)

    nodes = {}
    for l in open(os.path.join(S0, "graph/nodes.jsonl")):
        r = json.loads(l)
        nodes[r["id"]] = r
    stats = collections.Counter()

    # ---- re-anchor regions to this tree ----
    regions = collections.defaultdict(list)
    for rg in ann.get("regions", []):
        it = nodes.get(rg["anchor_item"])
        if it is None:
            stats["regions_lost_item"] += 1
            continue
        if it["hash"] == rg["anchor_hash"]:
            shift = it["line_start"] - rg["anchor_a"]
        else:
            q = norm(rg["evidence"][0]["quote"]) if rg.get("evidence") else ""
            ls = file_lines(it["file"]) or []
            want = rg["evidence"][0]["line"] - rg["anchor_a"] if rg.get("evidence") else None
            hits = [n for n in range(it["line_start"], min(it["line_end"], len(ls)) + 1) if q and q in norm(ls[n - 1])]
            if not hits or want is None:
                stats["regions_lost_quote"] += 1
                continue
            best = min(hits, key=lambda n: abs((n - it["line_start"]) - want))
            shift = best - rg["evidence"][0]["line"]
        r2 = dict(rg, a=rg["a"] + shift, b=rg["b"] + shift)
        if not (it["line_start"] <= r2["a"] and r2["b"] <= it["line_end"]):
            stats["regions_lost_bounds"] += 1
            continue
        regions[it["file"]].append(r2)
        stats["regions"] += 1
    for f in regions:
        regions[f].sort(key=lambda r: (r["a"], -r["b"]))
    file_defaults = {fd["file"]: fd for fd in ann.get("file_defaults", [])}

    def region_at(f, line):
        best = None
        for rg in regions.get(f, ()):
            if rg["a"] > line:
                break
            if rg["a"] < line <= rg["b"] and rg.get("kind") != "none":
                if best is None or rg["b"] - rg["a"] <= best["b"] - best["a"]:
                    best = rg
        return best

    def region_mask(rg):
        if rg.get("door") and rg["door"].get("default_reached") is False:
            return 0
        return V.mask(rg.get("families"), rg.get("routes"), rg.get("devices"))

    # ---- devices per item ----
    live = json.load(open(os.path.join(S0, "liveness.json")))
    live_lines = collections.defaultdict(lambda: collections.defaultdict(set))
    for variant, d in live.items():
        if variant not in DEVICE_OF_VARIANT:
            continue
        for u in d["units"]:
            for p in u["passes"].values():
                if not isinstance(p, dict) or "error" in p:
                    continue
                for f, rs in p.items():
                    s = live_lines["crates/memra-engine/" + f][variant]
                    for a_, b_ in rs:
                        s.update(range(a_, b_ + 1))

    def device_mask(r):
        if r["lang"] == "cuda":
            hit = set()
            for variant, lines in live_lines.get(r["file"], {}).items():
                if any(n in lines for n in range(r["line_start"], r["line_end"] + 1)):
                    hit.update(DEVICE_OF_VARIANT[variant])
            return V.mask(None, None, sorted(hit)) if hit else 0
        devs = set(DEVICES)
        for c in r.get("cfg", []):
            if c == "memra_cutlass" or c.startswith("all(memra_cutlass"):
                return 0  # MEMRA_CUTLASS is a build-time door: no default cell
            for k, ds in CFG_DEVICES.items():
                if c == k:
                    devs &= ds
                elif c == f"not({k})":
                    devs -= ds
        return V.mask(None, None, sorted(devs)) if devs else 0

    dev_mask = {i: device_mask(r) for i, r in nodes.items()}

    # ---- trait impl families, served binaries ----
    type_fams = collections.defaultdict(set)
    for f in tables.get("families", []):
        for t in f.get("engine_types") or []:
            type_fams[re.split(r"[<\s]", t.split("::")[-1])[0]].add(f["family"])
    for ld in tables.get("loader_dispatch", []):
        t = re.split(r"[<\s]", (ld.get("engine_type") or "").split("::")[-1])[0]
        for fam in ld.get("families") or []:
            if fam in V.fi:
                type_fams[t].add(fam)
    # an engine type the loader table builds only for some families: its methods can only run for
    # them (an instance exists only after that loader ran), whoever calls them
    ctor_fams = collections.defaultdict(set)
    type_names = {r["name"] for r in nodes.values() if r["lang"] == "rust" and r["kind"] in ("struct", "enum")}
    for ld in tables.get("loader_dispatch", []):
        fams = ld.get("families") or ["*"]
        et = ld.get("engine_type") or ""
        enum_paths = set(re.findall(r"([A-Z][A-Za-z0-9_]+)::", et))  # `Enum::Variant` names a variant, not the type
        for t in (set(re.findall(r"[A-Z][A-Za-z0-9_]+", et)) & type_names) - enum_paths:
            ctor_fams[t].update(["*"] if "*" in fams else [f for f in fams if f in V.fi] or ["*"])
    for t in list(ctor_fams):
        if verified_types is not None and t not in verified_types:
            del ctor_fams[t]  # a type restriction is a narrowing claim: honored once verified
    type_mask = {t: V.mask(sorted(f)) for t, f in ctor_fams.items() if "*" not in f and f}
    node_mask = {}
    for i, r in nodes.items():
        if r["lang"] != "rust":
            continue
        for t, msk in type_mask.items():
            if f"::{t}::" in r["qual"] or f" for {t}>" in r["qual"]:
                node_mask[i] = msk
                break
    stats["type_restricted_items"] = len(node_mask)
    served_files = {nodes[b["main_item"]]["file"] for b in tables.get("binaries", []) if b.get("main_item") in nodes
                    and re.search(r"serv", b.get("serves") or "", re.I)}

    def offline(i):
        r = nodes[i]
        if r.get("test"):
            return True
        f = r["file"]
        if "/tests/" in f or "/examples/" in f or "/benches/" in f or f.endswith("/build.rs"):
            return True
        if "/src/bin/" in f or f.endswith("/main.rs"):
            return f not in served_files
        return False

    # ---- edges ----
    adj = collections.defaultdict(list)
    fan_in = collections.Counter()
    writes = collections.defaultdict(set)  # mutable static -> items that write it
    for l in open(os.path.join(S0, "graph/edges.tsv")):
        s, d, k, n, lines = l.rstrip("\n").split("\t")
        if s not in nodes or d not in nodes or offline(s):
            continue
        fan_in[d] += 1
        src = nodes[s]
        ls = [int(x) for x in lines.split(",") if x]
        if k == "kernel" and not ls:
            name = nodes[d]["name"]
            fl = file_lines(src["file"]) or []
            ls = [i for i in range(src["line_start"], min(src["line_end"], len(fl)) + 1) if f'"{name}"' in fl[i - 1]]
        cond = 0
        for ln in ls:
            rg = region_at(src["file"], ln)
            cond |= region_mask(rg) if rg else V.ALL
        if not ls:
            cond = V.ALL
        if k == "trait":
            m_ = re.search(r"<[A-Za-z0-9_]+ for ([A-Za-z0-9_]+)", nodes[d]["qual"])
            if m_ and type_fams.get(m_.group(1)):
                cond &= V.mask(sorted(type_fams[m_.group(1)]))
        adj[s].append((d, cond))
        if nodes[d]["kind"] == "static" and ls:
            fl = file_lines(src["file"]) or []
            name = nodes[d]["name"]
            for ln in ls:
                text = fl[ln - 1] if 0 < ln <= len(fl) else ""
                if name in text and WRITE.search(text.split(name, 1)[1][:120]):
                    writes[d].add(s)
                    break
    for de in ann.get("dynamic_edges", []):
        if de.get("from_item") in nodes and de.get("to") in nodes:
            adj[de["from_item"]].append((de["to"], V.ALL))
            fan_in[de["to"]] += 1

    # ---- propagate from wide roots ----
    reach = collections.defaultdict(int)
    work = collections.deque()
    for i, r in nodes.items():
        if offline(i) or r["kind"] in ("use", "impl", "trait") or fan_in[i]:
            continue
        m = dev_mask[i] & node_mask.get(i, V.ALL)
        if m:
            reach[i] = m
            work.append(i)
            stats["roots"] += 1
    while work:
        x = work.popleft()
        rx = reach[x]
        for d, cond in adj.get(x, ()):
            m = rx & cond & dev_mask[d] & node_mask.get(d, V.ALL)
            if m & ~reach[d]:
                reach[d] |= m
                work.append(d)
                stats["propagation_steps"] += 1

    # ---- observed execution (kernel traces): a kernel that ran for a cell has that cell ----
    observed_n = 0
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        by_name = collections.defaultdict(list)
        for i, r in nodes.items():
            if r["lang"] == "cuda" and r["kind"] == "fn":
                by_name[r["name"]].append(i)
        launchers = collections.defaultdict(set)
        for src_, outs in adj.items():
            for d_, _ in outs:
                if nodes[d_]["lang"] == "cuda" and nodes[src_]["lang"] == "rust":
                    launchers[d_].add(src_)
        for ob in json.load(open(sys.argv[3])).get("observed", []):
            m_ = V.mask([ob["family"]], [ob["route"]], [ob["device"]])
            for k_ in by_name.get(ob["kernel"], ()):
                for t_ in {k_} | launchers.get(k_, set()):
                    if m_ & ~reach[t_]:
                        reach[t_] |= m_
                        observed_n += 1
    stats["observed_widenings"] = observed_n

    # ---- data flow through process-global state ----
    # A static is shared by every model the process serves; the item that WRITES it changes what
    # every reader sees, so a change in the writer reaches the static's whole reach (the union of
    # its referrers). Execution reach (used for propagation) is untouched; only the writer's
    # change reach widens.
    callers = collections.defaultdict(set)
    for src_, outs in adj.items():
        for d_, _ in outs:
            callers[d_].add(src_)
    change_extra = collections.defaultdict(int)
    for st, ws in writes.items():
        r_st = reach.get(st, 0)
        for w in ws:
            change_extra[w] |= r_st
            # a small setter's callers choose the value it stores: they are writers too
            nw = nodes[w]
            if nw["kind"] == "fn" and nw["line_end"] - nw["line_start"] <= 15:
                for c in callers.get(w, ()):
                    change_extra[c] |= r_st
    stats["static_writers"] = len(change_extra)

    exact = {x["item"] for x in ann.get("exact_items", [])}
    sets, items = {}, {}
    for i, r in nodes.items():
        if offline(i) or r["kind"] in ("use",):
            continue
        m = reach.get(i, 0) | change_extra.get(i, 0)
        src = "propagated"
        if not m:
            fd = file_defaults.get(r["file"])
            if fd:
                m = V.mask(fd.get("families"), fd.get("routes"), fd.get("devices")) & dev_mask[i]
                src = "file_default"
            if not m:
                m = dev_mask[i] & node_mask.get(i, V.ALL)
                src = "unreached_wide" if m else "no_default_build"
            stats["items_" + src] += 1
        # effects only ADD: an effect label is not a verified claim, so dropping numeric, speed,
        # memory or policy needs a skeptic-verified output-neutral item; CUDA device code changes
        # output and time by construction
        claimed = set()
        for rg in regions.get(r["file"], ()):
            if rg["a"] <= r["line_end"] and rg["b"] >= r["line_start"]:
                claimed.update(rg.get("effects") or [])
        if i in exact:
            eff = {"observability"}
        elif r["lang"] == "cuda":
            eff = {"numeric", "speed"} | (claimed - {"none", "observability"})
        else:
            eff = {"numeric", "speed", "memory", "policy"}
        if m not in sets:
            sets[m] = len(sets)
        items[i] = {"f": r["file"], "a": r["line_start"], "b": r["line_end"], "h": r["hash"], "k": r["kind"],
                    "r": sets[m], "e": sorted(eff), "s": src}
    reach_sets = [None] * len(sets)
    for m, idx in sets.items():
        reach_sets[idx] = V.summarize(m)
    regions_out = []
    for f, rs in sorted(regions.items()):
        for rg in rs:
            regions_out.append({k: rg.get(k) for k in ("file", "a", "b", "kind", "families", "routes", "devices", "formats",
                                                       "door", "effects", "claim", "verdict")})
    stats["items"] = len(items)
    stats["reach_sets"] = len(sets)
    data = {"schema": "memra-impact-map-v1",
            "source_commit": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
            "annotated_at": ann.get("annotated_at"),
            "vocab": {"families": V.families, "routes": V.routes, "devices": V.devices},
            "reach_sets": reach_sets, "items": items, "regions": regions_out, "stats": dict(stats)}
    with open(out_path, "w") as w:
        json.dump(data, w, separators=(",", ":"), sort_keys=True)
    print(json.dumps(dict(stats), indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()
