#!/usr/bin/env python3
"""Stage-0 fact sheets for the annotation agents, plus mechanical findings candidates.

Writes ~/.cache/memra-impact-map/slices/<SID>/{facts.tsv,hints.tsv,candidates.json,files.txt}
and stage0/{vocab.json,candidates-global.json,kernels.tsv}.
"""
import collections
import datetime
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import rust_items  # noqa: E402

HOME = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
WT = os.environ.get("MEMRA_IMPACT_REPO", str(Path(__file__).resolve().parents[2]))
S0 = os.path.join(HOME, "stage0")
SL = os.path.join(HOME, "slices")
TODAY = datetime.date(2026, 11, 3)

E = "crates/memra-engine/src/"
CU = "crates/memra-engine/cu/"
SV = "crates/memra-server/src/"

# slice id -> (title, [file globs or (file, lo, hi) line-range tuples])
SLICES = {
    "S01": ("engine lib.rs, first half (Engine core, module/kernel resolution, launch helpers)", [(E + "lib.rs", 1, None)]),
    "S02": ("engine lib.rs, second half", [(E + "lib.rs", None, 10 ** 9)]),
    "S03": ("hybrid_forward.rs, first half", [(E + "hybrid_forward.rs", 1, None)]),
    "S04": ("hybrid_forward.rs, second half + hybrid_forward/", [(E + "hybrid_forward.rs", None, 10 ** 9), E + "hybrid_forward/*.rs"]),
    "S05": ("hybrid loader, model, decode, decode_batch, prime graph, parallel",
            [E + f for f in ("hybrid.rs", "model.rs", "decode.rs", "decode_batch.rs", "qwen_prime_graph.rs", "hyper.rs", "parallel.rs", "graph*.rs", "decode_*.rs")]),
    "S06": ("DeepSeek-V4 GPU model (dsv4_gpu.rs)", [E + "dsv4_gpu.rs"]),
    "S07": ("DeepSeek-V4 rest: engine dsv4_* + FFI, gguf dsv4 forward/decode, server dsv4 route",
            [E + "dsv4_*.rs", "crates/memra-gguf/src/dsv4_*.rs", SV + "dsv4_*.rs"]),
    "S08": ("qwen4exp GPU model and siblings", [E + "qwen4exp*.rs"]),
    "S09": ("multi-card: tp, tp_ar, tp_transport, glm5_tp*, pp", [E + f for f in ("tp.rs", "tp_*.rs", "glm5_tp*.rs", "pp.rs", "pp_*.rs", "ep*.rs")]),
    "S10": ("speculative core: spec.rs, spec/, eagle", [E + "spec.rs", E + "spec/*.rs", E + "eagle*.rs", E + "mtp*.rs"]),
    "S11": ("spec routes: dflash, glm_spec, gemma_spec, glm5 decode graph, mimo mtp",
            [E + f for f in ("dflash*.rs", "glm_spec.rs", "glm_spec/*.rs", "gemma_spec*.rs", "glm5_decode_graph.rs", "glm5_*.rs", "mimo_mtp*.rs")]),
    "S12": ("memory tiers: tier_transfer, moe_cache, banked_residency, spill, cpu experts, pinned host + memra-tier",
            [E + f for f in ("tier_transfer*.rs", "moe_cache*.rs", "banked_residency*.rs", "banked_residency/*.rs", "spill*.rs", "cpu_experts*.rs", "pinned_host*.rs", "model_memory*.rs")]
            + ["crates/memra-tier/src/**"]),
    "S13": ("engine FFI shims, vision, mimo (non-MTP), kda, and every other engine file", [E + "*.rs"]),  # remainder
    "S14": ("CUDA matvec/GEMM: qmatvec, qmatvec_gemm, mmq_*, fp8/f16 prefill, headers",
            [CU + f for f in ("qmatvec*.cu", "mmq_*.cu", "mmq_*.cuh", "mma_tile.cuh", "fp8_*.cu", "f16_prefill.cu", "wgmma_common.cuh", "sm100_blockscale_layout.cuh", "cutlass_fp4_sm120.cu")]),
    "S15": ("CUDA attention and core kernels: flash_attn, mla_attn, fa3, fattn_vendor, kernels.cu",
            [CU + f for f in ("flash_attn*.cu", "mla_attn.cu", "fa3_prefill.cu", "fattn_vendor.cu", "kernels.cu", "memra_pdl_chain.cuh")]),
    "S16": ("CUDA model-specific and support kernels: dsv4, hybrid, kda, moe, router, spec_sample, tp_ar, mimo, tier_receipt",
            [CU + "*"]),  # remainder of cu/
    "S17": ("server worker.rs, first half", [(SV + "worker.rs", 1, None)]),
    "S18": ("server worker.rs, second half", [(SV + "worker.rs", None, 10 ** 9)]),
    "S19": ("server lib.rs and the rest of memra-server", [SV + "**"]),
    "S20": ("memra-gguf (plan, tensor contract, packs, manifest, config, source) + memra-cli",
            ["crates/memra-gguf/src/**", "crates/memra-cli/src/**"]),
    "S21": ("host crates: reference, tokenizer, sampling, kv, lanes, runtime, validate, net-guard, probe",
            ["crates/memra-reference/src/**", "crates/memra-tokenizer/src/**", "crates/memra-sampling/src/**", "crates/memra-kv/src/**",
             "crates/memra-lanes/src/**", "crates/memra-runtime/src/**", "crates/memra-validate/src/**", "crates/memra-net-guard/src/**", "crates/memra-probe/src/**"]),
    "S22": ("build and gates: build.rs files, battery binaries (kernel_check, run_gen, run_spec), release battery tools, validation_plan, fast-gate, #1030 tools",
            ["crates/memra-engine/build.rs", "crates/memra-server/build.rs", "crates/memra-probe/build.rs",
             E + "bin/kernel_check.rs", E + "bin/run_gen.rs", E + "bin/run_spec.rs", E + "bin/argmax_margin_probe.rs", E + "bin/tok_parity.rs",
             "tools/release-battery.sh", "tools/release-roster.tsv", "tools/kernel-check-*.cells", "tools/serve-smoke.sh", "tools/validation_plan.py",
             "tools/fast-gate/*", "tools/release_qualification.py", "tools/qualify-release.py", "tools/check_hardware_gate.py", "tools/hooks/pre-push",
             "tools/argmax-margin-gate.sh", "tools/release-coverage.py"]),
}

HINTS = {
    "arch": re.compile(r"cuda_arch|BUILT_CUDA_ARCH|compute_cap|sm_count|multiProcessorCount|\"1[02][01]a\"|\"90a\"|\"89\"|is_blackwell|is_hopper|card_class|device_name|total_mem|vram|cfg!\(memra_|cfg\(memra_|__CUDA_ARCH__|MEMRA_PORTABLE_CUDA|MEMRA_HOPPER_MMA|sm_?1[02]0|sm_?90|GB10|B200|H100|PRO 6000|5090"),
    "env": re.compile(r"MEMRA_[A-Z0-9_]+"),
    "family": re.compile(r"Arch::|\bfamily\b|arch_name|general\.architecture|\"(qwen|gemma|glm|deepseek|step|mimo|hy3|minimax|olmoe|llama|nemotron|whisper)|is_moe|n_expert|n_routed|\bmla\b|\bgdn\b|\bdsa\b|\bswa\b|sliding|is_dsv4|is_glm|is_gemma|is_qwen|is_step|is_mimo"),
    "route": re.compile(r"dflash|dspark|\bdraft|verify|\bmtp\b|_mtp|mtp_|\btp_|_tp\b|\bpp_|_pp\b|pipeline|tensor_parallel|vision|image|audio|asr|tts|embed|rerank|graph|capture|prime|chunk|eagle|ngram|batch"),
    "format": re.compile(r"GgmlType::|qtype|Q4_K|Q5_K|Q6_K|Q8_0|Q4_0|IQ4_XS|IQ3_S|NVFP4|nvfp4|\bfp8\b|FP8|bf16|kv_fmt|kfmt|vfmt|e4m3|mxfp4|MXFP4"),
    "plan": re.compile(r"\bplan\.|ModelPlan|PlanOp|capabilit|manifest|has_op|ExecutionManifest"),
}


def git_files():
    out = os.popen(f"git -C {WT} ls-files").read().split("\n")
    return [f for f in out if f and (f.startswith("crates/") or f.startswith("tools/"))]


def main():
    files = git_files()
    nodes = {}
    for l in open(os.path.join(S0, "graph/nodes.jsonl")):
        r = json.loads(l)
        nodes[r["id"]] = r
    fan_in = collections.Counter()
    fan_out = collections.Counter()
    callers = collections.defaultdict(list)
    callees = collections.defaultdict(list)
    for l in open(os.path.join(S0, "graph/edges.tsv")):
        s, d, k, n, line = l.rstrip("\n").split("\t")
        if s in nodes and nodes[s].get("test"):
            continue
        fan_in[d] += 1
        fan_out[s] += 1
        callers[d].append((s, k))
        callees[s].append((d, k))

    by_file = collections.defaultdict(list)
    for r in nodes.values():
        by_file[r["file"]].append(r)

    # split big files at the item boundary nearest the midpoint
    split_at = {}
    for path in (E + "lib.rs", E + "hybrid_forward.rs", SV + "worker.rs"):
        items = sorted((r for r in by_file[path] if r["kind"] not in ("impl", "trait")), key=lambda r: r["line_start"])
        # cut at the runtime-lines midpoint (test items excluded), on an item boundary
        rt = [r for r in items if not r.get("test")]
        half = sum(r["line_end"] - r["line_start"] + 1 for r in rt) / 2
        acc = 0
        cut = rt[-1]["line_start"]
        for r in rt:
            acc += r["line_end"] - r["line_start"] + 1
            if acc >= half:
                cut = r["line_end"] + 1
                break
        split_at[path] = cut

    assigned = {}
    assigned_files = set()
    slice_files = collections.defaultdict(list)  # sid -> [(file, lo, hi)]
    order = [s for s in SLICES if s not in ("S13", "S16", "S19")] + ["S13", "S16", "S19"]
    for sid in order:
        title, pats = SLICES[sid]
        for p in pats:
            if isinstance(p, tuple):
                path, lo, hi = p
                cut = split_at[path]
                lo = 1 if lo == 1 else cut
                hi = cut - 1 if hi is None else 10 ** 9
                slice_files[sid].append((path, lo, hi))
                assigned[(path, lo)] = sid
                assigned_files.add(path)
                continue
            for f in files:
                if f in assigned_files:
                    continue
                if "/bin/" in f and sid not in ("S22",):
                    continue
                if "/tests/" in f or "/benches/" in f or "/examples/" in f:
                    continue
                if fnmatch.fnmatch(f, p) or (p.endswith("/**") and f.startswith(p[:-2])):
                    slice_files[sid].append((f, 1, 10 ** 9))
                    assigned[(f, 1)] = sid
                    assigned_files.add(f)

    # unassigned runtime files (report)
    runtime_rs = [f for f in files if f.startswith("crates/") and f.endswith(".rs") and "/tests/" not in f
                  and "/bin/" not in f and "/examples/" not in f and "/benches/" not in f and not f.endswith("build.rs")]
    cu_files = [f for f in files if f.startswith(CU)]
    covered = {f for v in slice_files.values() for (f, _, _) in v}
    missing = [f for f in runtime_rs + cu_files if f not in covered]

    # ---- candidates (global) ----
    cands = []
    flags_rows = {}
    flags_text = open(os.path.join(WT, "docs/FLAGS.md")).read()
    for m in re.finditer(r"^\| `(MEMRA_[A-Z0-9_*]+)` \|(.*)$", flags_text, re.M):
        flags_rows[m.group(1)] = m.group(2)
    for name, row in flags_rows.items():
        m = re.search(r"decide-by:?\s*(20\d\d-\d\d-\d\d)", row)
        if m and datetime.date.fromisoformat(m.group(1)) < TODAY:
            ln = flags_text[:flags_text.find(f"| `{name}` |")].count("\n") + 1
            cands.append(dict(kind="stale_flag", file="docs/FLAGS.md", line=ln,
                              note=f"{name}: decide-by {m.group(1)} has passed ({(TODAY - datetime.date.fromisoformat(m.group(1))).days} days); AGENTS.md says remove without a receipt"))
    # env reads per item
    env_items = collections.defaultdict(set)
    for r in nodes.values():
        if r["lang"] != "rust" or r.get("test"):
            continue
        for s in r.get("strings", ()):
            for m in re.finditer(r"MEMRA_[A-Z0-9_]+", s):
                env_items[m.group(0)].add(r["id"])
    # big functions
    for r in nodes.values():
        if r["kind"] == "fn" and not r.get("test") and "/bin/" not in r["file"]:
            n = r["line_end"] - r["line_start"] + 1
            if n >= 400:
                cands.append(dict(kind="big_fn", file=r["file"], line=r["line_start"], item=r["id"],
                                  note=f"{n}-line function, fan_in={fan_in[r['id']]}, fan_out={fan_out[r['id']]}"))
    # dead code: non-test rust fns with no incoming edge; cuda kernels with no launcher
    for r in nodes.values():
        if r.get("test") or "/bin/" in r["file"] or "/tests/" in r["file"]:
            continue
        if r["kind"] != "fn" or fan_in[r["id"]]:
            continue
        if r["lang"] == "rust":
            if r["name"] in ("main", "new", "default", "drop", "fmt", "from", "clone", "eq", "hash", "deref", "deref_mut", "index", "next", "into_iter") or "<" in r["qual"]:
                continue
            if r.get("container") == "extern":
                continue
        else:
            if "__global__" not in r.get("quals", []) and not r.get("extern_c"):
                continue
        cands.append(dict(kind="dead_candidate", file=r["file"], line=r["line_start"], item=r["id"],
                          note=f"no incoming reference in the stage-0 graph ({r['lang']} {r['kind']}); check pub API, macros, dynamic names"))
    # exact duplicate bodies (same token hash, different item, >= 25 lines)
    by_hash = collections.defaultdict(list)
    for r in nodes.values():
        if r["kind"] == "fn" and not r.get("test") and r["line_end"] - r["line_start"] >= 24:
            by_hash[(r["lang"], r["hash"])].append(r)
    for (lang, h), group in by_hash.items():
        if len(group) > 1:
            g = sorted(group, key=lambda r: r["id"])
            cands.append(dict(kind="exact_duplicate", file=g[0]["file"], line=g[0]["line_start"], item=g[0]["id"],
                              note="identical token stream: " + ", ".join(f"{x['file']}:{x['line_start']}" for x in g)))

    # near duplicates: token-shingle MinHash over rust and cuda fns >= 40 lines
    near = near_duplicates(nodes)
    for a, b, j in near:
        ra, rb = nodes[a], nodes[b]
        cands.append(dict(kind="near_duplicate", file=ra["file"], line=ra["line_start"], item=a,
                          note=f"~{j:.2f} shingle similarity with {rb['file']}:{rb['line_start']} ({rb['qual']})"))

    # arch checks per file (scatter)
    arch_sites = collections.Counter()
    for f in runtime_rs:
        try:
            for i, line in enumerate(open(os.path.join(WT, f), errors="replace"), 1):
                if HINTS["arch"].search(line) and not line.lstrip().startswith("//"):
                    arch_sites[f] += 1
        except FileNotFoundError:
            pass
    scattered = [(n, f) for f, n in arch_sites.items()]
    scattered.sort(reverse=True)
    for n, f in scattered[:40]:
        cands.append(dict(kind="arch_check_scatter", file=f, line=1, note=f"{n} device/arch-check lines in this file"))

    # KERNELS.md drift: symbols in KERNELS.md first column vs scanned kernels
    ktext = open(os.path.join(WT, "docs/KERNELS.md")).read()
    kdoc = set(re.findall(r"^\| `([A-Za-z0-9_]+)` \|", ktext, re.M))
    kern = {r["name"] for r in nodes.values() if r["lang"] == "cuda" and "__global__" in r.get("quals", [])}
    ext = {r["name"] for r in nodes.values() if r["lang"] == "cuda" and r.get("extern_c")}
    for s in sorted(kdoc - kern - ext):
        ln = ktext[:ktext.find(f"| `{s}` |")].count("\n") + 1
        cands.append(dict(kind="stale_doc", file="docs/KERNELS.md", line=ln, note=f"KERNELS.md row `{s}` names no scanned kernel or extern C symbol"))
    undocumented = sorted(k for k in kern if k not in kdoc)
    cands.append(dict(kind="stale_doc", file="docs/KERNELS.md", line=1,
                      note=f"{len(undocumented)} __global__ kernels have no KERNELS.md row (first 40: {', '.join(undocumented[:40])})"))

    os.makedirs(SL, exist_ok=True)
    json.dump(cands, open(os.path.join(S0, "candidates-global.json"), "w"), indent=0)

    # kernels table
    with open(os.path.join(S0, "kernels.tsv"), "w") as kf:
        kf.write("kernel\tfile\tline\trust_sites\n")
        for r in sorted((r for r in nodes.values() if r["lang"] == "cuda" and "__global__" in r.get("quals", [])), key=lambda r: r["id"]):
            sites = [s for s, k in callers[r["id"]] if k in ("kernel", "launch")]
            kf.write(f"{r['name']}\t{r['file']}\t{r['line_start']}\t{','.join(short(nodes, s) for s in sites[:8])}{'...' if len(sites) > 8 else ''}\n")

    # per-slice sheets
    summary = {}
    for sid in SLICES:
        d = os.path.join(SL, sid)
        os.makedirs(d, exist_ok=True)
        ranges = slice_files[sid]
        in_slice = []
        for f, lo, hi in ranges:
            for r in by_file.get(f, ()):
                if r["kind"] in ("use",) or r.get("test"):
                    continue
                if lo <= r["line_start"] <= hi:
                    in_slice.append(r)
        ids = {r["id"] for r in in_slice}
        lines = 0
        with open(os.path.join(d, "facts.tsv"), "w") as w:
            w.write("# id\tkind\tlines\tn\tfan_in\tfan_out\tkernels\tenv\tcallers_outside(top)\tcallees_outside(top)\n")
            for r in sorted(in_slice, key=lambda r: (r["file"], r["line_start"])):
                n = r["line_end"] - r["line_start"] + 1
                lines += n if r["kind"] not in ("impl", "trait", "type") else 0
                kern_names = sorted({nodes[x]["name"] for x, k in callees[r["id"]] if k in ("kernel", "launch") and x in nodes})
                envs = sorted({m for s in r.get("strings", ()) for m in re.findall(r"MEMRA_[A-Z0-9_]+", s)})
                outside_callers = [s for s, k in callers[r["id"]] if s not in ids]
                outside_callees = [x for x, k in callees[r["id"]] if x not in ids and k in ("ref", "trait", "ffi", "kernel", "launch", "cucall")]
                w.write("\t".join([
                    r["id"], r["kind"], f"L{r['line_start']}-{r['line_end']}", str(n), str(fan_in[r["id"]]), str(fan_out[r["id"]]),
                    ",".join(kern_names[:8]) + ("..." if len(kern_names) > 8 else ""),
                    ",".join(envs[:8]) + ("..." if len(envs) > 8 else ""),
                    ",".join(short(nodes, s) for s in top(outside_callers, 5)),
                    ",".join(short(nodes, s) for s in top(outside_callees, 5)),
                ]) + "\n")
        with open(os.path.join(d, "hints.tsv"), "w") as w:
            w.write("# file:line\tcategories\ttext\n")
            for f, lo, hi in ranges:
                try:
                    src = open(os.path.join(WT, f), errors="replace").read().split("\n")
                except FileNotFoundError:
                    continue
                for i, line in enumerate(src, 1):
                    if not (lo <= i <= hi):
                        continue
                    s = line.strip()
                    if s.startswith("//") or s.startswith("*") or s.startswith("///"):
                        continue
                    cats = [c for c, rx in HINTS.items() if rx.search(line)]
                    if cats:
                        w.write(f"{f}:{i}\t{','.join(cats)}\t{s[:150]}\n")
        with open(os.path.join(d, "files.txt"), "w") as w:
            for f, lo, hi in ranges:
                w.write(f"{f}\t{lo}\t{hi if hi < 10 ** 8 else 'end'}\n")
        sc = [c for c in cands if any(c["file"] == f and lo <= c["line"] <= hi for f, lo, hi in ranges)]
        json.dump(sc, open(os.path.join(d, "candidates.json"), "w"), indent=0)
        summary[sid] = dict(title=SLICES[sid][0], files=len(ranges), items=len(in_slice), lines=lines, candidates=len(sc))
    json.dump(dict(slices=summary, unassigned=missing), open(os.path.join(SL, "summary.json"), "w"), indent=1)
    for sid, s in summary.items():
        print(f"{sid} files={s['files']:3d} items={s['items']:5d} lines={s['lines']:6d} cands={s['candidates']:4d}  {s['title'][:70]}")
    print("unassigned runtime files:", len(missing), missing[:10])
    print("global candidates:", collections.Counter(c["kind"] for c in cands))


def top(ids, k):
    c = collections.Counter(ids)
    return [x for x, _ in c.most_common(k)]


def short(nodes, i):
    r = nodes.get(i)
    if not r:
        return i
    return f"{r['file'].split('/')[-1]}::{r['qual'].split('::')[-1] if r['lang'] == 'cuda' else '::'.join(r['qual'].split('::')[-2:])}"


def near_duplicates(nodes):
    """MinHash over identifier-normalized 6-token shingles for fns >= 40 lines. Returns (a, b, jaccard)."""
    import random
    rng = random.Random(7)
    K = 64
    P = (1 << 61) - 1
    coeffs = [(rng.randrange(1, P), rng.randrange(0, P)) for _ in range(K)]
    sigs = {}
    shingle_sets = {}
    for r in nodes.values():
        if r["kind"] != "fn" or r.get("test") or "/bin/" in r["file"] or "/tests/" in r["file"]:
            continue
        if r["line_end"] - r["line_start"] < 40:
            continue
        path = os.path.join(WT, r["file"])
        try:
            src = open(path, errors="replace").read().split("\n")
        except FileNotFoundError:
            continue
        text = "\n".join(src[r["line_start"] - 1:r["line_end"]])
        if r["lang"] == "rust":
            toks = [t.t if t.k == "p" or t.t in KEYWORDS else ("I" if t.k == "id" else "L") for t in rust_items.lex(text)]
        else:
            toks = [t if t in "{}()[];,<>=+-*/&|!.:" or t in KEYWORDS else ("I" if re.match(r"[A-Za-z_]", t) else "L")
                    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d[\w.]*|\S", re.sub(r"//.*|/\*.*?\*/", "", text, flags=re.S))]
        sh = {hash(tuple(toks[i:i + 6])) & ((1 << 61) - 1) for i in range(max(0, len(toks) - 5))}
        if len(sh) < 60:
            continue
        shingle_sets[r["id"]] = sh
        sigs[r["id"]] = tuple(min((a * x + b) % P for x in sh) for a, b in coeffs)
    buckets = collections.defaultdict(list)
    for i, s in sigs.items():
        for band in range(16):
            buckets[(band, s[band * 4:band * 4 + 4])].append(i)
    pairs = set()
    for ids in buckets.values():
        if 1 < len(ids) <= 40:
            for x in range(len(ids)):
                for y in range(x + 1, len(ids)):
                    pairs.add(tuple(sorted((ids[x], ids[y]))))
    out = []
    for a, b in pairs:
        sa, sb = shingle_sets[a], shingle_sets[b]
        j = len(sa & sb) / len(sa | sb)
        if j >= 0.7 and nodes[a]["hash"] != nodes[b]["hash"]:
            out.append((a, b, j))
    out.sort(key=lambda x: -x[2])
    return out


KEYWORDS = {"fn", "let", "mut", "if", "else", "match", "for", "while", "loop", "return", "break", "continue", "in", "as",
            "struct", "enum", "impl", "pub", "unsafe", "const", "static", "self", "Self", "true", "false", "Some", "None",
            "Ok", "Err", "__global__", "__device__", "float", "int", "half", "void", "template", "typename", "__syncthreads",
            "threadIdx", "blockIdx", "blockDim", "__shfl_xor_sync", "unsigned", "uint32_t", "int32_t", "bool"}

if __name__ == "__main__":
    main()
