#!/usr/bin/env python3
"""Phase-bucket report for the diet-battery censuses (the launch-diet census script's
exact bucket map, run standalone because the box's memra-server needs duration-bounded
nsys collection — TERM kills it without the CUPTI atexit flush, receipted trap).

Usage: c8_buckets.py <kernsum.csv> <apisum.csv> <completion_tokens>
NOTE launch/tok mixes prefill+decode instances (the census's own caveat); subtract the
prefill share via the per-chunk counts in PREFILL-GAP.md 1.1 before quoting per-decode
figures. The ship trace also folds drafter + verify kernels into the same names.
"""
import csv, sys, collections, io


def read_report(path):
    """nsys stats 2026.1.3 writes preamble lines before the csv header — skip to it."""
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        if l.startswith("Time (%)"):
            return csv.DictReader(io.StringIO("\n".join(lines[i:])))
    return csv.DictReader(io.StringIO(""))

buckets = [
 ("kda-proj",     ("qmatvec_q8_0_mmvq", "quantize_q8_1", "qmatvec_kda6")),
 ("kda-scan+conv",("kda_scan", "kda_conv", "kda_gate", "kda_gated", "l2_norm", "sigmoid_f32")),
 ("mhc-sites",    ("dsv4_",)),
 ("rms-norm",     ("rms_norm", "add_rms")),
 ("mla+indexer",  ("mla_", "fwht", "kpool", "indexer", "nvfp4")),
 ("moe",          ("moe_", "qmatvec_expert", "swiglu", "axpy_f32", "router", "sigmoid_dot")),
 ("lm_head",      ("q5_K", "q5_k")),
 ("cublas-f32",   ("sgemm", "gemv", "dot_kernel", "reduce_1Block", "cutlass", "splitKreduce")),
 ("bf16-mmv",     ("matvec_bf16",)),
]
tot = collections.Counter(); cnt = collections.Counter(); grand_ns = 0; grand_n = 0
for row in read_report(sys.argv[1]):
    name = row.get("Name", ""); ns = float(row.get("Total Time (ns)", 0) or 0)
    inst = int(row.get("Instances", 0) or 0); grand_ns += ns; grand_n += inst
    for b, keys in buckets:
        if any(k in name for k in keys):
            tot[b] += ns; cnt[b] += inst; break
    else:
        tot["other"] += ns; cnt["other"] += inst
ct = int(sys.argv[3]) or 1
print(f"{'bucket':16} {'gpu_ms':>10} {'launches':>10} {'launch/tok':>10} {'gpu_share':>9}")
for b, ns in tot.most_common():
    print(f"{b:16} {ns/1e6:10.1f} {cnt[b]:10d} {cnt[b]/ct:10.1f} {ns/grand_ns*100:8.1f}%")
print(f"{'TOTAL':16} {grand_ns/1e6:10.1f} {grand_n:10d} {grand_n/ct:10.1f}")
print()
print("cuda-api families (host side: the allocs+syncs bucket):")
api_keys = ("cuLaunchKernel", "cudaLaunchKernel", "MemAlloc", "MemFree",
            "Memcpy", "Memset", "Synchronize", "StreamWait", "EventSynchronize")
for row in read_report(sys.argv[2]):
    name = row.get("Name", "")
    if any(k in name for k in api_keys):
        ns = float(row.get("Total Time (ns)", 0) or 0)
        n = int(row.get("Num Calls", 0) or 0)
        print(f"  {name:34} {ns/1e6:10.1f} ms {n:10d} calls {n/ct:8.1f}/tok")
