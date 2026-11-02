#!/usr/bin/env python3
"""Markdown tables from the release-profile A/B raw rows.
usage: summarize.py DIR   (DIR holds results.jsonl, host_results.jsonl, host_tok_results.jsonl)"""
import json, statistics as st, sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "results.jsonl")]
host = [json.loads(l) for l in open(d / "host_results.jsonl")]
tok = [json.loads(l) for l in open(d / "host_tok_results.jsonl")] if (d / "host_tok_results.jsonl").exists() else []
med = lambda xs: st.median(xs)
pct = lambda a, b: f"{(b - a) / a * 100:+.1f}%"

print("## Build (2 jobs, nice 15, no sccache, balanced platform profile)\n")
print("| what | N per arm | default wall s (median) | fat LTO wall s (median) | delta | default peak rustc RSS MB | fat LTO peak rustc RSS MB |")
print("|---|---|---|---|---|---|---|")
g = defaultdict(list)
for r in rows:
    assert r["rc"] == 0, r
    assert r["platform_profile"] == r["platform_profile_end"] == "balanced", r
    g[(r["kind"], r["arm"])].append(r)
labels = {"cold": "cold `-p memra-server` (deps + nvcc kernels + server)",
          "touch-server": "rebuild after touching memra-server/src/main.rs",
          "touch-engine": "rebuild after touching memra-engine/src/lib.rs",
          "gatebins": "qualified gate binaries (kernel-check, run-gen, run-spec, argmax-margin-probe, tok-parity), incremental",
          "testbuild-cpu": "`cargo test --release --no-run` CPU crates (reference, sampling, tokenizer, gguf, validate)",
          "testbuild-engine": "`cargo test --release --no-run -p memra-engine --lib`",
          "testrun-cpu": "run those CPU-crate tests",
          "testrun-engine": "run the engine lib tests",
          "cpu-native-check-build": "`cpu_native_check` bin, incremental",
          "wsbins": "`--workspace --bins` (CI full-mode build, INSTALLATION.md), from the server-built state"}
for kind in labels:
    a, b = g.get((kind, "A"), []), g.get((kind, "B"), [])
    if not a or not b:
        continue
    wa, wb = med([r["wall_s"] for r in a]), med([r["wall_s"] for r in b])
    print(f"| {labels[kind]} | {len(a)}/{len(b)} | {wa:.1f} ({', '.join(f'{r['wall_s']:.1f}' for r in a)}) | "
          f"{wb:.1f} ({', '.join(f'{r['wall_s']:.1f}' for r in b)}) | {pct(wa, wb)} | "
          f"{max(r['max_single_rss_mb'] for r in a)} | {max(r['max_single_rss_mb'] for r in b)} |")
sa = {r["server_bytes"] for r in rows if r["arm"] == "A" and r["server_bytes"]}
sb = {r["server_bytes"] for r in rows if r["arm"] == "B" and r["server_bytes"]}
print(f"\nmemra-server binary bytes: default {sorted(sa)}, fat LTO {sorted(sb)}"
      f" ({pct(min(sa), min(sb))})")
print(f"max coretemp during builds: {max(r['max_coretemp_c'] for r in rows)} C")

print("\n## Serving host paths, RTX 5090 Laptop, Qwen3.5-9B NVFP4 MTP GGUF (rounds interleaved ABBAAB)\n")
def col(arm, f):
    return [f(r) for r in host if r["arm"] == arm]
metrics = [("boot to /health 200 (s)", lambda r: r["boot_s"]),
           ("/v1/tokenize, %d-token chat, median ms" % host[0]["tokenize_tokens"], lambda r: r["tokenize_ms"]["median"]),
           ("streamed TTFT, short cached prompt, p50 ms (30/round)", lambda r: r["ttft_ms"]["p50"]),
           ("streamed TTFT p95 ms", lambda r: r["ttft_ms"]["p95"]),
           ("short request E2E p50 ms", lambda r: r["short_e2e_ms"]["p50"]),
           ("greedy 512-token stream wall s (median of 3)", lambda r: med([x["wall_s"] for x in r["greedy"]]))]
if tok:
    metrics += [("tokenize-only rounds: /v1/tokenize median ms (60/round)", None),
                ("tokenize-only rounds: server CPU ms per tokenize (10 ms clock ticks)", None),
                ("tokenize-only rounds: boot to /health 200 (s)", None)]
print("| metric | default (per round) | fat LTO (per round) | median delta |")
print("|---|---|---|---|")
for name, f in metrics:
    if f is None:
        src = tok
        f = ((lambda r: r["tokenize_ms"]["median"]) if "median ms" in name else
             (lambda r: r["boot_s"]) if "boot" in name else (lambda r: r["tokenize_server_cpu_ms"]["median"]))
    else:
        src = host
    a = [f(r) for r in src if r["arm"] == "A"]; b = [f(r) for r in src if r["arm"] == "B"]
    print(f"| {name} | {', '.join(str(round(x, 3)) for x in a)} | {', '.join(str(round(x, 3)) for x in b)} | {pct(med(a), med(b))} |")
print("\nBoot includes model load; the first round of each campaign (A) ran on a colder page cache.")
print("\n## Correctness\n")
na = Path(d / "logs/cpu_native_check-A.log").read_text().splitlines()
nb = Path(d / "logs/cpu_native_check-B.log").read_text().splitlines()
diff = [i for i, (x, y) in enumerate(zip(na, nb)) if x != y]
print(f"- cpu_native_check: {len(na)} lines per arm, differing lines: {[na[i].split()[0] for i in diff]} (timing profile only)" if len(na) == len(nb) else "- cpu_native_check: line counts differ")
for key in ("tokenize_ids_sha256", "greedy_sha256", "sampled_sha256"):
    vals = {r[key] for r in host} | ({r[key] for r in tok} if key == "tokenize_ids_sha256" and tok else set())
    print(f"- {key}: {len(vals)} distinct value(s) across both arms and all rounds: {sorted(vals)}")
print(f"- greedy identical within every round: {all(r['greedy_identical_within_arm'] for r in host)}; "
      f"sampled (seed 1234) identical within every round: {all(r['sampled_identical_within_arm'] for r in host)}")
print(f"- platform profile on every serving round: {sorted({r['platform_profile'] for r in host + tok})}")
print(f"- server binary sha256: default {sorted({r['binary_sha256'] for r in host if r['arm']=='A'})}, "
      f"fat LTO {sorted({r['binary_sha256'] for r in host if r['arm']=='B'})}")
