#!/usr/bin/env bash
# THE DECODE LAUNCH CENSUS (launch-diet lane, 2026-08-30) — the decode twin of
# ../prefill-gap-20260829/profile-prime-phases.sh, and the first cell of the
# step37 transfer map's lever 1 (TRANSFER-MAP.md, "What to start tomorrow").
#
# The one question this answers: the residency config's 33.39 ms/token splits
# "~0 staging / 15.9 roofline / 17.2 launch" (decode-attribution ROADMAP step 3),
# and the launch term is ~3200 launches/token at ~5.3 us. HOW does that 17.2 ms
# decompose across the per-token kernel chains — KDA projections / KDA scan+conv
# / mHC mixes / Sinkhorn / collapse+post / MoE / MLA / lm_head / allocs+syncs —
# in counts AND in wall-minus-GPU gap? The split sizes the fusion boundaries
# (which chains are worth an epilogue-degree fusion vs a QKV_FUSED-degree one)
# and answers the map's honest finding: what fusion degree closes 62-66 -> 90.
# It also re-answers the graph question with data: launch latency (gap between
# tiny kernels) vs dependency latency (long kernels serialized) — step37's -20%
# graph verdict does not transfer by analogy (glm5 is ~3200 true launches vs
# step37's ~405 dependency-bound children; TRANSFER-MAP do-not-transfer #1).
#
# BOX REQUIREMENTS (ask the owner for the window; the box is running the L2/L3
# A/B — never take it):
#   * bench box, NEVER a prod serving box (prod-serving-boxes-untouchable law);
#   * the serving card class: RTX PRO 6000 Blackwell 96 GB, enough cards for the
#     full-residency placement the ROADMAP step-3 cell used (the residency-cell
#     receipts ran CUDA_VISIBLE_DEVICES=0,1; the box A/B ran PP3 on 4 cards —
#     mirror whichever placement the owner's window carries, and RECORD it);
#   * the real NVFP4 artifact (~191 GB) at $MODEL_DIR;
#   * nsys (Nsight Systems CLI) on PATH;
#   * a freshly built memra-server from THIS lane's HEAD (git log -1 in the
#     receipt — rebuild-after-checkout-attribution law);
#   * ~30 min: one warm boot + one profiled boot + one decode pass.
#
# Config: the residency config with the FUSED EPILOGUE ON (the lever-1 brief
# pins this: the census must measure the residue AFTER the epilogue removes the
# MoE launch share, or the MoE term double-counts what lane/glm53-epilogue
# already owns). MEMRA_PREFIX_CACHE_MB=0 pinned like every glm5 receipt.
#
# Usage:
#   MODEL_DIR=~/models/glm53-nvfp4 BIN=~/memra/target/release/memra-server \
#   PP_ENV="MEMRA_PP_SHARD=... MEMRA_ST_PINNED=1 MEMRA_MOE_SLOTS=12000 MEMRA_MOE_HARD_VRAM_FRAC=0.95" \
#   bash census-decode-phases.sh
# PP_ENV carries the box's residency placement env verbatim from the window's
# serving recipe; it is recorded into the provenance receipt.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the glm53 NVFP4 artifact dir}"
BIN="${BIN:?set BIN to a freshly built memra-server (git log -1 in the receipt!)}"
PORT="${PORT:-18412}"
OUT="${OUT:-decode-launch-census}"
DECODE_TOKENS="${DECODE_TOKENS:-192}"
PP_ENV="${PP_ENV:-}"

# 1. Receipt header: binary + placement provenance.
{
  echo "== provenance =="
  (cd "$(dirname "$BIN")" && git -C "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" log -1 --format='%H %s' 2>/dev/null) || true
  ls -la "$BIN"
  sha256sum "$BIN" | cut -c1-16
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/[0-9a-f:]*$//'
  echo "PP_ENV=${PP_ENV}"
  echo "MEMRA_MOE_FUSED_EPI=1 (pinned by the lever-1 brief)"
} | tee "${OUT}-provenance.txt"

# 1b. THIS BOX'S launch-cost constant (step37's launch_econ instrument): us/launch
#     eager vs graph replay, so the per-family ms arithmetic below uses the box's
#     own measured number instead of inheriting the residency cell's ~5.3 us. The
#     graph column also re-prices the bounded-capture question for free.
LAUNCH_ECON="${LAUNCH_ECON:-$(dirname "$BIN")/launch-econ}"
if [ -x "$LAUNCH_ECON" ]; then
  "$LAUNCH_ECON" 3200 | tee "${OUT}-launch-econ.txt"
else
  echo "launch_econ binary not found at $LAUNCH_ECON — build it (cargo build --release --bin launch-econ) and re-run; the per-launch constant is part of this receipt" | tee "${OUT}-launch-econ.txt"
fi

# 2. A REAL short prompt (the attribution cells' p5 shape: ~200 prompt tokens,
#    192 completion). Real, never synthetic (greedy-loop law: real prompts +
#    capped max_tokens bound loop damage). Greedy IS the instrument here — this
#    is a launch census, not a serving-decision cell; no perf claim leaves this
#    receipt without the sampled twin caveat stated.
python3 - > /tmp/census-decode-prompt.json <<'PY'
import json, glob
text = ""
for f in sorted(glob.glob("/root/prompts/*.txt") + glob.glob("./prompts/*.txt")):
    text += open(f, errors="ignore").read()
if not text:
    for f in sorted(glob.glob("**/*.rs", recursive=True))[:8]:
        text += open(f, errors="ignore").read()
text = text[:700]  # ~200 tokens at 3.5 chars/token
json.dump({"model": "zai/glm-5.3-flash",
           "messages": [{"role": "user",
                         "content": "Summarize what this code does and list its main risks:\n" + text}],
           "max_tokens": 192, "temperature": 0.0, "stream": False}, open("/tmp/census-decode-prompt.json", "w"))
PY

# 3. Warm boot (page cache), then the profiled boot. One request only: the
#    prefill kernels share names with decode's, so the report separates phases
#    by the per-token count arithmetic below, not by time-slicing the trace.
MEMRA_BASE=(MEMRA_ADDR="127.0.0.1:${PORT}" MEMRA_COMPAT=openai MEMRA_CTX=8192
            MEMRA_MAX_SESSIONS=1 MEMRA_PREFIX_CACHE_MB=0 MEMRA_MOE_FUSED_EPI=1
            NVIDIA_TF32_OVERRIDE=0
            MEMRA_MODELS="zai/glm-5.3-flash=${MODEL_DIR}")

run_server() {  # $1 = "profile" | "warm"
  if [ "$1" = profile ]; then
    env "${MEMRA_BASE[@]}" ${PP_ENV} nsys profile --trace=cuda,osrt --cuda-flush-interval=8000 -o "${OUT}" "$BIN" &
  else
    env "${MEMRA_BASE[@]}" ${PP_ENV} "$BIN" &
  fi
  SRV=$!
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "server never became ready" >&2; kill "$SRV" 2>/dev/null; return 1
}

run_server warm
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @/tmp/census-decode-prompt.json >/dev/null
kill -TERM "$SRV"; wait "$SRV" || true

run_server profile
T0=$(date +%s.%N)
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @/tmp/census-decode-prompt.json \
  > "${OUT}-response.json"
T1=$(date +%s.%N)
python3 - "${OUT}-response.json" "$T0" "$T1" <<'PY' | tee "${OUT}-wall.txt"
import json, sys
r = json.load(open(sys.argv[1]))
u = r.get("usage", {})
wall = float(sys.argv[3]) - float(sys.argv[2])
print(f"wall_s={wall:.3f} prompt_tokens={u.get('prompt_tokens')} "
      f"completion_tokens={u.get('completion_tokens')}")
PY
sleep 12; kill -INT "$SRV"; wait "$SRV" || true

# 4. Reports. Three summaries: kernels (bucketed), memcpy/memset, host API
#    (cuLaunchKernel / cuMemAllocAsync / synchronize counts+times = the
#    allocs+syncs family the brief names).
nsys stats --report cuda_gpu_kern_sum   --format csv "${OUT}.nsys-rep" > "${OUT}-kernsum.csv"
nsys stats --report cuda_gpu_mem_time_sum --format csv "${OUT}.nsys-rep" > "${OUT}-memsum.csv" || true
nsys stats --report cuda_api_sum        --format csv "${OUT}.nsys-rep" > "${OUT}-apisum.csv" || true

python3 - "${OUT}-kernsum.csv" "${OUT}-apisum.csv" "${OUT}-wall.txt" <<'PY' | tee "${OUT}-phase-buckets.txt"
import csv, sys, collections, re

# THE DECODE PHASE MAP, from source (kda.rs / hyper.rs / hybrid_forward.rs /
# moe paths, this lane's study — see LANE.md "the decode program, counted"):
#   kda-proj      : qmatvec_q8_0_mmvq* (wq/wk/wv + f_b/g_b + wo, Q8_0 re-encode),
#                   quantize_q8_1 (3x per layer stage-1 today, 1x under the fused
#                   door), qmatvec_kda6_q8f32_mmvq (the fused door, if ON arm)
#   kda-scan+conv : memra_kda_scan_s128, memra_kda_conv_silu_decode, kda_gate,
#                   kda_gated_rmsnorm, l2_norm, sigmoid
#   mhc-mixes+f32 : cuBLASLt f32 GEMV/sgemm class (dot/reduce/sgemm/cutlass_simt).
#                   AMBIGUOUS BY NAME: the same cuBLASLt class serves BOTH the
#                   mHC mixes GEMM (90 calls/token: 2 sites x 45 layers) AND the
#                   KDA f32 projections f_a/g_a/b_proj (102/token: 3 x 34) —
#                   split them by the count arithmetic, stated in the report.
#   mhc-sites     : memra_dsv4_rowsq_scale, memra_dsv4_hc_sinkhorn_m,
#                   memra_dsv4_hc_collapse, memra_dsv4_hc_post,
#                   memra_dsv4_hc_expand, memra_dsv4_hc_mean
#   rms-norm      : rms_norm family (2/layer between mHC pre and the mixers)
#   mla+indexer   : memra_mla_*, fwht/kpool/indexer kernels, qmatvec_nvfp4*
#                   (the MLA projections are NVFP4 in our mint)
#   moe           : moe_*, qmatvec_expert*, swiglu*, axpy_f32, router/sigmoid_dot
#   lm_head       : q5_K matvec class (the head is Q5_K per the loader law)
buckets = [
 ("kda-proj",     ("qmatvec_q8_0_mmvq", "quantize_q8_1", "qmatvec_kda6")),
 ("kda-scan+conv",("kda_scan", "kda_conv", "kda_gate", "kda_gated", "l2_norm", "sigmoid_f32")),
 ("mhc-sites",    ("dsv4_",)),
 ("rms-norm",     ("rms_norm", "add_rms")),
 ("mla+indexer",  ("mla_", "fwht", "kpool", "indexer", "nvfp4")),
 ("moe",          ("moe_", "qmatvec_expert", "swiglu", "axpy_f32", "router", "sigmoid_dot")),
 ("lm_head",      ("q5_K", "q5_k")),
 ("cublas-f32",   ("sgemm", "gemv", "dot_kernel", "reduce_1Block", "cutlass", "splitKreduce")),
]
tot = collections.Counter(); cnt = collections.Counter(); grand_ns = 0; grand_n = 0
for row in csv.DictReader(open(sys.argv[1])):
    name = row.get("Name", ""); ns = float(row.get("Total Time (ns)", 0) or 0)
    inst = int(row.get("Instances", 0) or 0); grand_ns += ns; grand_n += inst
    for b, keys in buckets:
        if any(k in name for k in keys):
            tot[b] += ns; cnt[b] += inst; break
    else:
        tot["other"] += ns; cnt["other"] += inst

wall = {}
for tokline in open(sys.argv[3]):
    for kv in tokline.split():
        k, _, v = kv.partition("=")
        wall[k] = v
ct = int(wall.get("completion_tokens") or 0) or 1

print(f"{'bucket':16} {'gpu_ms':>10} {'launches':>10} {'launch/tok':>10} {'gpu_share':>9}")
for b, ns in tot.most_common():
    print(f"{b:16} {ns/1e6:10.1f} {cnt[b]:10d} {cnt[b]/ct:10.1f} {ns/grand_ns*100:8.1f}%")
print(f"{'TOTAL':16} {grand_ns/1e6:10.1f} {grand_n:10d} {grand_n/ct:10.1f}")
print()
print("NOTE launch/tok mixes prefill+decode instances; subtract the prefill")
print("share using the prompt-token count and the per-chunk launch counts in")
print("PREFILL-GAP.md 1.1 before quoting a per-decode-token figure.")
print()
print("cuda-api families (host side: the allocs+syncs bucket):")
api_keys = ("cuLaunchKernel", "cudaLaunchKernel", "MemAlloc", "MemFree",
            "Memcpy", "Memset", "Synchronize", "StreamWait", "EventSynchronize")
for row in csv.DictReader(open(sys.argv[2])):
    name = row.get("Name", "")
    if any(k in name for k in api_keys):
        ns = float(row.get("Total Time (ns)", 0) or 0)
        n = int(row.get("Num Calls", 0) or 0)
        print(f"  {name:34} {ns/1e6:10.1f} ms {n:10d} calls {n/ct:8.1f}/tok")
print()
print("THE GAP TERM: wall_s minus (TOTAL gpu_ms + prefill wall) is the")
print("launch/host-sync gap — the decode-attribution X-term. Attribute it")
print("per family in proportion to launch counts ONLY as a first cut, and")
print("say so; the api-sum sync rows above are the direct evidence for the")
print("alloc/sync share (step37 DECODE_V2 precedent: 81% of v1's layer wall")
print("was allocs + host round-trips, invisible in gpu_ms).")
PY

echo "Receipts: ${OUT}-provenance.txt ${OUT}-wall.txt ${OUT}-phase-buckets.txt ${OUT}-kernsum.csv ${OUT}-memsum.csv ${OUT}-apisum.csv ${OUT}-response.json"
