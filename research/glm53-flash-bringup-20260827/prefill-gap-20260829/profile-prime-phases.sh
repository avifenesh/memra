#!/usr/bin/env bash
# FIRST ACTION OF THE NEXT BOX WINDOW (prefill-gap lane, 2026-08-29).
#
# The one question this answers: of the ~12.3 ms/token glm5_next prefill wall,
# how many ms sit in (a) per-token MoE expert matvec launches, (b) the KDA
# sequential scan, (c) f32/BF16-expansion trunk GEMMs, (d) the per-layer host
# router readback, (e) everything else. PREFILL-GAP.md sizes (a) as dominant
# from source + launch-count receipts; this profile is the receipt that either
# confirms the ranking or reorders levers 2 and 3 before the arc is sequenced.
#
# No code change needed: one nsys pass over a single cold 4096-token prime,
# bucketed by kernel name. Run on a bench box (never a prod serving box), with
# the real NVFP4 artifact, serving config mirrored from the ctxprobe arms
# (MEMRA_PREFIX_CACHE_MB=0 pinned, MEMRA_CTX=8192).
#
# Usage: MODEL_DIR=~/models/glm53-nvfp4 BIN=~/memra/target/release/memra-server \
#        bash profile-prime-phases.sh
set -euo pipefail

MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the glm53 NVFP4 artifact dir}"
BIN="${BIN:?set BIN to a freshly built memra-server (git log -1 in the receipt!)}"
PORT="${PORT:-18411}"
OUT="${OUT:-prime-phase-profile}"
PROMPT_TOKENS="${PROMPT_TOKENS:-4096}"

# 1. Receipt header: binary provenance (LAW: rebuild-after-checkout-attribution).
{
  echo "== provenance =="
  (cd "$(dirname "$BIN")" && git -C "$(git rev-parse --show-toplevel 2>/dev/null || echo .)" log -1 --format='%H %s' 2>/dev/null) || true
  ls -la "$BIN"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/[0-9a-f:]*$//'
} | tee "${OUT}-provenance.txt"

# 2. Build a ~4096-token REAL prompt (never synthetic: greedy-loop law does not
#    apply to prefill, but routing entropy does; a repeated-token prompt routes
#    to few experts and understates the grouped win). Use the banked sxc/agentic
#    prompt pools if present; fall back to concatenated repo source.
python3 - "$PROMPT_TOKENS" > /tmp/prime-prompt.json <<'PY'
import json, sys, glob
target = int(sys.argv[1])
text = ""
for f in sorted(glob.glob("/root/prompts/*.txt") + glob.glob("./prompts/*.txt")):
    text += open(f, errors="ignore").read()
if not text:
    for f in sorted(glob.glob("**/*.rs", recursive=True))[:80]:
        text += open(f, errors="ignore").read()
# ~3.5 chars/token heuristic; the server-side token count lands in the receipt.
text = text[: int(target * 3.5)]
json.dump({"model": "zai/glm-5.3-flash",
           "messages": [{"role": "user", "content": text}],
           "max_tokens": 1, "stream": False}, sys.stdout)
PY

# 3. Boot under nsys, fire ONE cold prime, stop. Kernel launches during load
#    are excluded by --capture-range with a cudaProfilerStart... which memra
#    does not emit, so instead: boot WITHOUT nsys once to warm the page cache,
#    then profile a fresh boot and subtract nothing -- the load phase has its
#    own kernel names (repack/quantize at m accumulating under load) and the
#    bucket report below keys on prime-phase kernel families only.
MEMRA_ENV=(MEMRA_ADDR="127.0.0.1:${PORT}" MEMRA_COMPAT=openai MEMRA_CTX=8192
           MEMRA_MAX_SESSIONS=1 MEMRA_PREFIX_CACHE_MB=0
           MEMRA_MODELS="zai/glm-5.3-flash=${MODEL_DIR}")

env "${MEMRA_ENV[@]}" nsys profile --trace=cuda,osrt -o "${OUT}" "$BIN" &
SRV=$!
for _ in $(seq 1 180); do
  curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
  sleep 2
done
T0=$(date +%s.%N)
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' -d @/tmp/prime-prompt.json \
  > "${OUT}-response.json"
T1=$(date +%s.%N)
echo "wall_s=$(echo "$T1 - $T0" | bc)" | tee "${OUT}-wall.txt"
kill -INT "$SRV"; wait "$SRV" || true

# 4. Bucket the kernel summary. THE PHASE MAP (from source, PREFILL-GAP.md §1):
#    moe-expert-matvec : qmatvec_expert_q8, moe_gate_up_*, moe_down8_*,
#                        swiglu_preclamped_mul*, axpy_f32, quantize_q8_1
#    kda-scan          : memra_kda_scan_s128
#    kda-conv+glue     : ssm_conv1d*, l2_norm, gated/sigmoid rmsnorm family
#    trunk-f32-gemm    : sgemm / cublasLt f32 kernels (ampere_sgemm*, sm1xx sgemm)
#    trunk-tc-gemm     : mmq/qmatvec_gemm/fp8 TN GEMM families
#    mla+indexer       : memra_mla_*
#    mhc               : memra_dsv4_*
#    router            : router_gemv, sigmoid_dot_rows, moe_router_*topk*
#    Everything else lands in 'other'; memcpy volume is reported separately
#    (staging vs launch attribution, the decode-attribution method).
nsys stats --report cuda_gpu_kern_sum --format csv "${OUT}.nsys-rep" > "${OUT}-kernsum.csv"
nsys stats --report cuda_gpu_mem_time_sum --format csv "${OUT}.nsys-rep" > "${OUT}-memsum.csv" || true
python3 - "${OUT}-kernsum.csv" <<'PY' | tee "${OUT}-phase-buckets.txt"
import csv, sys, collections
buckets = [
 ("moe-expert-matvec", ("qmatvec_expert", "moe_gate_up", "moe_down8", "swiglu_preclamped", "axpy_f32", "quantize_q8_1")),
 ("kda-scan",          ("kda_scan",)),
 ("kda-glue",          ("ssm_conv1d", "l2_norm", "kda_",)),
 ("trunk-f32-gemm",    ("sgemm", "gemm_f32", "cutlass_80_simt", "simt_sgemm")),
 ("trunk-tc-gemm",     ("mmq", "qmatvec_gemm", "e4m3", "fp8", "nvfp4", "16816", "mma")),
 ("mla+indexer",       ("mla_",)),
 ("mhc",               ("dsv4_",)),
 ("router",            ("router", "sigmoid_dot",)),
]
tot = collections.Counter(); cnt = collections.Counter(); grand = 0
for row in csv.DictReader(open(sys.argv[1])):
    name = row.get("Name", ""); ns = float(row.get("Total Time (ns)", 0) or 0)
    inst = int(row.get("Instances", 0) or 0); grand += ns
    for b, keys in buckets:
        if any(k in name for k in keys):
            tot[b] += ns; cnt[b] += inst; break
    else:
        tot["other"] += ns; cnt["other"] += inst
print(f"{'bucket':20} {'ms':>12} {'launches':>10} {'share':>7}")
for b, ns in tot.most_common():
    print(f"{b:20} {ns/1e6:12.1f} {cnt[b]:10d} {ns/grand*100:6.1f}%")
print(f"{'TOTAL':20} {grand/1e6:12.1f} {sum(cnt.values()):10d}")
print("\nNOTE: GPU-time shares understate LAUNCH-GAP overhead (idle between")
print("tiny kernels is invisible here). Cross-check: wall_s from -wall.txt")
print("minus TOTAL GPU ms = the launch/host-sync gap term, the decode-")
print("attribution X-term method applied to prefill.")
PY
echo "Receipts: ${OUT}-provenance.txt ${OUT}-wall.txt ${OUT}-phase-buckets.txt ${OUT}-kernsum.csv ${OUT}-memsum.csv"
