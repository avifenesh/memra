#!/usr/bin/env bash
# Launch census for the sampled-spec round: how many CUDA API calls / kernels the
# host issues per verify round, and where the host API time goes. Sizes the
# de-CPU-bind options (graph capture vs fusion vs device-accept).
set -uo pipefail
M=/data/memra/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
R=/data/memra/models/ornith15/ornith15-ranks-owngen-32768.gguf
BIN=${BIN:-/data/memra/memra-src/target/release/memra-server}
NSYS=/opt/nvidia/nsight-systems/2025.6.3/bin/nsys
PORT=18399
OUT=/root/nsys-sampled-round
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo "gpu lock held"; exit 1; }

$NSYS profile -o $OUT --force-overwrite true -t cuda,nvtx --delay 150 --duration 20 \
  env CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS=m=$M MEMRA_ADDR=127.0.0.1:$PORT \
  MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=24 MEMRA_FRSPEC_TRIM=$R MEMRA_PRIME_CHUNK=0 \
  MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_PREFIX_CACHE_MB=24576 \
  MEMRA_SPEC_PHASE=1 \
  "$BIN" > /tmp/census-serve.log 2>&1 &
NPID=$!
for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 2; done
# drive decode continuously across the capture window
python3 - <<'PYEOF'
import json, time, urllib.request
url = "http://127.0.0.1:18399/v1/chat/completions"
doc = "\n".join(f"def process_batch_{i}(items):\n    for it in items:\n        it.normalize(mode={i%4})\n        commit(it, retries=3)" for i in range(80))
m = [{"role": "system", "content": "You are a senior engineer reviewing this repository.\n" + doc},
     {"role": "user", "content": "Explain function process_batch_7 line by line and suggest one optimization."}]
t_end = time.time() + 190
n = 0
while time.time() < t_end:
    body = {"model": "m", "messages": m, "max_tokens": 512, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 20260821}
    try:
        urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600).read()
        n += 1
    except Exception as e:
        print("err", e); time.sleep(3)
print("drove", n, "completions")
PYEOF
for _ in $(seq 60); do kill -0 $NPID 2>/dev/null || break; sleep 5; done
kill $NPID 2>/dev/null; sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); do kill -9 $p 2>/dev/null; done
echo "=== CUDA API summary (host-side calls) ==="
$NSYS stats --report cuda_api_sum --format table $OUT.nsys-rep 2>/dev/null | head -22
echo "=== kernel summary (top) ==="
$NSYS stats --report cuda_gpu_kern_sum --format table $OUT.nsys-rep 2>/dev/null | head -18
echo "=== phase lines ==="
grep -a "spec-phase" /tmp/census-serve.log | tail -4
