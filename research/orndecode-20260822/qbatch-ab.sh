#!/usr/bin/env bash
# MEMRA_SPEC_QBATCH: batched q gather in the sampled accept walk.
#
# Identity here is stronger than the verify-graph lane's could be: this change lives ON the
# sampled path, and that path is seed-pinned (sp_seed + a uniform counter), so a fixed-seed
# SAMPLED completion must hash identically with the batch on and off. If the batch moved any
# q value or consumed the uniforms in a different order, this catches it directly.
#
# Arms (balanced, both orders twice):
#   base = MEMRA_SPEC_QBATCH=0   (per-position gather + one sync each)
#   qb   = default               (one staged gather + one sync per round)
#   both = qb + MEMRA_SPEC_VERIFY_GRAPH=1 (the two de-CPU-bind halves composed)
set -uo pipefail
M=/data/memra/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
R=/data/memra/models/ornith15/ornith15-ranks-owngen-32768.gguf
BIN=${BIN:-/data/memra/memra-src/target/release/memra-server}
PORT=18399
OUT=/root/qbatch-ab-results.txt
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo "gpu lock held"; exit 1; }

cat > /tmp/qb_probe.py <<'PYEOF'
import hashlib, json, sys, time, urllib.request
tag, mode = sys.argv[1], sys.argv[2]
url = "http://127.0.0.1:18399/v1/chat/completions"
doc = "\n".join(f"def process_batch_{i}(items):\n    for it in items:\n        it.normalize(mode={i%4})\n        if it.stale(threshold={i*3}):\n            continue\n        commit(it, retries=3)" for i in range(80))
m = [{"role": "system", "content": "You are a senior engineer reviewing this repository.\n" + doc},
     {"role": "user", "content": "Explain function process_batch_7 line by line and suggest one optimization."}]
def call(mt):
    body = {"model": "m", "messages": m, "max_tokens": mt, "temperature": 0.6,
            "top_p": 0.95, "top_k": 20, "seed": 20260821}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    msg = r["choices"][0]["message"]
    txt = (msg.get("reasoning") or "") + "\x00" + (msg.get("content") or "")
    return time.time()-t0, txt, r["usage"]
if mode == "identity":
    # sampled, seed-pinned: the arms must agree token-for-token
    _, txt, u = call(192)
    print(f"IDENTITY {tag} tokens={u['completion_tokens']} sha={hashlib.sha256(txt.encode()).hexdigest()[:16]}", flush=True)
else:
    call(16)
    for rep in range(4):
        wall, _, u = call(512)
        print(f"SPEED {tag} rep{rep}: {u['completion_tokens']/(wall-0.06):.1f} tok/s", flush=True)
PYEOF

boot_probe() { # $1=tag $2=mode $3=arm
  local tag=$1 mode=$2 arm=$3 env_extra=""
  case $arm in
    base) env_extra="MEMRA_SPEC_QBATCH=0" ;;
    qb)   env_extra="" ;;
    both) env_extra="MEMRA_SPEC_VERIFY_GRAPH=1" ;;
  esac
  env CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS=m=$M MEMRA_ADDR=127.0.0.1:$PORT \
    MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=24 MEMRA_FRSPEC_TRIM=$R MEMRA_PRIME_CHUNK=0 \
    MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_PREFIX_CACHE_MB=24576 \
    MEMRA_SPEC_PHASE=1 $env_extra \
    "$BIN" > /tmp/qb-$tag.log 2>&1 &
  local pid=$! up=0
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
  if [ "$up" = 1 ]; then
    python3 /tmp/qb_probe.py "$tag" "$mode" >> $OUT
    grep -a "spec-phase" /tmp/qb-$tag.log | tail -2 >> $OUT
  else
    echo "$tag BOOT FAIL: $(tail -3 /tmp/qb-$tag.log | tr '\n' ' ')" >> $OUT
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

: > $OUT
echo "== identity (SAMPLED, seed-pinned — the arms must agree) ==" >> $OUT
boot_probe id-base identity base
boot_probe id-qb   identity qb
boot_probe id-both identity both
echo "== speed, balanced (base vs qb, both orders twice) ==" >> $OUT
boot_probe s-base-1 speed base
boot_probe s-qb-1   speed qb
boot_probe s-qb-2   speed qb
boot_probe s-base-2 speed base
echo "== composed arm (qb + verify-graph) vs base, adjacent ==" >> $OUT
boot_probe s-base-3 speed base
boot_probe s-both-1 speed both
boot_probe s-both-2 speed both
boot_probe s-base-4 speed base
echo QB-AB-DONE >> $OUT
cat $OUT
