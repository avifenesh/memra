#!/usr/bin/env bash
# MEMRA_SPEC_VERIFY_GRAPH forced ON/OFF: identity first, then interleaved A/B.
#
# Identity arm: greedy, fixed seed, same prompt — the captured trunk must emit the
# SAME tokens as the eager walk (the door's whole claim is a byte-identical body).
# Speed arm: ABBA x2 on the cached-long serve shape, vendor sampling, 4 timed reps
# per boot after a cache-priming request.
set -uo pipefail
M=/data/memra/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
R=/data/memra/models/ornith15/ornith15-ranks-owngen-32768.gguf
BIN=${BIN:-/data/memra/memra-src/target/release/memra-server}
PORT=18399
OUT=/root/vgraph-ab-results.txt
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo "gpu lock held"; exit 1; }

cat > /tmp/vg_probe.py <<'PYEOF'
import json, sys, time, urllib.request
tag, mode = sys.argv[1], sys.argv[2]
url = "http://127.0.0.1:18399/v1/chat/completions"
doc = "\n".join(f"def process_batch_{i}(items):\n    for it in items:\n        it.normalize(mode={i%4})\n        if it.stale(threshold={i*3}):\n            continue\n        commit(it, retries=3)" for i in range(80))
m = [{"role": "system", "content": "You are a senior engineer reviewing this repository.\n" + doc},
     {"role": "user", "content": "Explain function process_batch_7 line by line and suggest one optimization."}]
def call(mt, greedy):
    body = {"model": "m", "messages": m, "max_tokens": mt, "seed": 20260821}
    if greedy:
        body["temperature"] = 0.0
    else:
        body.update(temperature=0.6, top_p=0.95, top_k=20)
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    msg = r["choices"][0]["message"]
    # This model thinks first: at 160 tokens the whole reply can be reasoning, so hash
    # BOTH streams or the identity arm compares two empty strings and always "passes".
    txt = (msg.get("reasoning") or "") + "\x00" + (msg.get("content") or "")
    return time.time()-t0, txt, r["usage"]
if mode == "identity":
    _, txt, u = call(160, True)
    print(f"IDENTITY {tag} tokens={u['completion_tokens']} sha={__import__('hashlib').sha256(txt.encode()).hexdigest()[:16]}", flush=True)
else:
    call(16, False)  # prime the prefix cache
    for rep in range(4):
        wall, _, u = call(512, False)
        print(f"SPEED {tag} rep{rep}: {u['completion_tokens']/(wall-0.06):.1f} tok/s", flush=True)
PYEOF

boot_probe() { # $1=tag $2=mode $3=flagvalue
  local tag=$1 mode=$2 flag=$3
  local env_extra=""
  [ "$flag" = on ] && env_extra="MEMRA_SPEC_VERIFY_GRAPH=1"
  env CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS=m=$M MEMRA_ADDR=127.0.0.1:$PORT \
    MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=24 MEMRA_FRSPEC_TRIM=$R MEMRA_PRIME_CHUNK=0 \
    MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_PREFIX_CACHE_MB=24576 \
    MEMRA_SPEC_PHASE=1 $env_extra \
    "$BIN" > /tmp/vg-$tag.log 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { up=1; break; }; sleep 2; done
  if [ "$up" = 1 ]; then
    python3 /tmp/vg_probe.py "$tag" "$mode" >> $OUT
    grep -a "spec-vg" /tmp/vg-$tag.log | head -2 >> $OUT
    grep -a "spec-phase" /tmp/vg-$tag.log | tail -2 >> $OUT
  else
    echo "$tag BOOT FAIL: $(tail -3 /tmp/vg-$tag.log | tr '\n' ' ')" >> $OUT
  fi
  kill $pid 2>/dev/null; wait $pid 2>/dev/null; sleep 3
}

: > $OUT
echo "== identity (greedy, seed-pinned) ==" >> $OUT
boot_probe id-off identity off
boot_probe id-on  identity on
echo "== speed balanced 8-boot (vendor sampling, cached-long) ==" >> $OUT
# Balanced 8-boot sequence, both orders twice. One ABBA pair was not enough: the first
# window read FLAT because one OFF boot sat in a host boost window and drifted 203->290
# across its own four reps. Per-round phase totals are recorded alongside tok/s because
# that ratio is internal to each boot and so survives a shifting clock level.
boot_probe s-off-1 speed off
boot_probe s-on-1  speed on
boot_probe s-on-2  speed on
boot_probe s-off-2 speed off
boot_probe s-off-3 speed off
boot_probe s-on-3  speed on
boot_probe s-on-4  speed on
boot_probe s-off-4 speed off
echo VG-AB-DONE >> $OUT
cat $OUT
