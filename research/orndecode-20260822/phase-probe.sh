#!/usr/bin/env bash
# Sampled-spec round anatomy on the dev box (old ORN box, Zen3 host — CPU-bound
# regime, which is what the de-CPU-bind lane targets).
# Boots one server with MEMRA_SPEC_PHASE=1, runs the cached-long c1 probe, prints
# the phase decomposition + a host perf profile of the decode window.
set -uo pipefail
M=/data/memra/models/ornith15/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
R=/data/memra/models/ornith15/ornith15-ranks-owngen-32768.gguf
BIN=${BIN:-/data/memra/memra-src/target/release/memra-server}
PORT=18399
exec 9>/tmp/memra-gpu.lock; flock -n 9 || { echo "gpu lock held"; exit 1; }

env CUDA_VISIBLE_DEVICES=0 MEMRA_MODELS=m=$M MEMRA_ADDR=127.0.0.1:$PORT \
  MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=24 MEMRA_FRSPEC_TRIM=$R MEMRA_PRIME_CHUNK=0 \
  MEMRA_SPEC_ADAPT=1 MEMRA_SPEC_PMIN=0.3 MEMRA_SPEC_PMIN0=1 MEMRA_PREFIX_CACHE_MB=24576 \
  MEMRA_SPEC_PHASE=1 MEMRA_SPEC_DEBUG=1 ${EXTRA_ENV:-} \
  "$BIN" > /tmp/phase-serve.log 2>&1 &
pid=$!
for _ in $(seq 150); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && break; sleep 2; done

cat > /tmp/phase_probe.py <<'PYEOF'
import json, sys, time, urllib.request
port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
doc = "\n".join(f"def process_batch_{i}(items):\n    for it in items:\n        it.normalize(mode={i%4})\n        if it.stale(threshold={i*3}):\n            continue\n        commit(it, retries=3)" for i in range(80))
m = [{"role": "system", "content": "You are a senior engineer reviewing this repository.\n" + doc},
     {"role": "user", "content": "Explain function process_batch_7 line by line and suggest one optimization."}]
def plain(mt):
    body = {"model": "m", "messages": m, "max_tokens": mt, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 20260821}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1200))
    u = r["usage"]
    return time.time()-t0, u["completion_tokens"], u.get("spec", {})
plain(16)
for rep in range(int(sys.argv[2])):
    wall, ct, s = plain(512)
    print(f"rep{rep}: {ct/(wall-0.06):.1f} tok/s acc={round(s.get('acceptance_rate',0),3)} d/r={s.get('drafted')}/{s.get('rounds')}", flush=True)
PYEOF

python3 /tmp/phase_probe.py $PORT 1
# host profile over a decode window
perf record -F 999 -g -p $pid -o /tmp/phase-perf.data -- sleep 12 >/dev/null 2>&1 &
PERF=$!
python3 /tmp/phase_probe.py $PORT 3
wait $PERF 2>/dev/null

echo "=== spec phase lines ==="
grep -a "spec-stats\|\[spec\] stream_on\|ph_\|phase" /tmp/phase-serve.log | tail -12
echo "=== host hotspots (self%) ==="
perf report -i /tmp/phase-perf.data --no-children --sort symbol,dso 2>/dev/null | head -28
kill $pid 2>/dev/null; sleep 3
