#!/bin/bash
# launch-diet window (2026-08-30): decode census + prefill attribution, one boot.
# Config: 2-card arm C (the adopted serving recipe: L2 serve2.sh placement +
# grouped prefill + BF16_MMV + PP_BF16) + MEMRA_MOE_FUSED_EPI=1 (lever-1 brief pin).
set -u
D=/root/launch-diet
BIN=/root/memra/target/release/memra-server
PORT=18409
EP="http://127.0.0.1:$PORT/v1/chat/completions"
PIDFILE=$D/server.pid
LOG=$D/serve.log
NSYS=/usr/local/bin/nsys

cd $D
{
  echo "== provenance =="
  git -C /root/memra log -1 --format="%H %s"
  sha256sum $BIN | cut -c1-16
  stat -c "bin mtime %y" $BIN
  $NSYS --version | head -1
  nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
  echo "config: 2-card arm C + MEMRA_MOE_FUSED_EPI=1 (PP_STAGES=2 SPLITS=24 DEVICES=0,1 RESIDENT_GB=98 SLOTS=16 CTX=8192 GROUPED_PREFILL=1 BF16_MMV=1 PP_BF16=1 PREFIX_CACHE=0 TF32=0)"
} > provenance.txt

# 0. This box's own launch-cost constant (device 0; timing microbench, marker honored).
touch /root/TIMING-IN-FLIGHT
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > cotenancy-launch-econ.txt
CUDA_VISIBLE_DEVICES=0 /root/memra/target/release/launch-econ 3200 > launch-econ.txt 2>&1
rm -f /root/TIMING-IN-FLIGHT
cat launch-econ.txt

# 1. Boot arm C under nsys launch (instrumented, NOT capturing: load phase excluded).
: > "$LOG"
env MEMRA_PREFIX_CACHE_MB=0 MEMRA_SPILL_STATS=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
  MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1 \
  CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=/root/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:$PORT \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
  MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1 MEMRA_MOE_FUSED_EPI=1 \
  "$NSYS" launch --session-new=census -t cuda,osrt "$BIN" > "$LOG" 2>&1 &
NSPID=$!
echo $NSPID > "$PIDFILE"
i=0
for i in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  grep -qE "panicked" "$LOG" && { echo BOOT PANIC; tail -20 "$LOG"; exit 1; }
  sleep 2
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "NEVER READY"; tail -30 "$LOG"; exit 1; }
echo "ready after ~$((i*2))s" | tee -a provenance.txt
grep -c 'bf16-mmv] RESIDENT' "$LOG" | xargs echo "bf16-mmv RESIDENT lines:" | tee -a provenance.txt
grep -m2 "resident-experts decision" "$LOG" | tee -a provenance.txt

# 2. Requests built from the BANKED real pool (l3-ab/prompts.json).
python3 <<'PY'
import json
p = json.load(open("/root/l3-ab/prompts.json"))
def req(prompt, mt):
    return {"model": "zai/glm-5.3-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": mt, "temperature": 0, "reasoning_effort": "low",
            "stream": False}
json.dump(req(p["WARM"], 32), open("/root/launch-diet/req-warm.json", "w"))
json.dump(req(p["WARM"] + "\n\nNow list the five most load-bearing claims above and rate each for evidence quality.", 192),
          open("/root/launch-diet/req-decode.json", "w"))
json.dump(req(p["A4630"], 1), open("/root/launch-diet/req-prime.json", "w"))
PY

# 3. Warm request (uncaptured; page cache + JIT warm).
curl -s "$EP" -H "Content-Type: application/json" -d @req-warm.json > warm-response.json
echo "warm done"

# 4. WINDOW 1 - decode census: fresh short prime (~460 prompt tokens) + 192 greedy steps.
#    reasoning_effort pinned low (fleet law).
touch /root/TIMING-IN-FLIGHT
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > cotenancy-decode-window.txt
"$NSYS" start --session=census -o "$D/decode-census" 2>&1 | tail -1
T0=$(date +%s.%N)
curl -s "$EP" -H "Content-Type: application/json" -d @req-decode.json > decode-response.json
T1=$(date +%s.%N)
"$NSYS" stop --session=census 2>&1 | tail -1
rm -f /root/TIMING-IN-FLIGHT
W=$(python3 -c "print(f'{$T1 - $T0:.3f}')")
python3 - "$W" <<'PY' | tee decode-wall.txt
import json, sys
r = json.load(open("/root/launch-diet/decode-response.json")); u = r.get("usage", {})
print(f"decode wall_s={sys.argv[1]} prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
PY

# 5. WINDOW 2 - prefill attribution: A4630 banked prompt, max_tokens=1, full cold prime.
touch /root/TIMING-IN-FLIGHT
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader > cotenancy-prime-window.txt
"$NSYS" start --session=census -o "$D/prefill-census" 2>&1 | tail -1
T0=$(date +%s.%N)
curl -s "$EP" -H "Content-Type: application/json" -d @req-prime.json > prime-response.json
T1=$(date +%s.%N)
"$NSYS" stop --session=census 2>&1 | tail -1
rm -f /root/TIMING-IN-FLIGHT
W=$(python3 -c "print(f'{$T1 - $T0:.3f}')")
python3 - "$W" <<'PY' | tee prime-wall.txt
import json, sys
r = json.load(open("/root/launch-diet/prime-response.json")); u = r.get("usage", {})
print(f"prime wall_s={sys.argv[1]} prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
PY

# 6. Stop: session shutdown, then PID-verified TERM if anything survives.
"$NSYS" shutdown --session=census --kill=sigterm 2>&1 | tail -1
sleep 5
SP=$(pgrep -f "target/release/memra-server" | head -1)
if [ -n "${SP:-}" ]; then kill -TERM "$SP"; sleep 8; fi
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 7. Reports.
for w in decode-census prefill-census; do
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv "$w.nsys-rep" > "$w-kernsum.csv" 2>/dev/null
  "$NSYS" stats --report cuda_api_sum --format csv "$w.nsys-rep" > "$w-apisum.csv" 2>/dev/null
  "$NSYS" stats --report cuda_gpu_mem_time_sum --format csv "$w.nsys-rep" > "$w-memsum.csv" 2>/dev/null
  echo "$w: kern rows $(wc -l < "$w-kernsum.csv")"
done
echo WINDOW-DONE
