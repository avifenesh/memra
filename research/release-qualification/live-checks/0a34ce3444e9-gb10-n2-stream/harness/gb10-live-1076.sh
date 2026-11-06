#!/usr/bin/env bash
# Runs ON a Spark under the caller's sparklock lease ($LEASE): memra#1076's GB10 live check.
# Qwen3.8-27B GGUF on the plain route (MEMRA_SERVE_SPEC=0, no draft: n > 1 shared prefill is
# offered only there; the spec and DSpark routes answer n > 1 with HTTP 400 by design),
# MEMRA_CTX=262144, derived unified-memory defaults (the 60 s SSE prefill commit and the derived
# streaming deadline).
# Per arm: fresh boot, /readyz (15 min cap), one n = 2 streaming request on a ~131k-token prompt
# (n2_stream_check.py), stop.
# usage: gb10-live-1076.sh OUTDIR ARM:BINDIR...
set -u
OUT=$1; shift
mkdir -p "$OUT"
export SPARKLOCK_LEASE=${LEASE:?lease}
H=$HOME/.cache/memra-land-1076
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
L=$OUT/chain.log
log(){ echo "$(date -Is) $*" | tee -a "$L"; }
i=0
for spec in "$@"; do
  i=$((i+1))
  arm=${spec%%:*}; bin=${spec#*:}
  C=$OUT/$(printf '%02d' $i)-$arm
  mkdir -p "$C"
  (
    for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
    export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8090 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32 MEMRA_KV_HOST_MB=0
    export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1
    export MEMRA_SERVE_SPEC=0
    { echo "cell=$i arm=$arm bin=$bin"; sha256sum "$bin/memra-server"; env | grep '^MEMRA_' | sort;
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; grep MemAvailable /proc/meminfo; } > "$C/env.txt"
    nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 250 > "$C/gpu.csv" 2>&1 &
    smi=$!
    "$bin/memra-server" > "$C/server.log" 2>&1 &
    pid=$!
    echo $pid > "$C/server.pid"
    if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
         --fail-if "! kill -0 $pid" --every 2s --timeout 15m > "$C/wait.log" 2>&1; then
      python3 "$H/n2_stream_check.py" "$C/check" --lines 4678 --n 2 \
        --nonce "memra-1076 live check $arm $(date +%s)" > "$C/check.out" 2>&1
      echo "check rc=$?" >> "$C/check.out"
    else
      echo "never ready" > "$C/check.out"
    fi
    kill -TERM $pid 2>/dev/null
    waitfor --gone $pid --timeout 3m > /dev/null 2>&1 || kill -KILL $pid 2>/dev/null
    wait $pid 2>/dev/null
    kill $smi 2>/dev/null
  )
  log "arm $i $arm: $(tail -n 2 "$C/check.out" | tr '\n' ' ')"
  log "  $(grep -aoE '\[deadline\][^\n]{0,240}' "$C/server.log" | head -1)"
  grep -aE '\[(admission|shed)\]' "$C/server.log" | head -5 | while IFS= read -r l; do log "  $l"; done
done
log LIVE_1076_DONE
