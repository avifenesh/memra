#!/usr/bin/env bash
# Runs ON a Spark under the caller's sparklock lease ($LEASE): memra#1073 review item 1, peer
# first-token time beside one ~131k cold prime, the GB10 prime chunk default 8192 vs an explicit
# MEMRA_PRIME_CHUNK=4096. Qwen3.8-27B GGUF + masked MTP draft (the GB10 recommended arm),
# MEMRA_CTX=262144, MEMRA_MAX_SESSIONS=32, the spec gate pinned (LOW=64 HIGH=65) so every request
# takes the same route on both arms. Fresh boot per cell; fairness_cell.py drives the requests.
# usage: gb10-fairness.sh OUTDIR CELL... with CELL = arm:bindir:K=V,K2=V2 (env may be empty)
set -u
OUT=$1; shift
mkdir -p "$OUT"
export SPARKLOCK_LEASE=${LEASE:?lease}
H=$HOME/.cache/memra-land-1076
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
L=$OUT/chain.log
log(){ echo "$(date -Is) $*" | tee -a "$L"; }
i=0
for cell in "$@"; do
  i=$((i+1))
  arm=${cell%%:*}; rest=${cell#*:}; bin=${rest%%:*}; envs=${rest#*:}
  [ "$envs" = "$rest" ] && envs=""
  C=$OUT/$(printf '%02d' $i)-$arm
  mkdir -p "$C"
  (
    for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
    export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8090 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32 MEMRA_KV_HOST_MB=0
    export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1
    export MEMRA_MTP_DRAFT="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
    export MEMRA_SPEC_GATE_LOW=64 MEMRA_SPEC_GATE_HIGH=65
    for kv in ${envs//,/ }; do export "$kv"; done
    { echo "cell=$i arm=$arm bin=$bin env=$envs"; sha256sum "$bin/memra-server"; env | grep '^MEMRA_' | sort;
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; grep MemAvailable /proc/meminfo; } > "$C/env.txt"
    nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 250 > "$C/gpu.csv" 2>&1 &
    smi=$!
    "$bin/memra-server" > "$C/server.log" 2>&1 &
    pid=$!
    echo $pid > "$C/server.pid"
    if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
         --fail-if "! kill -0 $pid" --every 2s --timeout 15m > "$C/wait.log" 2>&1; then
      python3 "$H/fairness_cell.py" "$C/rows.jsonl" --tag "$arm" > "$C/cell.out" 2>&1
      echo "cell rc=$?" >> "$C/cell.out"
    else
      echo "never ready" > "$C/cell.out"
    fi
    kill -TERM $pid 2>/dev/null
    waitfor --gone $pid --timeout 3m > /dev/null 2>&1 || kill -KILL $pid 2>/dev/null
    wait $pid 2>/dev/null
    kill $smi 2>/dev/null
  )
  log "cell $i $arm env=$envs: $(grep -aoE '\[prime-chunk\][^\n]{0,120}' "$C/server.log" | head -1)"
  log "  $(grep -aoE '\[prime-walk\][^\n]{0,160}' "$C/server.log" | head -1)"
  while IFS= read -r l; do log "  $l"; done < "$C/cell.out"
done
log FAIRNESS_DONE
