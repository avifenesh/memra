#!/usr/bin/env bash
# Runs ON a Spark under the caller's sparklock lease ($LEASE): memra#1075's GB10 boot effects.
# One fresh boot per cell, Qwen3.8-27B GGUF (NVFP4+Q5_K trunk), derived unified-memory defaults,
# MEMRA_CTX=262144, MEMRA_MAX_SESSIONS=32. Two routes:
#   mtp     GGUF + masked (sxc32768) GGUF MTP draft (spec-cost calibration runs)
#   dflash  GGUF + DFlash2 drafter, FR-Spec trim, MEMRA_MTP_SKIP=1, DSpark restore doors on
# Each boot: wait for /readyz (15 min cap), one 32-token greedy chat request, stop.
# usage: gb10-boot-ab.sh OUTDIR CELL... with CELL = route:arm:bindir (order as given)
set -u
OUT=$1; shift
mkdir -p "$OUT"
export SPARKLOCK_LEASE=${LEASE:?lease}
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF
DF=$HOME/models/memra-q38/Qwen3.8-27B-DFlash2-memra
RANKS="$G8/q38-ranks-sxc32768.gguf.txt"
MASK="$G8/mtp-Qwen3.8-27B-NVFP4-frspec-sxc32768.gguf"
L=$OUT/chain.log
log(){ echo "$(date -Is) $*" | tee -a "$L"; }
i=0
for cell in "$@"; do
  i=$((i+1))
  route=${cell%%:*}; rest=${cell#*:}; arm=${rest%%:*}; bin=${rest#*:}
  C=$OUT/$(printf '%02d' $i)-$route-$arm
  mkdir -p "$C"
  (
    for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
    export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8090 MEMRA_CTX=262144 MEMRA_MAX_SESSIONS=32 MEMRA_KV_HOST_MB=0
    export MEMRA_MODELS="q38=$G8/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_PREFIX_STABLE_BOUNDARY=1
    case $route in
      mtp) export MEMRA_MTP_DRAFT="$MASK" ;;
      dflash) export MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" MEMRA_FRSPEC_TRIM="$RANKS" \
                MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 ;;
    esac
    { echo "cell=$i route=$route arm=$arm bin=$bin"; sha256sum "$bin/memra-server"; env | grep '^MEMRA_' | sort;
      nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; grep MemAvailable /proc/meminfo; } > "$C/env.txt"
    nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader -lms 250 > "$C/gpu.csv" 2>&1 &
    smi=$!
    t0=$(date +%s.%N)
    "$bin/memra-server" > "$C/server.log" 2>&1 &
    pid=$!
    echo $pid > "$C/server.pid"
    if waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
         --fail-if "! kill -0 $pid" --every 2s --timeout 15m > "$C/wait.log" 2>&1; then
      t1=$(date +%s.%N)
      echo "ready_s=$(python3 -c "print(round($t1 - $t0, 1))")" > "$C/ready.txt"
      curl -s -o "$C/req.json" -w '%{http_code}\n' http://127.0.0.1:8090/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{"model":"q38","messages":[{"role":"user","content":"Name three primary colors."}],"max_tokens":32,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        > "$C/req.code" 2>&1
      grep MemAvailable /proc/meminfo >> "$C/ready.txt"
    else
      echo "never ready" > "$C/ready.txt"
    fi
    kill -TERM $pid 2>/dev/null
    waitfor --gone $pid --timeout 3m > /dev/null 2>&1 || kill -KILL $pid 2>/dev/null
    wait $pid 2>/dev/null
    kill $smi 2>/dev/null
  )
  log "cell $i $route $arm: $(tr '\n' ' ' < "$C/ready.txt") http=$(cat "$C/req.code" 2>/dev/null)"
  log "  $(grep -aoE '\[prefix-cache\] (on: budget [0-9]+MB|WARNING: derived budget is 0 B)' "$C/server.log" | head -1)"
  log "  $(grep -aoE 'transient floor [0-9]+MB \(static[^)]*\)' "$C/server.log" | head -1)"
  log "  $(grep -aoE '\[wave-probe\][^\n]{0,200}' "$C/server.log" | head -1)"
  log "  $(grep -aoE '\[spec-cost\] (calibration|model=)[^\n]{0,160}' "$C/server.log" | head -1)"
  log "  oom_lines=$(grep -acE 'CUDA_ERROR_OUT_OF_MEMORY|out of memory' "$C/server.log")"
done
log BOOT_AB_DONE
