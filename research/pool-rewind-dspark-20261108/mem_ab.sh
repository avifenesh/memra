#!/usr/bin/env bash
# memra#1093 follow-up (a): does parking rewound DFlash sessions cost the RTX 5090 memory?
# main (BASE, recurrent DFlash sessions freed at retire) against the rewind (FIX, rewound and
# parked), both at the 5090 default (MEMRA_DSPARK_PREFIX_RESTORE off), with the 2026-11-05
# decision cell's harness and config (research/dflash-restorable-hit-20261103/rig5090-flip/):
# Qwen3.8-27B NVFP4+Q5_K + DFlash2 q4 + FR-Spec trim, MEMRA_MTP_SKIP=1, ctx 32,768, 8 sessions,
# prefix cache 4,000 MB, no host tier, T=1 seed 1234, thinking on, 768 tokens.
# Workloads: full (29 agentic bodies, 5fe1afe2afdc) and le8k (the 22 shortest, 331f269100d2).
# Order: FIX full c1 c2 c4 c8, BASE full, BASE le8k c2 c4 c8, FIX le8k. Single runs per arm.
# usage: mem_ab.sh BASE_BIN FIX_BIN OUT
set -u
BASE=$1 FIX=$2 O=$3
M=/data/ai-ml/models/q38-gguf
H=$HOME/.cache/model-scout/harness
WF=$HOME/.cache/memra-1025/ab-workload-5090.json
WL=$HOME/.cache/memra-pools/switch/ab-workload-5090-le8k.json
mkdir -p "$O"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$O/chain.log"; }
arm() { # tag BIN workload users
  local tag=$1 bin=$2 W=$3 users=$4 D=$O/$1
  if ss -ltn | grep -q ':8093 '; then log "port 8093 busy before $tag; refusing"; exit 4; fi
  mkdir -p "$D"
  { echo "arm $tag users=$users $(date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    sha256sum "$bin" "$W"; nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit --format=csv,noheader; } > "$D/arm.txt"
  env -u MEMRA_DSPARK_PREFIX_RESTORE \
    MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
    MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt MEMRA_PREFIX_STABLE_BOUNDARY=0 \
    "$bin" > "$D/serve.log" 2>&1 &
  local sp=$!
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader -lms 250 > "$D/gpu-250ms.csv" 2>/dev/null &
  local tel=$!
  if ! waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet; then
    log "boot FAILED $tag"; kill $tel; kill $sp; wait $sp; exit 3
  fi
  log "start $tag"
  for u in $users; do
    (cd $H && CLIENT_MAX_TIME=1800 BENCH_BASE=http://127.0.0.1:8093 BENCH_MODEL=q38 BENCH_TOKENIZE_PATH=/v1/tokenize \
       BENCH_EMULATE_CONTINUOUS_USAGE=1 timeout 3600 python3 ab_agent.py run "$W" \
       "$D" "$tag" --users $u --think 1 --temperature 1.0 --seed 1234 >> "$D/cells.log" 2>&1) || log "$tag c$u rc=$?"
    log "$tag c$u $(tail -n 1 "$D/cells.log" | cut -c1-200)"
  done
  curl -s http://127.0.0.1:8093/metrics > "$D/metrics.json" 2>/dev/null
  kill -TERM $sp; waitfor --gone $sp --timeout 90s --quiet || kill -KILL $sp
  wait $sp 2>/dev/null
  kill $tel 2>/dev/null
  { echo "end $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    grep -a -c 'capacity reject' "$D/serve.log" | sed 's/^/capacity-rejects=/'
    grep -a -c 'VRAM defer' "$D/serve.log" | sed 's/^/vram-defer-lines=/'
    grep -a -c 'dspark parked sessions (global LRU)' "$D/serve.log" | sed 's/^/reclaim-lines=/'
    grep -a -c 'dspark-park REWOUND' "$D/serve.log" | sed 's/^/dspark-park-rewound=/'
    grep -a -c 'dspark-park SKIPPED' "$D/serve.log" | sed 's/^/dspark-park-skipped=/'
    grep -a -c 'at the prime boundary (parked at' "$D/serve.log" | sed 's/^/dspark-rewound-resumes=/'
    grep -a -c -i 'out of memory' "$D/serve.log" | sed 's/^/oom-lines=/'
    grep -a -o 'MB prefill-workspace + [0-9]*MB fixed' "$D/serve.log" | sort | uniq -c | sort -rn | head -3 | sed 's/^/fixed: /'
    grep -a -m1 'transient floor' "$D/serve.log" | cut -c1-200 | sed 's/^/floor: /'; } >> "$D/arm.txt"
  log "$tag done: $(tail -n 11 "$D/arm.txt" | tr '\n' ' ' | cut -c1-400)"
}
log "mem A/B start base $(sha256sum "$BASE" | cut -c1-12) fix $(sha256sum "$FIX" | cut -c1-12) arms ${MEM_ARMS:-1 2 3 4}"
# MEM_ARMS picks a subset (the run yielded the card between arms 1 and 2 once and resumed).
for a in ${MEM_ARMS:-1 2 3 4}; do
  case $a in
    1) arm 1-fix-full "$FIX" "$WF" "1 2 4 8" ;;
    2) arm 2-base-full "$BASE" "$WF" "1 2 4 8" ;;
    3) arm 3-base-le8k "$BASE" "$WL" "2 4 8" ;;
    4) arm 4-fix-le8k "$FIX" "$WL" "2 4 8" ;;
  esac
done
log MEM_AB_DONE
