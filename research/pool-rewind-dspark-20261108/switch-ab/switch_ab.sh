#!/usr/bin/env bash
# memra#1093 follow-up (b): MEMRA_DSPARK_PREFIX_RESTORE on vs the RTX 5090 default (off), on current main,
# with the 2026-11-05 decision cell's harness and config (research/dflash-restorable-hit-20261103/
# rig5090-flip/): Qwen3.8-27B NVFP4+Q5_K + DFlash2 q4 + FR-Spec trim, MEMRA_MTP_SKIP=1, ctx 32,768,
# 8 sessions, prefix cache 4,000 MB, no host tier, T=1 seed 1234, thinking on, 768 tokens.
# Workloads: full (29 agentic bodies, 5fe1afe2afdc) and le8k (the 22 shortest, 331f269100d2).
# Order: ON c1 c2 c4 c8 full, OFF c1 c2 c4 c8 full, OFF c1, ON c1 (c1 ABBA), ON c2 c4 c8 le8k, OFF le8k.
# usage: switch_ab.sh BIN OUT
set -u
BIN=$1 O=$2
M=/data/ai-ml/models/q38-gguf
H=$HOME/.cache/model-scout/harness
WF=$HOME/.cache/memra-1025/ab-workload-5090.json
WL=$HOME/.cache/memra-pools/switch/ab-workload-5090-le8k.json
mkdir -p "$O"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$O/chain.log"; }
arm() { # tag ON|OFF workload users
  local tag=$1 sw=$2 W=$3 users=$4 D=$O/$1
  if ss -ltn | grep -q ':8093 '; then log "port 8093 busy before $tag; refusing"; exit 4; fi
  mkdir -p "$D"
  { echo "arm $tag switch=$sw users=$users $(date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    sha256sum "$BIN" "$W"; nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit --format=csv,noheader; } > "$D/arm.txt"
  local extra=""
  [ "$sw" = ON ] && extra="MEMRA_DSPARK_PREFIX_RESTORE=1"
  env -u MEMRA_DSPARK_PREFIX_RESTORE \
    MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=8 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
    MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt MEMRA_PREFIX_STABLE_BOUNDARY=0 $extra \
    "$BIN" > "$D/serve.log" 2>&1 &
  local sp=$!
  nvidia-smi --query-gpu=timestamp,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader -lms 250 > "$D/gpu-250ms.csv" 2>/dev/null &
  local tel=$!
  if ! waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet; then
    log "boot FAILED $tag"; kill $tel; kill $sp; wait $sp; exit 3
  fi
  log "start $tag $(grep -a 'prefix restore switch default' "$D/serve.log" | head -1 | cut -c1-160)"
  for u in $users; do
    (cd $H && CLIENT_MAX_TIME=1800 BENCH_BASE=http://127.0.0.1:8093 BENCH_MODEL=q38 BENCH_TOKENIZE_PATH=/v1/tokenize \
       BENCH_EMULATE_CONTINUOUS_USAGE=1 timeout 3600 python3 ab_agent.py run "$W" \
       "$D" "$tag" --users $u --think 1 --temperature 1.0 --seed 1234 >> "$D/cells.log" 2>&1) || log "$tag c$u rc=$?"
    log "$tag c$u $(tail -n 1 "$D/cells.log" | cut -c1-200)"
  done
  kill -TERM $sp; waitfor --gone $sp --timeout 90s --quiet || kill -KILL $sp
  wait $sp 2>/dev/null
  kill $tel 2>/dev/null
  { echo "end $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    grep -a -c 'cache-preferred' "$D/serve.log" | sed 's/^/cache-preferred=/'
    grep -a -c 'DSPARK restore' "$D/serve.log" | sed 's/^/dspark-restore=/'
    grep -a -c 'cold-drafter restore' "$D/serve.log" | sed 's/^/cold-drafter-restore=/'
    grep -a -c 'dspark-park SKIPPED' "$D/serve.log" | sed 's/^/dspark-park-skipped=/'
    grep -a -c -i 'out of memory' "$D/serve.log" | sed 's/^/oom-lines=/'
    grep -a -m1 'transient floor' "$D/serve.log" | cut -c1-200 | sed 's/^/floor: /'
    grep -a -m1 '\[prefix-cache\] on:' "$D/serve.log" | cut -c1-160 | sed 's/^/prefix: /'; } >> "$D/arm.txt"
  log "$tag done: $(tail -n 8 "$D/arm.txt" | tr '\n' ' ' | cut -c1-300)"
}
log "switch A/B start $(sha256sum "$BIN" | cut -c1-12)"
arm 1-on ON "$WF" "1 2 4 8"
arm 2-off OFF "$WF" "1 2 4 8"
arm 3-off-c1 OFF "$WF" "1"
arm 4-on-c1 ON "$WF" "1"
arm 5-on-le8k ON "$WL" "2 4 8"
arm 6-off-le8k OFF "$WL" "2 4 8"
log SWITCH_AB_DONE
