#!/usr/bin/env bash
# RTX 5090 Laptop (24 GB): concurrency follow-up to rig5090_ab.sh. On the full 29-body workload the
# 17-18k-token bodies are refused at c2+ (HTTP 400 context_length_exceeded: they do not fit an idle
# 24 GB box with another session's residue), doors on from c2, doors off from c4. This run uses the
# 22 bodies whose prompts are at most ~8.1k tokens (ab-workload-5090-le8k.json, the 7 longest
# dropped), same binary, config and telemetry; c2, c4, c8 per arm, fresh boot, order DEF then OFF.
set -u
BIN=$HOME/.cache/memra-1031-rebase/memra-server-flip-6561499516
M=/data/ai-ml/models/q38-gguf
H=$HOME/.cache/model-scout/harness
W=$HOME/.cache/memra-1031-rebase/ab-workload-5090-le8k.json
O=$HOME/.cache/memra-1031-rebase/rig5090
mkdir -p "$O"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$O/chain.log"; }
OFF="MEMRA_DSPARK_PREFIX_RESTORE=0 MEMRA_DSPARK_PARTIAL_RESTORE=0 MEMRA_DSPARK_BOUNDARY_CAPTURE=0"
arm() { # tag doors users
  local tag=$1 doors=$2 users=$3 D=$O/$1
  if ss -ltn | grep -q ':8093 '; then log "port 8093 busy before $tag; refusing"; exit 4; fi
  [ -d "$D" ] && mv "$D" "$D-prev-$(date +%H%M%S)"
  mkdir -p "$D"
  { echo "arm $tag doors=$doors users=$users $(date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) load=$(cut -d' ' -f1-3 /proc/loadavg)"
    sha256sum "$BIN"; nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit --format=csv,noheader; } > "$D/arm.txt"
  local extra=""
  [ "$doors" = OFF ] && extra=$OFF
  env -u MEMRA_DSPARK_PREFIX_RESTORE -u MEMRA_DSPARK_PARTIAL_RESTORE -u MEMRA_DSPARK_BOUNDARY_CAPTURE \
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
  log "start $tag $(grep -a 'prefix restore doors default' "$D/serve.log" | head -1)"
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
    grep -a -c -i 'out of memory' "$D/serve.log" | sed 's/^/oom-lines=/'
    grep -a -c ' 429' "$D/serve.log" | sed 's/^/429-lines=/'; } >> "$D/arm.txt"
  log "$tag done: $(tail -n 5 "$D/arm.txt" | tr '\n' ' ')"
}
log "rig5090 concurrency start $(sha256sum "$BIN" | cut -c1-12) workload $(sha256sum "$W" | cut -c1-12)"
arm 5-def-le8k DEF "2 4 8"
arm 6-off-le8k OFF "2 4 8"
log RIG5090_CONC_DONE
