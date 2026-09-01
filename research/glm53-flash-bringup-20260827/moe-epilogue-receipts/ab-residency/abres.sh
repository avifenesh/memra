#!/bin/bash
# One boot of the residency-config A/B arm for MEMRA_MOE_FUSED_EPI. usage: abres.sh <tag> <0|1>
# Reconstructed R2 env; the =0 arm must reproduce R2's banked shas before any number is quoted.
set -u
TAG=$1; EPI=$2
ROOT=$HOME/memra
BIN=$ROOT/target/release/memra-server
PIDFILE=$HOME/.memra-ab-server.pid
LOG=$HOME/cell-$TAG.log
bash ~/idle-check.sh || { echo "ABORT $TAG: not idle"; exit 2; }
if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE")
  if [ -r "/proc/$pid/cmdline" ] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q "memra/target/release/memra-server"; then
    kill -TERM "$pid"; for _ in $(seq 1 60); do [ -d "/proc/$pid" ] || break; sleep 1; done
    [ -d "/proc/$pid" ] && kill -KILL "$pid"; sleep 2
  fi
  rm -f "$PIDFILE"
fi
: > "$LOG"
env CUDA_VISIBLE_DEVICES=0,1 MEMRA_SPILL_STATS=1 MEMRA_COMPAT=openai \
  MEMRA_MODELS="zai/glm-5.3-flash=/root/models/glm53-nvfp4" MEMRA_ADDR=127.0.0.1:18400 \
  MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
  MEMRA_PREFIX_CACHE_MB=0 \
  MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PP_SPLITS=24 \
  MEMRA_MOE_SLOTS=18144 MEMRA_MOE_HARD_VRAM_FRAC=0.95 \
  MEMRA_MOE_FUSED_EPI="$EPI" \
  setsid nohup "$BIN" > "$LOG" 2>&1 < /dev/null &
SPID=$!; echo "$SPID" > "$PIDFILE"
for i in $(seq 1 900); do
  grep -q 'listening on' "$LOG" && break
  grep -qE '^\[server\] .*(error|failed)|panicked' "$LOG" && break
  [ -d "/proc/$SPID" ] || break
  sleep 2
done
grep -q 'listening on' "$LOG" || { echo "LOAD FAILED after $((i*2))s"; grep -iE "error|panic|refus" "$LOG" | head -5; exit 1; }
echo "LOAD $TAG ready ~$((i*2))s (pid $SPID) EPI=$EPI"
grep -E "\[pp\] cross-device transport|resident-experts decision" "$LOG" | head -4
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
for i in 0 2 5 7 9; do python3 ~/probe.py w-$i greedy 24 $i > /dev/null 2>&1; done
python3 ~/steady.py "$LOG" $TAG-p5-greedy  greedy  5 192 4
python3 ~/steady.py "$LOG" $TAG-p5-sampled sampled 5 192 4
python3 ~/steady.py "$LOG" $TAG-p7-greedy  greedy  7 192 4
echo "### engagement (full = 42.0/token)"
grep -E "\[moe-fused-epi\] snapshot" "$LOG" | tail -1
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "ARMDONE-$TAG"
