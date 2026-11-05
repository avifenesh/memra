#!/usr/bin/env bash
# Rig-side chain for the non-exact research arms on the GB10 box (default untouched):
#   C: attention partition (MEMRA_FA_SPLIT=$S on the default binary): coherence probe base vs arm,
#      then the served cell c1..c32 with the 264-task quality panel.
#   B: W4A4 decode (research/w4a4-decode, MEMRA_W4A4_DECODE=1): build + tick A/B + bench accuracy,
#      coherence probe, served c1..c32 + quality panel.
# Hard stop: nothing may run on the box from 09:40 Jerusalem (another session's benchmark); each
# long step starts only if it can finish by 09:35.
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-batchspec-gb10-optc
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
DEADLINE=$(TZ=Asia/Jerusalem date -d "2026-11-05 09:35" +%s)
fits(){ [ $(( $(date +%s) + $1 * 60 )) -le $DEADLINE ]; }
DEF=655a07cb82
W4=a53944e7aa
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)

boot(){ # bin extra
  if ss -ltn | grep -q ':8090 '; then log "rig port 8090 busy; refusing"; exit 4; fi
  ssh spark-cb33 "(ARM=gguf-mtpmask BIN=$1 LEASE=$LEASE EXTRA='$2' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-opts.out 2>&1 < /dev/null & echo started"
  waitfor --ssh spark-cb33 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED"; exit 3; }
  ssh -o ExitOnForwardFailure=yes -N -L 8090:127.0.0.1:8090 spark-cb33 &
  TUN=$!
  waitfor --http http://127.0.0.1:8090/v1/models --timeout 60s --quiet
}
halt(){ # tag
  kill $TUN 2>/dev/null
  local pid
  pid=$(ssh spark-cb33 'cat ~/memra-run/serve-bs-gguf-mtpmask.pid')
  ssh spark-cb33 "kill $pid"
  waitfor --ssh spark-cb33 "test ! -e /proc/$pid" --timeout 3m --quiet
  scp -q "spark-cb33:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/server-$1.log" 2>>"$OUT/scp.err" || scp -q "spark-cb33:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/server-$1.log" 2>>"$OUT/scp.err"
}
log "start; waiting for the split sweep and the chosen split"
waitfor --ssh spark-cb33 'grep -q STEP_DONE ~/memra-run/step-fasplit.out' --timeout 120m --every 60s --quiet || { log "sweep wait failed"; exit 2; }
waitfor --file "$H/C_SPLIT" --timeout 60m --quiet || { log "no C_SPLIT"; exit 2; }
S=$(cat "$H/C_SPLIT")
log "C split = $S"
# --- C ---
boot "~/memra-run/bin-batchspec-$DEF" ""
python3 "$H/coherence.py" http://127.0.0.1:8090 q38 "$OUT/coherence-base.jsonl" >> "$OUT/chain.log" 2>&1
halt coh-base
log "coherence base done"
boot "~/memra-run/bin-batchspec-$DEF" "MEMRA_FA_SPLIT=$S"
python3 "$H/coherence.py" http://127.0.0.1:8090 q38 "$OUT/coherence-c.jsonl" >> "$OUT/chain.log" 2>&1
if fits 150; then
  BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 NODE=spark-cb33 QUALITY=1 USERS="1 2 4 8 16 32" \
    BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
    bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/c-split$S" "c-split$S"
  log "C served + quality done"
else
  log "C served skipped: would cross the 09:40 quiet window"
fi
halt c
echo C_DONE >> "$OUT/chain.log"
