#!/usr/bin/env bash
# Rig-side: combined build on the GB10 box: build + profile (cb33_combo.sh), then the served
# agentic cell c1..c32 on a fresh boot with MEMRA_GDN_STATE_BF16=1 (the TC b32 tier is chosen
# by the boot probe as on the default build).
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-batchspec-gb10-combo
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
SHA=$(cat $H/COMBO_SHA)
rsync -a --exclude=.git --exclude=target --exclude=research "$HOME/.cache/memra-batchspec/" spark-cb33:projects/memra-batchspec/
ssh spark-cb33 "echo $SHA > ~/projects/memra-batchspec/BATCHSPEC_SHA"
scp -q "$H/cb33_combo.sh" spark-cb33:memra-run/
ssh spark-cb33 '(cd ~/memra-run && nohup bash cb33_combo.sh) > ~/memra-run/step-combo.out 2>&1 < /dev/null & echo started'
waitfor --ssh spark-cb33 'grep -q STEP_DONE ~/memra-run/step-combo.out' --timeout 150m --every 60s --quiet || { log "combo step wait failed"; exit 2; }
log "combo step done"
if ss -ltn | grep -q ':8090 '; then log "rig port 8090 busy; refusing"; exit 4; fi
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)
tag=memra-combo-bf16
ssh spark-cb33 "(ARM=gguf-mtpmask BIN=~/memra-run/bin-batchspec-$SHA LEASE=$LEASE EXTRA='MEMRA_GDN_STATE_BF16=1' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-$tag.out 2>&1 < /dev/null & echo started"
waitfor --ssh spark-cb33 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED"; exit 3; }
ssh -o ExitOnForwardFailure=yes -N -L 8090:127.0.0.1:8090 spark-cb33 &
TUN=$!
waitfor --http http://127.0.0.1:8090/v1/models --timeout 60s --quiet
BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 NODE=spark-cb33 QUALITY=0 USERS="1 2 4 8 16 32" \
  BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
  bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/$tag" "$tag"
log "done $tag"
kill $TUN 2>/dev/null
PID=$(ssh spark-cb33 'cat ~/memra-run/serve-bs-gguf-mtpmask.pid')
ssh spark-cb33 "kill $PID"
waitfor --ssh spark-cb33 "test ! -e /proc/$PID" --timeout 3m --quiet
scp -q "spark-cb33:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/$tag/server.log" 2>/dev/null
echo COMBO_DONE >> "$OUT/chain.log"
