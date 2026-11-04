#!/usr/bin/env bash
# Rig-side: after the memset step, build the default and combo servers on the GB10 box, then
# serve on fresh boots: (A) default fp32 build c1..c32 (the default-build rerun; covers c2/c8),
# (B) combo build with MEMRA_GDN_STATE_BF16=1 at c2 and c8 (gate-fix check on the bf16 arm).
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-batchspec-gb10-far
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
DEF=f276032189
COMBO=ea2344b203
log "start"
rsync -a --exclude=.git --exclude=target --exclude=research "$HOME/.cache/memra-batchspec/" spark-cb33:projects/memra-batchspec/
ssh spark-cb33 "echo $DEF > ~/projects/memra-batchspec/BATCHSPEC_SHA"
scp -q "$H/combo-src.tar" spark-cb33:memra-run/combo-src.tar
ssh spark-cb33 "tar -xf ~/memra-run/combo-src.tar -C ~/projects && echo $COMBO > ~/projects/memra-combo/COMBO_SHA && rm -f ~/memra-run/combo-src.tar"
scp -q "$H/cb33_servebuild.sh" spark-cb33:memra-run/
ssh spark-cb33 '(cd ~/memra-run && nohup bash cb33_servebuild.sh) > ~/memra-run/step-far.out 2>&1 < /dev/null & echo started'
waitfor --ssh spark-cb33 'grep -q BUILDS_DONE ~/memra-run/step-far.out' --timeout 90m --every 60s --quiet || { log "build wait failed"; exit 2; }
log "builds done"
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)
serve() { # tag bin extra users
  local tag=$1 bin=$2 extra=$3 users=$4
  if ss -ltn | grep -q ':8090 '; then log "rig port 8090 busy; refusing"; exit 4; fi
  ssh spark-cb33 "(ARM=gguf-mtpmask BIN=$bin LEASE=$LEASE EXTRA='$extra' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-$tag.out 2>&1 < /dev/null & echo started"
  waitfor --ssh spark-cb33 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED $tag"; exit 3; }
  ssh -o ExitOnForwardFailure=yes -N -L 8090:127.0.0.1:8090 spark-cb33 &
  local tun=$!
  waitfor --http http://127.0.0.1:8090/v1/models --timeout 60s --quiet
  BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 NODE=spark-cb33 QUALITY=0 USERS="$users" \
    BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
    bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/$tag" "$tag"
  log "done $tag"
  kill $tun 2>/dev/null
  local pid
  pid=$(ssh spark-cb33 'cat ~/memra-run/serve-bs-gguf-mtpmask.pid')
  ssh spark-cb33 "kill $pid"
  waitfor --ssh spark-cb33 "test ! -e /proc/$pid" --timeout 3m --quiet
  scp -q "spark-cb33:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/$tag/server.log" 2>/dev/null
}
serve fp32-r1 "~/memra-run/bin-batchspec-$DEF" "" "1 2 4 8 16 32"
serve bf16-r1 "~/memra-run/bin-combo-$COMBO" "MEMRA_GDN_STATE_BF16=1" "1 2 4 8 16 32"
serve bf16-r2 "~/memra-run/bin-combo-$COMBO" "MEMRA_GDN_STATE_BF16=1" "1 2 4 8 16 32"
serve fp32-r2 "~/memra-run/bin-batchspec-$DEF" "" "1 2 4 8 16 32"
echo FAR_DONE >> "$OUT/chain.log"
