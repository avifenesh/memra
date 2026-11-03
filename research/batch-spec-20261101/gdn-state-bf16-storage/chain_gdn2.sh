#!/usr/bin/env bash
# Rig-side: after the tensor-core served cells, the bf16-stored GDN state arm on the GB10 box:
# build + identity + decode A/B (cb33_gdn2.sh), then a clean served A/B: two fresh boots of the
# same binary, flag off then on, agentic c1..c32 each, nothing sent to a server before its cells.
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-batchspec-gb10-gdn2
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
waitfor --file "$HOME/.cache/memra-batchspec-gb10-tc/chain.log" --grep TC_SERVED_DONE --timeout 150m --quiet || { log "tc served wait failed"; exit 2; }
rsync -a --exclude=.git --exclude=target --exclude=research "$HOME/.cache/memra-gdnbf16/" spark-cb33:projects/memra-gdnbf16/
scp -q "$H/cb33_gdn2.sh" spark-cb33:memra-run/
ssh spark-cb33 '(cd ~/memra-run && nohup bash cb33_gdn2.sh) > ~/memra-run/step-gdn2.out 2>&1 < /dev/null & echo started'
waitfor --ssh spark-cb33 'grep -q STEP_DONE ~/memra-run/step-gdn2.out' --timeout 120m --every 60s --quiet || { log "gdn2 step wait failed"; exit 3; }
log "gdn2 bench done: $(ssh spark-cb33 'head -n 2 ~/memra-run/step-gdn2/ab.txt | tr "\n" " "')"
if ss -ltn | grep -q ':8090 '; then log "rig port 8090 busy; refusing"; exit 4; fi
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)
for arm in off on; do
  EXTRA=""; [ $arm = on ] && EXTRA="MEMRA_GDN_STATE_BF16=1"
  tag="memra-gdn-$arm"
  ssh spark-cb33 "(ARM=gguf-mtpmask BIN=~/memra-run/bin-gdnbf16 LEASE=$LEASE EXTRA='$EXTRA' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-$tag.out 2>&1 < /dev/null & echo started"
  waitfor --ssh spark-cb33 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED $tag"; continue; }
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
done
echo GDN2_DONE >> "$OUT/chain.log"
