#!/usr/bin/env bash
# Rig-side chain for the bf16 GDN state research arm on the GB10 box: wait for the raw-V A/B,
# build the arm, coherence probe on the base and the arm, then the agentic cell c1..c32 plus the
# 264-task quality panel on the arm.
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-batchspec-gb10-gdn
mkdir -p "$OUT"
NODE=spark-cb33
LEASE=$(cat ~/.cache/model-scout/memra-cb33-lease.txt)
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
waitfor --ssh $NODE 'grep -q STEP_DONE ~/memra-run/step-rawv.out' --timeout 90m --every 60s --quiet || { log "rawv wait failed"; exit 2; }
log "rawv done; building gdn arm"
rsync -a --exclude=.git --exclude=target --exclude=research "$HOME/.cache/memra-gdnbf16/" $NODE:projects/memra-gdnbf16/
scp -q "$H/cb33_gdn_build.sh" $NODE:memra-run/
ssh $NODE '(cd ~/memra-run && nohup bash cb33_gdn_build.sh) > ~/memra-run/step-gdn.out 2>&1 < /dev/null & echo started'
waitfor --ssh $NODE 'grep -q GDN_BUILD_DONE ~/memra-run/step-gdn.out' --timeout 120m --every 60s --quiet || { log "gdn build wait failed"; exit 3; }
log "gdn build + bench done"
boot() { # tag bin extra
  ssh $NODE "(ARM=gguf-mtpmask BIN=$2 LEASE=$LEASE EXTRA='$3' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-gdn-$1.out 2>&1 < /dev/null & echo started"
  waitfor --ssh $NODE 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED $1"; return 1; }
  ssh -o ExitOnForwardFailure=yes -N -L 8090:127.0.0.1:8090 $NODE &
  TUN=$!
  waitfor --http http://127.0.0.1:8090/v1/models --timeout 60s --quiet
}
stop() { # tag
  kill $TUN 2>/dev/null
  PID=$(ssh $NODE 'cat ~/memra-run/serve-bs-gguf-mtpmask.pid')
  ssh $NODE "kill $PID"
  waitfor --ssh $NODE "test ! -e /proc/$PID" --timeout 3m --quiet
  scp -q "$NODE:memra-run/serve-bs-gguf-mtpmask.log" "$OUT/server-$1.log" 2>/dev/null
}
L5=$(cat $H/L5_SHA)
boot base "~/memra-run/bin-batchspec-$L5" "" && {
  python3 "$H/coherence.py" http://127.0.0.1:8090 q38 "$OUT/coherence-base.jsonl" >> "$OUT/chain.log" 2>&1
  stop base
}
log "base coherence done"
boot bf16 "~/memra-run/bin-gdnbf16" "MEMRA_GDN_STATE_BF16=1" && {
  python3 "$H/coherence.py" http://127.0.0.1:8090 q38 "$OUT/coherence-bf16.jsonl" >> "$OUT/chain.log" 2>&1
  BENCH_BASE=http://127.0.0.1:8090 BENCH_MODEL=q38 NODE=$NODE QUALITY=1 USERS="1 2 4 8 16 32" \
    BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
    bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/memra-gdnbf16" "memra-gdnbf16"
  stop bf16
}
log "bf16 arm done"
echo GDN_CHAIN_DONE >> "$OUT/chain.log"
