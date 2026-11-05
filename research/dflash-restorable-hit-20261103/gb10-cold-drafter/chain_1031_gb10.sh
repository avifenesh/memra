#!/usr/bin/env bash
# Rig-side chain on spark-b23d for memra#1031's second half: exactness gate for
# MEMRA_DSPARK_COLD_DRAFTER=1, then the GB10 GGUF DFlash2 agentic cell c1..c32 with the door on
# and off (same binary, fresh boot each), port 8096 on the rig.
set -u
H=$HOME/.cache/memra-batchspec-harness
OUT=$HOME/.cache/memra-1031-gb10
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/chain.log"; }
NODE=spark-b23d
SHA=$(cat "$H/SHA_1031")
LEASE=claude-code-20261103T022643-3047127
waitfor --ssh $NODE 'grep -q BUILD_DONE ~/memra-run/step-1031.out' --timeout 60m --every 30s --quiet || { log "build wait failed"; exit 2; }
log "build done $SHA ($(ssh $NODE 'tail -n 2 ~/memra-run/step-1031/build.log | tr "\n" " "'))"
ssh $NODE "cd ~/memra-run && M=~/models/memra-q38 GGUF=~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf \
  DF=~/models/memra-q38/Qwen3.8-27B-DFlash2-memra RANKS=~/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/q38-ranks-sxc32768.gguf.txt \
  GATE=~/memra-run/dspark-cold-drafter-gate.py SPARKLOCK_LEASE=$LEASE \
  timeout 7200 bash ~/memra-run/cold-drafter-gate.sh ~/memra-run/bin-1031-$SHA/memra-server ~/memra-run/gate-1031 > ~/memra-run/gate-1031.out 2>&1"
rsync -a $NODE:memra-run/gate-1031/ "$OUT/gate/"
log "gate: $(grep -E 'GATE|=' "$OUT/gate/compare.txt" | tr '\n' ' ')"
serve() { # tag extra users
  local tag=$1 extra=$2 users=$3
  if ss -ltn | grep -q ':8096 '; then log "rig port 8096 busy; refusing"; exit 4; fi
  ssh $NODE "(ARM=gguf-dflash BIN=~/memra-run/bin-1031-$SHA LEASE=$LEASE EXTRA='$extra' nohup bash ~/memra-run/serve-batchspec.sh) > ~/memra-run/serve-$tag.out 2>&1 < /dev/null & echo started"
  waitfor --ssh $NODE 'curl -sf http://127.0.0.1:8090/v1/models >/dev/null' --timeout 20m --every 15s --quiet || { log "boot FAILED $tag"; exit 3; }
  ssh -o ExitOnForwardFailure=yes -N -L 8096:127.0.0.1:8090 $NODE &
  local tun=$!
  waitfor --http http://127.0.0.1:8096/v1/models --timeout 60s --quiet
  BENCH_BASE=http://127.0.0.1:8096 BENCH_MODEL=q38 NODE=$NODE QUALITY=0 USERS="$users" \
    BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 CLIENT_MAX_TIME=1800 \
    bash ~/.cache/memra-spark/b23d/bench_memra.sh "$OUT/$tag" "$tag"
  log "done $tag"
  kill $tun 2>/dev/null
  local pid
  pid=$(ssh $NODE 'cat ~/memra-run/serve-bs-gguf-dflash.pid')
  ssh $NODE "kill $pid"
  waitfor --ssh $NODE "test ! -e /proc/$pid" --timeout 3m --quiet
  scp -q "$NODE:memra-run/serve-bs-gguf-dflash.log" "$OUT/$tag/server.log" 2>>"$OUT/scp.err" || true
}
if grep -q "GATE PASS" "$OUT/gate/compare.txt"; then
  serve dflash-colddrafter "MEMRA_DSPARK_COLD_DRAFTER=1" "1 2 4 8 16 32"
  serve dflash-base "" "1 2 4 8 16 32"
else
  log "gate did not pass; no served runs"
fi
echo CHAIN_1031_DONE >> "$OUT/chain.log"
