#!/usr/bin/env bash
# cb33: boot memra-server on the Hy4-flash artifact, run the served-answer gates (chat_gate.py)
# and the readings (bench.py) against it, stop it. One server at a time; it waits for the box
# to hold no other GPU process and the weights' memory.
# usage: serve_gate.sh TAG [gate|bench|both] [extra env assignments for the server...]
# reads: $D/target/release/memra-server, $D/tapes.json, $D/route-gen.json, the PR 7 requests
set -u
tag=$1; what=${2:-both}; shift 2 || shift $#
D=$HOME/hy4flash/pr9
S=$D/src/research/modelplan-onboarding-hy4flash-20261119/serve/scripts
out=$D/serve/$tag
mkdir -p "$out"
port=18391
key=hy4f-serve-gate
log=$out/server.log
waitfor --quiet --timeout 30m --every 10s --no-proc 'target/release/(hy4f-gpu|hy4f-run|memra-server)' > /dev/null
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
echo "START $(date '+%F %T') tag=$tag what=$what env=$*" > "$out/run.txt"
sha256sum $D/target/release/memra-server >> "$out/run.txt"
free -g | head -2 >> "$out/run.txt"
env "$@" MEMRA_COMPAT=memra MEMRA_MODELS="hy4=$HOME/hy4flash/hf-stage" MEMRA_ADDR="127.0.0.1:$port" MEMRA_API_KEY=$key \
  systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra-server > "$log" 2>&1 &
pid=$!
echo "server pid $pid" >> "$out/run.txt"
if ! waitfor --quiet --timeout 15m --every 5s --http "http://127.0.0.1:$port/readyz" > /dev/null; then
  echo "NOT READY $(date '+%F %T')" >> "$out/run.txt"
  kill $pid; wait $pid
  tail -n 20 "$log" >> "$out/run.txt"
  exit 1
fi
echo "READY $(date '+%F %T')" >> "$out/run.txt"
rc=0
if [ "$what" = gate ] || [ "$what" = both ]; then
  python3 $S/chat_gate.py 127.0.0.1:$port $key hy4 $HOME/hy4flash/pr7/requests $D/tapes.json \
    $D/route-gen.json "$out/chat-gate.json" "$log" ${QUICK:+--quick} > "$out/chat-gate.log" 2>&1
  rc=$?
  echo "GATE rc=$rc $(date '+%F %T')" >> "$out/run.txt"
fi
if [ "$what" = bench ] || [ "$what" = both ]; then
  python3 $S/bench.py 127.0.0.1:$port $key hy4 $HOME/hy4flash/pr7/requests "$out/bench.jsonl" \
    > "$out/bench.log" 2>&1
  echo "BENCH rc=$? $(date '+%F %T')" >> "$out/run.txt"
fi
grep -c 'hy4f-step' "$log" >> "$out/run.txt"
kill -TERM $pid
waitfor --quiet --timeout 3m --every 2s --gone $pid > /dev/null || kill -9 $pid
wait $pid 2>/dev/null
echo "STOPPED $(date '+%F %T')" >> "$out/run.txt"
exit $rc
