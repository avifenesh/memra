#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: memra-server under Nsight Systems, one cold served
# prompt of TARGET tokens with 512 max tokens (the lane's nsys cell, research/gb10-prefill-20261106/
# tail-fold/profile_arm2.sh: CUDA buffers flushed every second, 30 s idle before the stop), then the
# publication reading (specpub_gap.py) and the lane's copy census (r6_big.py).
# usage: specpub_nsys.sh NAME BIN_DIR TARGET
set -u
D=$HOME/.cache/memra-1221
NAME=$1; BIN=$D/bins/$2; TARGET=${3:-131072}
OUT=$D/runs/nsys/$NAME; mkdir -p "$OUT"
NSYS=/usr/local/cuda/bin/nsys
echo "$(date -Is) nsys cell OUT=$OUT BIN=$BIN TARGET=$TARGET" | tee "$OUT/run.log"
WRAP="$NSYS profile -t cuda,nvtx -s none --cpuctxsw=none --cuda-graph-trace=node --cuda-flush-interval=1000 --force-overwrite=true -o $OUT/prof"
OUT=$OUT BIN=$BIN EXTRA="MEMRA_TICK_TRACE=1" WRAP=$WRAP bash "$D/box/serve.sh" &
sp=$!
waitfor --file "$OUT/server.pid" --timeout 1m > /dev/null 2>&1
if ! waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
     --fail-if "! kill -0 $sp" --every 5s --timeout 20m >> "$OUT/run.log" 2>&1; then
  echo "$(date -Is) server never became ready" | tee -a "$OUT/run.log"
  kill -INT "$(cat "$OUT/server.pid")"; wait $sp; exit 3
fi
python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets "$TARGET" --reps 1 --max-tokens 512 --tag "$NAME" >> "$OUT/run.log" 2>&1
echo "$(date -Is) bench rc=$?" | tee -a "$OUT/run.log"
sleep 30
kill -INT "$(cat "$OUT/server.pid")"   # nsys ends the session and SIGTERMs the server
waitfor --until "test -f $OUT/prof.nsys-rep && ! kill -0 $sp 2>/dev/null && exit 0 || exit 1" --every 10s --timeout 20m >> "$OUT/run.log" 2>&1
wait $sp
$NSYS export --type sqlite --force-overwrite=true -o "$OUT/prof.sqlite" "$OUT/prof.nsys-rep" >> "$OUT/run.log" 2>&1
ttft=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).read().splitlines()[-1])['ttft_s'])" "$OUT/pf.jsonl")
python3 "$D/box/specpub_gap.py" "$OUT/prof.sqlite" "$ttft" > "$OUT/gap.txt" 2>&1
python3 "$D/box/r6_big.py" "$OUT/prof.sqlite" "$ttft" > "$OUT/big.txt" 2>&1
grep -E '\[tick-spec\]|\[prefix-cache\]|\[spec\]' "$OUT/server.log" > "$OUT/trace.txt"
cat "$OUT/gap.txt"
echo "$(date -Is) done" | tee -a "$OUT/run.log"
