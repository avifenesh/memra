#!/usr/bin/env bash
# Runs ON the GB10 host: memra-server under Nsight Systems with osrt and CPU sampling (the host time before the first prime kernel), one cold served prompt, then stop.
# usage: profile_arm.sh OUT BIN "EXTRA env" TARGET
set -u
OUT=$1; BIN=$2; EXTRA=${3:-}; TARGET=${4:-131072}
D=$HOME/.cache/memra-gb10-r4
NSYS=/usr/local/cuda/bin/nsys
mkdir -p "$OUT"
echo "$(date -Is) profile_arm OUT=$OUT BIN=$BIN EXTRA=$EXTRA TARGET=$TARGET" | tee "$OUT/run.log"
WRAP="$NSYS profile -t cuda,osrt,nvtx -s cpu --cpuctxsw=none --force-overwrite=true -o $OUT/prof"
OUT=$OUT BIN=$BIN EXTRA=$EXTRA WRAP=$WRAP bash "$D/box/serve.sh" &
sp=$!
sleep 5
if ! waitfor --until 'curl -sf -o /dev/null http://127.0.0.1:8090/readyz && exit 0 || exit 1' \
     --fail-if "! kill -0 $sp" --every 5s --timeout 20m >> "$OUT/run.log" 2>&1; then
  echo "$(date -Is) server never became ready" | tee -a "$OUT/run.log"
  kill -INT "$(cat "$OUT/server.pid")"; wait $sp; exit 3
fi
echo "$(date -Is) ready" | tee -a "$OUT/run.log"
python3 "$D/box/pf_long_memra.py" "$OUT/pf.jsonl" --targets "$TARGET" --reps 1 --tag "$(basename "$OUT")" >> "$OUT/run.log" 2>&1
echo "$(date -Is) bench rc=$?" | tee -a "$OUT/run.log"
kill -INT "$(cat "$OUT/server.pid")"   # nsys ends the session and SIGTERMs the server
waitfor --until "test -f $OUT/prof.nsys-rep && ! kill -0 $sp 2>/dev/null && exit 0 || exit 1" --every 10s --timeout 20m >> "$OUT/run.log" 2>&1
wait $sp
ls -la "$OUT"/prof.* >> "$OUT/run.log" 2>&1
$NSYS export --type sqlite --force-overwrite=true -o "$OUT/prof.sqlite" "$OUT/prof.nsys-rep" >> "$OUT/run.log" 2>&1
echo "$(date -Is) done" | tee -a "$OUT/run.log"
