#!/usr/bin/env bash
# prof-served.sh <receipt-dir> <binary> [VAR=value ...]
# nsys over the SERVED route: memra-server under `nsys launch`, warmup, then a capture window
# around 2 greedy ignore-eos requests of 256 tokens. Holds /tmp/memra-gpu.lock for the boot.
set -uo pipefail
receipt=$1; bin=$2; shift 2
mkdir -p "$receipt"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || { echo SLOT_BUSY; exit 75; }
export PATH=/usr/local/cuda/bin:/usr/local/bin:$PATH MEMRA_GPU_LOCK=/tmp/memra-gpu.lock
nvidia-smi --query-compute-apps=pid --format=csv,noheader > "$receipt/pre-apps.csv"
[[ ! -s "$receipt/pre-apps.csv" ]] || { echo GPU_NOT_IDLE; exit 76; }
sha256sum "$bin" > "$receipt/binary.sha256"; printf '%s\n' "$@" > "$receipt/env.txt"
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')
KEY="dsv4f-prof-$RANDOM$RANDOM"; SES="srv$RANDOM"
env "$@" MEMRA_MODELS="dsv4f=/data/dsv4f/nvfp4" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" \
  setsid nohup nsys launch --session-new="$SES" -t cuda,nvtx,osrt --cuda-graph-trace=node "$bin" \
  > "$receipt/serve.log" 2>&1 &
lp=$!
ready=0; t0=$(date +%s)
for _ in $(seq 1 1800); do
  kill -0 $lp 2>/dev/null || break
  [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/readyz" || true)" == 200 ]] && { ready=1; break; }
  sleep 1
done
[[ $ready == 1 ]] || { echo BOOT_FAIL; tail -30 "$receipt/serve.log"; nsys shutdown --session="$SES" --kill sigkill; exit 71; }
echo "READY after=$(( $(date +%s) - t0 ))s" | tee "$receipt/controller.log"
python3 /root/box/bench.py --port $PORT --key $KEY --out "$receipt/warm.jsonl" --label warmup --conc 1 --n 1 --max-tokens 32 --greedy 2>&1 | tee -a "$receipt/controller.log"
nsys start --session="$SES" -o "$receipt/nsys" -f true >> "$receipt/controller.log" 2>&1
python3 /root/box/bench.py --port $PORT --key $KEY --out "$receipt/cells.jsonl" --label prof-greedy --conc 1 --n 2 --max-tokens 256 --greedy --ignore-eos --keep-text 2>&1 | tee -a "$receipt/controller.log"
nsys stop --session="$SES" >> "$receipt/controller.log" 2>&1
nsys shutdown --session="$SES" --kill sigterm >> "$receipt/controller.log" 2>&1
for _ in $(seq 1 90); do kill -0 $lp 2>/dev/null || break; sleep 1; done
kill -0 $lp 2>/dev/null && kill -KILL $lp
sleep 5
nvidia-smi --query-compute-apps=pid,process_name --format=csv > "$receipt/terminal-apps.csv"
nsys export --type sqlite -o "$receipt/nsys.sqlite" -f true "$receipt/nsys.nsys-rep" > /dev/null 2>&1
python3 /root/box/nsys-ana.py "$receipt/nsys.sqlite" 510 > "$receipt/ana.txt" 2>&1
echo "PROF_DONE $(head -3 "$receipt/ana.txt" | tr '\n' ' ')" | tee -a "$receipt/controller.log"
