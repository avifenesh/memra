#!/usr/bin/env bash
# q-v5w.sh (SE pair): #500 past the stall bound. The shipped timeout_ms ceiling is 90 s, under the
# 120 s stall bound, so this boot sets the measurement override MEMRA_TIMEOUT_MS_MAX=600000 and
# primes 48k tokens with a 600 s deadline on the naked main binary.
set -u
S=/root/rcpt/q-v5w.summary; R5=/root/rcpt/route-receipt-v5w; mkdir -p $R5
until grep -q V5V_DONE /root/rcpt/q-v5v.summary 2>/dev/null; do sleep 60; done
M=/data/dsv4f/nvfp4; X=/root/lane/target-push/release/memra-server
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'); KEY="route-$RANDOM"
env MEMRA_ENV_AUDIT=on MEMRA_TIMEOUT_MS_MAX=600000 MEMRA_MODELS="dsv4f=$M" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" MEMRA_GPU_LOCK=/tmp/memra-gpu.lock setsid nohup $X > $R5/serve.log 2>&1 9>&- &
pid=$!
for _ in $(seq 1 900); do sleep 1; [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/readyz)" == 200 ]] && break; done
python3 /root/box/routecell3.py --port $PORT --key $KEY --out $R5/cells.jsonl > $R5/routecell.out 2>&1
kill -TERM $pid; for _ in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
echo "route 500 $(grep -c '"poll"' $R5/cells.jsonl) polls, $(grep '"request done"' $R5/cells.jsonl | cut -c1-160)" >> $S
echo V5W_DONE >> $S
