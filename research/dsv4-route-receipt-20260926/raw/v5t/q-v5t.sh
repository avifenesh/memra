#!/usr/bin/env bash
# q-v5t.sh (SE pair): the DSv4 route's two-card receipt (#500 #501 #503) on main's naked defaults
# (the push lane's top 8a14c2c95 is main's DSv4 tree). The server's log carries the [admit-mem]
# verdicts; nvidia-smi before and after each phase gives the measured device deltas.
set -u
S=/root/rcpt/q-v5t.summary; R=/root/rcpt/route-receipt-v5t; mkdir -p $R
X=/root/lane/target-push/release/memra-server; M=/data/dsv4f/nvfp4
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'); KEY="route-$RANDOM"
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $R/vram-pre-boot.csv
env MEMRA_ENV_AUDIT=on MEMRA_MODELS="dsv4f=$M" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" MEMRA_GPU_LOCK=/tmp/memra-gpu.lock setsid nohup $X > $R/serve.log 2>&1 9>&- &
pid=$!
( for _ in $(seq 1 900); do curl -s --max-time 5 http://127.0.0.1:$PORT/health; echo; sleep 1; [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/readyz)" == 200 ]] && break; done ) > $R/health-during-load.txt 2>&1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $R/vram-ready.csv
python3 /root/box/routecell.py --port $PORT --key $KEY --out $R/cells.jsonl > $R/routecell.out 2>&1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > $R/vram-after.csv
curl -s --max-time 10 -H "authorization: Bearer $KEY" http://127.0.0.1:$PORT/metrics > $R/metrics-final.txt
kill -TERM $pid; for _ in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
echo "route receipt $(grep -c '"cell"' $R/cells.jsonl) rows, admit-mem lines $(grep -c 'admit-mem' $R/serve.log)" >> $S
echo V5T_DONE >> $S
