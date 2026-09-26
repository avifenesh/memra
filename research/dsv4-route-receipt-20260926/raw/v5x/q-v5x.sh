#!/usr/bin/env bash
# q-v5x.sh (SE pair): the #501 estimate fix $1 on the route receipt's own shape: a 12000-word prime,
# then 16 concurrent short requests (v5t shed eight of them on a 138 s estimate).
set -u
RSHA=$1
S=/root/rcpt/q-v5x.summary; R=/root/rcpt/route-receipt-v5x; mkdir -p $R
until grep -q V5W_DONE /root/rcpt/q-v5w.summary 2>/dev/null; do sleep 60; done
. /root/.cargo/env; export PATH=/usr/local/cuda/bin:$PATH MEMRA_CUDA_ARCH=120a MEMRA_NVCC=/usr/local/cuda/bin/nvcc
cd /root/lane/memra && git fetch -q origin lane/dsv4-route-receipt-20260926
git worktree add -f /root/lane/t-rr $RSHA > /dev/null 2>&1; git -C /root/lane/t-rr checkout -q --detach $RSHA
[[ -d /root/lane/target-rr ]] || cp -a /root/lane/target-pmain /root/lane/target-rr
bash /root/box/build.sh /root/lane/t-rr /root/lane/target-rr rr
echo "build rr $(git -C /root/lane/t-rr rev-parse --short HEAD) $(grep -hE 'EXIT' /root/build-rr.log | tr '\n' ' ')" >> $S
M=/data/dsv4f/nvfp4; X=/root/lane/target-rr/release/memra-server
sha256sum $X > $R/binary.sha256
exec 9>/tmp/memra-gpu.lock; while ! flock -n 9; do sleep 10; done
PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()'); KEY="route-$RANDOM"
env MEMRA_ENV_AUDIT=on MEMRA_MODELS="dsv4f=$M" MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_API_KEY="$KEY" MEMRA_GPU_LOCK=/tmp/memra-gpu.lock setsid nohup $X > $R/serve.log 2>&1 9>&- &
pid=$!
for _ in $(seq 1 900); do sleep 1; [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/readyz)" == 200 ]] && break; done
python3 /root/box/routecell4.py --port $PORT --key $KEY --out $R/cells.jsonl > $R/routecell.out 2>&1
curl -s --max-time 10 -H "authorization: Bearer $KEY" http://127.0.0.1:$PORT/metrics > $R/metrics-final.txt
kill -TERM $pid; for _ in $(seq 1 90); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
echo "c16: $(grep -c '"status": 200' $R/cells.jsonl) rows at 200, $(grep -c '"status": 429' $R/cells.jsonl) at 429" >> $S
echo V5X_DONE >> $S
