#!/usr/bin/env bash
# GB10 box, batch lane: nsys of a served c=N decode. Boots memra-server (BIN) under `nsys launch`,
# sends N chat requests at once (256 tokens each, sampled, seeded), captures DUR seconds of
# their decode from 4 s after they start, stops the server.
# usage: prof-serve.sh TAG BIN N DUR [env assignments...]
set -u
tag=$1; bin=$2; nreq=$3; dur=$4; shift 4
D=$HOME/hy4flash/batch
P=$D/prof; mkdir -p $P
model=$HOME/hy4flash/hf-stage
port=18393
key=hy4f-batch
log=$P/$tag-server.log
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null || { echo REFUSED; exit 2; }
echo "START $(date '+%F %T') tag=$tag n=$nreq dur=$dur env=$*"
env "$@" MEMRA_COMPAT=memra MEMRA_MODELS="hy4=$model" MEMRA_ADDR="127.0.0.1:$port" MEMRA_API_KEY=$key \
  systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -- \
  nsys launch --session-new=hy4b-$tag -t cuda,osrt --cuda-memory-usage=false \
  $bin > "$log" 2>&1 &
pid=$!
waitfor --quiet --timeout 15m --every 5s --http "http://127.0.0.1:$port/readyz" > /dev/null || { echo NOT-READY; kill $pid; exit 1; }
cat > $P/$tag-req.py <<PY
import json, sys, threading, urllib.request
def one(i):
    body = {"model": "hy4", "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper, at least 600 words."}],
            "max_tokens": 256, "seed": 20261121 + i, "temperature": 0.9, "reasoning_effort": "no_think", "ignore_eos": True}
    req = urllib.request.Request("http://127.0.0.1:$port/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer $key", "Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=3600).read()
ts = [threading.Thread(target=one, args=(i,)) for i in range($nreq)]
[t.start() for t in ts]; [t.join() for t in ts]
print("REQ-DONE")
PY
python3 $P/$tag-req.py > $P/$tag-req.log 2>&1 &
rq=$!
sleep 4
nsys start --session=hy4b-$tag -s none --cpuctxsw=none -o $P/$tag -f true
sleep $dur
nsys stop --session=hy4b-$tag
wait $rq
kill -TERM $pid; waitfor --quiet --timeout 3m --every 2s --gone $pid > /dev/null || kill -9 $pid
wait $pid 2>/dev/null
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum -f csv -o $P/$tag $P/$tag.nsys-rep > $P/$tag-stats.log 2>&1
echo "PROF-DONE $(date '+%F %T')"
