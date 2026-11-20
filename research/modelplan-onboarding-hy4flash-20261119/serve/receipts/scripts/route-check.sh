#!/usr/bin/env bash
# GB10 box: the route after the session split, whole and chunked prefills, against PR 8's tapes.
# usage: route-check.sh   (writes ~/hy4flash/pr9/native/<tag>, log logs/pr9-route-check.log)
set -u
D=$HOME/hy4flash/pr9
R=$HOME/hy4flash/pr7/requests
L=$HOME/hy4flash/logs/pr9-route-check.log
mkdir -p $D/native
run() {
  tag=$1 req=$2; shift 2
  waitfor --quiet --timeout 30m --every 10s --no-proc 'target/release/(hy4f-gpu|hy4f-run|memra-server)' > /dev/null
  waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
  echo "RUN $tag $req $* $(date '+%F %T')" >> $L
  systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
    $D/target/release/hy4f-gpu $HOME/hy4flash/hf-stage --request $R/$req.json --tape $D/native/$tag "$@" \
    > $HOME/hy4flash/logs/pr9-gpu-$tag.log 2>&1
  echo "EXIT $tag rc=$? $(date '+%F %T')" >> $L
  python3 -c "import json;v=json.load(open('$D/native/$tag/tape.json'));print('SHA $tag',v['logits']['sha256'],'chunk',v.get('prefill_chunk'),'calls',len(v['call_seconds']))" >> $L 2>&1
  rm -f $D/native/$tag/logits.f32
}
: > $L
for spec in "${@:-whole}"; do :; done
run long-whole long
run long-c500 long --prefill-chunk 500
run long-c1000 long --prefill-chunk 1000
run long-c2048 long --prefill-chunk 2048
run long-c511 long --prefill-chunk 511
run long-split-c512 long-split --prefill-chunk 512
run long-split-c7 long-split --prefill-chunk 7
run chat-forced chat --force-selections $HOME/hy4flash/pr7/tapes/chat-f64
run chat-c5 chat --prefill-chunk 5
echo "ROUTE_CHECK_DONE $(date '+%F %T')" >> $L
