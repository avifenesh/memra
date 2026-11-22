#!/usr/bin/env bash
# GB10 box, batch lane: memra model verify checkpoint (native runner hy4f-gpu) and verify serve
# (memra-server, drafts on with the gate) with bins-TAG. usage: verify.sh TAG
set -u
tag=$1
D=$HOME/hy4flash/batch
B=$D/bins-$tag
L=$HOME/hy4flash/logs
out=$D/verify/$tag
rm -rf $out; mkdir -p $D/verify
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $B/memra model verify checkpoint $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --oracle $HOME/hy4flash/pr7/oracle-set --native-runner $B/hy4f-gpu > $L/batch-verify-$tag.log 2>&1
echo "CHECKPOINT rc=$? $(date '+%F %T')"
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
MEMRA_SERVE_VERIFY_TIMEOUT_S=900 systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $B/memra model verify serve $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --native-runner $B/memra-server >> $L/batch-verify-$tag.log 2>&1
echo "SERVE rc=$? $(date '+%F %T')"
cat "$out/gates.txt" 2>/dev/null
(cd $D && sha256sum bins-$tag/memra bins-$tag/hy4f-gpu bins-$tag/memra-server > "$out/bins.sha256")
