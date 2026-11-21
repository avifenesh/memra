#!/usr/bin/env bash
# GB10 box: memra model verify checkpoint (native runner hy4f-gpu) then verify serve (memra-server)
# on the same bundle. usage: verify.sh TAG
set -u
D=$HOME/hy4flash/pr10
tag=$1
out=$D/verify/$tag
mkdir -p $D/verify
rm -rf "$out"
log=$HOME/hy4flash/logs/pr10-verify-$tag.log
waitfor --quiet --timeout 30m --every 10s --no-proc 'target/release/(hy4f-gpu|hy4f-run|hy4f-spec|memra-server)' > /dev/null
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
echo "START $(date '+%F %T') tag=$tag" > $log
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra model verify checkpoint $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --oracle $HOME/hy4flash/pr7/oracle-set --native-runner $D/target/release/hy4f-gpu >> $log 2>&1
echo "CHECKPOINT rc=$? $(date '+%F %T')" >> $log
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
MEMRA_SERVE_VERIFY_TIMEOUT_S=900 systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra model verify serve $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --native-runner $D/target/release/memra-server >> $log 2>&1
echo "SERVE rc=$? $(date '+%F %T')" >> $log
cat "$out/gates.txt" >> $log 2>/dev/null
sha256sum $D/target/release/memra $D/target/release/hy4f-gpu $D/target/release/memra-server > "$out/bins.sha256"
