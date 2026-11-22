#!/usr/bin/env bash
# GB10 box, step 11 chain 5: at the head, memra model verify checkpoint (native runner hy4f-gpu)
# then verify serve (memra-server, the draft gate on), and clippy -D warnings on the touched
# crates. Log: logs/s11-chain5.log.
set -u
D=$HOME/hy4flash/s11
L=$HOME/hy4flash/logs
cd $D
echo "CHAIN5 START $(date '+%F %T') $(cat src/HEAD.txt)"
. "$HOME/.cargo/env"
export PATH=/usr/local/cuda-13.0/bin:$PATH CARGO_TARGET_DIR=$D/target CARGO_BUILD_JOBS=16
(cd src && nice -n 10 cargo build --release -p memra-engine --bin hy4f-gpu --bin hy4f-spec -p memra-server --bin memra-server -p memra-cli --bin memra) > $L/s11-build-h5.log 2>&1
echo "BUILD rc=$?"
out=$D/verify/head
rm -rf $out; mkdir -p $D/verify
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
systemd-run --user --scope -q -p MemoryMax=114G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra model verify checkpoint $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --oracle $HOME/hy4flash/pr7/oracle-set --native-runner $D/target/release/hy4f-gpu > $L/s11-verify.log 2>&1
echo "CHECKPOINT rc=$? $(date '+%F %T')"
waitfor --quiet --timeout 10m --every 10s --until $HOME/hy4flash/pr8/memwait.sh > /dev/null
MEMRA_SERVE_VERIFY_TIMEOUT_S=900 systemd-run --user --scope -q -p MemoryMax=119G -p MemorySwapMax=0 -p CPUWeight=50 -- \
  $D/target/release/memra model verify serve $HOME/hy4flash/hf-stage --against hy_v4_flash \
  --out "$out" --native-runner $D/target/release/memra-server >> $L/s11-verify.log 2>&1
echo "SERVE rc=$? $(date '+%F %T')"
cat "$out/gates.txt" 2>/dev/null
sha256sum $D/target/release/memra $D/target/release/hy4f-gpu $D/target/release/memra-server > "$out/bins.sha256"
echo "CLIPPY on cb33"

echo "CHAIN5-DONE $(date '+%F %T')"
