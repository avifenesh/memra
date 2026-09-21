#!/usr/bin/env bash
# CPU expert companion bank on the target rig's host (lane/spill-c-20260919 day 17, memra#586): the
# prefetch accounting cells (tools/test_cpu_expert_prefetch.sh, the production translation unit
# included, source fixture on the root filesystem and its mirror on /dev/shm, both O_DIRECT) and the
# CI CPU bank (cpu_native_check against the companion built with the production flags). CPU-only,
# pass/fail, NOT timed; run through the collector for the lock proof and the record only.
# usage: cpubank-cell.sh <lockfd> [cell-name]
set -uo pipefail
fd=$1
cell=${2:-cpubank}
R=/root/spill-receipts/c-day17
EV=$R/$cell/ev
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
cd /root/wt-c
mkdir -p "$EV" "$EV/src"
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$EV/LOCK.json"
sha256sum $R/bins/libmemra-cpu-experts.so $R/bins/cpu_native_check | tee "$EV/binaries.sha256"
git rev-parse HEAD | tee "$EV/tree.sha"
uname -r | tee "$EV/kernel.txt"
rc=0
tools/test_cpu_expert_prefetch.sh "$EV/prefetch-logs" "$EV/src" /dev/shm > "$EV/prefetch-test.log" 2>&1 || rc=1
grep -v '^\[memra-cpu' "$EV/prefetch-test.log" | tail -20
MEMRA_CPU_EXPERT_LIB=$R/bins/libmemra-cpu-experts.so $R/bins/cpu_native_check > "$EV/cpu-native-check.log" 2>&1 || rc=1
tail -4 "$EV/cpu-native-check.log"
rm -rf "$EV/src"
echo "cpubank cell rc=$rc"
exit $rc
