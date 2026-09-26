#!/usr/bin/env bash
# DAY64 section 7: F's registered (a) includes the worker's span cells (`option_c_span_*`), which F's sitting's
# unit-cells.sh omitted; this step completes it. The worker's door cells `option_b_ option_c_` (the 18, the span cells
# among them) run serially from a server test executable built at <sha> (F's production code with the corrected worker
# cell arithmetic, 411177fea and 28c7aa6c1), in its own clone (/root/wt-a-funit). `bash unit-worker.sh build <sha>`,
# then `tools/tier-battery.py --rig pro-single --external-lock --execute bash unit-worker.sh @COLLECTOR_LOCK_FD@`.
set -uo pipefail
R=/root/spill-receipts/a-f/unit-worker
W=/root/wt-a-funit
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R"
if [ "${1:-}" = build ]; then
  [ -d "$W/.git" ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git "$W" >> "$R/build.log" 2>&1
  cd "$W" || exit 1
  git fetch -q origin lane/spill-a-20260919 >> "$R/build.log" 2>&1
  git checkout -q -B funit "$2" >> "$R/build.log" 2>&1 || { echo "rc=2 (checkout)" >> "$R/build.log"; exit 2; }
  git rev-parse HEAD > "$R/tree.sha"
  nice -n 5 cargo test -p memra-server --lib --no-run >> "$R/build.log" 2>&1 || { echo "rc=1" >> "$R/build.log"; exit 1; }
  echo "rc=0" >> "$R/build.log"
  exit 0
fi
fd=$1
cd "$W" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock /tmp/memra-gpu.lock --owner collector > "$R/LOCK.json"
CUDA_VISIBLE_DEVICES=0 cargo test -p memra-server --lib -- --ignored --test-threads=1 option_b_ option_c_ > "$R/door-cells.log" 2>&1
rc=$?
echo "UNIT-WORKER door-rc=$rc $(grep -h '^test result' "$R/door-cells.log")" | tee -a "$R/run.log"
exit $rc
