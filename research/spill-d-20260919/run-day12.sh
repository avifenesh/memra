#!/usr/bin/env bash
# Day 12 driver on the single-PRO clone: build receipt (both gate binaries, clippy, CPU tests),
# the seven fault arms through the collector, the two tier-transfer-gate cases, then the root
# validation over every cell. Cells run only against a green build receipt; a red build stops
# the run and says so. Runs detached under tmux; every step's exit is recorded.
# usage: run-day12.sh [worktree] [receipt-root]
set -uo pipefail
wt=${1:-/root/wt-d}
root=${2:-/root/spill-receipts/d-day12}
artifact=/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd "$wt" || exit 2
mkdir -p "$root/build"
bash research/spill-d-20260919/build-day12.sh "$wt" "$root/build" > "$root/build/driver.log" 2>&1
build=$(cat "$root/build/exit" "$root/build/clippy.exit" "$root/build/tests.exit" | tr '\n' ' ')
echo "build receipt exits: $build" >> "$root/build/driver.log"
if [[ "$build" != "0 0 0 " ]]; then
  echo "REFUSED: build receipt not green (exits $build); no cell launched" > "$root/ARMS-DONE"
  exit 0
fi
bash research/spill-d-20260919/run-fault-arms.sh "$root" "$artifact" "$wt/target/release/kv-tier-gate" \
  > "$root/arms-driver.log" 2>&1
bash research/spill-d-20260919/run-transfer-gate.sh "$root/transfer-gate" "$wt/target/release/tier-transfer-gate" \
  > "$root/transfer-driver.log" 2>&1
set +e
python3 tools/tier-battery.py --validate "$root" > "$root/validate.json" 2> "$root/validate.err"
echo $? > "$root/validate.exit"
set -e
date -u +%FT%TZ > "$root/ARMS-DONE"
