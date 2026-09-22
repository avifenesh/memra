#!/usr/bin/env bash
# WP-B day 29: reproduce A's day-18 27B twin-gate `V3=FAIL` on the local RTX 5090 Laptop GPU (DAY29.md 2).
# The gate command is A's day-18 driver's `twin27-off` cell verbatim (research/spill-a-20260919/rtx5090-day18/
# driver.sh line 50), door OFF (MEMRA_KV_HOST_CONTRACTS unset). Beside it: `nvidia-smi --query-compute-apps`
# and driver free sampled before the boot, at 1 Hz through every turn, and after. The gate takes the canonical
# lock itself (`flock -n` on MEMRA_GPU_LOCK); a busy lock is `REFUSED` rc=2 and is retried boundedly, never
# signalling the holder. No process this lane did not start is touched.
# usage: run-day29-repro.sh <run-name> [server_bin]     (receipts: $RIGDIR/<run-name>/)
set -uo pipefail
WT=${WT:-/home/avifenesh/projects/wt-spill-b}
RIGDIR=${RIGDIR:-$WT/research/spill-b-20260919/rtx5090-day29}
MODEL27=${MODEL27:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
BIN=${2:-$WT/target/release/memra-server}
name=${1:?run-name}
export MEMRA_GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
cd "$WT"
R=$RIGDIR/$name; mkdir -p "$R"
git rev-parse HEAD > "$R/tree.sha"; git status --short | grep -v '^??' > "$R/tree.dirty" || true
sha256sum "$BIN" > "$R/binary.sha256"
snap() { # label
    { echo "== $1 $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader;
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader; } >> "$R/card-$1.txt"
}
snap before
# 1 Hz sampler for the whole cell (my own process; stopped below).
( while :; do echo "$(date -u +%FT%T.%3NZ) | $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader) | $(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tr '\n' ';')"; sleep 1; done ) >> "$R/samples.log" &
SAMPLER=$!
rc=1
for try in $(seq 1 15); do
    rm -rf "$R/twin27-off"
    echo "$(date -u +%FT%TZ) gate start (try $try)" >> "$R/driver.log"
    python3 tools/prefix-newest-turn-fits-gate.py --model "$MODEL27" --bin "$BIN" --out "$R/twin27-off" > "$R/twin27-off.log" 2>&1
    rc=$?
    echo "$(date -u +%FT%TZ) gate end rc=$rc" >> "$R/driver.log"
    if [[ $rc -eq 2 ]] && grep -q 'REFUSED: canonical GPU lock busy' "$R/twin27-off.log"; then
        echo "$(date -u +%FT%TZ) lock busy, retry $try/15 in 120 s" >> "$R/driver.log"; sleep 120; continue
    fi
    break
done
kill "$SAMPLER" 2>/dev/null; wait "$SAMPLER" 2>/dev/null
snap after
echo "$rc" > "$R/twin27-off.exit"
grep -h 'PREFIX-NEWEST-TURN-FITS\|REFUSED' "$R/twin27-off.log" | tail -2
echo "day29-repro $name rc=$rc"
