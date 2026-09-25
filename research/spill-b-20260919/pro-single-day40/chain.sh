#!/usr/bin/env bash
# WP-B day 40 target-card half (DAY40.md 1.2): one RTX PRO 6000 Blackwell Workstation Edition, the 27B at the
# checkpoint's context, burst 64; day 36's target-card chain on S40's binary (build-arms.sh, arm tip). One collector hold
# per order on /tmp/memra-gpu.lock after a bounded idle wait; never a signal to anything. Then day34-compare.py
# (unchanged), day31-faults.py and day40-read.py. Receipts under /root/spill-receipts/b-day40.
set -uo pipefail
R=/root/spill-receipts/b-day40; mkdir -p "$R/bins" "$R/boots"
WT=/root/wt-b
MODEL=${MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
S40=${S40:-b46ae200e7638f9c55688d2d339de2872f4762a6}
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/chain.log"; }
cd "$WT" || { log "no $WT"; exit 1; }
git fetch -q origin lane/spill-b-20260919 && git merge --ff-only -q FETCH_HEAD >> "$R/fetch.log" 2>&1 || { log "fetch/ff failed"; exit 1; }
git rev-parse HEAD > "$R/source.txt"
[ "$(sha256sum "$MODEL" | cut -d' ' -f1)" = "$WANT_MODEL" ] || { log "model sha256 mismatch; not run"; exit 1; }
nvidia-smi --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
command grep -q "RTX PRO 6000 Blackwell Workstation" "$R/card.csv" || { log "not an RTX PRO 6000 Blackwell Workstation: $(tail -1 "$R/card.csv")"; exit 1; }
[ "${DRY_RUN:-0}" = 1 ] && { log "dry run done"; exit 0; }
export WT
TARGET=$WT/target WRAP="nice -n 10" bash research/spill-b-20260919/build-arms.sh "$R/bins" "$S40" tip > "$R/bins/build.out" 2>&1 \
  || { log "build failed: $(tail -1 "$R/bins/build.out")"; exit 1; }
log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
idle() { flock -n /tmp/memra-gpu.lock true || return 1; [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1; }
export MODEL MODEL_KEY=q38 NO_SCOPE=1
for order in O1 O2; do
  deadline=$((SECONDS + 7200))
  until idle; do [ $SECONDS -ge $deadline ] && { log "order $order: card not idle after 7200 s; not run"; exit 3; }; sleep 30; done
  log "order $order: collector start"
  rm -rf "$R/collector-$order"
  env R="$R" RIG_LOCK=/tmp/memra-gpu.lock BIN="$R/bins/tip/memra-server" BURST=64 \
    python3 tools/tier-battery.py --rig pro-single --timeout "${ORDER_TIMEOUT:-43200}" --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ $order > "$R/collector-$order.log" 2>&1
  log "order $order: collector exit=$?"
done
B=$R/boots
python3 research/spill-b-20260919/day34-compare.py --card pro6000 --served-ctx 262144 --model-ctx 262144 --registry-value 32768 \
  --survey context O1:$B/O1-off O1:$B/O1-on2048 O1:$B/O1-on8192 O1:$B/O1-on32768 O2:$B/O2-off O2:$B/O2-on2048 \
  O2:$B/O2-on8192 O2:$B/O2-on32768 > "$R/SUMMARY.txt" 2>&1
python3 research/spill-b-20260919/day31-faults.py "$R/boots" > "$R/FAULTS.txt" 2>&1
python3 research/spill-b-20260919/day40-read.py pro6000 "$R/boots" > "$R/READINGS.txt" 2>&1
( cd "$R" && find . -type f ! -path './bins/*' -print0 | sort -z | xargs -0 sha256sum > "$R/MANIFEST.sha256" )
log "LANE-B-DAY40-BOX-DONE"
