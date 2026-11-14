#!/usr/bin/env bash
# Runs ON GB10 unit-b (memra#1258 capacity A/B, the second point beside the RTX 5090), inside a
# sparklock lease: the first prime call's pool draw beside a decoding keeper, two boots of
# 12 (~5.7k tokens) or 10 (~8k) cold primes (b-spread.py, mode 0), for one arm. The boot
# calibration line gives the rest-mapped bytes. The margin sweeps are not run here: on unified
# memory their ballast would take the shared host's memory down to the admission margin.
# FILL variant (memra#1257): before each boot, fill_cache2.sh fills the page cache until MemFree is
# under 3 GB, the state in which the raw driver reading sits near admission's 2 GiB floor.
# usage: gb10-cap-run-fill.sh ARM LEN   (LEN 5.7k|8k)
set -u
ARM=$1; LEN=$2
R=$HOME/.cache/memra-1258
M=$HOME/models/memra-q38
GG=$M/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
BIN=$R/bins/memra-server-$ARM-121a
[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
if [ "$LEN" = 8k ]; then SP="--n 10 --words 5112 --seed0 800"; else SP="--n 12"; fi
OUT=$R/cap-gb10/spread-$ARM-$LEN-fill-$(date +%H%M%S)
mkdir -p "$OUT"
{ date -Is; hostname; sha256sum "$BIN"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader
  free -m | head -n 2; } > "$OUT/guard.log"
echo "$$" >> "$R/pids.txt"
rc=0
for i in 1 2; do
  bash "$R/fill_cache2.sh" 3000 300 4 >> "$OUT/guard.log" 2>&1
  # shellcheck disable=SC2086
  setsid python3 "$R/cap-gb10/b-spread.py" --bin "$BIN" --model "$GG" --draft "$DR" --out "$OUT/b$i-mode0" \
    --mode 0 $SP > "$OUT/b$i-mode0.log" 2>&1 &
  P=$!
  trap 'kill -TERM -- -"$P" 2>/dev/null; exit 143' TERM INT
  wait "$P" || rc=$?
  kill -- -"$P" 2>/dev/null
  echo "boot $i done rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
done
exit $rc
