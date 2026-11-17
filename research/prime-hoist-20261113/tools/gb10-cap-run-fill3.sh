#!/usr/bin/env bash
# Runs ON GB10 unit-b (memra#1258 capacity A/B, the second point beside the RTX 5090), inside a
# sparklock lease: the first prime call's pool draw beside a decoding keeper, two boots of
# 12 (~5.7k tokens) or 10 (~8k) cold primes (b-spread.py, mode 0), for one arm. The boot
# calibration line gives the rest-mapped bytes. The margin sweeps are not run here: on unified
# memory their ballast would take the shared host's memory down to the admission margin.
# FILL variant (memra#1257): before each boot, fill_cache2.sh fills the page cache until MemFree is
# under 3 GB, the state in which the raw driver reading sits near admission's 2 GiB floor.
# Second form (memra#1321 fold): the fill gets FILL_CAP seconds (default 600) and FILL_READERS
# readers (default 8), because on 2026-11-14 10:48 four readers left MemFree at 14 and 9 GB after
# 300 s on this unit; each boot's guard line says whether MemFree went under 3 GB.
# Third form (2026-11-14, short leases): ONE boot per call, so a lease is 10 minutes at most and
# the cell can be stopped within minutes: the fill gets FILL_CAP seconds (default 240), then one
# spread boot (about 2.5 minutes at 5.7k). The two boots of an arm go into one directory:
# cap-gb10/spread-ARM-LEN-fill3/bBOOT-mode0, each with its own guard-bBOOT.log.
# usage: gb10-cap-run-fill3.sh ARM LEN BOOT   (LEN 5.7k|8k; BOOT 1|2)
set -u
ARM=$1; LEN=$2; i=${3:?boot number}
R=$HOME/.cache/memra-1258
M=$HOME/models/memra-q38
GG=$M/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
BIN=$R/bins/memra-server-$ARM-121a
[ -x "$BIN" ] || { echo "missing $BIN"; exit 2; }
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
if [ "$LEN" = 8k ]; then SP="--n 10 --words 5112 --seed0 800"; else SP="--n 12"; fi
OUT=$R/cap-gb10/spread-$ARM-$LEN-fill3
G=$OUT/guard-b$i.log
mkdir -p "$OUT"
{ date -Is; hostname; sha256sum "$BIN"; nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader
  free -m | head -n 2; } > "$G"
echo "$$" >> "$R/pids.txt"
rc=0
bash "$R/fill_cache2.sh" 3000 "${FILL_CAP:-240}" "${FILL_READERS:-8}" >> "$G" 2>&1
awk '/^MemFree/{printf "boot %s starts with MemFree %d MB: %s\n", b, $2/1024, ($2/1024 < 3000 ? "filled" : "NOT under 3000 MB")}' b="$i" /proc/meminfo >> "$G"
# shellcheck disable=SC2086
setsid python3 "$R/cap-gb10/b-spread.py" --bin "$BIN" --model "$GG" --draft "$DR" --out "$OUT/b$i-mode0" \
  --mode 0 $SP > "$OUT/b$i-mode0.log" 2>&1 &
P=$!
echo "$P" >> "$R/pids.txt"
trap 'kill -TERM -- -"$P" 2>/dev/null; exit 143' TERM INT
wait "$P" || rc=$?
kill -- -"$P" 2>/dev/null
echo "boot $i done rc=$rc $(date -Is)" | tee -a "$G"
exit $rc
