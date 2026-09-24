#!/usr/bin/env bash
# DAY39 local chain (1.3, 1.4): the eight 5090 boots in the registered order on target/day39/{red,green}
# (day39-build.sh), the admission gate on green at its defaults, then day33-compare.py (unchanged) and day39-read.py.
# Every GPU step waits for an idle rig and holds /tmp/memra-5090.lock alone. Never a signal to anything.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919
R=$D/rtx5090-day39
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
log() { echo "$(date -u +%FT%TZ) $*" >> "$R/run.log"; }
idle() {
  flock -n /tmp/memra-5090.lock true || return 1
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || return 1
  [ "$(free -g | awk '/^Mem:/{print $7}')" -ge 24 ] || return 1
}
cp target/day39/SHA256SUMS "$R/binaries.sha256"
cp target/day39/green/source.commit "$R/green.source"; cp target/day39/red/source.commit "$R/red.source"
log "chain start HEAD=$(git rev-parse HEAD) $(tr '\n' ' ' < "$R/binaries.sha256")"
bash "$D/day39-run.sh" "$R" red-G2:red:G2 red-L64:red:L64 green-G2-r1:green:G2 green-L64-r1:green:L64 \
  red-off:red:off green-off:green:off green-G2-r2:green:G2 green-L64-r2:green:L64 || exit $?
deadline=$((SECONDS + 7200))
until idle; do [ $SECONDS -ge $deadline ] && { log "gate: rig not idle after 7200 s; not run"; exit 3; }; sleep 30; done
log "gate start (green, defaults)"
flock -w 7200 /tmp/memra-5090.lock tools/admit-mem-burst-gate.sh "$MODEL" "$WT/target/day39/green/memra-server" "$R/gate" > "$R/gate.log" 2>&1
log "gate rc=$? $(tail -1 "$R/gate.log")"
B=$R/boots
python3 "$D/day33-compare.py" --card rtx5090 red:G2:$B/red-G2 red:L64:$B/red-L64 green:G2:$B/green-G2-r1 \
  green:L64:$B/green-L64-r1 red:off:$B/red-off green:off:$B/green-off green:G2:$B/green-G2-r2 green:L64:$B/green-L64-r2 \
  > "$R/SUMMARY.txt" 2>&1
python3 "$D/day39-read.py" rtx5090 $B/red-G2 $B/red-L64 $B/green-G2-r1 $B/green-L64-r1 $B/green-G2-r2 $B/green-L64-r2 \
  $B/red-off $B/green-off > "$R/READINGS.txt" 2>&1
log "chain done"
