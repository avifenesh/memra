#!/usr/bin/env bash
# WP-A day 34, the 5090 hit gates rerun (DAY34.md section 6): in card-run.sh the hit gate's own stop found no server to
# signal (under an external lock it addresses its server only while /proc/<pid>/comm reads `memra-server`, and the
# binary was named `memra-server-d34`), so the spec-on boot's server outlived it and the spec-off twin refused the port.
# The same binary byte for byte under the name `memra-server`, the same two gates through --external-lock 9, one bounded
# hold (30 x 120 s). usage: hit-rerun.sh <out_root> <model.gguf> <bin named memra-server>
set -uo pipefail
ROOT=$1; MODEL=$2; BIN=$3
HERE=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$HERE" || exit 1
export MEMRA_GPU_LOCK=/tmp/memra-5090.lock
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ROOT/run.log"; }
exec 9>"$MEMRA_GPU_LOCK"
held=0
for attempt in $(seq 1 30); do flock -w 120 9 && { held=1; break; }; log "hit rerun: hold attempt $attempt busy"; done
[ "$held" = 1 ] || { log "hit rerun NOT RUN"; exit 2; }
log "hit rerun hold taken; binary $(sha256sum "$BIN" | cut -c1-16)"
for arm in off on; do
  extra=""; [ $arm = on ] && extra="MEMRA_KV_HOST_CONTRACTS=1"
  OUT=$ROOT/hit-$arm-rerun; mkdir -p "$OUT"
  env $extra bash tools/spec-on-cache-hit-gate.sh --external-lock 9 qwen "$MODEL" "$BIN" "$OUT/ev" > "$OUT/gate.log" 2>&1; rc=$?
  echo "$rc" > "$OUT/gate.exit"
  log "gate hit-$arm-rerun rc=$rc $(grep -hE 'GATE: ' "$OUT/gate.log" | tail -1)"
done
flock -u 9
log "hit rerun hold released"
