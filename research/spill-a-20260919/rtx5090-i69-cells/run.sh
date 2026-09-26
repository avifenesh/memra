#!/usr/bin/env bash
# integ69's worker GPU span cells on the local RTX 5090 (DAY63 section 7): the frozen test executable of the integ69
# branch with the fix (6c60d798f), serial, `option_b_ option_c_ --ignored`, in one bounded hold of /tmp/memra-5090.lock
# (60 x 120 s) with the idle rule (no compute app, >= 20000 MiB free; 15 x 60 s). Pass/fail only, not timed.
set -uo pipefail
O=/home/avifenesh/spill-a-cells/i69-gpu
E=/home/avifenesh/spill-a-cells/i69fix-branch-server-tests
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$O/run.log"; }
sha256sum "$E" > "$O/binary.sha256"
exec 9>/tmp/memra-5090.lock
held=0
for a in $(seq 1 60); do if flock -n 9; then held=1; break; fi; log "lock busy, attempt $a of 60"; sleep 120; done
[ $held = 1 ] || { log "NOT RUN: lock busy"; exit 2; }
ok=0
for a in $(seq 1 15); do
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1); free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if [ -z "$apps" ] && [ "${free:-0}" -ge 20000 ] 2>/dev/null; then ok=1; break; fi
  log "card not idle (apps=[${apps//$'\n'/; }] free=$free), attempt $a of 15"; sleep 60
done
[ $ok = 1 ] || { log "NOT RUN: card never idle"; exit 2; }
log "hold taken"
CUDA_VISIBLE_DEVICES=0 "$E" --ignored --test-threads=1 option_b_ option_c_ > "$O/cells.log" 2>&1; rc=$?
log "worker-span-cells rc=$rc $(grep -h '^test result' "$O/cells.log")"
