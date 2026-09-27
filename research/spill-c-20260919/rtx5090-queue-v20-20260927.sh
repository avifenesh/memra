#!/usr/bin/env bash
# Lane C RTX 5090 queue v20 (2026-09-27): DAY88 section 4's local check (day88-cpu/gpu-check.sh: the promoted
# run-gen-p88 by default against run-gen-i22 as qualified, both orders, then the rollback), its reader into
# day88-cpu/gpu-check.log. Detached so it outlives the calling shell. Never signals another process.
# usage: nohup setsid bash rtx5090-queue-v20-20260927.sh > <log> 2>&1 < /dev/null &
set -uo pipefail
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
O=$L/day88-cpu/gpu-check
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v20 start"
mkdir -p "$O"; printf '*.log -whitespace\n*.sha256 -whitespace\n*.wall -whitespace\n*.exit -whitespace\n' > "$O/.gitattributes"
bash "$L/day88-cpu/gpu-check.sh" "$O" > "$O/driver.log" 2>&1
log "check rc=$?"
{ echo "# DAY88 local GPU check (the development host's RTX 5090 under its lock): run-gen-p88 (0155bc69f) by default, run-gen-i22 (4b378a064) as qualified, order p88 i22 i22 p88, then p88 with MEMRA_EXPERTS_VIA_TIER=0; raw logs in gpu-check/"
  cat "$O/binary.sha256"; /usr/bin/python3 "$L/day88-cpu/gpu-check-read.py" "$O"; } > "$L/day88-cpu/gpu-check.log" 2>&1
log "read rc=$?"
touch "$O/v20.done"
log "queue v20 done"
