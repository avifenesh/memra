#!/usr/bin/env bash
# Lane C RTX 5090 queue v16 (2026-09-26): DAY82 section 2's local check (day82-cpu/gpu-check.sh: the door at I18 and
# at I20, 32 tokens each, order i18, i20, i20, i18), then its reader into day82-cpu/gpu-check.log. The check waits for
# an idle card (the lock free, no compute app, 40 GiB MemAvailable) up to 48 h and holds /tmp/memra-5090.lock for its
# four runs inside the 1200% CPU cap. Detached so it outlives the calling shell. Never signals another process.
# usage: nohup setsid bash rtx5090-queue-v16-20260926.sh > <log> 2>&1 &
set -uo pipefail
export PATH=/usr/bin:$HOME/.cargo/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH
L=/home/avifenesh/projects/wt-spill-c/research/spill-c-20260919
O=$L/day82-cpu/gpu-check
log() { echo "$(date -u +%FT%TZ) $*"; }
log "queue v16 start"
mkdir -p "$O"; printf '*.log -whitespace\n*.sha256 -whitespace\n' > "$O/.gitattributes"
bash "$L/day82-cpu/gpu-check.sh" "$O" > "$O/driver.log" 2>&1
log "check rc=$?"
{ echo "# DAY82 local GPU check (the development host's RTX 5090 under its lock): the door at I18 (run-gen-i18 = c7294b912) and at I20 (run-gen-i20 = 8efea3a54), 32 tokens each, order i18, i20, i20, i18; raw logs in gpu-check/"
  cat "$O/binary.sha256"; /usr/bin/python3 "$L/day82-cpu/gpu-check-read.py" "$O"; } > "$L/day82-cpu/gpu-check.log" 2>&1
log "read rc=$?"
touch "$O/v16.done"
log "queue v16 done"
