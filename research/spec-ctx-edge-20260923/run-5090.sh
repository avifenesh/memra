#!/usr/bin/env bash
# memra#659 on the local RTX 5090: the red arm (the unfixed integ49 merge tree 40891cf2d) then the
# green arm twice (the fix tree), all under ONE hold of /tmp/memra-5090.lock (15 x `flock -w 120`),
# then an idle check under the hold (no compute app, >= 20000 MiB free, 15 x 60 s). Either bound
# running out records NOT RUN. Pre-registration: PREREG.md in this directory.
# usage: run-5090.sh <out_root> <model.gguf> <red_bin> <green_bin>
set -uo pipefail
ROOT=$1; MODEL=$2; RED=$3; GREEN=$4
cd "$(dirname "$0")/../.." || exit 1
mkdir -p "$ROOT"
exec 9>/tmp/memra-5090.lock
held=0
for attempt in $(seq 1 15); do
    if flock -w 120 9; then held=1; break; fi
    echo "$(date -u +%FT%TZ) hold attempt $attempt: the lock stayed busy for 120 s" | tee -a "$ROOT/run.log"
done
[ "$held" = 1 ] || { echo "$(date -u +%FT%TZ) NOT RUN: the lock never came free" | tee -a "$ROOT/run.log"; exit 0; }
echo "$(date -u +%FT%TZ) hold taken tree=$(git rev-parse HEAD)" | tee -a "$ROOT/run.log"
idle=0
for attempt in $(seq 1 15); do
    apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1)
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    if [ -z "$apps" ] && [ "${free_mib:-0}" -ge 20000 ]; then idle=1; break; fi
    echo "$(date -u +%FT%TZ) idle wait $attempt: free=${free_mib}MiB apps=[${apps//$'\n'/; }]" | tee -a "$ROOT/run.log"
    sleep 60
done
[ "$idle" = 1 ] || { echo "$(date -u +%FT%TZ) NOT RUN: the card never went idle" | tee -a "$ROOT/run.log"; exit 0; }
for arm in red:"$RED" green1:"$GREEN" green2:"$GREEN"; do
    name=${arm%%:*}; bin=${arm#*:}
    echo "$(date -u +%FT%TZ) start $name bin=$bin" | tee -a "$ROOT/run.log"
    bash tools/spec-ctx-edge-gate.sh "$MODEL" "$bin" "$ROOT/$name" > "$ROOT/$name.log" 2>&1
    echo "$(date -u +%FT%TZ) done $name rc=$? $(tail -n 1 "$ROOT/$name.log")" | tee -a "$ROOT/run.log"
done
flock -u 9
echo "$(date -u +%FT%TZ) hold released; SPEC-CTX-EDGE-5090-DONE" | tee -a "$ROOT/run.log"
