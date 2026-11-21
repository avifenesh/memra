#!/bin/bash
# Lane M (nvfp4-rig) GPU cells, one lease (gpulock run --perf). Logs and receipts under gpu-run/.
# 1 clock check; 250 ms telemetry for the whole lease
# 2 kernel cells (ink2-gpu-gate: every ink2g cell, the nvfp4-rig ones included)
# 3 full-tensor golden: lane F's quantizer on CUDA vs the overlay manifest
# 4 route correctness under nvfp4-rig: chunkings bit-identity, greedy on the p1 tape
# 5 corpus hidden dumps (nvfp4-rig and shipped) on rawdoc and chatwrap, scored with lane F's metric
# 6 lane F's own harness on the nvfp4-rig set (rawdoc, chatwrap)
# 7 decode speed, nvfp4-rig vs shipped, interleaved ABBAAB, clamp check before each take
set -u
L=$(cd "$(dirname "$0")" && pwd)
O=$L/gpu-run
mkdir -p "$O"
W=$HOME/.cache/inkling2mini/wt-mint
BIN=$W/target/release
M=$HOME/models/inkling2-mini
T=$HOME/.cache/inkling2mini/oracle/tapes
K=$HOME/.cache/inkling2mini/klsens
PY=$K/venv/bin/python
CC=$HOME/.cache/inkling2mini/ceiling/clampcheck.sh
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$O/run.log"; }
clamp() {
  bash "$CC" >> "$O/clamp.log" 2>&1; rc=$?
  if [ $rc -eq 1 ]; then
    log "clamped: restarting nvidia-powerd"
    sudo -n systemctl restart nvidia-powerd >> "$O/clamp.log" 2>&1
    bash "$CC" >> "$O/clamp.log" 2>&1; rc=$?
  fi
  log "clampcheck rc=$rc"
  return $rc
}
nvidia-smi --query-gpu=timestamp,pstate,utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm,clocks.mem,clocks_event_reasons.active --format=csv -lms 250 > "$O/telemetry.csv" 2>/dev/null &
TEL=$!
trap 'kill $TEL 2>/dev/null' EXIT
step=${1:-all}
run() { local name=$1; shift; log "START $name"; "$@" > "$O/$name.log" 2>&1; local rc=$?; log "END $name rc=$rc"; return 0; }

if [ "$step" = all ] || [ "$step" = cells ]; then
  clamp
  run kernel-cells "$BIN/ink2-gpu-gate"
  if [ ! -s "$L/golden-full-cuda.json" ]; then
    run golden-full bash -c "cd $K && KLH_DEV=cuda $PY $L/golden_full.py $M.nvfp4-rig $O/golden-full-cuda.json"
  fi
  run chunkings-rig "$BIN/ink2-gpu" "$M" --class nvfp4-rig --tape "$T/p1_chat.actq_f32" --chunkings 26,1 --limit 57 --out "$O/chunkings-rig.json"
  run greedy-rig "$BIN/ink2-gpu" "$M" --class nvfp4-rig --tape "$T/p1_chat.actq_f32" --gen 32 --out "$O/greedy-rig.json"
fi
if [ "$step" = all ] || [ "$step" = kl ]; then
  for set in rawdoc chatwrap; do
    for cls in nvfp4-rig shipped; do
      run "hidden-$set-$cls" "$BIN/ink2-gpu" "$M" --class "$cls" --corpus "$K/corpus/$set.json" --hidden-out "$O/hidden-$set-$cls.safetensors" --chunk 512 --context 2048
      run "score-$set-$cls" $PY "$L/score_kl.py" "$set" "$O/hidden-$set-$cls.safetensors" "$cls" "$M.nvfp4-rig" "$O/kl-$set-$cls.json"
    done
  done
  run lanef-harness bash -c "cd $K && $PY klrun.py --jobs $L/lanef-jobs.json --budget-s 1500"
  for set in rawdoc chatwrap; do
    cp "$K/results/arms/$set/${set}_lanem_nvfp4rig.json" "$O/" 2>/dev/null
    run "lanef-stats-$set" $PY "$L/lanef_stats.py" "$K/results/arms/$set/${set}_lanem_nvfp4rig.npz" "$O/lanef-$set-nvfp4rig.json"
  done
fi
if [ "$step" = all ] || [ "$step" = speed ]; then
  i=0
  for cls in nvfp4-rig shipped shipped nvfp4-rig nvfp4-rig shipped; do
    i=$((i+1))
    clamp
    run "speed-$i-$cls" "$BIN/ink2-gpu" "$M" --class "$cls" --tape "$T/p1_chat.actq_f32" --gen 128 --out "$O/speed-$i-$cls.json"
  done
fi
log DONE
