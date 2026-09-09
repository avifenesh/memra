#!/usr/bin/env bash
# Native beam-1 transcription sweep over the pinned rental oracle clips.
#
# One process, niced, pinned to the machine's efficiency cores so the owner keeps every
# performance core. Each clip banks its own directory and the sweep skips clips that already
# finished, so an interrupted run resumes without repeating proven work.
#
# usage: whisper_transcribe_sweep.sh ORACLE CHECKPOINT BINARY OUT_DIR NUMERIC [CLIP...]
set -euo pipefail

ORACLE=$1; CHECKPOINT=$2; BINARY=$3; OUT=$4; NUMERIC=$5; shift 5
CLIPS=("$@")
if [ ${#CLIPS[@]} -eq 0 ]; then
  mapfile -t CLIPS < <(ls "$ORACLE/ct2")
fi

: "${MEMRA_SPEECH_THREADS:=16}"
: "${SWEEP_CPUS:=8-23}"
export MEMRA_SPEECH_THREADS
mkdir -p "$OUT"

for clip in "${CLIPS[@]}"; do
  if [ -f "$OUT/$clip/windows.tsv" ]; then
    echo "skip $clip (already banked)"
    continue
  fi
  rm -rf "${OUT:?}/$clip"
  pcm="$OUT/.$clip.pcm.f32"
  python3 - "$ORACLE/ct2/$clip/input.pcm.s16le" "$pcm" <<'PY'
import sys,numpy as np
np.fromfile(sys.argv[1],dtype='<i2').astype(np.float32).__itruediv__(32768.0).tofile(sys.argv[2])
PY
  echo "=== $clip $(date -u +%FT%TZ) threads=$MEMRA_SPEECH_THREADS cpus=$SWEEP_CPUS"
  /usr/bin/time -f "$clip cpu_seconds=%U+%S wall=%e" \
    nice -n 15 taskset -c "$SWEEP_CPUS" \
    "$BINARY" transcribe "$CHECKPOINT" "$pcm" "$OUT/$clip" "$NUMERIC"
  rm -f "$pcm"
done
echo "sweep complete $(date -u +%FT%TZ)"
