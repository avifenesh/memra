#!/bin/bash
# Feature-capture worker: one GPU, a list of scoring-prompt names.
# For each name, runs the patched run-safetensors over S = prompt_ids + cont_ids and dumps
# the contracted layer rows for the DFlash2 target layers to /root/dfp2/cap/<name>/.
# usage: capture_worker.sh <gpu> <name> [name...]
set -u
GPU=$1; shift
for NAME in "$@"; do
  OUT=/root/dfp2/cap/$NAME
  mkdir -p "$OUT"
  IDS=$(python3 - "$NAME" <<'EOF'
import json, sys
name = sys.argv[1]
sp = {p["name"]: p for p in json.load(open("/root/dfp2/scoring_prompts.json"))}
ro = {r["name"]: r for r in json.load(open("/root/dfp2/rollouts.json"))}
ids = sp[name]["ids"] + ro[name]["cont_ids"]
print(" ".join(str(i) for i in ids))
EOF
)
  echo "[gpu$GPU] $NAME: $(echo $IDS | wc -w) ids -> $OUT"
  CUDA_VISIBLE_DEVICES=$GPU \
  MEMRA_TRACE_LAYER_ROWS=$OUT MEMRA_TRACE_LAYER_ROWS_LAYERS=5,14,24,33,42 \
  MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=12000 NVIDIA_TF32_OVERRIDE=0 \
  /root/memra/target/release/run-safetensors /root/models/glm53-nvfp4 $IDS \
    > "$OUT/run.log" 2>&1
  RC=$?
  tail -2 "$OUT/run.log" | sed "s/^/[gpu$GPU] $NAME: /"
  [ $RC -ne 0 ] && echo "[gpu$GPU] $NAME: FAILED rc=$RC"
done
echo "[gpu$GPU] done"
