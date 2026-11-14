#!/usr/bin/env bash
# memra#1258 step 1 (laptop, one gpulock lease on the RTX 5090): exactness of the FFN-into-slabs
# change against its parent. Arms A and B name binaries run-gen-<arm>-120a, run-spec-<arm>-120a,
# dspark-q38-gate-<arm>-120a. Prompts of 2,000 / 4,106 / 8,200 / 16,500 tokens (ids from run-gen's
# prefill-only mode on a prose text, texts from the GGUF vocabulary): run-gen argmax at 4,106 and
# 8,200, run-spec K=1..8 at 8,200, dspark_q38_gate at all four. The arms' streams and digests must
# be equal. usage: b-exact2.sh A B
set -uo pipefail
A=$1; Bm=$2
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
GG=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/exact2/$A-$Bm-$(date +%H%M%S)
mkdir -p "$OUT/prompts"
{ date -Is; for a in "$A" "$Bm"; do sha256sum "$S/bin/run-gen-$a-120a" "$S/bin/run-spec-$a-120a" "$S/bin/dspark-q38-gate-$a-120a"; done
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; echo "profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; } > "$OUT/guard.log"
python3 - "$OUT/long.txt" <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", Path.home() / ".cache/memra-1258/harness/same-pass-admission-gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
Path(sys.argv[1]).write_text(g.prose(71, 11000))
EOF
MEMRA_PROMPT_FILE=$OUT/long.txt MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
  timeout 5m "$S/bin/run-gen-$A-120a" "$GG" > "$OUT/ids.log" 2>&1
python3 "$B/gguf-detok.py" "$GG" "$OUT/ids.log" "$OUT/prompts" 2000 4106 8200 16500 | tee -a "$OUT/guard.log" || exit 2
rc=0
for arm in "$A" "$Bm"; do
  for n in 4106 8200; do
    ids=$(cat "$OUT/prompts/p$n.ids")
    # shellcheck disable=SC2086
    MEMRA_NGEN=32 timeout 12m "$S/bin/run-gen-$arm-120a" "$GG" $ids > "$OUT/$arm-rungen-$n.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$arm run-gen $n rc=$r $(grep -E 'MATCH|MISMATCH|FLIP-NEARTIE' "$OUT/$arm-rungen-$n.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  done
  ids=$(cat "$OUT/prompts/p8200.ids")
  # shellcheck disable=SC2086
  MEMRA_SPEC_TEMP=0 timeout 10m "$S/bin/run-spec-$arm-120a" "$GG" $ids > "$OUT/$arm-runspec-8200.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm run-spec 8200 rc=$r $(grep -E 'STEP-IDENTITY|=== SELF-CONSISTENCY' "$OUT/$arm-runspec-8200.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  MEMRA_PROMPT_DIR=$OUT/prompts timeout 8m "$S/bin/dspark-q38-gate-$arm-120a" "$GG" "$DR" 96 > "$OUT/$arm-dspark.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm dspark rc=$r $(grep -E '== dspark_q38_gate' "$OUT/$arm-dspark.log" | cut -c1-160)" | tee -a "$OUT/guard.log"
  grep -E '^\[dspark-q38-gate\]' "$OUT/$arm-dspark.log" | sed -E 's/ prime_s=.*//' > "$OUT/$arm-dspark-digests.txt"
  grep -E '^ *tokens: |self-consistency' "$OUT/$arm-rungen-4106.log" "$OUT/$arm-rungen-8200.log" "$OUT/$arm-runspec-8200.log" \
    | sed -E 's|^[^:]*/[^-]*-||' > "$OUT/$arm-streams.txt"
done
if cmp -s "$OUT/$A-dspark-digests.txt" "$OUT/$Bm-dspark-digests.txt" && cmp -s "$OUT/$A-streams.txt" "$OUT/$Bm-streams.txt"; then
  echo "ARMS-EQUAL" | tee -a "$OUT/guard.log"
else
  echo "ARMS-DIFFER" | tee -a "$OUT/guard.log"; rc=1
fi
echo "EXACT2-DONE rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
exit $rc
