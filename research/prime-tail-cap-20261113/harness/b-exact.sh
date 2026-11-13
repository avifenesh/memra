#!/usr/bin/env bash
# memra#1258 (laptop, one gpulock lease on the RTX 5090): exactness of the fix-A prime split on
# prompts of 4,100 / 4,106 / 4,111 tokens (main: one 4,1xx-row call; fix A: two calls, the
# boundary at t - PRIME_MIN_T then aligned to the GDN grid). For main's schedule (1258base) and
# fix A (1258): run-gen argmax gate at 4,106 (prefill vs tokenwise decode, batched prime vs
# tokenwise), run-spec K=1..8 greedy at 4,106, and dspark_q38_gate (DFlash2 vs plain greedy) at all
# three lengths. The two arms' token streams and digests must also be equal.
set -uo pipefail
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
M=$HOME/models/q38-gguf
GG=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
DR=$M/Qwen3.8-27B-DFlash2-memra
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/exact/$(date +%H%M%S)
mkdir -p "$OUT/prompts"
{ date -Is; sha256sum "$S"/bin/run-gen-1258*-120a "$S"/bin/run-spec-1258*-120a "$S"/bin/dspark-q38-gate-1258*-120a
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; } > "$OUT/guard.log"
python3 - "$OUT/long.txt" <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", Path.home() / ".cache/memra-1255/wt/tools/same-pass-admission-gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
Path(sys.argv[1]).write_text(g.prose(67, 3000))
EOF
# Token ids of the long text (prefill-only mode prints them and exits after one rep).
MEMRA_PROMPT_FILE=$OUT/long.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
  timeout 5m "$S/bin/run-gen-1258-120a" "$GG" > "$OUT/ids.log" 2>&1
python3 "$B/gguf-detok.py" "$GG" "$OUT/ids.log" "$OUT/prompts" 4100 4106 4111 | tee -a "$OUT/guard.log" || exit 2
rc=0
for arm in 1258base 1258; do
  ids=$(cat "$OUT/prompts/p4106.ids")
  # shellcheck disable=SC2086
  MEMRA_NGEN=32 timeout 8m "$S/bin/run-gen-$arm-120a" "$GG" $ids > "$OUT/$arm-rungen-4106.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm run-gen rc=$r $(grep -E 'MATCH|MISMATCH|FLIP-NEARTIE' "$OUT/$arm-rungen-4106.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  # shellcheck disable=SC2086
  MEMRA_SPEC_TEMP=0 timeout 8m "$S/bin/run-spec-$arm-120a" "$GG" $ids > "$OUT/$arm-runspec-4106.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm run-spec rc=$r $(grep -E 'STEP-IDENTITY|=== SELF-CONSISTENCY' "$OUT/$arm-runspec-4106.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  MEMRA_PROMPT_DIR=$OUT/prompts timeout 6m "$S/bin/dspark-q38-gate-$arm-120a" "$GG" "$DR" 96 > "$OUT/$arm-dspark.log" 2>&1
  r=$?; [ $r = 0 ] || rc=1
  echo "$arm dspark rc=$r $(grep -E '^\[dspark-q38-gate\]|== dspark_q38_gate' "$OUT/$arm-dspark.log" | cut -c1-220 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
done
echo "EXACT-DONE rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
exit $rc
