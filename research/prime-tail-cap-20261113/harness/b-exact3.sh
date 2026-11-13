#!/usr/bin/env bash
# #1274 review item 5 (laptop, one gpulock lease on the RTX 5090): run-gen, main's schedule
# (1258base) against the head (1258b), bit for bit, at 8,200 tokens on Qwen3.8-27B (its last call
# was 4,104 rows on main) and at 4,106 tokens on the MoE trunk Qwen3.6-35B-A3B (one 4,106-row call
# on main). Each arm dumps the serving prime's last-row logits (MEMRA_PP_PRIME, MEMRA_PP_LOGITS) and prints
# its argmax lines and 16 greedy tokens; the files are compared with cmp and the largest logit
# difference is reported. usage: b-exact3.sh
set -uo pipefail
[ -x "$HOME/.cache/memra-1255/bin/run-gen-1258b-120a" ] || { echo "run-gen-1258b missing: not run"; exit 2; }
S=$HOME/.cache/memra-1255
B=$HOME/.cache/memra-1258
Q38=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
Q35=/data/ai-ml/hf-models/qwen36-35b-moe/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_GPU_LOCK$'); do unset "$v"; done
OUT=$B/exact3/$(date +%H%M%S)
mkdir -p "$OUT"
{ date -Is; sha256sum "$S/bin/run-gen-1258base-120a" "$S/bin/run-gen-1258b-120a"
  nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.draw,enforced.power.limit,memory.used --format=csv,noheader; echo "profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; } > "$OUT/guard.log"
python3 - "$OUT" <<'EOF'
import importlib.util, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("gate", Path.home() / ".cache/memra-1255/wt/tools/same-pass-admission-gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
out = Path(sys.argv[1])
(out / "q38-long.txt").write_text(g.prose(77, 5600))
(out / "q35-long.txt").write_text(g.prose(79, 3000))
EOF
rc=0
for spec in "q38 $Q38 8200" "q35 $Q35 4106"; do
  set -- $spec
  M=$1; GG=$2; N=$3
  MEMRA_PROMPT_FILE=$OUT/$M-long.txt MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
    timeout 6m "$S/bin/run-gen-1258b-120a" "$GG" > "$OUT/$M-ids.log" 2>&1
  python3 "$B/gguf-detok.py" "$GG" "$OUT/$M-ids.log" "$OUT/$M-prompts" "$N" | tee -a "$OUT/guard.log" || { rc=2; continue; }
  ids=$(cat "$OUT/$M-prompts/p$N.ids")
  for arm in 1258base 1258b; do
    # The serving prime's last-row logits (prime_cache on a fresh cache, chunked as served).
    # shellcheck disable=SC2086
    MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 MEMRA_PP_LOGITS=$OUT/$M-$arm.logits \
      timeout 6m "$S/bin/run-gen-$arm-120a" "$GG" $ids > "$OUT/$M-$arm-pp.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    # The argmax gate (prefill vs tokenwise decode, batched prime vs tokenwise) and 16 tokens.
    # shellcheck disable=SC2086
    MEMRA_NGEN=16 timeout 12m "$S/bin/run-gen-$arm-120a" "$GG" $ids > "$OUT/$M-$arm.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$M $arm rc=$r $(grep -E 'MATCH|MISMATCH|FLIP-NEARTIE' "$OUT/$M-$arm.log" | cut -c1-170 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
  done
  if cmp -s "$OUT/$M-1258base.logits" "$OUT/$M-1258b.logits"; then
    echo "$M logits BIT-EQUAL ($(stat -c %s "$OUT/$M-1258b.logits") bytes)" | tee -a "$OUT/guard.log"
  else
    echo "$M logits DIFFER" | tee -a "$OUT/guard.log"; rc=1
  fi
  python3 - "$OUT/$M-1258base.logits" "$OUT/$M-1258b.logits" <<'EOF' | tee -a "$OUT/guard.log"
import struct, sys
a = open(sys.argv[1], "rb").read(); b = open(sys.argv[2], "rb").read()
n = min(len(a), len(b)) // 4
fa = struct.unpack(f"<{n}f", a[:4 * n]); fb = struct.unpack(f"<{n}f", b[:4 * n])
d = max((abs(x - y) for x, y in zip(fa, fb)), default=float("nan"))
print(f"logit maxdiff main vs head over {n} logits: {d:.6g}")
EOF
  if diff <(grep -E '^ *tokens: ' "$OUT/$M-1258base.log") <(grep -E '^ *tokens: ' "$OUT/$M-1258b.log") > /dev/null; then
    echo "$M generated tokens EQUAL" | tee -a "$OUT/guard.log"
  else
    echo "$M generated tokens DIFFER" | tee -a "$OUT/guard.log"; rc=1
  fi
done
echo "EXACT3-DONE rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
exit $rc
