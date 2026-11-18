#!/usr/bin/env bash
# memra#1258 part 2 (one card, under maxq-cells.sh): exactness of one arm against another, the
# cell of b-exact2.sh with the kit's own paths. Arms A and B name binaries run-gen-<arm>-120a,
# run-spec-<arm>-120a, dspark-q38-gate-<arm>-120a under $KIT/bin; the model and the draft are
# MAXQ_GG and MAXQ_DR; the output directory is OUT. The `[prime-ws]` lines of every process are
# counted in guard.log: they say whether a workspace was dropped under the floor.
# EXACT_ONLY names one item (rungen4106, rungen8200, runspec8200, dspark) to run alone. The card's
# used memory is sampled twice a second and each process's peak is written to guard.log; with
# PEAK_DIR set, PEAK_DIR/ITEM.tight gets the larger arm's peak plus PEAK_PAD MiB (default 768):
# the free memory a later step holds the card to, so that the item's prime calls end under the
# park floor (maxq-cells.sh mode emul:@FILE). PROMPTS_FROM names the prompts/ directory of an
# earlier run to reuse instead of deriving the ids again. Prompts of 2,000 / 4,106 / 8,200 / 16,500 tokens (ids from run-gen's
# prefill-only mode on a prose text, texts from the GGUF vocabulary): run-gen argmax at 4,106 and
# 8,200, run-spec K=1..8 at 8,200, dspark_q38_gate at all four. The arms' streams and digests must
# be equal. usage: OUT=DIR KIT=DIR MAXQ_GG=GGUF MAXQ_DR=DIR exact-maxq.sh A B
set -uo pipefail
A=$1; Bm=$2
S=${KIT:?kit dir}
B=$S
GG=${MAXQ_GG:?model}
DR=${MAXQ_DR:?draft dir}
OUT=${OUT:?out dir}
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
mkdir -p "$OUT/prompts"
ONLY=${EXACT_ONLY:-}
want() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }
card_used() { nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
BASE_USED=$(card_used)
echo 0 > "$OUT/.peak"
( while :; do u=$(card_used); m=$(cat "$OUT/.peak" 2>/dev/null); [ "${u:-0}" -gt "${m:-0}" ] && echo "$u" > "$OUT/.peak"; sleep 0.5; done ) &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null' EXIT
peak_reset() { echo 0 > "$OUT/.peak"; }
peak_note() {  # peak_note ARM ITEM: this process's peak, and the tight target when PEAK_DIR is set
  local p; p=$(cat "$OUT/.peak"); p=$(( p - BASE_USED ))
  echo "peak $1 $2: about $p MiB (card used before the runs $BASE_USED MiB)" | tee -a "$OUT/guard.log"
  if [ -n "${PEAK_DIR:-}" ]; then
    mkdir -p "$PEAK_DIR"
    local t=$(( p + ${PEAK_PAD:-768} )) old; old=$(cat "$PEAK_DIR/$2.tight" 2>/dev/null || echo 0)
    [ "$t" -gt "$old" ] && echo "$t" > "$PEAK_DIR/$2.tight"
  fi
}
{ date -Is; for a in "$A" "$Bm"; do sha256sum "$S/bin/run-gen-$a-120a" "$S/bin/run-spec-$a-120a" "$S/bin/dspark-q38-gate-$a-120a"; done
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,driver_version,temperature.gpu,power.draw,enforced.power.limit,memory.total,memory.used --format=csv,noheader; } >> "$OUT/guard.log"
if [ -n "${PROMPTS_FROM:-}" ]; then
  # The prompts of an earlier run of this cell (the same ids: they are a function of the GGUF and
  # the prose seed). A step held to a tight free-memory target cannot run the 16.5k-token ids pass.
  cp -a "$PROMPTS_FROM"/. "$OUT/prompts/" || exit 2
  (cd "$OUT/prompts" && sha256sum p*.ids) | tee -a "$OUT/guard.log"
else
KIT=$S python3 - "$OUT/long.txt" <<'EOF'
import importlib.util, sys
from pathlib import Path
import os
spec = importlib.util.spec_from_file_location("gate", Path(os.environ["KIT"]) / "harness/same-pass-admission-gate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
Path(sys.argv[1]).write_text(g.prose(71, 11000))
EOF
MEMRA_PROMPT_FILE=$OUT/long.txt MEMRA_PP_ONLY=1 MEMRA_PP_PRIME=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
  timeout 5m "$S/bin/run-gen-$A-120a" "$GG" > "$OUT/ids.log" 2>&1
python3 "$B/gguf-detok.py" "$GG" "$OUT/ids.log" "$OUT/prompts" 2000 4106 8200 16500 | tee -a "$OUT/guard.log" || exit 2
  (cd "$OUT/prompts" && sha256sum p*.ids) | tee -a "$OUT/guard.log"
fi
rc=0
for arm in "$A" "$Bm"; do
  for n in 4106 8200; do
    want rungen$n || continue
    ids=$(cat "$OUT/prompts/p$n.ids")
    peak_reset
    # shellcheck disable=SC2086
    MEMRA_NGEN=32 timeout 12m "$S/bin/run-gen-$arm-120a" "$GG" $ids > "$OUT/$arm-rungen-$n.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$arm run-gen $n rc=$r $(grep -E 'MATCH|MISMATCH|FLIP-NEARTIE' "$OUT/$arm-rungen-$n.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
    peak_note "$arm" rungen$n
  done
  if want runspec8200; then
    ids=$(cat "$OUT/prompts/p8200.ids")
    peak_reset
    # shellcheck disable=SC2086
    MEMRA_SPEC_TEMP=0 timeout 10m "$S/bin/run-spec-$arm-120a" "$GG" $ids > "$OUT/$arm-runspec-8200.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$arm run-spec 8200 rc=$r $(grep -E 'STEP-IDENTITY|=== SELF-CONSISTENCY' "$OUT/$arm-runspec-8200.log" | cut -c1-160 | tr '\n' ' ')" | tee -a "$OUT/guard.log"
    peak_note "$arm" runspec8200
  fi
  if want dspark; then
    peak_reset
    MEMRA_PROMPT_DIR=$OUT/prompts timeout 8m "$S/bin/dspark-q38-gate-$arm-120a" "$GG" "$DR" 96 > "$OUT/$arm-dspark.log" 2>&1
    r=$?; [ $r = 0 ] || rc=1
    echo "$arm dspark rc=$r $(grep -E '== dspark_q38_gate' "$OUT/$arm-dspark.log" | cut -c1-160)" | tee -a "$OUT/guard.log"
    peak_note "$arm" dspark
  fi
  grep -h -E '^\[dspark-q38-gate\]' "$OUT/$arm-dspark.log" 2>/dev/null | sed -E 's/ prime_s=.*//' > "$OUT/$arm-dspark-digests.txt"
  grep -E '^ *tokens: |self-consistency' "$OUT/$arm-rungen-4106.log" "$OUT/$arm-rungen-8200.log" "$OUT/$arm-runspec-8200.log" 2>/dev/null \
    | sed -E 's|^[^:]*/[^-]*-||' > "$OUT/$arm-streams.txt"
done
if [ ! -s "$OUT/$A-dspark-digests.txt" ] && [ ! -s "$OUT/$A-streams.txt" ]; then
  echo "ARMS-NOTHING-COMPARED" | tee -a "$OUT/guard.log"; rc=1
elif cmp -s "$OUT/$A-dspark-digests.txt" "$OUT/$Bm-dspark-digests.txt" && cmp -s "$OUT/$A-streams.txt" "$OUT/$Bm-streams.txt"; then
  echo "ARMS-EQUAL" | tee -a "$OUT/guard.log"
else
  echo "ARMS-DIFFER" | tee -a "$OUT/guard.log"; rc=1
fi
for f in "$OUT"/*.log; do
  case "$(basename "$f")" in guard.log|step.log|ballast-fixed.log) continue ;; esac
  echo "prime-ws lines $(basename "$f"): $(grep -c '^\[prime-ws\]' "$f") $(grep -o '^\[prime-ws\] dev[0-9]* GDN scan workspace [a-z ]*' "$f" | sort | uniq -c | tr '\n' ';')" >> "$OUT/guard.log"
done
echo "EXACT2-DONE rc=$rc $(date -Is)" | tee -a "$OUT/guard.log"
exit $rc
