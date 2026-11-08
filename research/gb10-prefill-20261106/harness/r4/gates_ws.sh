#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: the exactness battery for the KV dequant workspace lever.
# usage: gates_ws.sh CAND_BIN BASE_BIN OUT TREE
#   kernel-check on CAND (the ARC B cells run the workspace + prime attention against the
#   inline-dequant kernel bit for bit); the calibrated run-gen argmax-margin gate on CAND;
#   prime-path (MEMRA_PP_PRIME=1) last-row logits at 16k and 33k for BASE and CAND, byte-compared.
set -u
C=$1; B=$2; OUT=$3; TREE=$4
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
L=$OUT/gates.log
GATE=$OUT/gate-root
rm -rf "$GATE"; mkdir -p "$GATE/target" "$GATE/research/e2e"
cp -a "$TREE/tools" "$GATE/tools"
cp -a "$TREE/research/e2e/prompts" "$GATE/research/e2e/prompts"
ln -s "$C" "$GATE/target/release"
{ echo "CAND $(head -1 "$C/COMMIT")"; echo "BASE $(head -1 "$B/COMMIT")"; } > "$OUT/COMMIT"
KC=(--require-manifest "$TREE/tools/kernel-check-27b.cells" --require-manifest "$TREE/tools/kernel-check-step35.cells")
(unset MEMRA_KC_FAST MEMRA_KC_ONLY; "$C/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check.log" 2>&1
echo "kernel-check rc=$? $(tail -1 "$OUT/kernel-check.log")" | tee -a "$L"
grep -E "fa_prefill_view_ws|FAIL" "$OUT/kernel-check.log" | head -n 20 | tee -a "$L"
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" | tee -a "$L"
P=$OUT/logits; mkdir -p "$P"
for n in 600 1240; do
  python3 - "$P/prompt-$n.txt" "$n" <<'PY'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
n = int(sys.argv[2])
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(n)]
open(sys.argv[1], "w").write("\n".join(lines))
PY
  for arm in B C; do
    bin=$C; [ $arm = B ] && bin=$B
    MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$P/prompt-$n.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
      MEMRA_PP_LOGITS=$P/$arm-$n.logits "$bin/run-gen" "$G8" > "$P/$arm-$n.log" 2>&1
    echo "prime logits $n $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm-$n.log") $(sha256sum "$P/$arm-$n.logits" | cut -c1-16)" | tee -a "$L"
  done
  if cmp -s "$P/B-$n.logits" "$P/C-$n.logits"; then r=IDENTICAL; else r=DIFFER; fi
  echo "prime logits $n B vs C: $r" | tee -a "$L"
done
echo GATES-WS-DONE | tee -a "$L"
