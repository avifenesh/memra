#!/usr/bin/env bash
# Runs ON a GB10 host: the exactness battery for the h2p binary on the Qwen3.8-27B GGUF.
# usage: gates_attn.sh CAND_BIN BASE_BIN OUT TREE
#   kernel-check x3 on CAND: device default (h2p), MEMRA_PRIME_KV_H2P=0 (h2), MEMRA_PRIME_KV_H2=0 (db);
#   the calibrated run-gen argmax-margin gate (default prompt) on CAND;
#   run-spec K=1..8 self-consistency (T=0, 32 tokens) on CAND;
#   the argmax-margin probe table on the 16k prompt for CAND and BASE, byte-compared;
#   run-gen prime-path last-row logits on a 16k and a 33k record prompt for BASE (h2), CAND (h2p)
#   and CAND with MEMRA_PRIME_KV_H2P=0 (h2), byte-compared.
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
{ echo "CAND $(cat "$C/COMMIT" | head -1)"; echo "BASE $(cat "$B/COMMIT" | head -1)"; } > "$OUT/COMMIT"
KC=(--require-manifest "$TREE/tools/kernel-check-27b.cells" --require-manifest "$TREE/tools/kernel-check-step35.cells")
for arm in default h2poff h2off; do
  case $arm in
    default) envs=() ;;
    h2poff) envs=(MEMRA_PRIME_KV_H2P=0) ;;
    h2off) envs=(MEMRA_PRIME_KV_H2=0) ;;
  esac
  (unset MEMRA_KC_FAST MEMRA_KC_ONLY; env "${envs[@]}" "$C/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check-$arm.log" 2>&1
  echo "kernel-check $arm rc=$? $(tail -1 "$OUT/kernel-check-$arm.log")" | tee -a "$L"
  grep -E "fa_prefill_view_ws vs inline" "$OUT/kernel-check-$arm.log" | tee -a "$L"
done
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" | tee -a "$L"
(unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY; MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$C/run-spec" "$G8") > "$OUT/run-spec.log" 2>&1
echo "run-spec rc=$? $(grep -E 'SELF-CONSISTENCY|acceptance' "$OUT/run-spec.log" | tail -3 | tr '\n' ' ')" | tee -a "$L"
# SKIP16K=1 drops the two 16k argmax-margin probes (about 36 min each on GB10); the logits compare
# below byte-checks the prime path at 16k and 33k either way.
for arm in C B; do
  [ "${SKIP16K:-0}" = 1 ] && break
  bin=$C; [ $arm = B ] && bin=$B
  R=$OUT/argmax16k-$arm; rm -rf "$R"; mkdir -p "$R/root/target" "$R/root/research/e2e"
  cp -a "$TREE/tools" "$R/root/tools"; cp -a "$TREE/research/e2e/prompts" "$R/root/research/e2e/prompts"
  ln -s "$bin" "$R/root/target/release"
  (cd "$R/root" && MEMRA_GATE_LOG_ROOT=$R bash tools/argmax-margin-gate.sh "$G8" --prompt research/e2e/prompts/p4-16k.txt --logdir "$R") > "$R/argmax.log" 2>&1
  echo "argmax-16k $arm rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$R/argmax.log" | tr '\n' ' ')" | tee -a "$L"
done
ta=$(ls "$OUT"/argmax16k-C/gate-*.log 2>/dev/null | head -n 1); tb=$(ls "$OUT"/argmax16k-B/gate-*.log 2>/dev/null | head -n 1)
if [ -n "$ta" ] && [ -n "$tb" ]; then
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$ta" > "$OUT/argmax16k-C.table"
  grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$tb" > "$OUT/argmax16k-B.table"
  if cmp -s "$OUT/argmax16k-C.table" "$OUT/argmax16k-B.table"; then r=IDENTICAL; else r=DIFFER; fi
  echo "argmax-16k probe table C vs B: $r ($(sha256sum < "$OUT/argmax16k-C.table" | cut -c1-16) / $(sha256sum < "$OUT/argmax16k-B.table" | cut -c1-16))" | tee -a "$L"
fi
P=$OUT/logits; mkdir -p "$P"
for n in 600 1240; do
  python3 - "$P/prompt-$n.txt" "$n" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
n = int(sys.argv[2])
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(n)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
  for arm in B C Coff; do
    bin=$C; envv=(); [ $arm = B ] && bin=$B; [ $arm = Coff ] && envv=(MEMRA_PRIME_KV_H2P=0)
    env "${envv[@]}" MEMRA_PROMPT_FILE=$P/prompt-$n.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
      MEMRA_PP_LOGITS=$P/$arm-$n.logits "$bin/run-gen" "$G8" > "$P/$arm-$n.log" 2>&1
    echo "logits $n $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm-$n.log") $(sha256sum "$P/$arm-$n.logits" | cut -c1-16)" | tee -a "$L"
  done
  if cmp -s "$P/B-$n.logits" "$P/C-$n.logits" && cmp -s "$P/B-$n.logits" "$P/Coff-$n.logits"; then r=IDENTICAL; else r=DIFFER; fi
  echo "logits $n B vs C vs Coff: $r" | tee -a "$L"
done
echo GATES-ATTN-DONE | tee -a "$L"
