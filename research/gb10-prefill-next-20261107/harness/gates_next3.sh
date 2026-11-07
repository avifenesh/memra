#!/usr/bin/env bash
# Runs ON a GB10 host: the exactness battery for this lane's binary on the Qwen3.8-27B GGUF.
# usage: gates_next.sh CAND_BIN BASE_BIN OUT TREE
#   kernel-check x2 on CAND: defaults, and every lever of the lane rolled back (OFF_ENV);
#   the calibrated run-gen argmax-margin gate (default prompt) on CAND;
#   run-spec K=1..8 self-consistency (T=0, 32 tokens) on CAND;
#   prime-path (MEMRA_PP_PRIME=1) last-row logits on a 16k and a 33k record prompt for BASE,
#   CAND and CAND with OFF_ENV, byte-compared; the monolithic forward's logits at 16k for BASE and
#   CAND, byte-compared.
set -u
C=$1; B=$2; OUT=$3; TREE=$4
OFF_ENV=${OFF_ENV:-"MEMRA_GDN_TRAFFIC=0 MEMRA_PRIME_NORM_FUSE=0 MEMRA_DP4A_TILED=0"}
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$OUT"
L=$OUT/gates.log
GATE=$OUT/gate-root
rm -rf "$GATE"; mkdir -p "$GATE/target" "$GATE/research/e2e"
cp -a "$TREE/tools" "$GATE/tools"
cp -a "$TREE/research/e2e/prompts" "$GATE/research/e2e/prompts"
ln -s "$C" "$GATE/target/release"
{ echo "CAND $(head -1 "$C/COMMIT")"; echo "BASE $(head -1 "$B/COMMIT")"; echo "OFF_ENV $OFF_ENV"; } > "$OUT/COMMIT"
read -r -a OFF <<< "$OFF_ENV"
KC=(--require-manifest "$TREE/tools/kernel-check-27b.cells" --require-manifest "$TREE/tools/kernel-check-step35.cells")
for arm in default off; do
  envs=(); [ $arm = off ] && envs=("${OFF[@]}")
  (unset MEMRA_KC_FAST MEMRA_KC_ONLY; env "${envs[@]}" "$C/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check-$arm.log" 2>&1
  echo "kernel-check $arm rc=$? $(tail -1 "$OUT/kernel-check-$arm.log")" | tee -a "$L"
  grep -E "GDN-DB-GQA|ADD-RMS-NORM-E4M3V2|GATED-RMSNORM-E4M3V2|MMQ-F8F4-V2-GATEUP|MMQ-F8F4-V2-RESID|MMQ-F8F4-V2 SUMMARY|NVFP4-DP4A-RP-T4|GDN-CONV-L2|SIG-MUL-E4M3V2|Q-SPLIT-NORM-ROPE|add_rms_norm fused" "$OUT/kernel-check-$arm.log" | tee -a "$L"
done
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" | tee -a "$L"
(unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY; MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$C/run-spec" "$G8") > "$OUT/run-spec.log" 2>&1
echo "run-spec rc=$? $(grep -E 'SELF-CONSISTENCY|acceptance' "$OUT/run-spec.log" | tail -3 | tr '\n' ' ')" | tee -a "$L"
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
    bin=$C; envv=(); [ $arm = B ] && bin=$B; [ $arm = Coff ] && envv=("${OFF[@]}")
    env "${envv[@]}" MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$P/prompt-$n.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
      MEMRA_PP_LOGITS=$P/$arm-$n.logits "$bin/run-gen" "$G8" > "$P/$arm-$n.log" 2>&1
    echo "prime logits $n $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm-$n.log") $(sha256sum "$P/$arm-$n.logits" | cut -c1-16)" | tee -a "$L"
  done
  if cmp -s "$P/B-$n.logits" "$P/C-$n.logits" && cmp -s "$P/B-$n.logits" "$P/Coff-$n.logits"; then r=IDENTICAL; else r=DIFFER; fi
  echo "prime logits $n B vs C vs Coff: $r" | tee -a "$L"
done
for arm in B C; do
  bin=$C; [ $arm = B ] && bin=$B
  MEMRA_PROMPT_FILE=$P/prompt-600.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
    MEMRA_PP_LOGITS=$P/fwd-$arm-600.logits "$bin/run-gen" "$G8" > "$P/fwd-$arm-600.log" 2>&1
  echo "forward logits 600 $arm rc=$? $(sha256sum "$P/fwd-$arm-600.logits" | cut -c1-16)" | tee -a "$L"
done
if cmp -s "$P/fwd-B-600.logits" "$P/fwd-C-600.logits"; then r=IDENTICAL; else r=DIFFER; fi
echo "forward logits 600 B vs C: $r" | tee -a "$L"
# the batched varlen prime (compact q/k at 3:1 reaches its *_vl kernels): batched against individual
# primes bit for bit (--exact), on the lane defaults and with every lever rolled back
for arm in C Coff; do
  envv=(); [ $arm = Coff ] && envv=("${OFF[@]}")
  env "${envv[@]}" "$C/prime-batch-gate" "$G8" --batch 3 --plen 300 --steps 16 --exact > "$OUT/prime-batch-$arm.log" 2>&1
  echo "prime-batch-gate $arm rc=$? $(grep -E 'PASS|FAIL' "$OUT/prime-batch-$arm.log" | tail -n 2 | tr '\n' ' ')" | tee -a "$L"
done
grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$OUT/prime-batch-C.log" > "$OUT/prime-batch-C.table"
grep -vE 'ms|s/tok|tok/s|elapsed|wall|time' "$OUT/prime-batch-Coff.log" > "$OUT/prime-batch-Coff.table"
if cmp -s "$OUT/prime-batch-C.table" "$OUT/prime-batch-Coff.table"; then r=IDENTICAL; else r=DIFFER; fi
echo "prime-batch-gate C vs Coff output: $r" | tee -a "$L"
echo GATES-NEXT-DONE | tee -a "$L"
