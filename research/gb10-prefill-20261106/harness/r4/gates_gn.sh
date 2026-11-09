#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: lever 4's exactness battery on its final head. Builds the
# gn2.bundle HEAD (gnf), then on it: kernel-check with the release manifests (GDN-K45-NORM
# included), the calibrated argmax-margin gate, run-spec K=1..8, and the prime-path last-row logits
# of B = gnb (the lever's base) against gnf on three qwen35-class GGUFs.
set -u
D=$HOME/.cache/memra-gb10-r4
OUT=$D/runs/gates-gn-unit1; mkdir -p "$OUT/logits"; L=$OUT/gates.log
git -C "$D/tree" fetch -q "$D/gn2.bundle" "HEAD:refs/heads/r4-lane-gnf" || { echo FETCH-FAIL > "$L"; exit 1; }
GF=$(git -C "$D/tree" rev-parse refs/heads/r4-lane-gnf)
git -C "$D/tree" checkout -q --detach "$GF" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" gnf > "$D/logs/build-gnf.out" 2>&1
B=$(ls -d "$D"/bins/gnb-* | tail -n 1); C=$(ls -d "$D"/bins/gnf-* 2>/dev/null | tail -n 1)
[ -n "$C" ] || { echo BUILD-FAIL >> "$L"; echo GATES-GN-DONE >> "$L"; exit 1; }
{ echo "CAND $(head -1 "$C/COMMIT")"; echo "BASE $(head -1 "$B/COMMIT")"; } > "$OUT/COMMIT"
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=$HOME/models/memra-gate/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
Q9=$HOME/models/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
KC=(--require-manifest "$D/tree/tools/kernel-check-27b.cells" --require-manifest "$D/tree/tools/kernel-check-step35.cells")
(unset MEMRA_KC_FAST MEMRA_KC_ONLY; "$C/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check.log" 2>&1
echo "kernel-check rc=$? $(tail -1 "$OUT/kernel-check.log")" >> "$L"
grep -E "GDN-K45-NORM|GDN-DB-GQA|GATED-RMSNORM-E4M3V2|FAIL" "$OUT/kernel-check.log" >> "$L"
GATE=$OUT/gate-root
rm -rf "$GATE"; mkdir -p "$GATE/target" "$GATE/research/e2e"
cp -a "$D/tree/tools" "$GATE/tools"
cp -a "$D/tree/research/e2e/prompts" "$GATE/research/e2e/prompts"
ln -s "$C" "$GATE/target/release"
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" >> "$L"
(unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY; MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$C/run-spec" "$G8") > "$OUT/run-spec.log" 2>&1
echo "run-spec rc=$? $(grep -E 'acceptance|SELF-CONSISTENCY' "$OUT/run-spec.log" | tail -3 | tr '\n' ' ')" >> "$L"
P=$OUT/logits
for n in 600 1240; do
  python3 - "$P/prompt-$n.txt" "$n" <<'PY'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
n = int(sys.argv[2])
open(sys.argv[1], "w").write("\n".join(f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(n)))
PY
done
for spec in "q27|$G8|600" "q27|$G8|1240" "ornith|$ORN|600" "q9|$Q9|600"; do
  IFS='|' read -r tag model n <<< "$spec"
  for arm in B C; do
    bin=$C; [ $arm = B ] && bin=$B
    MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$P/prompt-$n.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
      MEMRA_PP_LOGITS=$P/$tag-$arm-$n.logits "$bin/run-gen" "$model" > "$P/$tag-$arm-$n.log" 2>&1
    echo "prime logits $tag $n $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$tag-$arm-$n.log") $(sha256sum "$P/$tag-$arm-$n.logits" 2>/dev/null | cut -c1-16)" >> "$L"
  done
  if cmp -s "$P/$tag-B-$n.logits" "$P/$tag-C-$n.logits"; then r=IDENTICAL; else r=DIFFER; fi
  echo "prime logits $tag $n B vs C: $r" >> "$L"
done
echo GATES-GN-DONE >> "$L"
