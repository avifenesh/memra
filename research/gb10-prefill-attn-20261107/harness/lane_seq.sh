#!/usr/bin/env bash
# Runs ON a GB10 host under this lane's lease, one GPU job at a time:
#   1. tree (detached at BASE_SHA) + warm target if missing; build BASE_SHA and the lane head from
#      the bundle (BRANCH) and bank both;
#   2. the exactness battery on the lane binary: kernel-check in h2p (default), h2
#      (MEMRA_PRIME_KV_H2P=0) and db (MEMRA_PRIME_KV_H2=0) modes, the calibrated argmax-margin
#      gate, run-spec K=1..8;
#   3. the prime-path logits byte compare (MEMRA_PP_PRIME=1): BASE, lane (h2p) and lane with
#      MEMRA_PRIME_KV_H2P=0 (h2) on a 16k and a 33k record prompt, plus their prime timings (median of 3);
#   4. the served campaign in ORDER (fresh boot per cell, 32k/131k/260k per boot).
# usage: lane_seq.sh REPO WARM BASE_SHA BRANCH TAG "ORDER"
set -u
D=$HOME/.cache/memra-attn-gb10
REPO=$1; WARM=$2; BASE=$3; BR=$4; TAG=$5; ORDER=$6
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$D/logs" "$D/bins" "$D/runs"
S=$D/logs/lane-$TAG.log
note() { echo "$(date -Is) $*" | tee -a "$S"; }
[ -d "$D/tree" ] || bash "$D/box/setup_tree.sh" "$REPO" "$WARM" >> "$S" 2>&1
[ -d "$D/target" ] || cp -a "$WARM" "$D/target"
git -C "$D/tree" fetch -q origin main || note "fetch origin main failed"
git -C "$D/tree" fetch -q "$D/h2p.bundle" "$BR:refs/heads/attn-lane" --force || { note "bundle fetch failed"; exit 1; }
git -C "$D/tree" checkout -q --detach "$BASE" || { note "checkout $BASE failed"; exit 1; }
JOBS=8 bash "$D/box/build.sh" "base" >> "$S" 2>&1
git -C "$D/tree" checkout -q --detach attn-lane || exit 1
JOBS=8 bash "$D/box/build.sh" "h2p" >> "$S" 2>&1
B=$D/bins/base-$(git -C "$D/tree" rev-parse --short=10 "$BASE")
C=$D/bins/h2p-$(git -C "$D/tree" rev-parse --short=10 attn-lane)
[ -x "$B/memra-server" ] && [ -x "$C/memra-server" ] || { note "missing binaries $B $C"; exit 1; }
note "base=$B lane=$C"
OUT=$D/runs/gates-$TAG; mkdir -p "$OUT"; L=$OUT/gates.log
GATE=$OUT/gate-root
rm -rf "$GATE"; mkdir -p "$GATE/target" "$GATE/research/e2e"
cp -a "$D/tree/tools" "$GATE/tools"; cp -a "$D/tree/research/e2e/prompts" "$GATE/research/e2e/prompts"
ln -s "$C" "$GATE/target/release"
{ echo "LANE $(head -1 "$C/COMMIT")"; echo "BASE $(head -1 "$B/COMMIT")"; } > "$OUT/COMMIT"
KC=(--require-manifest "$D/tree/tools/kernel-check-27b.cells" --require-manifest "$D/tree/tools/kernel-check-step35.cells")
for arm in default h2poff h2off; do
  case $arm in default) envs=() ;; h2poff) envs=(MEMRA_PRIME_KV_H2P=0) ;; h2off) envs=(MEMRA_PRIME_KV_H2=0) ;; esac
  (unset MEMRA_KC_FAST MEMRA_KC_ONLY; env "${envs[@]}" "$C/kernel-check" "$G8" "${KC[@]}") > "$OUT/kernel-check-$arm.log" 2>&1
  echo "kernel-check $arm rc=$? $(tail -1 "$OUT/kernel-check-$arm.log")" | tee -a "$L"
  grep -E "fa_prefill_view_ws vs inline" "$OUT/kernel-check-$arm.log" | tee -a "$L"
done
MEMRA_GATE_LOG_ROOT=$OUT bash "$GATE/tools/argmax-margin-gate.sh" "$G8" > "$OUT/argmax.log" 2>&1
echo "argmax rc=$? $(grep -E 'SUMMARY|PASS|FAIL' "$OUT/argmax.log" | tr '\n' ' ')" | tee -a "$L"
cp /tmp/memra-argmax-margin-gate/gate-Qwen3.8-27B-NVFP4-Q5K-mtp-board-2048.log "$OUT/argmax-gate-raw.log" 2>/dev/null
(unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY; MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 "$C/run-spec" "$G8") > "$OUT/run-spec.log" 2>&1
echo "run-spec rc=$? $(grep -E 'SELF-CONSISTENCY|acceptance' "$OUT/run-spec.log" | tail -3 | tr '\n' ' ')" | tee -a "$L"
P=$OUT/prime-logits; mkdir -p "$P"
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
    env "${envv[@]}" MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$P/prompt-$n.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=3 MEMRA_PP_WARMUP=1 \
      MEMRA_PP_LOGITS=$P/$arm-$n.logits "$bin/run-gen" "$G8" > "$P/$arm-$n.log" 2>&1
    echo "prime-logits $n $arm rc=$? $(grep -E 'pp-only MEDIAN' "$P/$arm-$n.log") $(sha256sum "$P/$arm-$n.logits" | cut -c1-16)" | tee -a "$L"
  done
  if cmp -s "$P/B-$n.logits" "$P/C-$n.logits" && cmp -s "$P/B-$n.logits" "$P/Coff-$n.logits"; then r=IDENTICAL; else r=DIFFER; fi
  echo "prime-logits $n B vs C vs Coff: $r" | tee -a "$L"
done
note "gates done; campaign served-$TAG"
bash "$D/box/served_ab.sh" "served-$TAG" "$B" "$C" "$ORDER" >> "$S" 2>&1
note "LANE-SEQ-DONE"
