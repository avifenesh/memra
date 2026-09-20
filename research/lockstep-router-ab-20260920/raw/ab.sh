#!/usr/bin/env bash
# Hy3 lockstep router A/B for memra #565: base (origin/main) vs fix (main + router commit).
# Cells per arm: M=1 (prompt P0), M=4 same prompt (identity gate), M=4 distinct prompts (stream 0 = P0).
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ART=/data/hy3; OUT=/root/lane/ab; mkdir -p "$OUT"
COMMON=( MEMRA_CHAT=1 MEMRA_MOE_CACHE=1 MEMRA_MOE_SIZE_AWARE=1 MEMRA_MOE_LFU=1 MEMRA_MOE_LFU_DECAY=1.0
  MEMRA_MOE_VRAM_FRAC=0.85 MEMRA_MOE_HARD_VRAM_FRAC=0.85 MEMRA_SPILL_IO=direct MEMRA_SPILL_PREAD_DEPTH=32
  MEMRA_SPILL_WORKER_EXPERT_WINDOW=8 MEMRA_CPU_EXPERT_LIB=/root/lane/libmemra-cpu-experts.so
  MEMRA_CPU_EXPERT_THREADS=16 MEMRA_CPU_EXPERT_IO_THREADS=8 MEMRA_CPU_EXPERT_CACHE_GB=60 MEMRA_CPU_EXPERT_RESERVE_GB=8
  MEMRA_CPU_EXPERT_IO=direct MEMRA_CPU_EXPERT_FREEZE_CACHE=1 MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128
  MEMRA_CPU_EXPERT_FREEZE_PROFILE=/root/lane/hy3.freeze NVIDIA_TF32_OVERRIDE=0 )
P0="Explain speculative decoding briefly."
printf '%s\n' "$P0" "Write a haiku about a lighthouse in winter." "List three causes of the French Revolution." "What does a mutex protect against, in one paragraph?" > "$OUT/prompts.txt"
run() { # name binary extra-env...
  local name=$1 bin=$2; shift 2
  echo "== $name start $(date -u +%T)"
  ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" "$@" "$bin" "$ART" ) > "$OUT/$name.log" 2>&1
  echo "== $name rc=$? end $(date -u +%T); tail: $(tail -1 "$OUT/$name.log" | cut -c1-120)"
}
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "$OUT/gpu.txt"
sha256sum /root/lane/target-base/release/run_lockstep /root/lane/target-fix/release/run_lockstep /root/lane/target-base/release/run-gen /root/lane/libmemra-cpu-experts.so > "$OUT/binaries.sha256"
(cd /root/lane/memra && git rev-parse HEAD; cd /root/lane/memra-fix && git rev-parse HEAD) > "$OUT/commits.txt"
[ -f /root/lane/hy3.freeze ] || run warmup-rungen /root/lane/target-base/release/run-gen MEMRA_NGEN=8 MEMRA_PROMPT="$P0"
for arm in base fix; do
  B=/root/lane/target-$arm/release/run_lockstep
  run "$arm-m1"       "$B" MEMRA_LOCKSTEP_M=1 MEMRA_NGEN=32 MEMRA_PROMPT="$P0"
  run "$arm-m4-same"  "$B" MEMRA_LOCKSTEP_M=4 MEMRA_NGEN=32 MEMRA_PROMPT="$P0"
  run "$arm-m4-mixed" "$B" MEMRA_LOCKSTEP_M=4 MEMRA_NGEN=32 MEMRA_PROMPTS_FILE="$OUT/prompts.txt"
done
echo "AB_DONE $(date -u +%T)"
