#!/usr/bin/env bash
# Exact multi-row CPU program receipt: lane (rows default) vs base (one job per row) on Hy3. Stream 0 = P0 everywhere.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ART=/data/hy3; OUT=/root/lane/abrows; mkdir -p "$OUT"
COMMON=( MEMRA_CHAT=1 MEMRA_MOE_CACHE=1 MEMRA_MOE_SIZE_AWARE=1 MEMRA_MOE_LFU=1 MEMRA_MOE_LFU_DECAY=1.0
  MEMRA_MOE_VRAM_FRAC=0.85 MEMRA_MOE_HARD_VRAM_FRAC=0.85 MEMRA_SPILL_IO=direct MEMRA_SPILL_PREAD_DEPTH=32
  MEMRA_SPILL_WORKER_EXPERT_WINDOW=8 MEMRA_CPU_EXPERT_LIB=/root/lane/libmemra-cpu-experts.so
  MEMRA_CPU_EXPERT_THREADS=16 MEMRA_CPU_EXPERT_IO_THREADS=8 MEMRA_CPU_EXPERT_CACHE_GB=60 MEMRA_CPU_EXPERT_RESERVE_GB=8
  MEMRA_CPU_EXPERT_IO=direct MEMRA_CPU_EXPERT_FREEZE_CACHE=1 MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128
  MEMRA_CPU_EXPERT_FREEZE_PROFILE=/root/lane/hy3.freeze NVIDIA_TF32_OVERRIDE=0 MEMRA_NGEN=32 )
P0="Explain speculative decoding briefly."
printf '%s\n' "$P0" "Write a haiku about a lighthouse in winter." "List three causes of the French Revolution." "What does a mutex protect against, in one paragraph?" > "$OUT/prompts.txt"
run() { # name binary extra-env...
  local name=$1 bin=$2; shift 2
  echo "== $name start $(date -u +%T)"
  ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" MEMRA_LOCKSTEP_LOGITS_DUMP="$OUT/$name.f32" "$@" "$bin" "$ART" ) > "$OUT/$name.log" 2>&1
  echo "== $name rc=$? end $(date -u +%T); $(grep -c 'stream0 step' "$OUT/$name.log") step lines; $(grep -h 'lockstep m=' "$OUT/$name.log" | cut -c1-60)"
}
BASE=/root/lane/target-base/release/run_lockstep; LANE=/root/lane/target-lane/release/run_lockstep
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "$OUT/gpu.txt"
sha256sum "$BASE" "$LANE" /root/lane/target-base/release/run-gen /root/lane/libmemra-cpu-experts.so > "$OUT/binaries.sha256"
(cd /root/lane/memra && echo "base $(git rev-parse HEAD)"; cd /root/lane/memra-lane && echo "lane $(git rev-parse HEAD)" && git log --oneline -2) > "$OUT/commits.txt"
python3 -c "import json; c=json.load(open('/data/hy3/config.json')); print(c.get('vocab_size'))" > "$OUT/n_vocab.txt"
[ -f /root/lane/hy3.freeze ] || { echo "== warmup"; ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" MEMRA_NGEN=8 MEMRA_PROMPT="$P0" /root/lane/target-base/release/run-gen "$ART" ) > "$OUT/warmup-rungen.log" 2>&1; echo "warmup rc=$?"; }
run base-m1        "$BASE" MEMRA_LOCKSTEP_M=1 MEMRA_PROMPT="$P0"
run lane-m1        "$LANE" MEMRA_LOCKSTEP_M=1 MEMRA_PROMPT="$P0"
run lane-m4-mixed  "$LANE" MEMRA_LOCKSTEP_M=4 MEMRA_PROMPTS_FILE="$OUT/prompts.txt" MEMRA_LOCKSTEP_CPU_ROWS=1
echo "ABROWS_DONE $(date -u +%T)"
