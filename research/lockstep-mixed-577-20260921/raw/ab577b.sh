#!/usr/bin/env bash
# memra#577 probe matrix on the probe binary (origin/main + diagnostics). Stream 0 = P0 in every cell.
set -uo pipefail
export PATH=/usr/local/cuda/bin:$PATH
ART=/data/hy3; OUT=/root/lane/ab577b; mkdir -p "$OUT"
COMMON=( MEMRA_CHAT=1 MEMRA_MOE_CACHE=1 MEMRA_MOE_SIZE_AWARE=1 MEMRA_MOE_LFU=1 MEMRA_MOE_LFU_DECAY=1.0
  MEMRA_MOE_VRAM_FRAC=0.85 MEMRA_MOE_HARD_VRAM_FRAC=0.85 MEMRA_SPILL_IO=direct MEMRA_SPILL_PREAD_DEPTH=32
  MEMRA_SPILL_WORKER_EXPERT_WINDOW=8 MEMRA_CPU_EXPERT_LIB=/root/lane/libmemra-cpu-experts.so
  MEMRA_CPU_EXPERT_THREADS=16 MEMRA_CPU_EXPERT_IO_THREADS=8 MEMRA_CPU_EXPERT_CACHE_GB=60 MEMRA_CPU_EXPERT_RESERVE_GB=8
  MEMRA_CPU_EXPERT_IO=direct MEMRA_CPU_EXPERT_FREEZE_CACHE=1 MEMRA_CPU_EXPERT_FREEZE_WARMUP_TOKENS=128
  MEMRA_CPU_EXPERT_FREEZE_PROFILE=/root/lane/hy3.freeze NVIDIA_TF32_OVERRIDE=0 MEMRA_NGEN=32 )
P0="Explain speculative decoding briefly."
printf '%s\n' "$P0" "Write a haiku about a lighthouse in winter." "List three causes of the French Revolution." "What does a mutex protect against, in one paragraph?" > "$OUT/prompts.txt"
B=/root/lane/target/release/run_lockstep
run() { # name extra-env...
  local name=$1; shift
  echo "== $name start $(date -u +%T)"
  ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" MEMRA_LOCKSTEP_LOGITS_DUMP="$OUT/$name.f32" MEMRA_MOE_TRACE="$OUT/$name.routes" "$@" "$B" "$ART" ) > "$OUT/$name.log" 2>&1
  echo "== $name rc=$? end $(date -u +%T); $(grep -c 'stream0 step' "$OUT/$name.log") step lines"
}
nvidia-smi --query-gpu=name,driver_version,memory.total,power.limit --format=csv > "$OUT/gpu.txt"
sha256sum "$B" /root/lane/target/release/run-gen /root/lane/libmemra-cpu-experts.so > "$OUT/binaries.sha256"
(cd /root/lane/memra && git rev-parse HEAD && git log --oneline -2) > "$OUT/commits.txt"
python3 -c "import json; c=json.load(open('/data/hy3/config.json')); print(c.get('vocab_size'))" > "$OUT/n_vocab.txt"
[ -f /root/lane/hy3.freeze ] || { echo "== warmup"; ( exec 9>/tmp/memra-gpu.lock; flock 9; env "${COMMON[@]}" MEMRA_NGEN=8 MEMRA_PROMPT="$P0" /root/lane/target/release/run-gen "$ART" ) > "$OUT/warmup-rungen.log" 2>&1; echo "warmup rc=$?"; }
run m1            MEMRA_LOCKSTEP_M=1 MEMRA_PROMPT="$P0"
run m4-mixed      MEMRA_LOCKSTEP_M=4 MEMRA_PROMPTS_FILE="$OUT/prompts.txt"
run m4-mixed-cpurows0 MEMRA_LOCKSTEP_M=4 MEMRA_PROMPTS_FILE="$OUT/prompts.txt" MEMRA_LOCKSTEP_CPU_ROWS=0
run m4-same       MEMRA_LOCKSTEP_M=4 MEMRA_PROMPT="$P0"
run m2-mixed      MEMRA_LOCKSTEP_M=2 MEMRA_PROMPTS_FILE="$OUT/prompts.txt"
run m3-mixed      MEMRA_LOCKSTEP_M=3 MEMRA_PROMPTS_FILE="$OUT/prompts.txt"
echo "AB577_DONE $(date -u +%T)"
