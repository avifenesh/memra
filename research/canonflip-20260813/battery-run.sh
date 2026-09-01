#!/usr/bin/env bash
# Q35-only engine battery on the live box. Yields immediately to real traffic.
set -uo pipefail
STOP=/root/BATTERY_STOP
out=/data/memra-battery-20260813
build=/data/memra-v0820-candidate-df7547bcd-build
repo=$build/src-tmp
bin=$repo/target/release
models=/scratch/models
q35=$models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
q35_draft=$models/draft-35b-owntrim-nvfp4head-q4blk.gguf
prompt=$repo/research/e2e/prompts/pp512.txt
mkdir -p "$out"
echo $$ > /root/battery.pgid   # this script is the process-group leader (launched with setsid)
export PATH=/root/.cargo/bin:/usr/local/cuda-13.2/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64

guard() { [[ -e "$STOP" ]] && { echo "ABORTED_BY_TRAFFIC before $1" | tee -a "$out/verdict.txt"; exit 99; }; return 0; }
cell() {
  local name=$1; shift
  guard "$name"
  echo "--- $name start $(date -u +%FT%TZ) ---" | tee -a "$out/verdict.txt"
  local t0=$SECONDS
  "$@" >"$out/$name.log" 2>&1
  local rc=$?
  echo "$name rc=$rc duration_s=$((SECONDS-t0))" | tee -a "$out/verdict.txt"
  return $rc
}

cell kernel-check timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_KC_MODELS_DIR="$models" "$bin/kernel-check"
grep -q 'ALL GREEN' "$out/kernel-check.log" && echo "kernel-check: ALL GREEN" >>"$out/verdict.txt" \
  || echo "kernel-check: NO ALL-GREEN LINE" >>"$out/verdict.txt"
grep -Eq '(^|[^A-Z])FAIL([^A-Z]|$)|MISMATCH' "$out/kernel-check.log" \
  && echo "kernel-check: FAILURE VERDICT PRESENT" >>"$out/verdict.txt"

cell run-gen-q35 timeout 2400 env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
  MEMRA_PROMPT_FILE="$prompt" MEMRA_CHAT=1 "$bin/run-gen" "$q35"
grep -q 'argmax=.*MATCH' "$out/run-gen-q35.log" && echo "run-gen-q35: argmax MATCH" >>"$out/verdict.txt" \
  || echo "run-gen-q35: NO argmax MATCH" >>"$out/verdict.txt"

cell run-spec-q35 timeout 4800 env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR \
  CUDA_VISIBLE_DEVICES=0 MEMRA_MTP_DRAFT="$q35_draft" MEMRA_NGEN=32 \
  MEMRA_PROMPT_FILE="$prompt" MEMRA_CHAT=1 "$bin/run-spec" "$q35"
n=$(grep -c 'self-consistency: PASS' "$out/run-spec-q35.log" 2>/dev/null || echo 0)
echo "run-spec-q35: self-consistency PASS count=$n (need 8)" >>"$out/verdict.txt"
grep -q '=== SELF-CONSISTENCY PASS ===' "$out/run-spec-q35.log" && echo "run-spec-q35: overall PASS" >>"$out/verdict.txt"

echo "BATTERY_COMPLETE $(date -u +%FT%TZ)" | tee -a "$out/verdict.txt"
