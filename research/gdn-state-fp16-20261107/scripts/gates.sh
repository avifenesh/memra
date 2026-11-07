#!/usr/bin/env bash
# Runs ON a GB10 unit under the caller's lease: the correctness gates of the fp16 GDN decode class
# (research/gdn-state-fp16-20261107) on Qwen3.8-27B GGUF.
#   1. kernel-check (synthetic cells incl. gdn_st16_*), tree binary
#   2. fp32 class byte identity: main vs tree with MEMRA_GDN_STATE_FP16=0 (run-gen numeric lines
#      + forward_last prefill logits); the tree's fp16 class must keep the prefill logits too
#      (the prime is fp32 in both classes) while its decode stream may move
#   3. run-spec K=1..8 (greedy, 32 tokens) in both classes: verify == plain decode
#   4. GPU tests in both classes: packed verify == chained rows; capture/restore cold-exact,
#      restore == unrestored through the decode-start rounding, rollback and widen round trips
#   usage: gates.sh BINTAG MAINBINDIR
set -u
R=$HOME/.cache/memra-gdnfp16
B=$R/bins/${1:?bintag}
MAIN=${2:?main bin dir}
OUT=$R/runs/gates-$1
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/gates.log"; }
Q=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
PROMPT="Write a Python function that returns the n-th Fibonacci number, then explain its time complexity in two sentences."
sha(){ sha256sum "$1" | cut -c1-16; }
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm --format=csv > "$OUT/gpu.txt" 2>&1
sha256sum "$B"/* "$B"/tests/* "$MAIN"/run-gen > "$OUT/bins.sha256" 2>&1

MEMRA_KC_FAST=1 timeout 900 "$B/kernel-check" > "$OUT/kc-fast.log" 2>&1
log "kernel-check fast rc=$? $(grep -cE ' OK$' "$OUT/kc-fast.log") OK, $(grep -cE ' FAIL$' "$OUT/kc-fast.log") FAIL; gdn_st16: $(grep -E '^gdn_st16' "$OUT/kc-fast.log" | grep -c ' OK$') OK $(grep -E '^gdn_st16' "$OUT/kc-fast.log" | grep -c ' FAIL$') FAIL"

for arm in main new0 new1; do
  bin=$B/run-gen; envs="MEMRA_GDN_STATE_FP16=${arm#new}"
  [ $arm = main ] && bin=$MAIN/run-gen && envs=""
  d=$OUT/ident/$arm; mkdir -p "$d"
  env $envs MEMRA_CHAT=1 MEMRA_NGEN=128 timeout 1200 "$bin" "$Q" --prompt "$PROMPT" > "$d/run-gen.log" 2>&1
  echo "rc=$?" >> "$d/run-gen.log"
  env $envs MEMRA_CHAT=1 MEMRA_PP_ONLY=1 MEMRA_PP_LOGITS=$d/pp.f32 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 timeout 1200 \
    "$bin" "$Q" --prompt "$PROMPT" > "$d/pp.log" 2>&1
  echo "rc=$?" >> "$d/pp.log"
  grep -E '^tokens: |argmax|maxdiff' "$d/run-gen.log" > "$d/numeric.txt"
  log "ident $arm pp_forward_last=$(sha $d/pp.f32) numeric=$(sha $d/numeric.txt) class=$(grep -c 'fp16 GDN decode class' "$d/run-gen.log") $(grep -E 'tok/s' "$d/run-gen.log" | tail -1 | cut -c1-80)"
done

for c in 0 1; do
  ( unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY
    MEMRA_GDN_STATE_FP16=$c MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 timeout 2400 "$B/run-spec" "$Q" ) > "$OUT/run-spec-fp16$c.log" 2>&1
  log "run-spec fp16=$c rc=$? $(grep -iE 'PASS|FAIL|accept' "$OUT/run-spec-fp16$c.log" | tail -n 3 | tr '\n' ' ' | cut -c1-300)"
done

for t in gdn_packed_gpu grid_capture_gpu; do
  for c in 0 1; do
    MEMRA_GDN_STATE_FP16=$c MEMRA_TEST_QWEN_GGUF=$Q timeout 2400 "$B/tests/$t" --ignored --test-threads=1 > "$OUT/test-$t-fp16$c.log" 2>&1
    log "test $t fp16=$c rc=$? $(grep -E '^test result' "$OUT/test-$t-fp16$c.log" | tail -1)"
  done
done
log GATES_DONE
