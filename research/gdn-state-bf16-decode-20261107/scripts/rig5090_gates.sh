#!/usr/bin/env bash
# RTX 5090 Laptop correctness gates for the bf16 GDN decode-class lane
# (research/gdn-state-bf16-decode-20261107). Runs under one gpulock lease (gpulock run).
#   1. fp32 class unchanged: run-gen prefill logits and the 128-token greedy stream (numeric lines),
#      main cf85a23ff5 vs this tree in its default class (fp32 on this card) and forced
#      MEMRA_GDN_STATE_BF16=0, byte for byte, on Qwen3.8-27B and Ornith-1.5-35B-A3B; plus the
#      tree forced to the bf16 decode class, whose prefill logits must not move (fp32 prime).
#   2. kernel-check, tree binary (all cells it can run, gdn_st16_* included).
#   3. GPU tests in both classes on Qwen3.5-9B: packed verify == chained decode rows; chunked
#      capture == prefix scan; capture/restore cold-exact; settle-then-resume; restore ==
#      unrestored through the decode-start rounding, rollback and widen round trips.
#   4. run-spec K=1..8 (greedy, 32 tokens) in both classes on Qwen3.8-27B and Ornith.
#   usage: rig5090_gates.sh BINTAG OUTDIR
set -u
R=$HOME/.cache/memra-gdnbf16-decode
NEW=$R/bins/${1:?bintag}
MAIN=$R/bins/main-cf85a23ff5-120a
OUT=${2:?outdir}
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/gates.log"; }
Q27=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
Q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
PROMPT="Write a Python function that returns the n-th Fibonacci number, then explain its time complexity in two sentences."
sha(){ sha256sum "$1" | cut -c1-16; }
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.limit --format=csv > "$OUT/gpu.txt"
sha256sum "$NEW"/* "$NEW"/tests/* "$MAIN"/* > "$OUT/bins.sha256" 2>&1

for m in q27:$Q27 orn:$ORN; do
  tag=${m%%:*}; g=${m#*:}
  for arm in main newd new0 new1; do
    bin=$NEW/run-gen; envs=""
    case $arm in
      main) bin=$MAIN/run-gen ;;
      new0) envs="MEMRA_GDN_STATE_BF16=0" ;;
      new1) envs="MEMRA_GDN_STATE_BF16=1" ;;
    esac
    d=$OUT/ident/$tag-$arm; mkdir -p "$d"
    env $envs MEMRA_CHAT=1 MEMRA_NGEN=128 timeout 900 "$bin" "$g" --prompt "$PROMPT" > "$d/run-gen.log" 2>&1
    echo "rc=$?" >> "$d/run-gen.log"
    env $envs MEMRA_CHAT=1 MEMRA_PP_ONLY=1 MEMRA_PP_LOGITS=$d/pp.f32 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 timeout 900 \
      "$bin" "$g" --prompt "$PROMPT" > "$d/pp.log" 2>&1
    echo "rc=$?" >> "$d/pp.log"
    grep -E '^tokens: |argmax|maxdiff' "$d/run-gen.log" > "$d/numeric.txt"
    log "ident $tag $arm pp_forward_last=$(sha $d/pp.f32) numeric=$(sha $d/numeric.txt) bf16class=$(grep -c 'bf16 GDN decode class' "$d/run-gen.log")"
  done
done

timeout 3600 "$NEW/kernel-check" > "$OUT/kernel-check.log" 2>&1
log "kernel-check rc=$? $(grep -cE ' OK$' "$OUT/kernel-check.log") OK, $(grep -cE ' FAIL$' "$OUT/kernel-check.log") FAIL; gdn_st16 $(grep -E '^gdn_st16' "$OUT/kernel-check.log" | grep -c ' OK$') OK; $(tail -n 1 "$OUT/kernel-check.log" | cut -c1-120)"

for t in gdn_packed_gpu grid_capture_gpu; do
  for c in 0 1; do
    MEMRA_GDN_STATE_BF16=$c MEMRA_TEST_QWEN_GGUF=$Q9 timeout 2400 "$NEW/tests/$t" --ignored --test-threads=1 > "$OUT/test-$t-bf16$c.log" 2>&1
    log "test $t bf16=$c rc=$? $(grep -E '^test result' "$OUT/test-$t-bf16$c.log" | tail -1)"
  done
done

for m in q27:$Q27 orn:$ORN; do
  tag=${m%%:*}; g=${m#*:}
  for c in 0 1; do
    ( unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY
      MEMRA_GDN_STATE_BF16=$c MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 timeout 2400 "$NEW/run-spec" "$g" ) > "$OUT/run-spec-$tag-bf16$c.log" 2>&1
    log "run-spec $tag bf16=$c rc=$? $(python3 "$R/wt/tools/release-coverage.py" spec < "$OUT/run-spec-$tag-bf16$c.log" 2>&1 | tail -n 1 | cut -c1-200)"
  done
done
log GATES_DONE
