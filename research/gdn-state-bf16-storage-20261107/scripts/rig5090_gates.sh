#!/usr/bin/env bash
# RTX 5090 Laptop correctness gates for the bf16 GDN state storage lane
# (research/gdn-state-bf16-storage-20261107). Runs under one gpulock lease.
#   1. fp32 class unchanged: run-gen prefill logits (verify class), prime_cache logits (serving
#      prime class) and the 128-token greedy stream, main c2e04719b6 vs this tree with
#      MEMRA_GDN_STATE_BF16=0, byte for byte, on Qwen3.8-27B and Ornith-1.5-35B-A3B.
#   2. GPU tests in both classes: packed verify == chained decode rows, chunked capture ==
#      prefix scan, model capture/restore cold-exact (9B), settle-then-resume cold-exact.
#   3. release battery (kernel-check incl. gdn_st16_* cells, calibrated argmax, run-spec K=1..8)
#      once per class.
set -u
R=$HOME/.cache/memra-gdnbf16-real
WT=$R/wt
NEW=$R/bins/dev1-120a/release
MAIN=$R/bins/main-c2e04719b6-120a
OUT=${1:?outdir}
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/gates.log"; }
Q27=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
Q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
PROMPT="Write a Python function that returns the n-th Fibonacci number, then explain its time complexity in two sentences."
sha(){ sha256sum "$1" | cut -c1-16; }
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,clocks.sm,power.limit --format=csv > "$OUT/gpu.txt"
sha256sum "$NEW"/{kernel-check,run-gen,run-spec,argmax-margin-probe,memra-server} "$MAIN"/{kernel-check,run-gen,run-spec} > "$OUT/bins.sha256"

# ---- 1. fp32 class byte identity vs main ----
for m in q27:$Q27 orn:$ORN; do
  tag=${m%%:*}; g=${m#*:}
  for arm in main new0; do
    bin=$MAIN/run-gen; envs=""
    [ $arm = new0 ] && bin=$NEW/run-gen && envs="MEMRA_GDN_STATE_BF16=0"
    d=$OUT/ident/$tag-$arm; mkdir -p "$d"
    env $envs MEMRA_CHAT=1 MEMRA_NGEN=128 timeout 900 \
      "$bin" "$g" --prompt "$PROMPT" > "$d/run-gen.log" 2>&1
    echo "rc=$?" >> "$d/run-gen.log"
    env $envs MEMRA_CHAT=1 MEMRA_PP_ONLY=1 MEMRA_PP_LOGITS=$d/pp.f32 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 timeout 900 \
      "$bin" "$g" --prompt "$PROMPT" > "$d/pp.log" 2>&1
    echo "rc=$?" >> "$d/pp.log"
    # numeric lines only (timing lines differ run to run): the token stream (prime_cache +
    # 128 decode steps), the tokenwise-vs-prefill and batched-prime argmax/maxdiff verdicts
    grep -E '^tokens: |argmax|maxdiff' "$d/run-gen.log" > "$d/numeric.txt"
    log "ident $tag $arm pp_forward_last=$(sha $d/pp.f32) numeric=$(sha $d/numeric.txt) $(grep -E 'tok/s' "$d/run-gen.log" | tail -1 | cut -c1-80)"
  done
done

# ---- 3. release battery per class ----
ln -sfn "$R/bins/dev1-120a" "$WT/target"
printf 'own\tornith-ai/ornith-1.5-35b-a3b\t%s\tself-published\nvendor\tqwen/qwen3.8-27b\t%s\tlocal copy\n' "$ORN" "$Q27" > "$OUT/roster.tsv"
for c in 0 1; do
  MEMRA_GDN_STATE_BF16=$c timeout 7200 bash "$WT/tools/release-battery.sh" --roster "$OUT/roster.tsv" \
    --evidence-dir "$OUT/battery-bf16$c" > "$OUT/battery-bf16$c.log" 2>&1
  log "battery bf16=$c rc=$? $(grep -E 'PASS|FAIL|REFUSED' "$OUT/battery-bf16$c.log" | tr '\n' ' ' | cut -c1-600)"
done
# ---- 2. GPU tests, both classes ----
T=$R/target-rig/release/deps
waitfor --file $R/logs/build-rig-tests.log --grep '^rc=0' --timeout 60m --quiet || log 'test build not ready'
for test in gdn_packed_gpu grid_capture_gpu; do
  tb=$(ls -t $T/${test}-* 2>/dev/null | grep -v '\.d$' | head -1)
  for c in 0 1; do
    MEMRA_GDN_STATE_BF16=$c MEMRA_TEST_QWEN_GGUF=$Q9 timeout 1800 "$tb" --ignored --test-threads=1 \
      > "$OUT/test-$test-bf16$c.log" 2>&1
    log "test $test bf16=$c rc=$? $(grep -E '^test result' "$OUT/test-$test-bf16$c.log" | tail -1)"
  done
done

log GATES_DONE
