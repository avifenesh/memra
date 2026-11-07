#!/usr/bin/env bash
# The rest of rig5090_gates.sh after its 10:16 stop at a step boundary (released for #1079):
# the GPU tests in both classes on Qwen3.5-9B and run-spec K=1..8 in both classes.
#   usage: rig5090_gates2.sh BINTAG OUTDIR
set -u
R=$HOME/.cache/memra-gdnbf16-decode
NEW=$R/bins/${1:?bintag}
OUT=${2:?outdir}
mkdir -p "$OUT"
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/gates.log"; }
Q27=$HOME/models/q38-gguf/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
ORN=/data/ai-ml/hf-models/ornith15-gguf/Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf
Q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
log "resume (part 2) after the step-boundary stop"
for t in gdn_packed_gpu grid_capture_gpu; do
  for c in 0 1; do
    MEMRA_GDN_STATE_BF16=$c MEMRA_TEST_QWEN_GGUF=$Q9 timeout 2400 "$NEW/tests/$t" --ignored --test-threads=1 --nocapture > "$OUT/test-$t-bf16$c.log" 2>&1
    log "test $t bf16=$c rc=$? $(grep -E '^test result' "$OUT/test-$t-bf16$c.log" | tail -1)"
  done
done
for m in q27:$Q27 orn:$ORN; do
  tag=${m%%:*}; g=${m#*:}
  for c in 0 1; do
    ( unset MEMRA_PROMPT_DIR MEMRA_SPEC_K MEMRA_GEN_ONLY
      MEMRA_GDN_STATE_BF16=$c MEMRA_SPEC_TEMP=0 MEMRA_NGEN=32 timeout 2400 "$NEW/run-spec" "$g" ) > "$OUT/run-spec-$tag-bf16$c.log" 2>&1
    rc=$?
    log "run-spec $tag bf16=$c rc=$rc $(python3 "$R/wt/tools/release-coverage.py" spec < "$OUT/run-spec-$tag-bf16$c.log" 2>&1 | tail -n 1 | cut -c1-200)"
  done
done
log GATES2_DONE
