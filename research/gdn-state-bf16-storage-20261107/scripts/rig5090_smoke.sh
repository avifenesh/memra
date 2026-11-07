#!/usr/bin/env bash
# serve-smoke (prebuilt lane binary, no in-script build) in both GDN state classes, plus the
# grid-capture model tests with --nocapture so the boot class line is in the log.
set -u
R=$HOME/.cache/memra-gdnbf16-real
OUT=${1:?outdir}; mkdir -p "$OUT"
Q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/smoke.log"; }
tb=$(ls $R/bins/dev1-120a/tests/grid_capture_gpu-*)
for c in 0 1; do
  MEMRA_GDN_STATE_BF16=$c MEMRA_TEST_QWEN_GGUF=$Q9 timeout 1200 "$tb" --ignored --test-threads=1 --nocapture > "$OUT/grid_capture_gpu-nocapture-bf16$c.log" 2>&1
  log "grid_capture_gpu bf16=$c rc=$? $(grep -E '^test result' "$OUT/grid_capture_gpu-nocapture-bf16$c.log") class-lines=$(grep -c '\[gdn-state\] bf16' "$OUT/grid_capture_gpu-nocapture-bf16$c.log")"
done
for c in 0 1; do
  MEMRA_GDN_STATE_BF16=$c timeout 3600 bash "$R/wt/tools/serve-smoke-nobuild.sh" > "$OUT/serve-smoke-bf16$c.log" 2>&1
  log "serve-smoke bf16=$c rc=$? $(grep -cE '^  ok: ' "$OUT/serve-smoke-bf16$c.log") ok, $(grep -cE '^  FAIL: ' "$OUT/serve-smoke-bf16$c.log") FAIL | $(tail -n 1 "$OUT/serve-smoke-bf16$c.log" | cut -c1-160)"
done
log SMOKE_DONE
