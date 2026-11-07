#!/usr/bin/env bash
# GPU tests in both GDN state classes on the RTX 5090 (the boot class via MEMRA_GDN_STATE_BF16 for
# the model-level tests; the synthetic tests pin both classes in code).
set -u
R=$HOME/.cache/memra-gdnbf16-real
OUT=${1:?outdir}; mkdir -p "$OUT"
Q9=$HOME/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" | tee -a "$OUT/tests.log"; }
for tb in $R/bins/dev1-120a/tests/gdn_packed_gpu-* $R/bins/dev1-120a/tests/grid_capture_gpu-*; do
  t=$(basename "$tb" | sed 's/-[0-9a-f]*$//')
  for c in 0 1; do
    MEMRA_GDN_STATE_BF16=$c MEMRA_TEST_QWEN_GGUF=$Q9 timeout 2400 "$tb" --ignored --test-threads=1 > "$OUT/$t-bf16$c.log" 2>&1
    log "$t bf16=$c rc=$? $(grep -E '^test result' "$OUT/$t-bf16$c.log" | tail -1) $(grep -cE '^test .* ok$' "$OUT/$t-bf16$c.log") ok"
  done
done
ln -sfn "$R/bins/dev1-120a" "$R/wt/target"
for c in 0 1; do
  MEMRA_GDN_STATE_BF16=$c timeout 3600 bash "$R/wt/tools/serve-smoke.sh" > "$OUT/serve-smoke-bf16$c.log" 2>&1
  log "serve-smoke bf16=$c rc=$? $(grep -cE '^  ok: ' "$OUT/serve-smoke-bf16$c.log") ok, $(grep -cE '^  FAIL: ' "$OUT/serve-smoke-bf16$c.log") FAIL $(tail -n 1 "$OUT/serve-smoke-bf16$c.log" | cut -c1-160)"
done
log TESTS_DONE
