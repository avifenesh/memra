#!/usr/bin/env bash
# RTX 5090 (laptop, 24 GB): MEMRA_DSPARK_COLD_DRAFTER A/B on the agentic c1 cell (memra#1031 half 2).
#   cold-drafter-ab-5090.sh <bin> <arm> ...   arm = on|off, run in the order given; same binary and config
# Qwen3.8-27B NVFP4+Q5_K GGUF + DFlash2 q4 drafter, MEMRA_MTP_SKIP=1, the restore/capture doors,
# context and session count cut to fit 24 GB. Log: ~/.cache/memra-1025/dflash-ab/<n>-<bin>/
set -u
M=/data/ai-ml/models/q38-gguf
H=~/.cache/model-scout/harness
BIN=$1; shift
O=~/.cache/memra-1031-5090/ab; mkdir -p "$O"
cat /sys/firmware/acpi/platform_profile > "$O/platform_profile.txt" 2>/dev/null
n=0
for ARM in "$@"; do
  n=$((n+1)); tag=$n-$ARM; D=$O/$tag; mkdir -p "$D"
  COLD=0; [ "$ARM" = on ] && COLD=1
  { echo "arm $tag $(date -Is) profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; sha256sum "$BIN";
    nvidia-smi --query-gpu=name,temperature.gpu,clocks.sm,power.limit --format=csv,noheader; } > "$D/arm.txt"
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 MEMRA_PREFIX_CACHE_MB=4000 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
    MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt MEMRA_DSPARK_PREFIX_RESTORE=1 \
    MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_DSPARK_COLD_DRAFTER=$COLD \
    "$BIN" > "$D/serve.log" 2>&1 &
  SP=$!
  if ! waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet; then
    echo "boot FAILED" >> "$D/arm.txt"; kill $SP; wait $SP; continue
  fi
  (cd $H && CLIENT_MAX_TIME=1800 BENCH_BASE=http://127.0.0.1:8093 BENCH_MODEL=q38 BENCH_TOKENIZE_PATH=/v1/tokenize \
     BENCH_EMULATE_CONTINUOUS_USAGE=1 timeout 3600 python3 ab_agent.py run ~/.cache/memra-1025/ab-workload-5090.json \
     "$D" "dflash-ab-$tag" --users 1 --think 1 --temperature 1.0 --seed 1234 > "$D/cells.log" 2>&1)
  tail -n 1 "$D/cells.log" | cut -c1-300 >> "$D/arm.txt"
  grep -a -c 'cache-preferred' "$D/serve.log" | sed 's/^/cache-preferred=/' >> "$D/arm.txt"
  grep -a -c 'restorable-preferred' "$D/serve.log" | sed 's/^/restorable-preferred=/' >> "$D/arm.txt"
  grep -a -c 'DSPARK restore' "$D/serve.log" | sed 's/^/dspark-restore=/' >> "$D/arm.txt"
  grep -a -c 'cold-drafter restore' "$D/serve.log" | sed 's/^/cold-drafter-restore=/' >> "$D/arm.txt"
  grep -a 'dspark-acc' "$D/serve.log" | tail -1 >> "$D/arm.txt"
  kill -TERM $SP; waitfor --gone $SP --timeout 90s --quiet || kill -KILL $SP
  wait $SP 2>/dev/null
done
cat "$O"/*/arm.txt
