#!/usr/bin/env bash
# RTX 5090 (24 GB): exactness gate for MEMRA_DSPARK_COLD_DRAFTER=1 (memra#1031 second half).
#   cold-drafter-gate-5090.sh <memra-server bin> <out dir>
# Arm: prefix cache + restore/partial/cold-drafter doors. Reference: prefix cache off, so every
# request cold-primes DFlash. Same model, drafter, seed and requests (tools/dspark-cold-drafter-gate.py).
set -u
BIN=$1; O=$2; mkdir -p "$O"
M=/data/ai-ml/models/q38-gguf
G=$(dirname "$0")/../../tools/dspark-cold-drafter-gate.py
boot() { # tag extra-env...
  local tag=$1; shift
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=16384 MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$M/Qwen3.8-27B-DFlash2-memra \
    MEMRA_FRSPEC_TRIM=$M/q38-ranks-sxc32768.gguf.txt "$@" \
    "$BIN" > "$O/serve-$tag.log" 2>&1 &
  SP=$!
  waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet
}
stop() { kill -TERM $SP; waitfor --gone $SP --timeout 90s --quiet || kill -KILL $SP; wait $SP 2>/dev/null; }
boot arm MEMRA_PREFIX_CACHE_MB=4000 MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 \
  MEMRA_DSPARK_PARTIAL_RESTORE=1 MEMRA_PREFIX_STABLE_BOUNDARY=1 MEMRA_DSPARK_COLD_DRAFTER=1 \
  && python3 "$G" http://127.0.0.1:8093 q38 "$O/arm.jsonl"
stop
boot ref MEMRA_PREFIX_CACHE_MB=0 && python3 "$G" http://127.0.0.1:8093 q38 "$O/ref.jsonl"
stop
python3 "$G" --compare "$O/arm.jsonl" "$O/ref.jsonl" | tee "$O/compare.txt"
for k in 'cache-preferred' 'cold-drafter restore' 'DSPARK restore' 'dspark-boundary'; do
  echo "$k=$(grep -a -c "$k" "$O/serve-arm.log")"; done | tee -a "$O/compare.txt"
