#!/usr/bin/env bash
# memra#1093, the cold-short rung on the RTX 5090 Laptop for Qwen3.8-27B (the table owns K
# there): twelve short prompts (PROMPT_SET=short). Arms over one binary: s3 (the default table, K=3 under
# 1,024 prompt tokens) and s2 (MEMRA_SPEC_K_COLD_SHORT=2). 12 boots, s3 s2 s2 s3 x3, one boot
# per lease, each from a card at 62 C or cooler, behind any landing ticket.
set -u
H=$HOME/.cache/memra-1093r/pk
O=$H/k27s2
BIN=$H/k9/memra-server
mkdir -p "$O/perf"
waitfor --file "$O/GO" --timeout 120m --quiet || exit 1
n=${1:-0}
for arm in s3 s2 s2 s3 s3 s2 s2 s3 s3 s2 s2 s3; do
  n=$((n + 1))
  case $arm in s3) extra= ;; s2) extra=MEMRA_SPEC_K_COLD_SHORT=2 ;; esac
  waitfor --until '! gpulock status 2>/dev/null | grep -qi "landing"' --timeout 180m --every 30s --quiet
  waitfor --until 'test "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62' --timeout 10m --every 20s --quiet
  echo "perf boot $n ($arm) start $(date -Is), card at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$O/perf/boots.log"
  gpulock run --ttl 15m --wait 300m --for "memra #1093 cold-short rung on the 27B, twelve short prompts, boot $n ($arm, one boot per lease)" -- \
    env MODE=perf PROMPT_SET=short CELL_PORT=18196 BOOT_START=$n \
      CELL_ENV="$extra" \
      python3 "$H/pk_k9.py" "$O/perf" "$BIN" "$arm" >> "$O/perf/boots.log" 2>&1
  echo "perf boot $n ($arm) exit $? $(date -Is)" >> "$O/perf/boots.log"
done
echo K27S2_DONE > "$O/done"
