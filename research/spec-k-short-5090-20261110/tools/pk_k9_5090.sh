#!/usr/bin/env bash
# memra#1093, the draft length where the spec cost model owns it on the RTX 5090: Qwen3.5-9B
# with its own-trim drafter. Arms over one binary: own (the cost model picks), k2 and k3
# (MEMRA_SPEC_K pinned). 12 boots, own k2 k3 k3 k2 own x2, one boot per lease, each from a card
# at 62 C or cooler, behind any landing ticket.
set -u
H=$HOME/.cache/memra-1093r/pk
O=$H/k9
BIN=$H/k9/memra-server
mkdir -p "$O/perf"
waitfor --file "$O/GO" --timeout 120m --quiet || exit 1
M=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf
n=${1:-0}
for arm in own k2 k3 k3 k2 own own k2 k3 k3 k2 own; do
  n=$((n + 1))
  case $arm in own) pin=() ;; k2) pin=(SPEC_K=2) ;; k3) pin=(SPEC_K=3) ;; esac
  waitfor --until '! gpulock status 2>/dev/null | grep -qi "landing"' --timeout 180m --every 30s --quiet
  waitfor --until 'test "$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" -le 62' --timeout 10m --every 20s --quiet
  echo "perf boot $n ($arm) start $(date -Is), card at $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C" >> "$O/perf/boots.log"
  gpulock run --ttl 12m --wait 300m --for "memra #1093 draft length on the 9B, perf boot $n ($arm, one boot per lease)" -- \
    env MODE=perf CELL_PORT=18196 BOOT_START=$n CELL_SESSIONS=4 CELL_MODEL="$M/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf" \
      CELL_ENV="MEMRA_MTP_DRAFT=$M/draft-9b-owntrim-nvfp4head-q4blk.gguf" "${pin[@]}" \
      python3 "$H/pk_k9.py" "$O/perf" "$BIN" "$arm" >> "$O/perf/boots.log" 2>&1
  echo "perf boot $n ($arm) exit $? $(date -Is)" >> "$O/perf/boots.log"
done
echo K9_DONE > "$O/done"
