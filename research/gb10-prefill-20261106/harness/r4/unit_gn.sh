#!/usr/bin/env bash
# Runs ON a GB10 host under this session's lease: lever 4 (the GDN out-norm inside the fused K4+K5
# pass). Fetches gn.bundle (HEAD = the lever, HEAD~1 = its base), builds the probe against the
# lever tree's cu/hybrid.cu and runs it, then builds B (base) and G (lever) and runs kernel-check's
# GDN-K45-NORM / GDN-DB-GQA / GATED-RMSNORM-E4M3V2 cells on G.
# usage: unit_gn.sh [probe|build|kc|all]
set -u
STEP=${1:-all}
D=$HOME/.cache/memra-gb10-r4
T=$D/tree
L=$D/logs/unit-gn.log
mkdir -p "$D/logs" "$D/probe-gn"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
echo "$(date -Is) unit_gn $STEP start" >> "$L"
git -C "$T" fetch -q "$D/gn.bundle" "HEAD:refs/heads/r4-lane-gn" || { echo "GN-FETCH-FAIL" >> "$L"; exit 1; }
G=$(git -C "$T" rev-parse refs/heads/r4-lane-gn)
B=$(git -C "$T" rev-parse "refs/heads/r4-lane-gn~1")
echo "B=$B G=$G" >> "$L"
if [ "$STEP" = probe ] || [ "$STEP" = all ]; then
  git -C "$T" checkout -q --detach "$G" || exit 1
  P=$D/probe-gn
  cp "$T/research/gb10-prefill-20261106/gdn-norm/probe/probe_gdn_norm.cu" "$P/"
  {
    date -Is; git -C "$T" log --oneline -1; sha256sum "$T/crates/memra-engine/cu/hybrid.cu" "$P/probe_gdn_norm.cu"
    nvcc --version | tail -n 1
    (cd "$P" && nice -n 10 nvcc -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a -I"$T/crates/memra-engine/cu" \
      --resource-usage probe_gdn_norm.cu -o probe_gdn_norm 2>&1 | grep -A2 -E "Function properties for (gdn_chunk_state_output|gated_rmsnorm_e4m3v2)" \
      | grep -E "properties|Used|spill")
    echo "nvcc rc=$?"
    nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,temperature.gpu,power.draw --format=csv,noheader
    for t in 4096 4032 43 16; do timeout 600 "$P/probe_gdn_norm" "$t" 5 2; echo "probe T=$t rc=$?"; done
    nvidia-smi --query-gpu=name,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
    date -Is
  } > "$D/logs/probe-gn.log" 2>&1
  echo "GN-PROBE-DONE" >> "$L"
fi
if [ "$STEP" = build ] || [ "$STEP" = all ]; then
  git -C "$T" checkout -q --detach "$B" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" gnb > "$D/logs/build-gnb.out" 2>&1
  git -C "$T" checkout -q --detach "$G" && JOBS=16 NOTOUCH=1 bash "$D/box/build.sh" gn > "$D/logs/build-gn.out" 2>&1
  tail -n 1 "$D/logs/build-gnb.out" "$D/logs/build-gn.out" >> "$L"
  echo "GN-BUILD-DONE" >> "$L"
fi
if [ "$STEP" = kc ] || [ "$STEP" = all ]; then
  GB=$(ls -d "$D"/bins/gn-"${G:0:10}" 2>/dev/null)
  if [ -z "$GB" ]; then echo "GN-KC-NOBIN" >> "$L"; exit 1; fi
  G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
  (cd "$T" && MEMRA_KC_FAST=1 timeout 1800 "$GB/kernel-check" "$G8" > "$D/logs/kc-gn.log" 2>&1; echo "kernel-check rc=$?" >> "$D/logs/kc-gn.log")
  grep -E "GDN-K45-NORM|GDN-DB-GQA|GATED-RMSNORM-E4M3V2|ALL GREEN|FAIL|rc=" "$D/logs/kc-gn.log" | head -n 40 >> "$L"
  echo "GN-KC-DONE" >> "$L"
fi
echo "$(date -Is) unit_gn $STEP end" >> "$L"
