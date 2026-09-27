#!/usr/bin/env bash
# Day 16 A/B cell driver, local RTX 5090: the day-15 policy A/B through the collector with the
# inherited canonical lock (`--rig rtx5090`, `/tmp/memra-5090.lock`), the two-arm binary built from
# the pre-registration ref 9466b8912 in the DETACHED worktree (build-day16.sh). The shape is the
# day-15 shape scaled to a 1024 MiB budget (DAY16.md states the scaling). Nothing here pre-creates
# --out: the collector creates it and refuses one that exists. Bounded lock retries, never kills a
# holder; every attempt leaves its own driver log. The whole cell runs under the owner's CPU quota.
# usage: run-day16-ab.sh <cell-name> <pairs-per-order> <collector-timeout-s>
set -uo pipefail
cell=${1:?cell}; pairs=${2:?pairs}; tmo=${3:?timeout}
WT=${WT:-$HOME/projects/wt-spill-b-ab}
R=${R:-$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day16}
BIN=${BIN:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
cd "$WT" || exit 1
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-policy-ab.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" \
      --budget-mib 1024 --cohort-tokens 1250,1350,1450,1550 \
      --turns 12 --start-tokens 11000 --grow-tokens 150 --return-every 3 --pairs "$pairs" \
      --ctx 16384 --max-tokens 8 --gpu-lock /tmp/memra-5090.lock > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && git rev-parse HEAD > "$out/gate-source.txt"
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
