#!/usr/bin/env bash
# Day 20 A/B cell driver, local RTX 5090: the day-15 policy A/B through the collector with the
# inherited canonical lock (`--rig rtx5090`, `/tmp/memra-5090.lock`), the two-arm binary built from
# the pre-registration ref 9466b8912 plus the day-18/19 capture-alignment commits in the DETACHED
# worktree (build-day20.sh). The shape is the day-16 shape moved onto the 32-token GDN prime grid with
# the byte shares preserved (DAY20.md states the scaling): cohort 1248/1344/1440/1536, loop 10912 + 160,
# budget 1024 MiB, ctx 16384. The fourth argument sets MEMRA_REUSE_POOL for the server the harness boots
# (`default` leaves the environment alone; `0` parks no whole session): the continuation pool never
# serves a prompt_ids replay (every row reads `plain-affinity: declined`) and on this card its parked
# sessions are the VRAM the admission reclaim ladder took from the prefix cache on day 16. Nothing here
# pre-creates --out: the collector creates it and refuses one that exists. Bounded lock retries, never
# kills a holder; every attempt leaves its own driver log. The whole cell runs under the owner's CPU quota.
# usage: run-day20-ab.sh <cell-name> <pairs-per-order> <collector-timeout-s> <reuse-pool: default|0>
set -uo pipefail
cell=${1:?cell}; pairs=${2:?pairs}; tmo=${3:?timeout}; pool=${4:?reuse-pool}
WT=${WT:-$HOME/projects/wt-spill-b-ab}
R=${R:-$HOME/projects/wt-spill-b/research/spill-b-20260919/rtx5090-day20}
BIN=${BIN:-$WT/target/release/memra-server}
MODEL=${MODEL:-/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
run() { systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=28G "$@"; }
envp=()
[ "$pool" != default ] && envp=(env "MEMRA_REUSE_POOL=$pool")
cd "$WT" || exit 1
for attempt in 0 1 2 3 4 5; do
  out=$R/$cell
  [ $attempt -gt 0 ] && out=$R/$cell-retry$attempt
  run "${envp[@]}" python3 tools/tier-battery.py --rig rtx5090 --timeout "$tmo" --out "$out" --external-lock \
    --execute python3 tools/prefix-policy-ab.py --external-lock @COLLECTOR_LOCK_FD@ \
      --model "$MODEL" --bin "$BIN" --out "$out/cell" \
      --budget-mib 1024 --cohort-tokens 1248,1344,1440,1536 \
      --turns 12 --start-tokens 10912 --grow-tokens 160 --return-every 3 --pairs "$pairs" \
      --ctx 16384 --max-tokens 8 --gpu-lock /tmp/memra-5090.lock > "$out-driver.log" 2>&1
  rc=$?
  echo $rc > "$out.exit"
  [ -d "$out" ] && { git rev-parse HEAD > "$out/gate-source.txt"; echo "MEMRA_REUSE_POOL=$pool" > "$out/env.txt"; }
  if grep -q "canonical rig lock\|BlockingIOError\|Resource temporarily unavailable" "$out-driver.log" && [ ! -d "$out/cell" ]; then
    echo "attempt $attempt: lock busy at $(date -u +%T), sleeping 90s" >> "$R/$cell-retries.log"
    sleep 90
    continue
  fi
  exit $rc
done
exit 3
