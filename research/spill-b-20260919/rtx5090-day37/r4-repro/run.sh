#!/usr/bin/env bash
# DAY37 addendum G: the repro of 2.7's A1 red on the 5090's pooled arm. Each gate waits for the card's lock (bounded) and
# for no compute app; never a signal.
set -uo pipefail
WT=$HOME/projects/wt-spill-b
D=$WT/research/spill-b-20260919/rtx5090-day37/r4-repro
MODEL=/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
cd "$WT" || exit 1
for spec in r4-1:target/day37/r4/memra-server v3-1:target/b2/v3/memra-server r4-2:target/day37/r4/memra-server; do
  name=${spec%%:*}; bin=${spec#*:}
  deadline=$((SECONDS + 7200))
  until flock -n /tmp/memra-5090.lock true && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do
    [ $SECONDS -ge $deadline ] && { echo "$(date -u +%FT%TZ) $name: rig not idle after 7200 s; not run" >> "$D/run.log"; exit 3; }
    sleep 20
  done
  echo "$(date -u +%FT%TZ) $name start bin=$(sha256sum "$bin" | cut -c1-16)" >> "$D/run.log"
  env -u MEMRA_KV_ALLOCATOR flock -w 600 /tmp/memra-5090.lock tools/admit-mem-burst-gate.sh "$MODEL" "$bin" "$D/$name" \
    > "$D/$name.gate.log" 2>&1
  echo "$(date -u +%FT%TZ) $name rc=$? $(tail -1 "$D/$name.gate.log")" >> "$D/run.log"
done
