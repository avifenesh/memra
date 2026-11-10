#!/usr/bin/env bash
# Rig: copy this lane's untimed cells off a unit, without binaries or profiles: the pair cells
# (cells/pairs-*: cmd.sh, rc, log, out/), the hit gate cells' cmd.sh, rc and log, the unit cells'
# directories (runs/unit-cells*) and the nsys cells' small files (runs/nsys/*: gap.txt, big.txt,
# pf.jsonl, guard.log, trace.txt, run.log). usage: fetch_small.sh UNIT
set -u
UNIT=$1
D=$HOME/.cache/memra-specpub/fetched/$UNIT; mkdir -p "$D"
rsync -a --prune-empty-dirs --include='cells/' --include='cells/pairs-*/***' --include='cells/hitgate-*/' \
  --include='cells/hitgate-*/cmd.sh' --include='cells/hitgate-*/rc' --include='cells/hitgate-*/log' \
  --include='runs/' --include='runs/unit-cells*/***' --include='runs/nsys/' --include='runs/nsys/*/' \
  --include='runs/nsys/*/gap.txt' --include='runs/nsys/*/big.txt' --include='runs/nsys/*/pf.jsonl' \
  --include='runs/nsys/*/guard.log' --include='runs/nsys/*/trace.txt' --include='runs/nsys/*/run.log' \
  --exclude='*' "spark-$UNIT:.cache/memra-specpub/" "$D/"
