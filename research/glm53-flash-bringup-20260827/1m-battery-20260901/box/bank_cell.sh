#!/usr/bin/env bash
# Bank one cell's receipts into the rig-side memra worktree, scrubbing identity.
# Called from the rig, not the box: the box only produces receipts.
# usage (on the box): bank_cell.sh <cell>   -> tars the cell's receipts to stdout
set -uo pipefail
CELL=$1
cd /root/out-1m
tar cf - "receipts/$CELL" \
  $(ls logs/${CELL}.log 2>/dev/null) \
  $(ls logs/boot-${CELL}*.log logs/boot-${CELL}*.gates logs/boot-${CELL}*.engage logs/boot-${CELL}*.identity logs/boot-${CELL}*.vram 2>/dev/null) \
  2>/dev/null
