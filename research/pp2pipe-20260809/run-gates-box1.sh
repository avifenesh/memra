#!/usr/bin/env bash
# Current-tip PP-2 exactness battery. The standing driver already contains the
# required production/canary rows; this wrapper only binds it to this lane's
# dedicated checkout and raw receipt directory.
set -euo pipefail

REPO=${REPO:-"$HOME/memra-pp2pipe"}
RAW=${RAW:-"$REPO/research/pp2pipe-20260809/raw/box1/gates"}

exec env REPO="$REPO" RAW="$RAW" \
  bash "$REPO/research/microchunk-20260808/run-gates-box1.sh"
