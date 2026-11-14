#!/usr/bin/env bash
# Runs on the workstation (memra#1307): the two cell arms' engine files, each with the
# [diag1307] lines (diag1307.py), into DEST/B/crates/memra-engine/src and DEST/T/... for
# build1307.sh.
#   B = BASE_REV's files (main with memra#1302);
#   T = the working tree WT (the head).
# usage: prep_arms.sh WT BASE_REV DEST
set -eu
WT=$1; BASE=$2; DEST=$3
H=$(cd "$(dirname "$0")" && pwd)
F="spec.rs dflash.rs unified_memory.rs dsv4_gpu.rs"
for a in B T; do mkdir -p "$DEST/$a/crates/memra-engine/src"; done
for f in $F; do
  git -C "$WT" show "$BASE:crates/memra-engine/src/$f" > "$DEST/B/crates/memra-engine/src/$f"
  cp "$WT/crates/memra-engine/src/$f" "$DEST/T/crates/memra-engine/src/$f"
done
for a in B T; do
  python3 "$H/diag1307.py" "$DEST/$a"
  (cd "$DEST/$a/crates/memra-engine/src" && sha256sum $F)
done
