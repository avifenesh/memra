#!/usr/bin/env bash
# RIG-side per-cell banking (run on the rig, not the box): pull a cell's receipts from
# box B into the battery worktree, scrub identity fields (system_fingerprint + request
# ids; boot nonces KEPT — arm-identity receipt), commit, push. One bank per cell close.
# Usage: SB_BOX=<ssh-dest> SB_PORT=<port> bank_cell.sh <cell-dir-name> "<commit message>"
# The box ssh destination is FLEET STATE (darklanes-private): it comes from the
# environment, never from this file (public-boundary law).
set -euo pipefail
CELL="${1:?cell dir (c1|c2|c3|c4|c5|logs|.)}"
MSG="${2:?commit message}"
BOX="${SB_BOX:?set SB_BOX to the box ssh destination (user@host)}"
PORT="${SB_PORT:?set SB_PORT to the box ssh port}"
WT=/tmp/wt-struct
DST=$WT/research/glm53-flash-bringup-20260827/struct-battery-20260831/receipts
SSH="/usr/bin/ssh -o IdentitiesOnly=yes -i $HOME/.ssh/id_ed25519 -p $PORT"
mkdir -p "$DST"
rsync -a -e "$SSH" "$BOX:/root/out-struct/$CELL" "$DST/" \
  --exclude '*.nsys-rep' --exclude 'server.pid'
python3 "$WT/research/glm53-flash-bringup-20260827/struct-battery-20260831/box/scrub_bank.py" "$DST/$CELL"
cd "$WT"
git add research/glm53-flash-bringup-20260827/struct-battery-20260831/
git -c user.name="Avi Fenesh" -c user.email="avifenesh@users.noreply.github.com" commit -q -m "$MSG"
git push -q origin lane/glm5-struct-battery
git log -1 --oneline
