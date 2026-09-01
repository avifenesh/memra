#!/usr/bin/env bash
# RIG-side per-cell banking (run on the rig, not the box): pull a cell's receipts from box B
# into the tpd-battery worktree, scrub identity fields (system_fingerprint + request ids;
# boot nonces KEPT — arm-identity receipt), commit, push. One bank per cell close.
# Usage: TPD_BOX=<ssh-dest> TPD_PORT=<port> TPD_WT=<worktree> bank_cell.sh <path-under-out-tpd> "<msg>"
# The box ssh destination is FLEET STATE (darklanes-private): it comes from the environment,
# never from this file (public-boundary law).
set -euo pipefail
CELL="${1:?path under /root/out-tpd (c0|c1-*|analysis|logs|served-cal|.)}"
MSG="${2:?commit message}"
BOX="${TPD_BOX:?set TPD_BOX to the box ssh destination (user@host)}"
PORT="${TPD_PORT:?set TPD_PORT to the box ssh port}"
WT="${TPD_WT:?set TPD_WT to the rig worktree path}"
DST=$WT/research/glm53-flash-bringup-20260827/tpd-battery-20260831/receipts
SSH="/usr/bin/ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -i $HOME/.ssh/id_ed25519 -p $PORT"
mkdir -p "$DST"
rsync -a -e "$SSH" "$BOX:/root/out-tpd/$CELL" "$DST/" \
  --exclude '*.nsys-rep' --exclude 'server.pid' --exclude '*.f32' --exclude 'prompts-*'
python3 "$WT/research/glm53-flash-bringup-20260827/tpd-battery-20260831/box/scrub_bank.py" \
  "$DST/$(basename "$CELL")"
cd "$WT"
git add research/glm53-flash-bringup-20260827/tpd-battery-20260831/
git -c user.name="Avi Fenesh" -c user.email="avifenesh@users.noreply.github.com" commit -q -m "$MSG"
git push -q origin HEAD:lane/glm5-tpd-battery
git log -1 --oneline
