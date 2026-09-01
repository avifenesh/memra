#!/usr/bin/env bash
# Round-2 ARM RUNNER, rig side. One arm per invocation, and the receipts are pulled the
# moment the arm exits — the lane's own hard lesson from the 2026-08-31 preemption: a
# receipt is banked when it is on the RIG, not when the box wrote it. The keeper mirrors
# ~/realgate every 2 min as a safety net; this is the per-arm copy that does not wait.
#
# Usage: Q4E_BOX=<user>@<host> Q4E_KEY=<path/to/key> run-arm.sh <tag> <remote command ...>
#   stdout+stderr land in round2-box-receipts/logs/<tag>.log on the rig,
#   and the whole ~/realgate tree is rsync'd down afterwards.
#
# Host and key path come from the ENVIRONMENT, never from this file: they are fleet state
# (which machine, which provider's key) and fleet state lives in darklanes, not in the
# public engine repo. The public-boundary gate enforces this — it refused a push over a key
# FILENAME that carried a provider name, and because that gate scans every blob version in
# the pushed range, the fix was to rewrite the unpushed commits rather than to grandfather
# the string in the allowlist. An allowlist entry would have been the wrong tool twice: the
# leak was accidental rather than deliberate, and allowlists outlive their reasons.
set -uo pipefail
BOX="${Q4E_BOX:?set Q4E_BOX=<user>@<host>}"
KEY="${Q4E_KEY:?set Q4E_KEY=<path to the box ssh key>}"
SSH=(/usr/bin/ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
     -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -i "$KEY")
DST="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$DST/logs"
tag="$1"; shift
log="$DST/logs/$tag.log"
{
  echo "=== arm=$tag start=$(date -u +%FT%TZ) ==="
  echo "=== cmd: $* ==="
} > "$log"
"${SSH[@]}" "$BOX" "$@" >> "$log" 2>&1
rc=$?
echo "=== arm=$tag rc=$rc end=$(date -u +%FT%TZ) ===" >> "$log"
# BANK NOW, whatever the rc: a failed arm's receipt is evidence too.
# Destination is this dir, matching the keeper's own rsync target exactly — two mirrors
# with different destinations produce two divergent copies of the same tree (it happened:
# the keeper wrote ./kvq2 while this wrote ./realgate/kvq2). One tree, two writers.
# ladder-ids.txt is EXCLUDED: it is 4.9 MB of derived corpus, exactly reproducible from
# yarn/make-ladder-ids.py at the corpus commit quoted in every ladder receipt.
rsync -az --timeout=120 -e "${SSH[*]}" \
  --exclude '*.bin' --exclude '*.pt' --exclude 'ladder-ids.txt' \
  "$BOX:~/realgate/" "$DST/" >/dev/null 2>&1 \
  && echo "banked: $DST (arm=$tag rc=$rc)" \
  || echo "BANK FAILED for arm=$tag (rc=$rc) — box may be gone"
exit $rc
