#!/bin/bash
# LAW:rebuild-after-checkout-attribution, and the two wrong-binary handoffs this week: a
# binary is trusted because it is NEWER than every source it was built from and because its
# OWN STRINGS carry the changes under test, never because of checkout provenance.
# Markers for this cell's binary:
#   MEMRA_TIMEOUT_MS_MAX          this lane's deadline override (must be >0)
#   MEMRA_DSA_INDEX_RING          ring flag present
#   rows still owed to unbuilt pools   post-ring-fix drain message (must be >0)
#   cannot cover pools from row        pre-fix guard message (must be 0)
#   MEMRA_MOE_FUSED_EPI           fused epilogue flag present
#   MEMRA_PP_STAGES               pp door present
# usage: verify-binary.sh <worktree>
set -u
W=${1:?worktree}
BIN=$W/target/release/memra-server

echo "=== git log -1 of the tree ==="
git -C "$W" log -1 --format="%H%n%s%n%ci"
echo "  dirty files: $(git -C "$W" status --porcelain | wc -l)"
echo
echo "=== binary identity ==="
ls -l --time-style=full-iso "$BIN"
echo "  sha256 $(sha256sum "$BIN" | cut -d' ' -f1)"
echo
echo "=== FRESHNESS: binary mtime vs newest source mtime ==="
NEWEST=$(find "$W/crates" -type f \( -name '*.rs' -o -name '*.cu' -o -name '*.toml' \) -printf '%T@ %p\n' | sort -rn | head -1)
NEWEST_T=${NEWEST%% *}; NEWEST_P=${NEWEST#* }
BIN_T=$(stat -c %Y "$BIN")
echo "  newest source: $NEWEST_P at $(date -u -d @${NEWEST_T%.*} +%FT%TZ)"
echo "  binary at $(date -u -d @"$BIN_T" +%FT%TZ)"
if [ "$BIN_T" -gt "${NEWEST_T%.*}" ]; then
  echo "  FRESH: binary is newer than every source"
else
  echo "  *** STALE: binary is NOT newer than its sources. REBUILD BEFORE TRUSTING. ***"
fi
echo
echo "=== strings census ==="
for m in "MEMRA_TIMEOUT_MS_MAX" "MEMRA_DSA_INDEX_RING" "rows still owed to unbuilt pools" \
         "cannot cover pools from row" "MEMRA_MOE_FUSED_EPI" "MEMRA_PP_STAGES"; do
  printf "  %-38s %s\n" "$m" "$(strings "$BIN" 2>/dev/null | grep -c "$m")"
done
echo "  expected: override>0 ring>0 post-fix>0 pre-fix=0 epi>0 pp>0"
