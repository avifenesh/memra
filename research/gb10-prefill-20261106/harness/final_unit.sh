#!/usr/bin/env bash
# Runs ON a GB10 host under a lease: build the heads named in heads.env if not banked, then the
# final-head cells. usage: final_unit.sh gates|served|both TAG
#   gates : gates_v2.sh on the v2 head (B = the base head for the 16k table compare)
#   served: P V V P at the defaults (4,096-row chunks), P = base head, V = v2 head
set -u
D=$HOME/.cache/memra-gemm-v2
MODE=$1; TAG=$2
. "$D/heads.env"
bash "$D/build_heads.sh" "$BASE_SHA" "$V2_SHA" || { echo "build failed"; exit 1; }
B=$D/bins/base-${BASE_SHA:0:10}; V=$D/bins/v2pr-${V2_SHA:0:10}
[ -d "$B" ] || B=$D/bins/lanehead-${BASE_SHA:0:10}
echo "V=$V B=$B"; cat "$V/COMMIT" "$B/COMMIT"
export TMPDIR=$D/tmp; mkdir -p "$TMPDIR"
if [ "$MODE" = gates ] || [ "$MODE" = both ]; then
  git -C "$D/tree" checkout -q --detach "$V2_SHA"
  bash "$D/gates_v2.sh" "$V" "$B" "$D/runs/F-gates-$TAG" "$D/tree"
fi
if [ "$MODE" = served ] || [ "$MODE" = both ]; then
  CD=$D/runs/F-served-$TAG
  bash "$D/campaign.sh" "$CD" "32768 131072 260000" "1 1 1" "P|$B|" "V|$V|" "V|$V|" "P|$B|"
  python3 "$D/summarize.py" "$CD" > "$CD/summary.txt" 2>&1
fi
echo FINAL-DONE
