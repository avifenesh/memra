#!/usr/bin/env bash
# Build host, memra#1266: the final CPU checks on the synced tree (the branch rebased onto the
# landing head): the full chain (dd_full.sh -> full.txt), then the red arms (dd_red.sh ->
# red3/summary.txt) against that head's main files (main.*.rs), one after the other.
# usage: dd_final.sh HEAD_SHA  -> <lane>/final.txt (last line FINAL-DONE)
set -u
R=<lane>
{
  echo "# $(date -Is) final CPU checks of memra#1266 at $1"
  bash "$R/dd_full.sh" "$R/src"; echo "full rc=$?"
  RED_OUT=$R/red3 bash "$R/dd_red.sh"; echo "red rc=$?"
  echo "FINAL-DONE $(date -Is)"
} > "$R/final.txt" 2>&1
