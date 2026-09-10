#!/bin/bash
# Replay of the repaired run-cell.sh lane-class engagement guard (memra #475) against every
# banked 256k row. No GPU. Both halves: ON must announce phase=prime, OFF must not announce.
set -uo pipefail
old() { grep -E "indexer=pool-split .*t=1 " "$1" >/dev/null && echo 0 || echo 93; }
new() { # log split
  local log=$1 split=$2 rc=0
  if [[ $split == 1 ]]; then
    grep -E "indexer=pool-split .*phase=prime" "$log" >/dev/null || rc=93
  else
    grep -F "[glm5-tp-indexer-split]" "$log" >/dev/null && rc=95
  fi
  echo "$rc"
}
R=/root/src/prime-lever/receipts
printf "%-24s %-6s %-9s %-9s %-9s\n" row split old_rc new_rc banked_rc
for r in idxsplit-check-256k:1 idxsplit-check2-256k:1 idxsplit-on-a-256k:1 idxsplit-on-b-256k:1 idxsplit-off-a-256k:0 idxsplit-off-b-256k:0; do
  n=${r%%:*}; s=${r##*:}; l=$R/$n/run.log
  printf "%-24s %-6s %-9s %-9s %-9s\n" "$n" "$s" "$(old "$l")" "$(new "$l" "$s")" "$(cat "$R/$n/exit")"
done
echo
echo "RED ARM, the guard must FAIL when the door did not do its job:"
printf "  OFF log fed to the split=1 arm (engine refused the split): rc=%s (expect 93)\n" "$(new $R/idxsplit-off-a-256k/run.log 1)"
printf "  OFF log fed to the split=1 arm (engine refused the split): rc=%s (expect 93)\n" "$(new $R/idxsplit-off-b-256k/run.log 1)"
printf "  ON  log fed to the split=0 arm (door leaked while OFF):    rc=%s (expect 95)\n" "$(new $R/idxsplit-on-a-256k/run.log 0)"
printf "  ON  log fed to the split=0 arm (door leaked while OFF):    rc=%s (expect 95)\n" "$(new $R/idxsplit-on-b-256k/run.log 0)"
