#!/usr/bin/env bash
# CPU build host: memra#1221 part 2 review round 3 on the rebased head (R3 replayed onto main 202e1411fe,
# landing 17; src holds it): after dd_r3_all.sh, the full CPU chain again, then the RTX 5090
# binaries again (cargo rebuilds what the rebase changed; the first build may have started
# before the sync). -> full.txt, build5090-r3.log
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
waitfor --grep 'R3-ALL-DONE' "$R/r3-all.txt" --timeout 120m --quiet || exit 1
bash "$R/dd_full.sh" "$R/src"
bash "$R/dd_build5090_r3.sh"
echo "R3B-ALL-DONE $(date -Is) at $(cat "$R/src/.head")" >> "$R/r3-all.txt"
