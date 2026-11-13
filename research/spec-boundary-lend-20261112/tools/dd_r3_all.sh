#!/usr/bin/env bash
# CPU build host: memra#1221 part 2 review round 3, in order on the R3 tree (src): the full CPU chain,
# then the round-3 red arms, then the RTX 5090 binaries (one cargo job at a time on this host).
R=$HOME/.cache/memra-1221b
bash "$R/dd_full.sh" "$R/src"
bash "$R/dd_red_r3.sh"
bash "$R/dd_build5090_r3.sh"
echo "R3-ALL-DONE $(date -Is)" >> "$R/r3-all.txt"
