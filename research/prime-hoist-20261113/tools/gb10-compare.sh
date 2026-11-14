#!/usr/bin/env bash
# Runs ON GB10 unit-b (memra#1258): compare two arms' gb10-exact.sh digests. usage: gb10-compare.sh A B
R=$HOME/.cache/memra-1258/run
if cmp -s "$R/exact-$1/$1-dspark-digests.txt" "$R/exact-$2/$2-dspark-digests.txt" \
  && cmp -s "$R/exact-$1/$1-streams.txt" "$R/exact-$2/$2-streams.txt"; then echo "ARMS-EQUAL $1 $2"; else echo "ARMS-DIFFER $1 $2"; fi
for a in "$1" "$2"; do grep -E "run-gen|run-spec|dspark rc|GB10-EXACT" "$R/exact-$a/guard.log"; done
