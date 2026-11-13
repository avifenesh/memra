#!/usr/bin/env bash
# CPU build host, memra#1266 review round 2: build F5 (the round-2 head) and the bench from it for the
# RTX 5090, then the final CPU checks (full chain + red arms, the round-2 arms included).
set -u
R=<lane>
bash "$R/dd_build.sh" F5
bash "$R/dd_bench.sh"
bash "$R/dd_final.sh" "$1"
