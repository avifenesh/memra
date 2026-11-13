#!/usr/bin/env bash
# CPU build host, memra#1266: build F4 (the head with the numeric-scope guard) for the RTX 5090, then the
# final CPU checks (full chain + red arms) on the same tree.
set -u
R=<lane>
bash "$R/dd_build.sh" F4
bash "$R/dd_final.sh" "$1"
