#!/usr/bin/env bash
# Part 2 of the park-compact Max-Q cell (a later window): main-O2-on main-O2-off, then the combined reading
# with the newest complete part 1 (or PART1_OUT=<dir>) in out-combined-<stamp>/VERDICT.txt.
# Usage: CARD=<idx> bash /mnt/Data/work/memra-maxq/park-compact/run-part2.sh
PART=2 exec bash /mnt/Data/work/memra-maxq/park-compact/run.sh
