#!/usr/bin/env bash
# Part 1 of the admit-w-release Max-Q cell when one gap cannot hold the whole cell: the DAY45 2.2 half (4 boots).
# usage: CARD=<idx> bash /mnt/Data/work/memra-maxq/admit-w-release/run-part1.sh
# The verdict combines with the newest out-*/day46 whose six boots all finished (run-part2.sh), in either order.
exec env PART=1 bash /mnt/Data/work/memra-maxq/admit-w-release/run.sh "$@"
