#!/usr/bin/env bash
# Part 2 of the admit-w-release Max-Q cell when one gap cannot hold the whole cell: the DAY46 2.2 half (6 boots).
# usage: CARD=<idx> bash /mnt/Data/work/memra-maxq/admit-w-release/run-part2.sh
# The verdict combines with the newest out-*/day45 whose four boots all finished (run-part1.sh), in either order.
exec env PART=2 bash /mnt/Data/work/memra-maxq/admit-w-release/run.sh "$@"
