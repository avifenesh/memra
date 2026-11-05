#!/usr/bin/env bash
# The timed readings of the park-compact Max-Q cell (DAY38 1.4 P3 park-copy ms per fed length, P4 idle device
# bytes and resumed-turn E2E), for a later QUIET slot: no compute app on any card of the box (checked), nothing
# training beside it. Runs the four main boots (both orders) in one run, budget 6 h by default (BUDGET_S), with
# the same binaries, card pin, lock and telemetry as run.sh. Its VERDICT line reads TIMED: the rule's verdict
# comes from the gap runs (run-part1.sh, run-part2.sh). Usage: CARD=<idx> bash /mnt/Data/work/memra-maxq/park-compact/run-timed.sh
MODE=timed PART=main exec bash /mnt/Data/work/memra-maxq/park-compact/run.sh
