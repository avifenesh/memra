#!/bin/bash
# The 5090 above-floor blocks: the scaled limit cell, then the mixed shape, A B B A each, one lease per arm.
set -u
H=<lane>/research/partial-reclaim-20261109/rtx5090-above-floor/harness
B=<bins>
R=<runs>
export CELL_A=$B/memra-server-fa0ee88acf CELL_B=$B/memra-server-ffb2b4b909
LIMIT_ARGS="--tok 26000 --fill 32 --fill-conc 4 --pressure 4 --repeat 6" bash $H/pairs.sh limit $R/limit MEMRA_MAX_SESSIONS=4
bash $H/pairs.sh mixed $R/mixed MEMRA_MAX_SESSIONS=4
