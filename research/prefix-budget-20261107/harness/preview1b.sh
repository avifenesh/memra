#!/usr/bin/env bash
# Preview (single runs, not scored): agentic c8 on main and on the first candidate 7b385110e4
# (GDS + partial reclaim, no clamp; MEMRA_ENV_AUDIT=warn for its older door registry).
set -u
H=/home/avifenesh/.cache/memra-prefix-budget/harness
B=/home/avifenesh/.cache/memra-prefix-budget/bins
O=/home/avifenesh/.cache/memra-prefix-budget/runs/preview1b
mkdir -p "$O"
bash $H/arm.sh c8-base $B/memra-server-base-6f5e03e74a $O agentic:8
bash $H/arm.sh c8-noclamp $B/memra-server-new-7b385110e4 $O agentic:8 MEMRA_ENV_AUDIT=warn
echo PREVIEW1B_DONE
