#!/usr/bin/env bash
# Rig: copy a served B/T campaign's small files off a unit: per cell ab.jsonl, guard.log, run.log
# and trace.txt (the server log's `[tick-spec]`, `[prefix-cache]`, spec K, draft load and cost
# calibration lines, cut on the unit), then campaign.log. Full server logs, GPU telemetry and
# memory samples stay on the unit. usage: fetch_cells.sh UNIT CAMPAIGN  -> ~/.cache/memra-specpub/runs/CAMPAIGN
set -u
UNIT=$1; CAMP=$2
R=.cache/memra-specpub/runs/$CAMP
/usr/bin/ssh -n "spark-$UNIT" "cd ~/$R && for c in [0-9][0-9]-[A-Z]; do [ -f \$c/server.log ] && grep -E '\[(tick-spec|prefix-cache|prefix-cache-split-state|spec-k|spec-cost|mtp-draft|device-trim|spec-pool|spec)\]' \$c/server.log > \$c/trace.txt; done"
mkdir -p "$HOME/$R"
rsync -a --prune-empty-dirs --include='*/' --include='ab.jsonl' --include='guard.log' --include='run.log' --include='trace.txt' --include='campaign.log' --exclude='*' "spark-$UNIT:$R/" "$HOME/$R/"
