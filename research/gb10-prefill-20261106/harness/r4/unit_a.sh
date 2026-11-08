#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: round-4 tree at origin/main, the attention ping-pong probe
# (timing + ragged byte compare), then main's build and the cold 131k nsys profile.
set -u
D=$HOME/.cache/memra-gb10-r4
mkdir -p "$D/logs"
bash "$D/box/setup_tree.sh" "$HOME/projects/memra-cells-repo" "$HOME/.cache/memra-base-target" > "$D/logs/setup.log" 2>&1
git -C "$D/tree" fetch -q origin main && git -C "$D/tree" checkout -q --detach origin/main
git -C "$D/tree" log --oneline -1 >> "$D/logs/setup.log"
bash "$D/probe/run_probe.sh" pp-r1 "$D/tree" 4096 "4096,16384,65536,131072" 3 "h2p,pp0,ppw,ppg,ppl,ppv,h2p,ppw,ppl" > /dev/null 2>&1
bash "$D/probe/edge.sh" edge-pp "h2p,ppw,ppg,ppl,ppv" > /dev/null 2>&1
echo PROBE-DONE >> "$D/logs/setup.log"
bash "$D/box/profile_main.sh"
echo UNIT-A-DONE >> "$D/logs/setup.log"
