#!/usr/bin/env bash
# Runs ON a GB10 host under this lane's lease: tree + warm target, base and h2p builds, the gate
# battery, then the served campaign in ORDER. usage: unit_all.sh REPO WARM_TARGET CAMPAIGN "ORDER"
set -u
D=$HOME/.cache/memra-attn-gb10
REPO=$1; WARM=$2; CAMP=$3; ORDER=$4
mkdir -p "$D/logs" "$D/bins" "$D/runs"
bash "$D/box/setup_tree.sh" "$REPO" "$WARM" > "$D/logs/setup.log" 2>&1
git -C "$D/tree" checkout -q --detach c2e04719b6581c557242d3cc4bec26cf2aae542e || exit 1
JOBS=8 bash "$D/box/build.sh" base > "$D/logs/build-base.out" 2>&1
echo BASE-BUILD-DONE >> "$D/logs/build-base.out"
bash "$D/box/build_cand.sh" h2p > "$D/logs/build-h2p.out" 2>&1
H=$(ls -d "$D"/bins/h2p-* | tail -n 1); B=$(ls -d "$D"/bins/base-* | tail -n 1)
echo "$(date -Is) base=$B h2p=$H" >> "$D/logs/unit_all.log"
SKIP16K=1 bash "$D/box/gates_attn.sh" "$H" "$B" "$D/runs/gates-h2p" "$D/tree" > "$D/logs/gates-h2p.out" 2>&1
echo "$(date -Is) gates done" >> "$D/logs/unit_all.log"
bash "$D/box/served_ab.sh" "$CAMP" "$B" "$H" "$ORDER" >> "$D/logs/unit_all.log" 2>&1
echo "$(date -Is) UNIT-ALL-DONE" >> "$D/logs/unit_all.log"
