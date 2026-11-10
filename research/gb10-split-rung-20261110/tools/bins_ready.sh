#!/usr/bin/env bash
# Rig: when the build unit has both arms of the rebased B/T (bins/head-HEAD with the gate binaries,
# bins/base-BASE's memra-server), hash them there, copy them to the other unit and check the hashes
# on both, link the head's gate binaries for argmax-margin-gate.sh, then touch the rig's flag file.
# usage: BUILD_UNIT=<sparklock name> OTHER_UNIT=<sparklock name> bins_ready.sh HEAD BASE
set -u
BU=${BUILD_UNIT:?}; OU=${OTHER_UNIT:?}; H=$1; B=$2
R=$HOME/.cache/memra-split-rung
L=$R/runs/bins-ready-$H.log
log() { echo "$(date -Is) $*" >> "$L"; }
waitfor --ssh "spark-$BU" "grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-base-$B.log && grep -q BUILD-DONE ~/.cache/memra-split-rung/logs/build-head-$H.log" --timeout 150m --every 30s --quiet || { log "builds missing"; exit 1; }
/usr/bin/ssh -n "spark-$BU" "cd ~/.cache/memra-split-rung/bins && sha256sum head-$H/* base-$B/* > rebased-$H.sha256 && cd .. && mkdir -p kt-head-$H/release && for b in bins/head-$H/*; do ln -sf \$HOME/.cache/memra-split-rung/\$b kt-head-$H/release/; done" || { log "hash failed"; exit 1; }
/usr/bin/ssh -n "spark-$OU" "mkdir -p ~/.cache/memra-split-rung/bins/head-$H ~/.cache/memra-split-rung/bins/base-$B"
scp -3 -q "spark-$BU:.cache/memra-split-rung/bins/head-$H/*" "spark-$OU:.cache/memra-split-rung/bins/head-$H/" &&
scp -3 -q "spark-$BU:.cache/memra-split-rung/bins/base-$B/*" "spark-$OU:.cache/memra-split-rung/bins/base-$B/" &&
scp -3 -q "spark-$BU:.cache/memra-split-rung/bins/rebased-$H.sha256" "spark-$OU:.cache/memra-split-rung/bins/" || { log "copy failed"; exit 1; }
/usr/bin/ssh -n "spark-$OU" "cd ~/.cache/memra-split-rung/bins && sha256sum -c rebased-$H.sha256" >> "$L" 2>&1 || { log "sha256 mismatch on the other unit"; exit 1; }
/usr/bin/ssh -n "spark-$BU" "cd ~/.cache/memra-split-rung/bins && sha256sum -c rebased-$H.sha256" >> "$L" 2>&1 || { log "sha256 mismatch on the build unit"; exit 1; }
/usr/bin/ssh -n "spark-$BU" "cat ~/.cache/memra-split-rung/bins/rebased-$H.sha256" >> "$L"
echo ready > "$R/runs/bins-ready-$H"  # waitfor --file needs a non-empty file
log "ready"
