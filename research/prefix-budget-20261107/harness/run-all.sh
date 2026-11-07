#!/usr/bin/env bash
# All 5090 measurement phases, each under its own gpulock lease (others may take the card between
# phases). NEW_LABEL names the candidate binary in bins/.
set -u
H=<harness>
L=<lane>
export NEW_LABEL=${NEW_LABEL:?}
for ph in ${PHASES:-phase1a phase1b phase1c}; do
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) start $ph"
  gpulock run --ttl 2h --wait 3h --for "memra prefix-budget lane: $ph" -- bash $H/$ph.sh > $L/$ph.log 2>&1
  echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) end $ph rc=$?"
done
echo RUN_ALL_DONE
