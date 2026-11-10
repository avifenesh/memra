#!/usr/bin/env bash
# Probe for waitfor --until: exit 0 once a lease or ticket whose reason says "landing" has shown in
# `gpulock status` and then none has for QUIET seconds (default 300), else exit 1. State: the last
# time one was seen (LANDING_SEEN); with no state yet the probe waits for a landing to appear.
F=${LANDING_SEEN:-$HOME/.cache/memra-split-rung/runs/landing-seen}
now=$(date +%s)
if gpulock status 2>/dev/null | grep -qi landing; then echo "$now" > "$F"; exit 1; fi
[ -f "$F" ] || exit 1
[ $(( now - $(cat "$F") )) -ge "${QUIET:-300}" ]
