#!/usr/bin/env bash
# Exit 0 once no gpulock holder or waiter has named a "landing" for QUIET seconds in a row
# (a landing leaves the queue between its own leases, for its cool-down, and must not find the
# card warmed by a cell that slipped in). State: the time "landing" was last seen.
QUIET=${QUIET:-780}
F=$HOME/.cache/memra-1160/run/.landing-last-seen
now=$(date +%s)
[ -f "$F" ] || echo "$now" > "$F"
if gpulock status 2>/dev/null | grep -qi "landing"; then echo "$now" > "$F"; exit 1; fi
[ $(( now - $(cat "$F") )) -ge "$QUIET" ]
