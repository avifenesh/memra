#!/usr/bin/env bash
# Probe for waitfor --until: exit 0 when no lease or ticket whose reason names a landing or a
# release shows on UNIT and none has shown for QUIET seconds (default 180) of observation, else
# exit 1. STATE holds the time one was last seen, or the time observation began (a missing STATE
# starts the clock: the unit must be watched quiet for QUIET seconds before a first take).
# usage: landing_quiet.sh UNIT STATE [QUIET]
s=$(sparklock status --unit "$1" 2>/dev/null) || exit 1
now=$(date +%s)
if echo "$s" | grep -qiE 'memra landing|release'; then echo "$now" > "$2"; exit 1; fi
[ -f "$2" ] || { echo "$now" > "$2"; exit 1; }
[ $(( now - $(cat "$2") )) -ge "${3:-180}" ]
