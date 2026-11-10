#!/usr/bin/env bash
# Probe for waitfor --until: exit 0 when no lease or ticket whose reason names a landing or a
# release ("landing", "release", "0.141.0") shows on UNIT and none has shown for QUIET seconds
# (default 300), else exit 1. A release runs several tickets in a row on a unit, and the unit is
# the release's between them: a cell of this lane starts only once the unit has been quiet. STATE
# holds the time one was last seen.
# usage: landing_quiet.sh UNIT STATE [QUIET]
s=$(sparklock status --unit "$1" 2>/dev/null) || exit 1
now=$(date +%s)
if echo "$s" | grep -qiE 'landing|release|0\.141\.0'; then echo "$now" > "$2"; exit 1; fi
[ -f "$2" ] || exit 0
[ $(( now - $(cat "$2") )) -ge "${3:-300}" ]
