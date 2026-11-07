#!/usr/bin/env bash
# Start a GB10 plan as soon as the queued sparklock acquire for UNIT returns a lease.
#   on_lease.sh UNIT LEASEFILE OUTROOT MODE...
set -u
UNIT=$1; F=$2; ROOT=$3; shift 3
waitfor --file "$F" --grep '^rc=' --timeout 320m --every 30s --quiet || exit 2
grep -q '^rc=0' "$F" || { echo "acquire failed: $(cat "$F")"; exit 3; }
LEASE=$(grep -E "^claude-code-[0-9T]+-[0-9]+$" "$F" | tail -1)
[ -n "$LEASE" ] || { echo "no lease id in $F"; exit 4; }
echo "lease $LEASE"
bash "$HOME/.cache/memra-gdnbf16-real/gates/gb10_plan.sh" "$UNIT" "$LEASE" dev1 "$ROOT" "$@"
sparklock release "$LEASE" --unit "$UNIT"
echo "released $LEASE"
