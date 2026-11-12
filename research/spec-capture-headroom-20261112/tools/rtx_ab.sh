#!/usr/bin/env bash
# Rig: memra#1221's RTX 5090 no-regression cells, one gpulock lease per cell (rtx_cell.sh). Before
# each cell the 5090 must have shown no landing or release lease or ticket for 3 minutes.
# usage: rtx_ab.sh CAMPAIGN "X=BIN_PATH ..." CELL...   (CELL = NN-X)
set -u
H=$(cd "$(dirname "$0")" && pwd); CAMP=$1; ARMS=$2; shift 2
L=$HOME/.cache/memra-1221/logs/measure-5090.log
SEEN=$HOME/.cache/memra-1221/logs/landing-last-seen-5090
cat > "$H/.rtx_quiet.sh" <<EOF
s=\$(gpulock status 2>/dev/null) || exit 1
now=\$(date +%s)
if echo "\$s" | grep -qiE 'memra landing|release'; then echo "\$now" > "$SEEN"; exit 1; fi
[ -f "$SEEN" ] || exit 0
[ \$(( now - \$(cat "$SEEN") )) -ge 180 ]
EOF
for c in "$@"; do
  a=${c#*-}; bin=""
  for kv in $ARMS; do [ "${kv%%=*}" = "$a" ] && bin=${kv#*=}; done
  [ -x "$bin" ] || { echo "$(date -Is) $CAMP-$c: no binary for $a" >> "$L"; exit 1; }
  waitfor --until "bash $H/.rtx_quiet.sh" --every 30s --timeout 300m --quiet >> "$L" 2>&1 || { echo "$(date -Is) $CAMP-$c: never quiet" >> "$L"; exit 1; }
  gpulock run --ttl 15m --for "memra #1221 spec capture trim: 5090 no-regression cell $CAMP-$c (one cell per lease)" --wait 120m -- \
    bash "$H/rtx_cell.sh" "$CAMP" "$c" "$bin" >> "$L" 2>&1; rc=$?
  echo "$(date -Is) $CAMP-$c rc=$rc" >> "$L"
  [ $rc = 0 ] || exit 1
done
echo "$(date -Is) $CAMP done: $*" >> "$L"
