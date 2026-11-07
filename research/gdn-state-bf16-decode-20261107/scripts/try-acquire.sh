#!/usr/bin/env bash
# Try to lease a free Spark unit (b23d first, then cb33) without waiting; on success write
# "UNIT LEASE" to lease-<slot>.txt and exit 0.  usage: try-acquire.sh SLOT [UNITS...]
set -u
SLOT=${1:?slot}; shift
UNITS=${*:-b23d cb33}
R=$HOME/.cache/memra-gdnbf16-decode
for u in $UNITS; do
  if grep -qs "^$u " "$R"/lease-*.txt; then continue; fi
  id=$(sparklock acquire --unit "$u" --ttl 5h --for "memra GDN bf16 decode-class lane (build, gates, quality panel, decode tick)" 2>/dev/null) || continue
  [ -n "$id" ] || continue
  echo "$u $id" > "$R/lease-$SLOT.txt"
  echo "acquired $u $id"
  exit 0
done
exit 1
