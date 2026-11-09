#!/usr/bin/env bash
# memra#1093: resume of c2_driver.sh after the 21:20 session restart killed it in boot 9 (no boot 9
# row was written). Boots 9..12 = k3 k2 k2 k3, same binary, one boot per gpulock lease.
# usage: c2_resume.sh BINARY OUT
set -u
B=$1 O=$2
H=$(cd "$(dirname "$0")" && pwd)
n=8
for arm in k3 k2 k2 k3; do
  n=$((n + 1))
  if [ -e "$O/YIELD" ]; then echo "yield before boot $n $(date -Is)"; break; fi
  if gpulock status 2>/dev/null | grep -qi "waiting:.*landing"; then
    echo "landing waiter queued; yield before boot $n $(date -Is)"; touch "$O/YIELDED-landing"; break
  fi
  echo "boot $n $arm start $(date -Is)"
  gpulock run --ttl 15m --wait 240m --for "memra #1093 cold-long K c2 A/B, boot $n ($arm)" -- \
    env CELL_PORT=18193 BOOT_START=$n python3 "$H/c2_ab.py" "$O" "$B" "$arm" >> "$O/boots.log" 2>&1
  echo "boot $n $arm exit $? $(date -Is)"
done
python3 "$H/c2_ab.py" "$O" - - > "$O/summary.log" 2>&1
echo AB_DONE
