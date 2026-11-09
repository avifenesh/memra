#!/usr/bin/env bash
# memra#1093 follow-up: the cold-long K=2 vs K=3 A/B on the RTX 5090, one boot (one arm) per lease,
# boots in the order k3 k2 k2 k3 x3. Before each lease: stop if a gpulock waiter says "landing" or a
# YIELD file exists in OUT. usage: ab_driver.sh BINARY OUT
set -u
B=$1 O=$2
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$O"
n=0
for arm in k3 k2 k2 k3 k3 k2 k2 k3 k3 k2 k2 k3; do
  n=$((n + 1))
  if [ -e "$O/YIELD" ]; then echo "yield before boot $n $(date -Is)"; break; fi
  if gpulock status 2>/dev/null | grep -qi "waiting:.*landing"; then
    echo "landing waiter queued; yield before boot $n $(date -Is)"; touch "$O/YIELDED-landing"; break
  fi
  echo "boot $n $arm start $(date -Is)"
  gpulock run --ttl 15m --wait 60m --for "memra #1093 cold-long K A/B, boot $n ($arm)" -- \
    env CELL_PORT=18193 BOOT_START=$n PROMPT_SET=${PROMPT_SET:-} python3 "$H/cold_long_ab.py" "$O" "$B" "$arm" >> "$O/boots.log" 2>&1
  echo "boot $n $arm exit $? $(date -Is)"
done
python3 "$H/cold_long_ab.py" "$O" "$B" - > "$O/summary.log" 2>&1
echo AB_DONE
