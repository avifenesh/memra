#!/usr/bin/env bash
# Abort the battery the instant REAL traffic arrives.
# Discriminator: the battery runs as tenant "battery"; real marketplace traffic is tenant "onlist".
# Any growth in t:onlist counters, or any unexpected new tenant, stops everything.
set -uo pipefail
STOP=/root/BATTERY_STOP
PGID_FILE=/root/battery.pgid
LOG=/root/battery-watch.log
T=$(cat /root/memra-secrets/metrics-token)
base=$(curl -sS -H "Authorization: Bearer $T" http://127.0.0.1:8002/metrics \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print(int(d.get('tenants',{}).get('t:onlist',{}).get('prompt_tokens_in',0)))")
echo "$(date -u +%FT%TZ) watch start: baseline t:onlist prompt_tokens_in=$base" >>"$LOG"
while :; do
  [[ -e "$STOP" ]] && { echo "$(date -u +%FT%TZ) STOP file present; watcher exiting" >>"$LOG"; exit 0; }
  read -r cur tenants < <(curl -sS --max-time 10 -H "Authorization: Bearer $T" http://127.0.0.1:8002/metrics \
    | python3 -c "
import json,sys
d=json.load(sys.stdin); t=d.get('tenants',{})
print(int(t.get('t:onlist',{}).get('prompt_tokens_in',0)), ','.join(sorted(t.keys())))" 2>/dev/null) || { sleep 2; continue; }
  unexpected=$(printf '%s' "$tenants" | tr ',' '\n' | grep -vE '^(t:battery|t:servetest|t:onlist)$' | head -1)
  if [[ "$cur" -gt "$base" || -n "$unexpected" ]]; then
    echo "$(date -u +%FT%TZ) REAL TRAFFIC DETECTED: t:onlist $base -> $cur unexpected='$unexpected' — aborting battery" >>"$LOG"
    touch "$STOP"
    if [[ -s "$PGID_FILE" ]]; then
      pgid=$(<"$PGID_FILE")
      kill -TERM "-$pgid" 2>/dev/null || true
      sleep 3
      kill -KILL "-$pgid" 2>/dev/null || true
      echo "$(date -u +%FT%TZ) killed battery pgid $pgid" >>"$LOG"
    fi
    exit 0
  fi
  sleep 2
done
