#!/usr/bin/env bash
# Runs ON the GB10 host: an interleaved served A/B campaign, one fresh boot per cell.
# usage: campaign.sh CAMPAIGN_DIR "TARGETS" "REPS" ARM_SPEC...
#   ARM_SPEC = name|bindir|K=V,K2=V2   (env list comma-separated, may be empty)
# Cells run in the order given (write the interleaving out explicitly, e.g. A B C C B A).
set -u
CD=$1; TARGETS=$2; REPS=$3; shift 3
D=$HOME/.cache/memra-prefill
mkdir -p "$CD"
i=0
for spec in "$@"; do
  i=$((i+1))
  name=${spec%%|*}; rest=${spec#*|}; bin=${rest%%|*}; envs=${rest#*|}
  [ "$envs" = "$rest" ] && envs=""
  extra=${envs//,/ }
  out=$CD/$(printf '%02d' $i)-$name
  echo "$(date -Is) cell $i arm=$name bin=$bin extra=$extra" | tee -a "$CD/campaign.log"
  bash "$D/run_arm.sh" "$out" "$bin" "$extra" "$TARGETS" "$REPS" >> "$CD/campaign.log" 2>&1
  grep '^{' "$out/run.log" | sed "s/^/$name /" >> "$CD/campaign.log"
done
echo "$(date -Is) campaign done" | tee -a "$CD/campaign.log"
