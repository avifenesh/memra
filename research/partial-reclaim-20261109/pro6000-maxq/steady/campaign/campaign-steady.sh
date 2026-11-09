#!/bin/bash
# Steady traffic at the limit (#1124), wrx GPU 0 only, under the tracked wrapper (one lease for the block), A B B A,
# a fresh boot per arm, MEMRA_CTX=32768 MEMRA_MAX_SESSIONS=12 (the card's derived 54.3 GB budget). No power raise.
# Run under taskset -c 24-31,56-63. Outputs on the root disk (the kit, nvme1).
#   campaign-steady.sh OUT [ORDER] [FIRST_INDEX] [LEASE_DIR]   (defaults: "A B B A" 1 lease; the wrapper's timeout is 2.5 h)
set -u
. <kit>/env.sh
R=$K/reclaim
O=$1; ORDER=${2:-A B B A}; FIRST=${3:-1}; LEASE=${4:-lease}
A=$R/bins/memra-server-fa0ee88acf
B=$R/bins/memra-server-merge-fa0ee88acf-3448243319
for b in "$A" "$B"; do [ -x "$b" ] || { echo "missing $b"; exit 3; }; done
mkdir -p "$O"
log() { echo "$(TZ=Asia/Jerusalem date '+%F %T') $*" | tee -a "$O/campaign.log" >> "$R/status.txt"; }
export STEADY_ARGS="--tok 24000 --fill 64 --fill-conc 4 --workers 8 --duration 1020 --repeat-every 4 --back-min 16 --back-max 48"
log "steady: start; affinity $(taskset -pc $$ | cut -d: -f2); STEADY_ARGS=$STEADY_ARGS"
python3 "$REPO/tools/memra-gpu-run" --gpus "$GPU0_UUID" --receipt "$O/$LEASE" --box-lock <box-locks>/gpu0.lock --wait 1800 --timeout 9000 -- \
  bash -c '
    i='"$((FIRST - 1))"'
    for a in '"$ORDER"'; do
      i=$((i + 1)); bin='"$A"'; [ "$a" = B ] && bin='"$B"'
      echo "$(TZ=Asia/Jerusalem date "+%F %T") steady $i $a start" | tee -a '"$O"'/campaign.log
      bash '"$R"'/arm-steady.sh "steady-$i-$a" "$bin" '"$O"' MEMRA_MAX_SESSIONS=12
      echo "$(TZ=Asia/Jerusalem date "+%F %T") steady $i $a rc=$? $(grep -a "oom_lines\|reclaim_on_defer_lines\|thermal stop" '"$O"'/steady-$i-$a/arm.txt | tr "\n" " ")" | tee -a '"$O"'/campaign.log
      grep -q "thermal stop" '"$O"'/steady-$i-$a/arm.txt && exit 143
    done
    exit 0'
log "steady: wrapper rc=$?"
