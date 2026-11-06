#!/usr/bin/env bash
# Exactness gate (RTX 5090 or GB10) for the cold-drafter conversion (memra#1031 second half).
#   cold-drafter-gate.sh <memra-server bin> <out dir>
# Arm: prefix cache + restore/partial doors (the conversion is on with them).
# Reference: prefix cache off, so every request cold-primes DFlash. Same model, drafter, seed and requests (tools/dspark-cold-drafter-gate.py).
# Sizes: GATE_P_PARAS / GATE_S_PARAS (60-word paragraphs). The defaults 10 / 12 fit the 27B on a
# 24 GB RTX 5090 (about 1.7k / 3.9k / 4.0k tokens on x0 / a / b; S leaves 2,129 suffix tokens over
# the 2,056-row drafter window). The GB10 receipts before 2026-11-06 ran 70 / 80 (about 26k on b),
# which a 24 GB card refuses. compare.txt starts with the sizes, context and doors used.
# Exit 0 only when both runs complete, every a/b completion is byte-identical to the reference
# and the arm's log has the serving shape (--check-log). Any failed step is named and exits 1.
set -u -o pipefail
BIN=$1; O=$2; mkdir -p "$O"
M=${M:-/data/ai-ml/models/q38-gguf}
GGUF=${GGUF:-$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
DF=${DF:-$M/Qwen3.8-27B-DFlash2-memra}
RANKS=${RANKS:-$M/q38-ranks-sxc32768.gguf.txt}
G=${GATE:-$(dirname "$0")/../../tools/dspark-cold-drafter-gate.py}
ADDR=${GATE_ADDR:-127.0.0.1:8093}
CTX=${GATE_CTX:-16384}
export GATE_P_PARAS=${GATE_P_PARAS:-10} GATE_S_PARAS=${GATE_S_PARAS:-12}
R=$O/compare.txt
note() { echo "$*" | tee -a "$R"; }
boot() { # tag extra-env...
  local tag=$1; shift
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$GGUF" MEMRA_ADDR="$ADDR" \
    MEMRA_CTX="$CTX" MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" \
    MEMRA_FRSPEC_TRIM="$RANKS" "$@" \
    "$BIN" > "$O/serve-$tag.log" 2>&1 &
  SP=$!
  waitfor --http "http://$ADDR/v1/models" --timeout 10m --every 5s --fail-after 120 --quiet
}
stop() { kill -TERM "$SP" 2>/dev/null; waitfor --gone "$SP" --timeout 90s --quiet || kill -KILL "$SP" 2>/dev/null; wait "$SP" 2>/dev/null; }
# DOORS=default leaves the three restore doors unset, so the arm runs on the device default.
if [ "${DOORS:-explicit}" = default ]; then
  ARMDOORS=""
else
  ARMDOORS="MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1"
fi
: > "$R"
note "gate sizes GATE_P_PARAS=$GATE_P_PARAS GATE_S_PARAS=$GATE_S_PARAS MEMRA_CTX=$CTX DOORS=${DOORS:-explicit}"
rm -f "$O/arm.jsonl" "$O/ref.jsonl"
rc=0
# shellcheck disable=SC2086
if ! { boot arm MEMRA_PREFIX_CACHE_MB=4000 $ARMDOORS MEMRA_PREFIX_STABLE_BOUNDARY=1 \
       && python3 "$G" "http://$ADDR" q38 "$O/arm.jsonl"; }; then
  note "arm run FAILED (serve-arm.log)"; rc=1
fi
stop
if ! { boot ref MEMRA_PREFIX_CACHE_MB=0 && python3 "$G" "http://$ADDR" q38 "$O/ref.jsonl"; }; then
  note "reference run FAILED (serve-ref.log)"; rc=1
fi
stop
python3 "$G" --compare "$O/arm.jsonl" "$O/ref.jsonl" | tee -a "$R" || rc=1
for k in 'cache-preferred' 'cold-drafter restore' 'DSPARK restore' 'dspark-boundary'; do
  echo "$k=$(grep -a -c "$k" "$O/serve-arm.log")"; done | tee -a "$R"
# The serving shape: one veto and one conversion, then tail restores (the conversion must publish
# its tail-carrying boundary entry; before the fix every step converted and none restored).
python3 "$G" --check-log "$O/serve-arm.log" | tee -a "$R" || rc=1
if [ $rc = 0 ]; then note "COLD-DRAFTER GATE PASS"; else note "COLD-DRAFTER GATE FAIL"; fi
exit $rc
