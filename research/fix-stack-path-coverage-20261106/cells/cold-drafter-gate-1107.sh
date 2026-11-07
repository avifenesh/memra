#!/usr/bin/env bash
# Exactness gate (RTX 5090 or GB10) for the cold-drafter conversion (memra#1031 second half).
#   cold-drafter-gate.sh <memra-server bin> <out dir>
# Arm: prefix cache + restore/partial doors (the conversion is on with them).
# Reference: prefix cache off, so every request cold-primes DFlash. Same model, drafter, seed and requests (tools/dspark-cold-drafter-gate.py).
# The default sizes (GATE_P_PARAS=70, GATE_S_PARAS=80, about 26k tokens on the b step) fit GB10. The
# 27B on a 24 GB RTX 5090 admits about 18.9k tokens: run it there with GATE_P_PARAS=30 GATE_S_PARAS=30
# (about 10k; S stays above the 2,056-row drafter window).
# Exit 0 only when both runs complete, every a/b completion is byte-identical and the arm took the
# conversion: at least one cache-preferred veto and one cold-drafter restore in its log.
set -u -o pipefail
BIN=$1; O=$2; mkdir -p "$O"
M=${M:-<models>/q38-gguf}
GGUF=${GGUF:-$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
DF=${DF:-$M/Qwen3.8-27B-DFlash2-memra}
RANKS=${RANKS:-$M/q38-ranks-sxc32768.gguf.txt}
G=${GATE:-$(dirname "$0")/../../tools/dspark-cold-drafter-gate.py}
boot() { # tag extra-env...
  local tag=$1; shift
  # refuse a foreign listener on the fixed port: waitfor --http would accept its answers
  if ss -ltn | grep -q ':8093 '; then echo "port 8093 already bound before the $tag boot; refusing"; return 4; fi
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$GGUF" MEMRA_ADDR=127.0.0.1:8093 \
    MEMRA_CTX=16384 MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT=$DF \
    MEMRA_FRSPEC_TRIM=$RANKS "$@" \
    "$BIN" > "$O/serve-$tag.log" 2>&1 &
  SP=$!
  waitfor --http http://127.0.0.1:8093/v1/models --timeout 10m --every 5s --fail-after 120 --quiet || return 3
  kill -0 $SP 2>/dev/null || { echo "the $tag server exited during boot"; return 3; }
  grep -a -q 'listening' "$O/serve-$tag.log" || { echo "the $tag server log has no listening line: another process answered"; return 3; }
}
stop() { kill -TERM $SP; waitfor --gone $SP --timeout 90s --quiet || kill -KILL $SP; wait $SP 2>/dev/null; }
# DOORS=default leaves the three restore doors unset, so the arm runs on the device default.
if [ "${DOORS:-explicit}" = default ]; then
  ARMDOORS=""
else
  ARMDOORS="MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1"
fi
rc=0
# shellcheck disable=SC2086
boot arm MEMRA_PREFIX_CACHE_MB=4000 $ARMDOORS MEMRA_PREFIX_STABLE_BOUNDARY=1 \
  && python3 "$G" http://127.0.0.1:8093 q38 "$O/arm.jsonl" || { echo "arm run FAILED"; rc=1; }
stop
boot ref MEMRA_PREFIX_CACHE_MB=0 && python3 "$G" http://127.0.0.1:8093 q38 "$O/ref.jsonl" \
  || { echo "reference run FAILED"; rc=1; }
stop
python3 "$G" --compare "$O/arm.jsonl" "$O/ref.jsonl" | tee "$O/compare.txt" || rc=1
for k in 'cache-preferred' 'cold-drafter restore' 'DSPARK restore' 'dspark-boundary'; do
  echo "$k=$(grep -a -c "$k" "$O/serve-arm.log")"; done | tee -a "$O/compare.txt"
for k in 'cache-preferred' 'cold-drafter restore'; do
  grep -a -q "$k" "$O/serve-arm.log" || { echo "the arm never logged '$k': the conversion path did not run" | tee -a "$O/compare.txt"; rc=1; }
done
python3 "$G" --check-log "$O/serve-arm.log" | tee -a "$O/compare.txt" || rc=1
[ $rc = 0 ] && echo "COLD-DRAFTER GATE PASS" || echo "COLD-DRAFTER GATE FAIL"
exit $rc
