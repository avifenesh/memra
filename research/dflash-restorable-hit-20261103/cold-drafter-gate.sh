#!/usr/bin/env bash
# Exactness gate (RTX 5090 or GB10) for the cold-drafter conversion (memra#1031 second half).
#   cold-drafter-gate.sh <memra-server bin> <out dir>
# Arm: prefix cache + restore/partial doors (the conversion is on with them).
# Reference: prefix cache off, so every request cold-primes DFlash. Same model, drafter, seed and requests (tools/dspark-cold-drafter-gate.py).
# Sizes: GATE_P_PARAS / GATE_S_PARAS (60-word paragraphs). The default follows the card's memory
# (nvidia-smi memory.total of the first visible device; GATE_VRAM_MIB overrides the reading).
# `nvidia-smi -i` counts in PCI bus order and CUDA_VISIBLE_DEVICES in CUDA_DEVICE_ORDER, so on a
# box with mixed cards set CUDA_DEVICE_ORDER=PCI_BUS_ID or GATE_VRAM_MIB:
#   a reading under 32,768 MiB (a 24 GB RTX 5090; a 32 GB card reads a little under its nominal
#     size and lands here too): 10 / 14, about 1.7k / 4.3k / 4.4k tokens on x0 / a / b, S above
#     the 2,056-row drafter window (30 / 30, about 10.8k on b, is refused there under the arm's
#     4,000 MB prefix cache);
#   GB10 (unified memory, memory.total [N/A]), an RTX PRO 6000 and larger: 70 / 80, about 26k on b.
# compare.txt starts with the sizes, memory reading, context and doors used.
# Port: GATE_PORT (default 8093). Each boot refuses a port something already listens on
# (tools/port-guard.sh), and after the ready wait the listener must be the server this boot
# started and its log must carry that server's own "[server] listening" line. Without that, on
# 2026-11-06 the reference boot never came up, the ready wait passed on a foreign server holding
# 8093, and ref.jsonl recorded that server's answers.
# Exit 0 only when both runs complete, every a/b completion is byte-identical to the reference
# and the arm's log has the serving shape (--check-log). Any failed step is named and exits 1.
set -u -o pipefail
BIN=$1; O=$2; mkdir -p "$O"
M=${M:-/data/ai-ml/models/q38-gguf}
GGUF=${GGUF:-$M/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}
DF=${DF:-$M/Qwen3.8-27B-DFlash2-memra}
RANKS=${RANKS:-$M/q38-ranks-sxc32768.gguf.txt}
G=${GATE:-$(dirname "$0")/../../tools/dspark-cold-drafter-gate.py}
PORT=${GATE_PORT:-8093}
ADDR=127.0.0.1:$PORT
PG=${PORT_GUARD:-$(dirname "$0")/../../tools/port-guard.sh}
# shellcheck source=tools/port-guard.sh
. "$PG" || { echo "cold-drafter-gate: FAIL: cannot load the port guard at $PG (set PORT_GUARD)"; exit 1; }
CTX=${GATE_CTX:-16384}
dev=${CUDA_VISIBLE_DEVICES:-0}; dev=${dev%%,*}
VRAM=${GATE_VRAM_MIB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$dev" 2>/dev/null | head -n 1 | tr -d ' ')}
case "$VRAM" in
  '' | *[!0-9]*) dP=70 dS=80 ;; # unified memory ([N/A]) or no reading
  *) if [ "$VRAM" -lt 32768 ]; then dP=10 dS=14; else dP=70 dS=80; fi ;;
esac
export GATE_P_PARAS=${GATE_P_PARAS:-$dP} GATE_S_PARAS=${GATE_S_PARAS:-$dS}
R=$O/compare.txt
note() { echo "$*" | tee -a "$R"; }
SP=
boot() { # tag extra-env...
  local tag=$1; shift
  memra_port_guard cold-drafter-gate "$PORT" GATE_PORT || return 1
  env MEMRA_COMPAT=openai MEMRA_MODELS="q38=$GGUF" MEMRA_ADDR="$ADDR" \
    MEMRA_CTX="$CTX" MEMRA_MAX_SESSIONS=1 MEMRA_KV_HOST_MB=0 \
    MEMRA_MTP_SKIP=1 MEMRA_DSPARK_SPEC=1 MEMRA_DSPARK_DRAFT="$DF" \
    MEMRA_FRSPEC_TRIM="$RANKS" "$@" \
    "$BIN" > "$O/serve-$tag.log" 2>&1 &
  SP=$!
  waitfor --http "http://$ADDR/v1/models" --timeout 10m --every 5s --fail-after 120 --quiet || return 1
  memra_port_owned cold-drafter-gate "$PORT" "$SP" || return 1
  grep -a -q -F "[server] listening on http://$ADDR" "$O/serve-$tag.log" || {
    echo "cold-drafter-gate: FAIL: $ADDR answers, but serve-$tag.log has no \"[server] listening on http://$ADDR\"" \
      "line: the responder is not the server this boot started."
    return 1
  }
}
stop() {
  [ -n "$SP" ] || return 0
  kill -TERM "$SP" 2>/dev/null; waitfor --gone "$SP" --timeout 90s --quiet || kill -KILL "$SP" 2>/dev/null; wait "$SP" 2>/dev/null
  SP=
}
# DOORS=default leaves the three restore doors unset, so the arm runs on the device default.
if [ "${DOORS:-explicit}" = default ]; then
  ARMDOORS=""
else
  ARMDOORS="MEMRA_DSPARK_PREFIX_RESTORE=1 MEMRA_DSPARK_BOUNDARY_CAPTURE=1 MEMRA_DSPARK_PARTIAL_RESTORE=1"
fi
: > "$R"
note "gate sizes GATE_P_PARAS=$GATE_P_PARAS GATE_S_PARAS=$GATE_S_PARAS memory.total=${VRAM:-unknown} MiB${GATE_VRAM_MIB:+ (GATE_VRAM_MIB override)} MEMRA_CTX=$CTX DOORS=${DOORS:-explicit} port=$PORT"
# A refused boot writes no server log, so a previous run's logs must not be read as this one's.
rm -f "$O/arm.jsonl" "$O/ref.jsonl" "$O/serve-arm.log" "$O/serve-ref.log"
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
if [ -f "$O/serve-arm.log" ]; then
  for k in '[dspark] cache-preferred:' 'DSPARK cold-drafter restore:' 'DSPARK restore:' 'insert (dspark-boundary):'; do
    echo "$k=$(grep -a -c -F "$k" "$O/serve-arm.log")"; done | tee -a "$R"
  # The serving shape: one veto and one conversion, then tail restores (the conversion must publish
  # its tail-carrying boundary entry; before the fix every step converted and none restored).
  python3 "$G" --check-log "$O/serve-arm.log" | tee -a "$R" || rc=1
else
  note "no serve-arm.log: the arm server never started, so there is no serving shape to check"; rc=1
fi
if [ $rc = 0 ]; then note "COLD-DRAFTER GATE PASS"; else note "COLD-DRAFTER GATE FAIL"; fi
exit $rc
