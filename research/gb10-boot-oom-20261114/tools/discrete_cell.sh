#!/usr/bin/env bash
# Runs on a host with a discrete card, under the caller's GPU lease (memra#1316): the changed
# boot path where the retry must not engage.
#   1. N boot-only starts of memra-server (server_boot.py: up to the worker's `Engine ready`
#      line, then killed): every one must come up with no `[ctx-retry]` line.
#   2. One start with a device ordinal the host does not have (MEMRA_PP_DEVICES=ORDINAL): it must
#      end with the step named, `cuDeviceGet on device ORDINAL failed`, and no `[ctx-retry]` line.
# Written to OUT: boots.jsonl, absent.log, run.log (the card, the driver, the binary's hash, the
# verdicts). The model file is only named to the server; a boot-only start reads no weights.
# usage: BIN=dir OUT=dir MODEL=file.gguf [N=10] [ORDINAL=7] discrete_cell.sh
set -u
BIN=${BIN:?}; OUT=${OUT:?}; MODEL=${MODEL:?}; N=${N:-10}; ORDINAL=${ORDINAL:-7}
H=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"
log() { echo "$(date -Is) $*" | tee -a "$OUT/run.log"; }
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*'); do unset "$v"; done
export MEMRA_COMPAT=openai MEMRA_ADDR=127.0.0.1:8091 MEMRA_MODELS="q38=$MODEL"
log "card: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader) integrated attribute: discrete card"
log "binary: $(sha256sum "$BIN/memra-server" | cut -c1-64)"
python3 "$H/server_boot.py" "$OUT/boots.jsonl" "$N" 1000 "$BIN" 2>&1 | tee -a "$OUT/run.log"
up=$(grep -c '"outcome": "ready"' "$OUT/boots.jsonl")
retried=$(grep -c 'ctx-retry' "$OUT/boots.jsonl")
log "boots: $N started, $up came up, $retried with a [ctx-retry] line"
MEMRA_PP_DEVICES=$ORDINAL timeout 60 "$BIN/memra-server" > "$OUT/absent.log" 2>&1
rc=$?
log "absent ordinal $ORDINAL: exit $rc: $(grep -m1 'FATAL' "$OUT/absent.log")"
named=$(grep -c "cuDeviceGet on device $ORDINAL failed" "$OUT/absent.log")
verdict=PASS
[ "$up" = "$N" ] && [ "$retried" = 0 ] && [ "$named" -ge 1 ] && ! grep -q 'ctx-retry' "$OUT/absent.log" || verdict=FAIL
log "verdict: $verdict"
[ "$verdict" = PASS ]
