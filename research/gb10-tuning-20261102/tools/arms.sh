#!/usr/bin/env bash
# Rig-side orchestrator for the gb10-b validation of memra PR #1025.
#   BIN=~/memra-run/bin-<sha12> LEASE=<lease> ARMS="..." [USERS="1 4 8 16 32"] [QUALITY_ARMS="..."] arms.sh
# Per arm: boot through serve-arm.sh on gb10-b, record the receipt lines, run bench_memra.sh
# (agentic cells, optional quality panel) through an SSH tunnel on 127.0.0.1:8091, stop.
set -u
BIN=${BIN:?bin dir on gb10-b}
LEASE=${LEASE:?sparklock lease id}
ARMS=${ARMS:?arms}
USERS=${USERS:-1 4 8 16 32}
QUALITY_ARMS=${QUALITY_ARMS:-}
TAGSFX=${TAGSFX:-gb10-b}
L=~/.cache/memra-1025/gb10-b/arms.log
R=~/.cache/model-scout/runs
H=~/.cache/model-scout/harness
PORT=8091
log(){ echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) $*" >> "$L"; }
S(){ /usr/bin/ssh -n -o BatchMode=yes gb10-b "$@"; }
stop_arm() {  # $1 = arm
  S "g=\$(cat ~/memra-run/guard-$1.pid 2>/dev/null); s=\$(cat ~/memra-run/serve-$1.pid 2>/dev/null); kill \$g 2>/dev/null; kill -TERM \$s 2>/dev/null; for i in \$(seq 1 60); do kill -0 \$s 2>/dev/null || break; sleep 2; done; kill -KILL \$s 2>/dev/null; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l"
}
tunnel() {
  if ! curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; then
    /usr/bin/ssh -f -N -o BatchMode=yes -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:8090" gb10-b || log "tunnel failed"
  fi
}
log "start BIN=$BIN ARMS=$ARMS USERS=$USERS QUALITY_ARMS=$QUALITY_ARMS"
tunnel
for arm in $ARMS; do
  tag=q27-memra-$arm-$TAGSFX
  log "boot $arm ($tag)"
  S "(ARM=$arm BIN=$BIN LEASE=$LEASE EXTRA='${EXTRA:-}' bash ~/memra-run/serve-arm.sh) > /dev/null 2>&1 < /dev/null & echo started" >> "$L"
  if ! waitfor --http "http://127.0.0.1:$PORT/v1/models" --timeout 10m --every 10s --fail-after 80 --quiet >> "$L" 2>&1; then
    log "boot $arm FAILED"; S "tail -20 ~/memra-run/serve-$arm.log" >> "$L"
    log "stop $arm: gpu procs left $(stop_arm "$arm")"; continue
  fi
  S "grep -a -E 'mtp-skip|stub|TRIMMED|frspec-trim|spec-k\]|DSPARK SPEC|dspark\] (precision|q38)|prefix-cache\] on|\[deadline\]|admission\]|route=|device-trim\] reason=calibration' ~/memra-run/serve-$arm.log | cut -c1-300; echo -n 'boot MemAvailable '; grep MemAvailable /proc/meminfo" >> "$L"
  case "$arm" in *dflash*)
    S "curl -s http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"q38\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain in three sentences why the sky is blue.\"}],\"max_tokens\":200}' | head -c 300; echo; sleep 2; grep -a -E 'dspark-acc|spec-k\\] model' ~/memra-run/serve-$arm.log | tail -3 | cut -c1-300" >> "$L" ;;
  esac
  if [ "${NONSTREAM:-0}" = 1 ]; then
    log "non-streaming long prompt ($arm): $(python3 ~/.cache/memra-1025/gb10-b/nonstream.py http://127.0.0.1:$PORT ${NONSTREAM_TOKENS:-131072} 2>&1 | tail -1 | cut -c1-400)"
    S "grep -a -E '\\[deadline\\]|shed|refused' ~/memra-run/serve-$arm.log | tail -5 | cut -c1-250" >> "$L"
  fi
  case "$arm" in *-noskip) log "receipt boot only"; log "stop $arm: gpu procs left $(stop_arm "$arm")"; continue ;; esac
  q=0; case " $QUALITY_ARMS " in *" $arm "*) q=1 ;; esac
  (cd $H && CLIENT_MAX_TIME=1800 QUALITY=$q BENCH_BASE=http://127.0.0.1:$PORT BENCH_MODEL=q38 BENCH_TOKENIZE_PATH=/v1/tokenize BENCH_EMULATE_CONTINUOUS_USAGE=1 \
    NODE=gb10-b USERS="$USERS" <harness>/bench_memra.sh $R/$tag $tag)
  log "bench $arm done: $(tail -n 4 $R/$tag/run.log | cut -c1-220 | tr '\n' ' ')"
  S "echo inserts \$(grep -a -c 'prefix-cache\] insert' ~/memra-run/serve-$arm.log); grep -a -E '^\[shed\]|refused (before|after)|unified memory: insert budget' ~/memra-run/serve-$arm.log | sort | uniq -c | sort -rn | head -20 | cut -c1-250; grep -a -E 'dspark-acc' ~/memra-run/serve-$arm.log | tail -2 | cut -c1-250" >> "$L"
  log "stop $arm: gpu procs left $(stop_arm "$arm")"
  S "tail -3 ~/memra-run/guard-$arm.log" >> "$L"
done
log "all arms done; memra stopped"
