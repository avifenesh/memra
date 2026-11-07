#!/usr/bin/env bash
# admit-by-memory Max-Q cell: the WP-B day 40 target-card cell (DAY40.md 1.2; Workstation receipt in DAY40.md 2.1,
# research/spill-b-20260919/pro-single-day40/box/) replayed on ONE RTX PRO 6000 Blackwell Max-Q Workstation Edition
# on <box>, on the same program (S40 = b46ae200e) and the same drivers (50e8cb345). Port of
# pro-single-day40/chain.sh (kept as registered-chain.sh.txt). Rule: PREREG.md. Every change: DEVIATIONS.md.
#
# usage, on <box>, by the session that holds the card's window:
#   CARD=<idx> bash /mnt/Data/work/memra-maxq/admit-by-memory/run.sh [next|part1|part2|check]
#   bash /mnt/Data/work/memra-maxq/admit-by-memory/run.sh selfcheck      (readers on the Workstation receipt; no GPU)
#   bash /mnt/Data/work/memra-maxq/admit-by-memory/run.sh read <out dir> (readers on a finished campaign; no GPU)
# next (default): part2 of the newest campaign whose O1 is done and O2 is not, else part1.
# part1: order O1 (off, on2048, on8192, on32768; inner order AB) in a new out-<UTC stamp>/. About 3.0-3.4 h on a Max-Q.
# part2: order O2 (on32768, on8192, on2048, off; inner order BA) in that same out dir, then the readers over all eight
#        boots: SUMMARY.txt, FAULTS.txt, READINGS.txt and VERDICT.txt (first line = the DAY34 V-DOOR line).
# check: every refusal check and the model and binary sha256, then exit; no boot, no out dir.
# env: CARD (required for a GPU mode), WINDOW_S (seconds this call may take, default 14400), OUT (part2: the campaign
#      dir; default the newest one waiting for O2), IDLE_WAIT_S (default 300).
# exit: 0 part done, 1 refused (card, model, binary, host), 3 card not idle, 4 window too short, 5 order did not
#       complete, 6 readers printed no V-DOOR line, 130 interrupted.
set -uo pipefail
KIT=/mnt/Data/work/memra-maxq/admit-by-memory
MODEL=/mnt/Data/work/memra-maxq/models/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
WANT_MODEL=1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a
BIN=$KIT/bins/s40/memra-server
WANT_BIN=7da1db366fbe34ca40440fc8e818b864c2bc9014db7e7330b08a929084e7cc45
MAXQ_NAME="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
RIG_LOCK=/tmp/memra-gpu.lock
PORT=18526
export PATH=/usr/local/cuda/bin:/usr/bin:/bin:/usr/sbin:/sbin
WINDOW_S=${WINDOW_S:-14400}
IDLE_WAIT_S=${IDLE_WAIT_S:-300}
# The Workstation receipt's orders took 9398 s (O1) and 9376 s (O2) at 600 W; a Max-Q order is not expected faster.
MIN_ORDER_S=9400
# Readers, manifest and cleanup after the order (the readers take under a minute on the receipt's 8 boots).
READ_MARGIN_S=600
MODE=${1:-next}
T0=$SECONDS
STARTED=0
R=""
BUDGET=0

say() { echo "$(date -u +%FT%TZ) $*"; }
log() { say "$*" | tee -a "$R/chain.log"; }
refuse() { local code=$1; shift; say "REFUSED: $*"; [ -n "$R" ] && [ -d "$R" ] && say "REFUSED: $*" >> "$R/chain.log"; exit "$code"; }
# The processes this kit starts, by the cell's own rule (run-day26-cell.sh own_server_pid: cwd is the kit dir) plus
# what they run: the server, the collector and its sampler, the order and cell scripts, the client, the log stamper.
# A shell that merely sits in the kit dir does not match.
own_procs() {
  local d p cmd
  for d in /proc/[0-9]*; do
    p=${d#/proc/}
    [ "$p" = "$$" ] && continue
    [ "$(readlink "$d/cwd" 2>/dev/null)" = "$KIT" ] || continue
    cmd=$(tr '\0' ' ' < "$d/cmdline" 2>/dev/null)
    [[ "$cmd" =~ memra-server|research/spill-b-20260919/|tools/tier-battery.py|tools/tier-lock-proof.py|nvidia-smi|import\ sys,time ]] && echo "$p"
  done
}
cleanup() {
  local rc=$? p
  trap - EXIT INT TERM HUP
  if [ "$STARTED" = 1 ]; then
    for p in $(own_procs); do kill -TERM "$p" 2>/dev/null; done
    for _ in $(seq 30); do [ -z "$(own_procs)" ] && break; sleep 1; done
    for p in $(own_procs); do kill -KILL "$p" 2>/dev/null; done
  fi
  # The only scratch any mode creates (selfcheck's symlink dirs); campaign outputs are evidence and stay.
  rm -rf /mnt/Data/work/memra-maxq/admit-by-memory/.selfcheck
  if [ -n "$R" ] && [ -d "$R" ]; then
    say "run.sh $MODE exit=$rc" >> "$R/chain.log"
    [ -s "$R/VERDICT.txt" ] || echo "INCOMPLETE: run.sh $MODE ended with exit $rc before its order finished; see chain.log" > "$R/VERDICT.txt"
  fi
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP

CALLER_PWD=$PWD
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[ "$HERE" = "$KIT" ] || refuse 1 "run.sh must run from $KIT (it is at $HERE)"
cd "$KIT" || refuse 1 "cannot cd $KIT"
# Clean environment: no inherited MEMRA_* door, no inherited device selection or library override.
for v in $(compgen -e); do case $v in MEMRA_*|CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|LD_LIBRARY_PATH|LD_PRELOAD) unset "$v";; esac; done

case $MODE in
  selfcheck)
    bash "$KIT/selfcheck.sh" "$KIT/selfcheck/ws-box" /mnt/Data/work/memra-maxq/admit-by-memory/.selfcheck
    exit $? ;;
  read)
    R=$(cd "$CALLER_PWD" && cd "${2:?usage: run.sh read <out dir>}" && pwd -P) || exit 1
    nice -n 10 bash "$KIT/readers.sh" "$R/boots" "$R"
    rrc=$?
    nice -n 10 python3 "$KIT/maxq-vs-ws.py" "$R/boots" "$KIT/selfcheck/ws-box/boots" > "$R/MAXQ-VS-WS.txt" 2>&1
    exit $rrc ;;
  next|part1|part2|check) ;;
  *) refuse 1 "unknown mode $MODE (next|part1|part2|check|selfcheck|read)" ;;
esac

[[ "${CARD:-}" =~ ^[0-9]+$ ]] || refuse 1 "CARD unset or not an nvidia-smi index: '${CARD:-}'"
export CARD CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD

WHY=""
idle_now() {
  local apps used
  flock -n "$RIG_LOCK" true || { WHY="$RIG_LOCK is held"; return 1; }
  apps=$(nvidia-smi -i "$CARD" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>&1 | grep -E '^[0-9]+,')
  [ -z "$apps" ] || { WHY="compute apps on card $CARD: $(echo "$apps" | tr '\n' ';')"; return 1; }
  used=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits 2>&1)
  [[ "$used" =~ ^[0-9]+$ ]] && [ "$used" -lt 2000 ] || { WHY="card $CARD memory.used=$used MiB, not under 2000"; return 1; }
}
preflight() {
  local name dl got avail free
  name=$(nvidia-smi -i "$CARD" --query-gpu=name --format=csv,noheader 2>&1)
  [ "$name" = "$MAXQ_NAME" ] || refuse 1 "card $CARD is '$name', not '$MAXQ_NAME'"
  dl=$((SECONDS + IDLE_WAIT_S))
  until idle_now; do [ $SECONDS -ge $dl ] && refuse 3 "card $CARD not idle after ${IDLE_WAIT_S} s: $WHY"; sleep 10; done
  ss -ltn | grep -q ":$PORT " && refuse 1 "port $PORT busy"
  [ -x "$BIN" ] || refuse 1 "no binary $BIN"
  got=$(sha256sum "$BIN" | cut -d' ' -f1)
  [ "$got" = "$WANT_BIN" ] || refuse 1 "binary sha256 $got, want $WANT_BIN"
  avail=$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)
  # DAY34.md 1.3's host-memory bound for the local chain (24 GB); the 27B file alone is 15.7 GB.
  [ "$avail" -ge 24 ] || refuse 1 "host MemAvailable ${avail} GiB < 24"
  free=$(df -P -BG "$KIT" | awk 'NR==2 {gsub("G","",$4); print $4}')
  # A part writes about 30 MB (the receipt's 8 boots are 50 MB); refuse under 1 GiB.
  [ "$free" -ge 1 ] || refuse 1 "$KIT has ${free} GiB free"
  [ -f "$MODEL" ] || refuse 1 "model absent: $MODEL"
  say "model sha256 check (15.7 GB) ..."
  got=$(nice -n 10 sha256sum "$MODEL" | cut -d' ' -f1)
  [ "$got" = "$WANT_MODEL" ] || refuse 1 "model sha256 $got, want $WANT_MODEL; not run"
  say "preflight ok: card $CARD '$name' idle, port $PORT free, binary and model sha256 match, MemAvailable ${avail} GiB"
}
set_budget() {
  BUDGET=$((WINDOW_S - (SECONDS - T0) - READ_MARGIN_S))
  [ "$BUDGET" -ge "$MIN_ORDER_S" ] || refuse 4 "window too short: ${BUDGET} s left for the order; the Workstation order alone took ${MIN_ORDER_S} s. Set WINDOW_S to the real gap."
}
latest_waiting_o2() {
  local d
  for d in $(ls -1d "$KIT"/out-* 2>/dev/null | sort -r); do
    [ -f "$d/PART1.done" ] && [ ! -f "$d/PART2.done" ] && { echo "$d"; return 0; }
  done
  return 1
}
since_last() { # file pattern: the file from its last line matching pattern on
  local n
  n=$(grep -n -- "$2" "$1" 2>/dev/null | tail -n 1 | cut -d: -f1)
  [ -n "$n" ] && tail -n "+$n" "$1"
}
record_card() { # tag
  nvidia-smi -i "$CARD" --query-gpu=name,power.limit,memory.total --format=csv > "$R/card-$1.csv"
  nvidia-smi -i "$CARD" --query-gpu=index,uuid,pci.bus_id,name,power.limit,power.default_limit,power.max_limit,memory.total,driver_version,temperature.gpu,clocks.max.sm --format=csv > "$R/card-$1-detail.csv"
}
record_host() { # tag
  { echo "loadavg $(cat /proc/loadavg)"; grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; echo "nproc $(nproc)"; } > "$R/host-$1.txt"
}
order_done() { # order: this attempt's collector exited 0 (the order script ran to its end inside the budget), every
  # boot left its exit code, and no boot was refused for landing off the card.
  local arm
  since_last "$R/chain.log" "order $1: collector start" | grep -q "order $1: collector exit=0\$" || return 1
  since_last "$R/order.log" "order=$1 lock-proof" | grep -q "order=$1 done" || return 1
  [ -e "$R/boots/PLACEMENT-FAILED" ] && return 1
  for arm in off on2048 on8192 on32768; do [ -s "$R/boots/$1-$arm.exit" ] || return 1; done
}
run_order() { # order
  local order=$1 crc cpid
  set_budget
  record_card "$order"
  record_host "$order-before"
  { env | grep -E '^(MEMRA_|CUDA_|CARD=|LD_)' | sort; echo "python3=$(python3 --version 2>&1)"; echo "nice=10"; } > "$R/env-$order.txt"
  ldd "$BIN" | grep -E 'cuda|cublas' > "$R/ldd-$order.txt"
  echo "$WANT_MODEL  $MODEL" > "$R/model-$order.sha256"
  log "order $order: collector start card=$CARD $(tail -n 1 "$R/card-$order.csv") budget=${BUDGET}s"
  STARTED=1
  # chain.sh:32-34 with the kit's paths, nice -n 10 and the collector's timeout set to what is left of the window. In
  # the background so a signal to this script runs cleanup at once; the outer timeout only backstops the collector.
  env R="$R" RIG_LOCK="$RIG_LOCK" BIN="$BIN" BURST=64 \
    timeout -k 60 $((BUDGET + 300)) nice -n 10 python3 tools/tier-battery.py --rig pro-single --timeout "$BUDGET" \
    --out "$R/collector-$order" --external-lock --execute \
    bash research/spill-b-20260919/day31-order.sh @COLLECTOR_LOCK_FD@ "$order" > "$R/collector-$order.log" 2>&1 &
  cpid=$!
  wait "$cpid"; crc=$?
  log "order $order: collector exit=$crc"
  if [ -n "$(own_procs)" ]; then log "order $order: kit processes outlived the collector: $(own_procs | tr '\n' ' ')stopping them"
  else STARTED=0; fi
  for p in $(own_procs); do kill -TERM "$p" 2>/dev/null; done
  record_host "$order-after"
  record_card "$order-after"
}
boot_lines() { # order
  local arm
  for arm in off on2048 on8192 on32768; do
    echo "$1-$arm exit=$(cat "$R/boots/$1-$arm.exit" 2>/dev/null || echo none) $(grep -h '^DAY31 V-BOOT' "$R/boots/$1-$arm/REPORT.txt" 2>/dev/null || echo 'no V-BOOT line')"
  done
}
manifest() { ( cd "$R" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 ); }
context() {
  local o
  echo "context (not verdicts): DAY40 target-card cell replayed on an RTX PRO 6000 Blackwell Max-Q Workstation Edition; reader label --card pro6000 kept as registered"
  for o in O1 O2; do [ -f "$R/card-$o-detail.csv" ] && echo "card $o: $(tail -n 1 "$R/card-$o-detail.csv")"; done
  echo "binary bins/s40/memra-server sha256=$WANT_BIN (S40 b46ae200e7638f9c55688d2d339de2872f4762a6); drivers $(cat "$KIT/SOURCE_SHA") + portability.patch"
  echo "model $(basename "$MODEL") sha256=$WANT_MODEL"
}

if [ "$MODE" = next ]; then
  if c=$(latest_waiting_o2); then MODE=part2; OUT=${OUT:-$c}; else MODE=part1; fi
  say "next -> $MODE${OUT:+ ($OUT)}"
fi

case $MODE in
  check)
    preflight
    set_budget
    say "CHECK OK: card $CARD, ${BUDGET} s of the ${WINDOW_S} s window left for an order; nothing was started"
    exit 0 ;;
  part1)
    preflight
    R=$KIT/out-$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$R/bins/s40" "$R/boots" || refuse 1 "cannot create $R"
    log "part1 (order O1) start; out=$R"
    cp "$KIT/SOURCE_SHA" "$R/source.txt"
    cp "$KIT/PROVENANCE.txt" "$KIT/PREREG.md" "$KIT/DEVIATIONS.md" "$R/"
    cp "$KIT/bins/build-s40.log" "$R/bins/s40/build.log"
    echo b46ae200e7638f9c55688d2d339de2872f4762a6 > "$R/bins/s40/source.commit"
    ( cd "$KIT/bins" && sha256sum ./s40/memra-server ) > "$R/bins/SHA256SUMS"
    nvidia-smi -i "$CARD" --query-gpu=name,power.limit,memory.total --format=csv > "$R/card.csv"
    log "chain start HEAD=$(cat "$R/source.txt") $(tr '\n' ' ' < "$R/bins/SHA256SUMS")"
    export WT=$KIT MODEL MODEL_KEY=q38 NO_SCOPE=1
    run_order O1
    if order_done O1; then
      { echo "O1 done $(date -u +%FT%TZ) card=$CARD"; boot_lines O1; } > "$R/PART1.done"
      nice -n 10 python3 research/spill-b-20260919/day31-faults.py "$R/boots/O1-off" "$R/boots/O1-on2048" \
        "$R/boots/O1-on8192" "$R/boots/O1-on32768" > "$R/FAULTS-O1.txt" 2>&1
      { echo "PENDING: part 1 of 2 done (order O1). Next window: CARD=<idx> bash $KIT/run.sh part2   (or run.sh next)"
        boot_lines O1; context; } > "$R/VERDICT.txt"
      manifest; cat "$R/VERDICT.txt"; log "part1 done"; echo "OUT=$R"; exit 0
    fi
    { echo "INCOMPLETE: order O1 did not complete (see chain.log, collector-O1.log); rerun part1 whole in a new window"
      boot_lines O1; context; } > "$R/VERDICT.txt"
    manifest; cat "$R/VERDICT.txt"; echo "OUT=$R"; exit 5 ;;
  part2)
    if [ -z "${OUT:-}" ]; then OUT=$(latest_waiting_o2) || refuse 1 "no campaign waiting for O2 (run part1 first)"; fi
    R=$(cd "$CALLER_PWD" && cd "$OUT" && pwd -P) || refuse 1 "no dir $OUT"
    [ -f "$R/PART1.done" ] || refuse 1 "$R has no PART1.done"
    [ -f "$R/PART2.done" ] && refuse 1 "$R already has PART2.done"
    grep -q "$WANT_BIN" "$R/binary-O1.sha256" || refuse 1 "O1 ran another binary: $(cat "$R/binary-O1.sha256")"
    preflight
    if [ -e "$R/collector-O2" ] || ls -d "$R"/boots/O2-* > /dev/null 2>&1; then
      # DAY40.md 1.5: a rerun happens only as a whole order; the earlier attempt is kept, never overwritten.
      F=$R/failed-O2-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$F/boots"
      mv "$R"/boots/O2-* "$F/boots/" 2>/dev/null
      for f in collector-O2 collector-O2.log LOCK-O2.json binary-O2.sha256 card-O2.csv card-O2-detail.csv; do
        [ -e "$R/$f" ] && mv "$R/$f" "$F/"
      done
      log "part2: earlier O2 attempt moved to $F"
    fi
    log "part2 (order O2) start; out=$R"
    export WT=$KIT MODEL MODEL_KEY=q38 NO_SCOPE=1
    run_order O2
    if ! order_done O2; then
      { echo "INCOMPLETE: order O2 did not complete (see chain.log, collector-O2.log); rerun part2 whole in a new window"
        boot_lines O2; context; } > "$R/VERDICT.txt"
      manifest; cat "$R/VERDICT.txt"; echo "OUT=$R"; exit 5
    fi
    { echo "O2 done $(date -u +%FT%TZ) card=$CARD"; boot_lines O2; } > "$R/PART2.done"
    log "readers start"
    nice -n 10 timeout -k 30 "$READ_MARGIN_S" bash "$KIT/readers.sh" "$R/boots" "$R" > "$R/readers.out" 2>&1
    log "readers exit=$?"
    # PREREG.md 7: cross-card identity against the Workstation receipt, a reading with no bound.
    nice -n 10 timeout -k 30 120 python3 "$KIT/maxq-vs-ws.py" "$R/boots" "$KIT/selfcheck/ws-box/boots" > "$R/MAXQ-VS-WS.txt" 2>&1
    log "maxq-vs-ws exit=$? $(tail -n 1 "$R/MAXQ-VS-WS.txt")"
    [ -s "$R/VERDICT.txt" ] && head -n 1 "$R/VERDICT.txt" | grep -q -v '^PENDING' \
      || echo "NO-VERDICT: the readers wrote no VERDICT.txt (see readers.out); rerun: bash $KIT/run.sh read $R" > "$R/VERDICT.txt"
    { cat "$R/VERDICT.txt"; boot_lines O1 | sed 's/^/boot /'; boot_lines O2 | sed 's/^/boot /'; context; } > "$R/VERDICT.tmp" \
      && mv "$R/VERDICT.tmp" "$R/VERDICT.txt"
    manifest
    log "LANE-B-DAY40-MAXQ-DONE"
    cat "$R/VERDICT.txt"; echo "OUT=$R"
    head -n 1 "$R/VERDICT.txt" | grep -q '^DAY34 V-DOOR ' || exit 6
    exit 0 ;;
esac
