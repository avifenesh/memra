#!/usr/bin/env bash
# HOST-MICRO AUDIT — the Box B window driver (lane/glm5-host-audit, 2026-09-01).
#
# ENGINE-WIDE lane, glm5 as the measurement vehicle: the interventions here are
# family-agnostic (they act on the OS's placement of the one GPU worker thread every
# family's decode tick runs on), so the arms are priced once on the heaviest available
# decode shape and the PROD-APPLICABILITY section of LANE.md says which prod stacks may
# adopt them through their own deploy seam.
#
# WHAT THIS BOX CANNOT DO, measured not assumed (probe receipts in LANE.md §B0):
#   * perf stat / perf trace / ANY PMU counter — `kernel.perf_event_paranoid = 4` in an
#     unprivileged container plus no kernel-matched linux-tools. So cache-misses,
#     LLC/L3-misses and the exact cpu-migrations counter are UNAVAILABLE. Migrations are
#     recovered by SAMPLING /proc/<tid>/stat field 39 (lower bound, labelled).
#   * chrt SCHED_FIFO and negative nice — no CAP_SYS_NICE (`chrt` returns EPERM,
#     `nice -n -10` returns EACCES). Arm (ii) therefore runs as the two things that ARE
#     permitted: a POSITIVE-nice control on the co-tenant noise, and the tokio worker-count
#     cap (the 184-thread runtime is the actual preemption source on this box).
#   * taskset and strace DO work, so arm (i) and the futex census are real.
#
# PROTOCOL: TIMED cells raise /root/TIMING-IN-FLIGHT and drop it after. Every stop is
# scoped to THIS lane's pidfile plus a /proc/<pid>/exe check — never pkill, never a
# basename kill (LAW gate-stop-pkill-basename-trap). Arms are BOOT-level (the flag is
# read once through OnceLock), interleaved x3 per the amended law, x5 on anomaly.
set -euo pipefail

R="${R:-/root/out-hostaudit}"
SRC="${SRC:-/root/memra-hostaudit}"
BIN="${BIN:-$SRC/target/release/memra-server}"
MODEL="${MODEL:-/root/models/glm53-nvfp4}"
PORT="${PORT:-18700}"
PIDFILE="$R/server.pid"
PROMPTS="${PROMPTS:-$SRC/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json}"
SAMPLER="${SAMPLER:-$SRC/research/glm53-flash-bringup-20260827/host-audit-20260901/host-sampler.py}"
MT="${MT:-192}"
REPS="${REPS:-3}"

mkdir -p "$R"

# STDERR. `boot()` echoes the server pid on stdout and the caller reads it through command
# substitution, so a progress line on stdout lands INSIDE the captured "pid". That is exactly
# the trap the sibling identity gate hit, and it is recorded twice because it bit twice: any
# helper whose stdout is a value must keep its narration on stderr.
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/window.log" >&2; }

# ---- stop(): PID-verified, exe-verified, never pkill --------------------------------
stop() {
    [ -f "$PIDFILE" ] || return 0
    local pid; pid=$(cat "$PIDFILE")
    if [ -n "$pid" ] && [ -d "/proc/$pid" ]; then
        local exe; exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null || echo "")
        if [ "$exe" = "$(readlink -f "$BIN")" ]; then
            log "stop: SIGTERM $pid ($exe)"
            kill "$pid" 2>/dev/null || true
            for _ in $(seq 1 60); do [ -d "/proc/$pid" ] || break; sleep 1; done
            [ -d "/proc/$pid" ] && { log "stop: SIGKILL $pid"; kill -9 "$pid" 2>/dev/null || true; }
        else
            log "stop: REFUSED — pid $pid exe '$exe' is not this lane's binary"
        fi
    fi
    rm -f "$PIDFILE"
}
trap 'stop; rm -f /root/TIMING-IN-FLIGHT' EXIT

# ---- the ship serving recipe, one place ---------------------------------------------
# 3-card PP shape = the glm5 serving recipe the flip/struct batteries priced. Every arm
# differs from baseline by EXACTLY ONE env var; nothing else moves.
ship_env() {
    echo "MEMRA_SPILL_STATS=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 \
CUDA_VISIBLE_DEVICES=0,1,2 MEMRA_COMPAT=openai \
MEMRA_MODELS=zai/glm-5.3-flash=$MODEL MEMRA_ADDR=127.0.0.1:$PORT \
MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 MEMRA_PREFIX_CACHE_MB=0"
}

# ---- boot(): fresh boot with a BOOT NONCE, arm identity asserted, not assumed --------
# LAW ab-arm-identity-not-liveness: a 200 on /health proves a listener, not WHICH server.
boot() {
    local arm="$1"; shift            # extra env for this arm in "$@"
    stop
    local nonce="$arm-$(date -u +%s)-$RANDOM"
    local slog="$R/serve-$arm.log"
    log "boot arm=$arm nonce=$nonce extra_env='$*'"
    ( env $(ship_env) MEMRA_BOOT_NONCE="$nonce" "$@" "$BIN" >"$slog" 2>&1 & echo $! > "$PIDFILE" )
    local pid; pid=$(cat "$PIDFILE")
    for _ in $(seq 1 900); do
        curl -sf "http://127.0.0.1:$PORT/readyz" >/dev/null 2>&1 && break
        [ -d "/proc/$pid" ] || { log "boot FAILED arm=$arm — server died, tail:"; tail -30 "$slog"; return 1; }
        sleep 2
    done
    # arm identity: the live listener is OUR pid, and the log is OUR nonce's boot.
    local lpid; lpid=$(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ "$lpid" = "$pid" ] || { log "ARM IDENTITY FAIL arm=$arm: port $PORT held by pid '$lpid', ours is $pid"; return 1; }
    grep -q "$nonce" "$slog" 2>/dev/null || log "note: boot nonce not echoed by the server (env-only marker)"
    echo "$pid"
}

# ---- engagement receipt: the arm must ANNOUNCE, a 200 is not a receipt --------------
assert_affinity_engaged() {
    # Split, NOT `local arm="$1" slog="...$arm..."`: bash declares every name in a single
    # `local` before assigning any of them, so $arm is still unset while $slog is expanded and
    # `set -u` aborts the whole cell. Cost one window boot to learn.
    local arm="$1"
    local want="$2"
    local slog="$R/serve-$arm.log"
    local line; line=$(grep -m1 '\[worker-affinity\]' "$slog" 2>/dev/null || true)
    # BOTH arms announce, so neither arm's identity rests on the absence of a line.
    [ -n "$line" ] || { log "ENGAGEMENT FAIL arm=$arm: no [worker-affinity] line at all"; return 1; }
    log "engagement arm=$arm: $line"
    if [ "$want" = "on" ]; then
        # The line reports the mask READ BACK from sched_getaffinity, never the request.
        echo "$line" | grep -q 'engaged .*effective=' || {
            log "ENGAGEMENT FAIL arm=$arm: not an engaged+readback line"; return 1; }
        echo "$line" | grep -q 'CLAMPED-BY-OUTER-CPUSET' && \
            log "WARN arm=$arm: an outer cpuset narrowed the mask — the row is the EFFECTIVE mask"
    else
        echo "$line" | grep -q '\[worker-affinity\] off' || {
            log "OFF-ARM FAIL arm=$arm: expected the off announce, got: $line"; return 1; }
    fi
}

# ---- decode rows: greedy = the instrument, vendor-default sampled = the product ------
rows() {
    local arm="$1" pid="$2" tag="$3"
    touch /root/TIMING-IN-FLIGHT
    PROMPTS_JSON="$PROMPTS" PORT="$PORT" python3 "$(dirname "$SAMPLER")/decode-rows.py" \
        --arm "$arm" --tag "$tag" --pid "$pid" --reps "$REPS" --max-tokens "$MT" \
        --sampler "$SAMPLER" --out "$R" | tee -a "$R/rows.jsonl"
    rm -f /root/TIMING-IN-FLIGHT
}

case "${1:-}" in
  census)   # B0: the host census on ONE baseline boot (no arms, no A/B)
    pid=$(boot base) || exit 1
    assert_affinity_engaged base off
    rows base "$pid" census
    log "census: bounded strace futex sample on the primary worker tid (INSTRUMENTED numbers)"
    wtid=$(for t in /proc/$pid/task/*; do [ "$(cat $t/comm)" = "memra-gpu-worke" ] && \
        echo "$(awk '{print $14+$15}' $t/stat) ${t##*/}"; done | sort -rn | head -1 | awk '{print $2}')
    log "primary worker tid=$wtid (highest utime+stime of the memra-gpu-worke set)"
    echo "$wtid" > "$R/worker.tid"
    ;;
  arm)      # one intervention boot: run-window.sh arm <name> ENV=VAL...
    name="$2"; shift 2
    pid=$(boot "$name" "$@") || exit 1
    rows "$name" "$pid" arm
    ;;
  stop) stop ;;
  *) echo "usage: run-window.sh {census|arm <name> ENV=VAL...|stop}"; exit 2 ;;
esac
