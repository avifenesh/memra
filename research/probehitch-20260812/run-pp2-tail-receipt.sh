#!/usr/bin/env bash
# Interleaved x5 PP-2 tail receipt. Requires the measurement-only source patch captured in
# raw/sbox/source.diff; that patch arms only the max rung immediately before a queued request.
set -uo pipefail

REPO=${REPO:-/opt/dl-image/nvme/cx-probehitch/memra}
BIN=${BIN:-$REPO/target/release/memra-server}
MODEL=${MODEL:-/opt/dl-image/nvme/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
PROMPT=${PROMPT:-$REPO/research/e2e/prompts/pp512.txt}
RAW=${RAW:-$REPO/research/probehitch-20260812/raw/sbox}
N=${N:-5}
PORT=${PORT:-18371}
BASE="http://127.0.0.1:$PORT"
STAMP=${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
CLIENT="$RAW/client-$STAMP.jsonl"
DRIVER="$RAW/driver-$STAMP.log"
SERVER_PID=

mkdir -p "$RAW"
cd "$REPO" || exit 1

snapshot() {
    printf 'snapshot %s\n' "$(date -u +%FT%TZ)"
    nvidia-smi \
        --query-gpu=index,name,temperature.gpu,clocks.sm,pstate,power.draw,memory.used,utilization.gpu \
        --format=csv,noheader || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
        --format=csv,noheader || true
}

require_idle() {
    local receipt_apps
    receipt_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
        --format=csv,noheader 2>/dev/null || true)
    if [[ -n "$receipt_apps" ]]; then
        printf 'GPU NOT IDLE AFTER LOCK ACQUISITION\n%s\n' "$receipt_apps"
        return 76
    fi
}

stop_server() {
    if [[ -n ${SERVER_PID:-} ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=
    fi
}

wait_idle() {
    local attempt receipt_apps
    for attempt in $(seq 1 120); do
        receipt_apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader 2>/dev/null || true)
        [[ -z "$receipt_apps" ]] && return 0
        sleep 1
    done
    printf 'GPU applications remained after server stop\n%s\n' "$receipt_apps"
    return 1
}

boot_server() {
    local arm=$1 pair=$2 gate=$3 server_log=$4
    env \
        MEMRA_MODELS="step35=$MODEL" \
        MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 \
        MEMRA_PP_DEVICES=0,1 \
        MEMRA_CTX=8192 \
        MEMRA_ADDR="127.0.0.1:$PORT" \
        MEMRA_TAG="probehitch-$arm-p$pair" \
        MEMRA_PROBEHITCH_RECEIPT="$arm" \
        MEMRA_PROBEHITCH_GATE="$gate" \
        "$BIN" >"$server_log" 2>&1 &
    SERVER_PID=$!

    local attempt
    for attempt in $(seq 1 240); do
        sleep 2
        if curl -sf "$BASE/readyz" >/dev/null 2>&1; then
            printf '%s pair=%s ready after <=%ss log=%s\n' \
                "$arm" "$pair" "$((attempt * 2))" "$server_log"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            printf '%s pair=%s SERVER DIED\n' "$arm" "$pair"
            tail -100 "$server_log"
            return 1
        fi
    done
    printf '%s pair=%s readiness timeout\n' "$arm" "$pair"
    tail -100 "$server_log"
    return 1
}

run_arm() {
    local arm=$1 pair=$2
    local gate="/tmp/cx-probehitch-$STAMP-$arm-p$pair.gate"
    local server_log="$RAW/server-$arm-p$pair-$STAMP.log"
    local client_log="$RAW/client-$arm-p$pair-$STAMP.jsonl"
    local console_log="$RAW/client-$arm-p$pair-$STAMP.log"
    local metrics_log="$RAW/metrics-$arm-p$pair-$STAMP.json"
    rm -f "$gate"

    printf '########## pair=%s arm=%s ##########\n' "$pair" "$arm"
    snapshot
    boot_server "$arm" "$pair" "$gate" "$server_log" || return 1
    python3 research/probehitch-20260812/probe-pp2-tail.py \
        --base "$BASE" \
        --model step35 \
        --prompt-file "$PROMPT" \
        --gate "$gate" \
        --cache-salt "probehitch-$arm-p$pair" \
        --label "$arm-p$pair" \
        --out "$client_log" 2>&1 | tee "$console_log"
    local probe_rc=${PIPESTATUS[0]}
    curl -sf "$BASE/metrics" >"$metrics_log" || true
    stop_server
    wait_idle || return 1
    rm -f "$gate"
    snapshot
    (( probe_rc == 0 )) || return "$probe_rc"

    if [[ "$arm" == before ]]; then
        grep -q '\[probehitch-receipt\] mode=before ran=true scheduler_idle=true' "$server_log" \
            || { printf 'before receipt did not run inline\n'; return 1; }
    else
        grep -q '\[probehitch-receipt\] mode=after ran=false scheduler_idle=false' "$server_log" \
            || { printf 'after receipt did not defer at the busy boundary\n'; return 1; }
        grep -q 'tokens=4096.*scheduler_idle=true' "$server_log" \
            || { printf 'after receipt did not drain the max rung at idle\n'; return 1; }
    fi
    jq -c --arg arm "$arm" --argjson pair "$pair" \
        'select(.phase == "measured") + {arm: $arm, pair: $pair}' "$client_log" >>"$CLIENT"
}

summarize() {
    python3 - "$CLIENT" "$N" <<'PY'
import json
import math
import statistics
import sys

path, expected = sys.argv[1], int(sys.argv[2])
rows = [json.loads(line) for line in open(path) if line.strip()]
by_pair = {}
for arm in ("before", "after"):
    measured = [row for row in rows if row["arm"] == arm]
    if len(measured) != expected:
        raise SystemExit(f"{arm}: expected N={expected}, got {len(measured)}")
    values = sorted(row["client_ttft_ms"] for row in measured)
    p95 = values[max(0, math.ceil(0.95 * len(values)) - 1)]
    print(json.dumps({
        "kind": "summary",
        "arm": arm,
        "n": len(values),
        "ttft_p50_ms": statistics.median(values),
        "ttft_p95_ms": p95,
        "ttft_min_ms": values[0],
        "ttft_max_ms": values[-1],
        "samples_ms": values,
        "cached_tokens": sorted({row["cached_tokens"] for row in measured}),
        "text_sha256": sorted({row["text_sha256"] for row in measured}),
    }, sort_keys=True))
    for row in measured:
        by_pair.setdefault(row["pair"], {})[arm] = row["client_ttft_ms"]

deltas = []
for pair in range(1, expected + 1):
    arms = by_pair.get(pair, {})
    if sorted(arms) != ["after", "before"]:
        raise SystemExit(f"pair {pair}: incomplete arms {arms}")
    deltas.append(arms["before"] - arms["after"])
print(json.dumps({
    "kind": "paired_delta",
    "definition": "before_ttft_ms - after_ttft_ms",
    "n": len(deltas),
    "median_ms": statistics.median(deltas),
    "min_ms": min(deltas),
    "max_ms": max(deltas),
    "samples_ms": deltas,
}, sort_keys=True))
PY
}

main() {
    : >"$CLIENT"
    printf '=== cx-probehitch PP-2 tail receipt %s\n' "$STAMP"
    printf 'repo=%s head=%s\n' "$REPO" "$(git rev-parse HEAD)"
    printf 'protocol=interleaved x5, one lock hold, warm cache, streaming first-visible-token TTFT\n'
    printf 'instrumentation=max rung armed only after queued command is received; before forces old inline decision; after uses busy decision then idle drain\n'

    (
        flock -w 21600 9 || { printf 'LOCK TIMEOUT\n'; exit 75; }
        trap stop_server EXIT
        printf 'lock acquired %s\n' "$(date -u +%FT%TZ)"
        snapshot
        require_idle || exit $?
        git diff --binary >"$RAW/source.diff"
        sha256sum "$PROMPT"
        sha256sum "${MODEL%00001-of-00003.gguf}"*.gguf

        /usr/bin/time -f 'build wall_s=%e max_rss_kb=%M exit=%x' \
            env RUSTC=/home/ubuntu/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/rustc \
            /home/ubuntu/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin/cargo \
            build --release -p memra-server --bin memra-server \
            2>&1 | tee "$RAW/build-$STAMP.log"
        local build_rc=${PIPESTATUS[0]}
        (( build_rc == 0 )) || exit "$build_rc"
        printf 'binary_sha256=%s\n' "$(sha256sum "$BIN" | awk '{print $1}')"

        local pair order arm
        for pair in $(seq 1 "$N"); do
            if (( pair % 2 == 1 )); then
                order='before after'
            else
                order='after before'
            fi
            for arm in $order; do
                run_arm "$arm" "$pair" || exit 1
            done
        done
        summarize | tee "$RAW/summary-$STAMP.jsonl"
        snapshot
        printf 'lock released %s\n' "$(date -u +%FT%TZ)"
    ) 9>/tmp/memra-gpu.lock
    local receipt_rc=$?
    printf '=== cx-probehitch PP-2 tail receipt rc=%s\n' "$receipt_rc"
    return "$receipt_rc"
}

exec > >(tee "$DRIVER") 2>&1
main
