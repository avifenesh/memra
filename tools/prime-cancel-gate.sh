#!/usr/bin/env bash
# PRIME CANCELLATION GATE (memra#536 item 2, lane/spill-a-20260919 day 16).
#
# WHAT IT PROVES. A client that disconnects while its prompt is being primed stops the prime at the
# next completed internal chunk (the engine's `progress::prime_cancel_point`, asked by the sequential
# chunk walks right where the odometer stamps), publishes nothing, and leaves the next requests'
# bytes untouched: the cold digest and the warm (prefix-hit) digest of the same prompt are identical
# to a control boot that never saw the disconnect. Before the cancellation point the worker's only
# check was the tick-top sweep, once per `prefill_tick` call, so a prime ran to the end of its take
# (the 2026-09-05 incident: 329 `[abort] client disconnected while queued` lines behind one prime).
#
# SHAPE. Two boots of the same binary and env:
#   control: r1 cold P (temperature 0, max_tokens 24) -> cold digest, r2 same P -> warm digest.
#   fault:   r0 a raw-socket streaming request for P, the socket closed after DISCONNECT_MS; then the
#            server log must show `[prime] cancelled at chunk K (R of T rows ...)` with R < T and the
#            `[abort] client disconnected` retirement, and no `[prefix-cache] insert` line before
#            the cancel; then r1 cold P (cached_tokens must be 0: nothing was published), r2 warm P.
# The boot pins one prime call over the whole prompt (`MEMRA_PREFILL_TICK=8192`, so the tick-top
# sweep cannot be what stopped it) and small internal chunks (`MEMRA_PRIME_CHUNK=256`), plain decode
# (`MEMRA_SERVE_SPEC=0`), prefix cache on (the warm arm needs the seed). The prompt is deterministic
# (about 6000 words of a 64-word list, MEMRA_PCG_WORDS). The disconnect delay (MEMRA_PCG_DISCONNECT_MS,
# default 300) must land inside the prime: a prime that finished before it is a FIXTURE verdict
# (`fixture: prime finished before the disconnect`), not an engine pass or fail.
#
# usage: prime-cancel-gate.sh [--external-lock FD] <model.gguf> <server_bin> <evidence_dir>
# Boots its own servers one at a time behind flock ${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
# (bounded wait MEMRA_PCG_LOCK_WAIT seconds, default 600) or the collector's inherited lock FD.
# Exit 0 = PASS (every clause), 1 = FAIL or fixture verdict (named), 2 = refused before any boot.
set -euo pipefail
LOCK_FD=""
LOCK_OWNER=internal-canonical
if [[ ${1:-} == --external-lock ]]; then
    [[ ${2:-} =~ ^[0-9]+$ ]] || { echo "REFUSED: inherited lock FD required" >&2; exit 2; }
    LOCK_FD=$2
    LOCK_OWNER=collector
    shift 2
fi
[[ $# == 3 ]] || { echo "REFUSED: expected MODEL BIN EV" >&2; exit 2; }
MODEL=$1
BIN=$2
EV=$3
GPU_LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}
PORT=${MEMRA_GATE_PORT:-18133}
HERE=$(cd "$(dirname "$0")" && pwd)
[ -f "$HERE/port-guard.sh" ] || {
    echo "prime-cancel-gate: FAIL: $HERE/port-guard.sh missing; refusing to bind unguarded" >&2
    exit 1
}
. "$HERE/port-guard.sh"
case "$GPU_LOCK" in
    /tmp/memra-5090.lock|/tmp/memra-gpu.lock) ;;
    *) echo "REFUSED: noncanonical GPU lock" >&2; exit 2 ;;
esac
if [[ $LOCK_OWNER == internal-canonical ]]; then
    exec 9>"$GPU_LOCK"
    LOCK_FD=9
    flock -w "${MEMRA_PCG_LOCK_WAIT:-600}" "$LOCK_FD" || { echo "REFUSED: canonical GPU lock busy" >&2; exit 2; }
fi
LOCK_PROOF=$(python3 "$HERE/tier-lock-proof.py" --fd "$LOCK_FD" --lock "$GPU_LOCK" --owner "$LOCK_OWNER")
mkdir -p "$EV"
printf '%s\n' "$LOCK_PROOF" > "$EV/LOCK.json"
sha256sum "$BIN" > "$EV/binary.sha256"
SERVER_PID=""
DISCONNECT_MS=${MEMRA_PCG_DISCONNECT_MS:-300}
WORDS=${MEMRA_PCG_WORDS:-6000}
boot() { # $1 log
    memra_port_guard prime-cancel-gate "$PORT" MEMRA_GATE_PORT || return 1
    if curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
        echo "port $PORT already serving, refusing to boot over it"
        return 1
    fi
    env CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
        MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" \
        "MEMRA_ADDR=127.0.0.1:$PORT" MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 \
        "MEMRA_PREFIX_CACHE_MB=${MEMRA_PCG_CACHE_MB:-2048}" MEMRA_KV_HOST_MB=0 \
        MEMRA_SERVE_SPEC=0 MEMRA_PREFILL_TICK=8192 MEMRA_PRIME_CHUNK=256 "$BIN" >"$1" 2>&1 &
    SERVER_PID=$!
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && return 0
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died during boot:"; tail -20 "$1"; return 1; }
        sleep 2
    done
    echo "server never became ready"
    return 1
}
stop() {
    [[ -n $SERVER_PID ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop EXIT

# One client for the whole gate: `req` (full response, digest and usage to a JSON file) and
# `disconnect` (raw socket, streaming request, closed after the delay, nothing read).
client() { # $1 mode  $2 out  [$3 delay-ms]
    python3 - "$PORT" "$WORDS" "$1" "$2" "${3:-0}" <<'PYEOF'
import hashlib, json, socket, sys, time, urllib.request
port, words, mode, out, delay_ms = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5])
W = ("river stone maple copper harbor signal ladder winter garden meadow anchor beacon canvas delta ember "
     "falcon granite hollow island jasper kettle lantern marble nickel orchid pepper quartz ribbon saddle "
     "timber umber velvet walnut yellow zephyr basket candle dagger engine fabric gutter hammer ingot jacket "
     "kernel locket magnet needle oyster pillar quiver rocket socket tablet uplink vessel window yonder "
     "zenith almond bridge cobalt dinghy pewter").split()
assert len(W) == 64
prompt = "Recite the inventory in order: " + " ".join(W[(i * 7 + 3) % 64] for i in range(words))
body = {"model": "gate", "prompt": prompt, "max_tokens": 24, "temperature": 0}
if mode == "disconnect":
    body["stream"] = True
    raw = json.dumps(body).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=30)
    head = (f"POST /v1/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n").encode()
    t0 = time.monotonic()
    s.sendall(head + raw)
    time.sleep(delay_ms / 1000.0)
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    s.close()
    json.dump({"mode": mode, "sent_bytes": len(head) + len(raw), "closed_after_ms": (time.monotonic() - t0) * 1e3,
               "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}, open(out, "w"), indent=1)
    sys.exit(0)
t0 = time.monotonic()
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                  headers={"Content-Type": "application/json"}), timeout=900)
resp = json.load(r)
text = resp["choices"][0]["text"]
usage = resp.get("usage", {})
cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", usage.get("cached_tokens", 0)) or 0
json.dump({"mode": mode, "wall_ms": (time.monotonic() - t0) * 1e3, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
           "text": text, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
           "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}, open(out, "w"), indent=1)
PYEOF
}
field() { python3 -c "import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print(v if v is not None else '')" "$1" "$2"; }

fails=0
ok() { echo "  ok: $1"; }
bad() { echo "  FAIL: $1"; fails=$((fails + 1)); }

echo "prime-cancel-gate: control boot"
boot "$EV/control-server.log" || exit 1
client req "$EV/control-r1.json"
client req "$EV/control-r2.json"
stop
C1=$(field "$EV/control-r1.json" text_sha256); C2=$(field "$EV/control-r2.json" text_sha256)
C1P=$(field "$EV/control-r1.json" prompt_tokens); C2C=$(field "$EV/control-r2.json" cached_tokens)
echo "  control: cold sha=${C1:0:16} prompt_tokens=$C1P; warm sha=${C2:0:16} cached_tokens=$C2C"
[ "$C2C" -gt 0 ] 2>/dev/null && ok "control warm arm is a prefix hit (cached_tokens=$C2C)" || bad "control warm arm did not hit (cached_tokens=$C2C); the fixture cannot prove the warm digest"
[ "$C1" = "$C2" ] && ok "control cold and warm digests agree (the hit gate's law, this fixture)" || echo "  note: control cold and warm digests differ (${C1:0:16} / ${C2:0:16}); the fault boot must reproduce both exactly"

echo "prime-cancel-gate: fault boot (disconnect after ${DISCONNECT_MS} ms)"
boot "$EV/fault-server.log" || exit 1
client disconnect "$EV/fault-r0.json" "$DISCONNECT_MS"
cancel=""
for _ in $(seq 1 120); do
    cancel=$(grep -m1 '\[prime\] cancelled at chunk' "$EV/fault-server.log" || true)
    [ -n "$cancel" ] && break
    if grep -q '\[abort\] client disconnected' "$EV/fault-server.log"; then
        # the sweep retired it: either the prime finished first (fixture) or the cancel never fired
        sleep 1
        cancel=$(grep -m1 '\[prime\] cancelled at chunk' "$EV/fault-server.log" || true)
        break
    fi
    sleep 0.5
done
abort=$(grep -m1 '\[abort\] client disconnected' "$EV/fault-server.log" || true)
echo "  server: ${cancel:-<no cancel line>}"
echo "  server: ${abort:-<no abort line>}"
if [ -z "$cancel" ]; then
    if [ -n "$abort" ] && grep -q '\[prefix-cache\] insert' "$EV/fault-server.log"; then
        echo "  fixture: prime finished before the disconnect (an insert published; raise MEMRA_PCG_WORDS or lower MEMRA_PCG_DISCONNECT_MS)"
    fi
    bad "no '[prime] cancelled at chunk' receipt (the cancellation point did not fire)"
else
    K=$(sed -E 's/.*cancelled at chunk ([0-9]+) \(([0-9]+) of ([0-9]+) rows.*/\1 \2 \3/' <<<"$cancel")
    read -r k r t <<<"$K"
    [ "$r" -lt "$t" ] && ok "stopped at chunk $k after $r of $t rows (within one chunk of the disconnect, before the take ended)" || bad "cancel receipt rows $r not below take $t"
    [ -n "$abort" ] && ok "session retired as a client abort (no park, cache released)" || bad "no '[abort] client disconnected' retirement after the cancel"
    if grep -q '\[prefix-cache\] insert' "$EV/fault-server.log"; then
        bad "a prefix-cache insert was published in the fault boot before the next request: $(grep -m1 '\[prefix-cache\] insert' "$EV/fault-server.log")"
    else
        ok "nothing published (no '[prefix-cache] insert' before the next request)"
    fi
    if grep -q 'prefill error' "$EV/fault-server.log"; then
        bad "the cancel surfaced as a 'prefill error' (it must be the typed abort)"
    else
        ok "no 'prefill error' path taken"
    fi
fi
client req "$EV/fault-r1.json"
client req "$EV/fault-r2.json"
stop
F1=$(field "$EV/fault-r1.json" text_sha256); F2=$(field "$EV/fault-r2.json" text_sha256)
F1P=$(field "$EV/fault-r1.json" prompt_tokens); F1C=$(field "$EV/fault-r1.json" cached_tokens); F2C=$(field "$EV/fault-r2.json" cached_tokens)
echo "  fault: cold sha=${F1:0:16} prompt_tokens=$F1P cached=$F1C; warm sha=${F2:0:16} cached_tokens=$F2C"
[ "${F1C:-0}" = "0" ] && ok "the next cold request restored nothing from the aborted prime (cached_tokens=0)" || bad "the next cold request hit a cache entry (cached_tokens=$F1C): the aborted prime published"
[ "$F1" = "$C1" ] && ok "cold digest unchanged versus the control boot" || bad "cold digest differs: control ${C1:0:16} fault ${F1:0:16}"
[ "$F2" = "$C2" ] && ok "warm digest unchanged versus the control boot" || bad "warm digest differs: control ${C2:0:16} fault ${F2:0:16}"
[ "$F1P" = "$C1P" ] && ok "prompt_tokens equal ($F1P)" || bad "prompt_tokens differ: control $C1P fault $F1P"
[ "$F2C" = "$C2C" ] && ok "warm cached_tokens equal ($F2C)" || bad "warm cached_tokens differ: control $C2C fault $F2C"

if [ "$fails" -eq 0 ]; then
    echo "PRIME-CANCEL GATE: PASS (disconnect_ms=$DISCONNECT_MS words=$WORDS cold=${C1:0:16} warm=${C2:0:16})"
    exit 0
fi
echo "PRIME-CANCEL GATE: FAIL ($fails clause(s))"
exit 1
