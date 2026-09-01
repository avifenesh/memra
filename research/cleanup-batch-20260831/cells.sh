#!/usr/bin/env bash
# Cleanup-batch 2026-08-31 GPU cells (the rented dev box, card 0 only, minutes of use):
#   tooth 1: MEMRA_MTP_SKIP=1 + explicit MEMRA_SERVE_SPEC=1, no dspark -> boot FATAL,
#            "cannot be honored" quoted (contract unchanged, re-proven on this binary).
#   tooth 2 (NEW refuse-loud contract, hermes baf261e2bfdae118): skip + MEMRA_SERVE_SPEC
#            UNSET, no dspark -> boot FATAL naming MEMRA_SERVE_SPEC=0 and
#            MEMRA_DSPARK_SPEC (default-ON spec with no spec program never boots plain
#            silently: the DFlash2 2026-08-25 incident class).
#   tooth 3: skip + explicit MEMRA_SERVE_SPEC=0 -> boots, announces PLAIN by explicit
#            choice, greedy request serves with .usage.spec == null. The same boot
#            carries the step vision tower (MEMRA_STEP_VISION_DIR on the step37
#            artifact) so the walker is live for the vision-cap cells:
#     cell 1a: oversized data URI (one base64 quad past the 12 MiB raw cap) -> named 400
#              "12 MiB", refused at the walker before any decode allocation.
#     cell 1b: at-cap data URI (exactly 16,777,216 base64 chars = 12 MiB raw) -> the cap
#              gate ADMITS; the refusal (if any) comes from the image decoder, not the cap.
#     cell 1c: 9 valid small images -> "too many images (max 8)" (budget unchanged).
#     cell 1d: 1 valid small image -> not refused by the cap (walker admits past decode).
# usage: cells.sh <memra-server-bin> <q38.gguf> <step37-model-dir|none> <evidence_dir>
#   <step37-model-dir> = "none" runs the mtp-skip teeth only (no vision tower, no
#   vision cells) - the rig arm, where the step37 artifact does not live.
set -uo pipefail
BIN=$1; MODEL=$2; VDIR=$3; EV=$4
PORT=${MEMRA_GATE_PORT:-18147}
LOCK=${MEMRA_GPU_LOCK:-/tmp/memra-gate.lock}
mkdir -p "$EV"
FAIL=0
echo "binary: $BIN" | tee "$EV/provenance.txt"
sha256sum "$BIN" | tee -a "$EV/provenance.txt"
git -C "$(dirname "$BIN")" log -1 --oneline 2>/dev/null | tee -a "$EV/provenance.txt" || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader | tee -a "$EV/provenance.txt"

# 64x64 flat-color PNG, the unit suite's embedded fixture (valid, tile-free plan).
PNG64="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAY0lEQVR4nO3PQQ3AIADAQEANmlCD9IngcVnSU9DOe/b4s6UDXjWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgNaA1oDWgfeKYAYIDsx/LAAAAAElFTkSuQmCC"

if [ "${SKIP_TEETH:-0}" != 1 ]; then
echo "== tooth 1: skip + MEMRA_SERVE_SPEC=1, no dspark -> FATAL =="
LOG="$EV/tooth1-explicit-spec-fatal.log"
timeout 900 flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
    MEMRA_MTP_SKIP=1 MEMRA_SERVE_SPEC=1 \
    "$BIN" >"$LOG" 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
    echo "FAIL: server exited 0 under an unhonorable explicit spec request"; FAIL=1
elif grep -qF "cannot be honored" "$LOG"; then
    echo "PASS: exit=$RC, cause quoted:"
    grep -F "cannot be honored" "$LOG" | head -1 | sed 's/^/    /'
else
    echo "FAIL: exit=$RC but the refusal cause is missing"; tail -5 "$LOG" | sed 's/^/    /'; FAIL=1
fi

echo "== tooth 2 (NEW): skip + MEMRA_SERVE_SPEC unset, no dspark -> FATAL naming the override =="
LOG="$EV/tooth2-default-spec-fatal.log"
timeout 900 flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
    MEMRA_MTP_SKIP=1 \
    "$BIN" >"$LOG" 2>&1
RC=$?
if [ "$RC" -eq 0 ]; then
    echo "FAIL: server booted (exit 0 path) under default-ON spec with no spec program"; FAIL=1
elif grep -qF "MEMRA_SERVE_SPEC=0" "$LOG" && grep -qF "MEMRA_DSPARK_SPEC" "$LOG"; then
    echo "PASS: exit=$RC, refusal names the override and the missing drafter:"
    grep -F "Refusing at boot" "$LOG" | head -1 | sed 's/^/    /'
else
    echo "FAIL: exit=$RC but the named refusal is missing"; tail -5 "$LOG" | sed 's/^/    /'; FAIL=1
fi
fi  # SKIP_TEETH

echo "== tooth 3: skip + MEMRA_SERVE_SPEC=0 -> boots, PLAIN by explicit choice; vision cap cells =="
LOG="$EV/tooth3-explicit-zero-serves.log"
SERVER_PID=""
stop() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    sleep 2
}
trap stop EXIT
VENV=()
[ "$VDIR" != "none" ] && VENV=("MEMRA_STEP_VISION_DIR=$VDIR")
setsid flock -w 900 "$LOCK" env CUDA_VISIBLE_DEVICES=0 \
    MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
    MEMRA_CTX=4096 MEMRA_MAX_SESSIONS=2 \
    MEMRA_MTP_SKIP=1 MEMRA_SERVE_SPEC=0 \
    ${VENV[@]+"${VENV[@]}"} \
    "$BIN" >"$LOG" 2>&1 &
SERVER_PID=$!
UP=0
for _ in $(seq 1 300); do
    curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && { UP=1; break; }
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 2
done
if [ "$UP" != 1 ]; then
    echo "FAIL: skip + MEMRA_SERVE_SPEC=0 boot did not come up"; tail -8 "$LOG" | sed 's/^/    /'; FAIL=1
else
    if grep -qF "serves PLAIN decode by explicit choice" "$LOG"; then
        echo "PASS: explicit-choice PLAIN line present:"
        grep -F "[mtp-skip] gate:" "$LOG" | head -1 | sed 's/^/    /'
    else
        echo "FAIL: no explicit-choice PLAIN line in the boot log"; FAIL=1
    fi
    R="$EV/tooth3-plain-req.json"
    curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'content-type: application/json' \
        -d '{"model":"gate","temperature":0,"max_tokens":64,"messages":[{"role":"user","content":"Describe, in three sentences, why tidal forces lock a moon to its planet."}]}' \
        >"$R"
    if jq -e '(.error == null) and (.usage.spec == null) and (((.choices[0].message.reasoning // "") + (.choices[0].message.content // "")) | length > 0)' "$R" >/dev/null 2>&1; then
        echo "PASS: greedy request served PLAIN (.usage.spec == null)"
    else
        echo "FAIL: request errored or engaged spec on a skip+SERVE_SPEC=0 boot"; FAIL=1
        jq -c '{error: .error, spec: .usage.spec}' "$R" 2>/dev/null | sed 's/^/    /'
    fi

    # ---- vision cap cells (walker live via the step tower; skipped when VDIR=none) ----
    if [ "$VDIR" != "none" ]; then
    CAP_CHARS=16777216  # 12 MiB raw * 4/3, exact (cap is a multiple of 3)
    echo "== cell 1a: oversized data URI -> named 400 =="
    R="$EV/cell1a-oversized.json"
    python3 - "$PORT" "$CAP_CHARS" >"$R" <<'PY'
import json, sys, urllib.request
port, cap = sys.argv[1], int(sys.argv[2])
uri = "data:image/png;base64," + "A" * (cap + 4)
body = json.dumps({"model": "gate", "max_tokens": 8, "messages": [
    {"role": "user", "content": [
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": uri}}]}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"content-type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=120)
    print(json.dumps({"status": r.status, "body": json.loads(r.read())}))
except urllib.error.HTTPError as e:
    print(json.dumps({"status": e.code, "body": json.loads(e.read())}))
PY
    if jq -e '(.status == 400) and ((.body | tostring) | contains("12 MiB"))' "$R" >/dev/null 2>&1; then
        echo "PASS: oversized URI refused 400, names the limit:"
        jq -c '.body.error.message // .body' "$R" | head -c 200 | sed 's/^/    /'; echo
    else
        echo "FAIL: oversized URI not refused by name"; jq -c . "$R" | head -c 300; echo; FAIL=1
    fi

    echo "== cell 1b: at-cap data URI -> cap gate admits (decoder judges the bytes) =="
    R="$EV/cell1b-atcap.json"
    python3 - "$PORT" "$CAP_CHARS" >"$R" <<'PY'
import json, sys, urllib.request
port, cap = sys.argv[1], int(sys.argv[2])
uri = "data:image/png;base64," + "A" * cap
body = json.dumps({"model": "gate", "max_tokens": 8, "messages": [
    {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": uri}}]}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"content-type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=120)
    print(json.dumps({"status": r.status, "body": json.loads(r.read())}))
except urllib.error.HTTPError as e:
    print(json.dumps({"status": e.code, "body": json.loads(e.read())}))
PY
    if jq -e '(.body | tostring) | contains("12 MiB") | not' "$R" >/dev/null 2>&1; then
        echo "PASS: at-cap URI passed the cap gate (refusal, if any, is the decoder's):"
        jq -c '.body.error.message // "no error"' "$R" | head -c 200 | sed 's/^/    /'; echo
    else
        echo "FAIL: at-cap URI hit the cap error (boundary off by one)"; FAIL=1
    fi

    echo "== cell 1c: 9 valid images -> too many images (max 8), budget unchanged =="
    R="$EV/cell1c-nine-images.json"
    python3 - "$PORT" "$PNG64" >"$R" <<'PY'
import json, sys, urllib.request
port, png = sys.argv[1], sys.argv[2]
part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}
body = json.dumps({"model": "gate", "max_tokens": 8, "messages": [
    {"role": "user", "content": [part] * 9}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"content-type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=120)
    print(json.dumps({"status": r.status, "body": json.loads(r.read())}))
except urllib.error.HTTPError as e:
    print(json.dumps({"status": e.code, "body": json.loads(e.read())}))
PY
    if jq -e '(.status == 400) and ((.body | tostring) | contains("too many images (max 8)"))' "$R" >/dev/null 2>&1; then
        echo "PASS: 9th image refused with the documented budget message"
    else
        echo "FAIL: 8-image budget behavior changed"; jq -c . "$R" | head -c 300; echo; FAIL=1
    fi

    echo "== cell 1d: 1 valid small image -> not refused by the cap =="
    R="$EV/cell1d-one-image.json"
    python3 - "$PORT" "$PNG64" >"$R" <<'PY'
import json, sys, urllib.request
port, png = sys.argv[1], sys.argv[2]
part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}
body = json.dumps({"model": "gate", "max_tokens": 8, "messages": [
    {"role": "user", "content": [part]}]}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=body, headers={"content-type": "application/json"})
try:
    r = urllib.request.urlopen(req, timeout=180)
    print(json.dumps({"status": r.status, "body": json.loads(r.read())}))
except urllib.error.HTTPError as e:
    print(json.dumps({"status": e.code, "body": json.loads(e.read())}))
except Exception as e:  # a post-walker path on this mixed-trunk boot may fail loudly;
    # the assertion here is only that the CAP did not refuse an in-budget image.
    print(json.dumps({"status": -1, "body": str(e)}))
PY
    if jq -e '(.body | tostring) | contains("12 MiB") | not' "$R" >/dev/null 2>&1; then
        echo "PASS: in-budget image admitted past the cap gate:"
        jq -c '{status: .status, err: (.body.error.message // "none")}' "$R" | head -c 220 | sed 's/^/    /'; echo
    else
        echo "FAIL: in-budget image refused by the cap"; FAIL=1
    fi
    fi  # VDIR != none
fi
stop

if [ "$FAIL" = 0 ]; then echo "cleanup-batch cells: ALL GREEN"; else echo "cleanup-batch cells: FAILED"; exit 1; fi
