#!/usr/bin/env bash
# Day 19 cell (lane/spill-c-20260919, research/spill-c-20260919/DAY19.md, pre-registered before any run).
# One collector lock hold (tools/tier-battery.py --external-lock passes the lock fd as $2).
# Cell serverdoor19: the day-18 `serverdoor` shape on the memra#617 fixed tree, three boots in one hold:
#   flag   : memra-server with `--experts-via-tier` on its argv (the day-18 shape; must now be refused)
#   envon  : memra-server without the flag, MEMRA_KV_HOST_CONTRACTS=1 (the door as an environment variable)
#   envbad : memra-server without the flag, MEMRA_KV_HOST_CONTRACTS=on (the documented parse refusal)
# Cell serverdoor19b (pre-registered after serverdoor19's result, DAY19.md): the same three arms with a host
# tier budget in every arm (MEMRA_KV_HOST_MB=$D19_HOST_MB), the shape the door's ON boot line needs.
# Environment (set by the driver, never hosts or ids): D19_R receipts root, D19_OUT the collector's --out dir
# (ev/ lands inside it; serverdoor19 wrote ev/ under $D19_R/<cell>/ and it was relocated by hand), D19_BINS
# binary dir, D19_TREE worktree, D19_ART artifact, D19_LOCK the rig lock path, D19_PORT server port,
# D19_MOE_ENV extra env, D19_HOST_MB the host tier budget (serverdoor19b only).
# usage: day19-cell.sh serverdoor19|serverdoor19b <lockfd>
set -uo pipefail
cell=$1; fd=$2
: "${D19_R:?}" "${D19_BINS:?}" "${D19_TREE:?}" "${D19_LOCK:?}" "${D19_ART:?}" "${D19_PORT:?}"
EV=${D19_OUT:-$D19_R/$cell}/ev
mkdir -p "$EV"
cd "$D19_TREE" || exit 1
python3 tools/tier-lock-proof.py --fd "$fd" --lock "$D19_LOCK" --owner collector > "$EV/LOCK.json"
git rev-parse HEAD | tee "$EV/tree.sha"
sha256sum "$D19_BINS/memra-server" | tee "$EV/binary.sha256"
sha256sum "$D19_ART" | tee "$EV/artifact.sha256"
mark() { printf '%s\t%s\n' "$(date -u +%FT%T.%3NZ)" "$1" >> "$EV/marks.tsv"; }
: > "$EV/marks.tsv"
case $cell in
serverdoor19) HOST_ENV=() ;;
serverdoor19b) : "${D19_HOST_MB:?}"; HOST_ENV=("MEMRA_KV_HOST_MB=$D19_HOST_MB") ;;
*) echo "unknown cell $cell"; exit 2 ;;
esac
. tools/port-guard.sh
memra_port_guard day19-serverdoor "$D19_PORT" D19_PORT || exit 1
if curl -s --max-time 1 "http://127.0.0.1:$D19_PORT/v1/models" >/dev/null 2>&1; then
    echo "port $D19_PORT already serving, refusing to boot over it"; exit 1
fi
# boot <label> <door-env-value|-> <argv...>: one server boot, readiness polled on /v1/models (up to 480 s),
# one completion when ready, TERM, exit code banked. A refusal exits on its own before readiness.
boot() {
    local label=$1 door=$2; shift 2
    local log=$EV/$label.log
    mark "$label boot"
    local -a denv=()
    [ "$door" != "-" ] && denv=("MEMRA_KV_HOST_CONTRACTS=$door")
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES=0 MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$D19_ART" "MEMRA_ADDR=127.0.0.1:$D19_PORT" \
        MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4 ${D19_MOE_ENV:-} "${HOST_ENV[@]}" "${denv[@]}" "$D19_BINS/memra-server" "$@" >"$log" 2>&1 &
    local pid=$!
    local ready=0
    for _ in $(seq 1 240); do
        curl -s --max-time 2 "http://127.0.0.1:$D19_PORT/v1/models" >/dev/null 2>&1 && { ready=1; break; }
        kill -0 "$pid" 2>/dev/null || break
        sleep 2
    done
    echo "ready=$ready" | tee "$EV/$label.ready"
    mark "$label ready=$ready"
    if [ "$ready" = 1 ]; then
        python3 - "$D19_PORT" "$EV/$label.r1.json" <<'PYEOF'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
body = {"model": "gate", "prompt": "Name three primary colors, comma-separated.", "max_tokens": 16, "temperature": 0}
r = urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600)
json.dump(json.load(r), open(out, "w"), indent=1)
PYEOF
        echo "request rc=$?" | tee "$EV/$label.request"
        mark "$label r1 done"
        kill -TERM "$pid" 2>/dev/null || true
    fi
    for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -KILL "$pid" 2>/dev/null || true
    wait "$pid"; local rc=$?
    echo "$rc" > "$EV/$label.exit"
    mark "$label stopped rc=$rc"
}
boot flag   -  --experts-via-tier
boot envon  1
boot envbad on
echo "$cell cell done: flag rc=$(cat "$EV/flag.exit") envon rc=$(cat "$EV/envon.exit") envbad rc=$(cat "$EV/envbad.exit")"
