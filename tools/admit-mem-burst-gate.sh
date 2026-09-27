#!/usr/bin/env bash
# tools/admit-mem-burst-gate.sh: the memory-admission burst gate (memra#680). Under
# MEMRA_ADMIT_BY_MEMORY=1 an open-output burst used to be admitted past the door's own estimate:
# the VRAM gate compared each arrival with one live reading, which cannot see the prefill
# workspace the sessions admitted earlier in the same burst still owe (it grows at prime time),
# so the burst filled the card and requests died in prefill with CUDA_ERROR_OUT_OF_MEMORY,
# returned as 503. This gate boots the real memra-server with the door armed, releases one burst
# of open requests on a barrier, and asserts that every request is either served or refused with
# a typed 429 before prefill, and that every admission fits the door's booked reading.
#
# Verdicts (one line each, verbatim, in `$OUT/VERDICTS.txt`):
#   no prefill OOM   zero CUDA_ERROR_OUT_OF_MEMORY lines in the server log, zero 503s.
#   typed refusals   every burst status is 200 or 429, at least one 200, every 429 carries
#                    Retry-After in 1..=60, and the 429 count equals the `verdict=refuse` lines.
#   booked admits    at least one `[admit-mem] ... verdict=admit` line, every one with
#                    est_bytes <= device_free and a pending_prime field, and at least as many admit
#                    lines as 200s.
#   boot census      alive after the burst, /health 200, no `panicked`, `argmax sentinel`,
#                    `[worker] FATAL`, `[worker] respawn` or `spec verify refused` line.
#
# Usage: tools/admit-mem-burst-gate.sh <model.gguf> <memra-server binary> [out_dir]
#   AMB_PORT (8194), AMB_OPEN (8192), AMB_BURST (64), AMB_CTX (65536, MEMRA_CTX). The defaults are the
#   shape that reproduced memra#680 on the local RTX 5090 (lane B day 33, shape G2: 34 prefill-OOM 503s of 64
#   on the unfixed tree).
#   The binary is an argument, not built here, so the red arm (the unfixed tree) and the green arm
#   run the same gate. Run under the rig lock (`flock /tmp/memra-5090.lock`); the gate boots one
#   server and never takes the lock itself. Exit 0 when no verdict FAILED; 1 on any FAIL; 2 on setup.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

MODEL="${1:-}"; BIN="${2:-}"
[ -n "$MODEL" ] && [ -n "$BIN" ] || { echo "usage: $0 <model.gguf> <memra-server> [out_dir]"; exit 2; }
[ -f "$MODEL" ] || { echo "admit-mem-burst-gate: SKIP (no model at $MODEL)"; exit 0; }
[ -x "$BIN" ] || { echo "admit-mem-burst-gate: FATAL: $BIN is not an executable"; exit 2; }
PORT="${AMB_PORT:-8194}"
ADDR=127.0.0.1:$PORT
BASE=http://$ADDR
OPEN="${AMB_OPEN:-8192}"
BURST="${AMB_BURST:-64}"
CTX="${AMB_CTX:-65536}"
OUT="${3:-/tmp/admit-mem-burst-gate-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$OUT"
VERDICTS="$OUT/VERDICTS.txt"
: > "$VERDICTS"
READY_WAIT_S="${AMB_READY_WAIT_S:-420}"

. tools/port-guard.sh
memra_port_guard admit-mem-burst-gate "$PORT" AMB_PORT || exit 2

sha256sum "$BIN" > "$OUT/binary.sha256"
{ echo "commit $(git rev-parse HEAD)"; echo "model $MODEL"; echo "binary $BIN"; echo "open $OPEN burst $BURST ctx $CTX"
  nvidia-smi --query-gpu=name,driver_version,power.limit --format=csv,noheader; date -u +%FT%TZ; } > "$OUT/source.txt"

PASSN=0; FAILN=0
verdict() { # <line ending in -> PASS|FAIL>
  echo "$1" | tee -a "$VERDICTS"
  case "$1" in *"-> PASS") PASSN=$((PASSN+1));; *) FAILN=$((FAILN+1));; esac
}
gpu_apps() { nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv > "$1" 2>&1; }

SPID=0
cleanup() { [ "$SPID" != 0 ] && kill -0 "$SPID" 2>/dev/null && [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ] && kill -TERM "$SPID" 2>/dev/null; true; }
trap cleanup EXIT

echo "== admit-mem-burst-gate: open=$OPEN burst=$BURST ctx=$CTX port=$PORT out=$OUT =="
gpu_apps "$OUT/compute-apps-before.csv"
env MEMRA_COMPAT=openai MEMRA_MODELS="amb=$MODEL" MEMRA_ADDR=$ADDR MEMRA_CTX="$CTX" \
  MEMRA_ADMIT_BY_MEMORY=1 MEMRA_ADMIT_OPEN_OUTPUT_TOKENS="$OPEN" MEMRA_TIMEOUT_MS_MAX=3600000 \
  "$BIN" > "$OUT/server.log" 2>&1 &
SPID=$!
t=0
until [ "$(curl -s -o /dev/null -w '%{http_code}' -m 2 "$BASE/readyz" 2>/dev/null)" = 200 ]; do
  if ! kill -0 "$SPID" 2>/dev/null; then verdict "AMB boot: server died during boot -> FAIL"; tail -n 5 "$OUT/server.log"; exit 1; fi
  t=$((t+1)); [ "$t" -gt $((READY_WAIT_S*10)) ] && { verdict "AMB boot: not ready within ${READY_WAIT_S}s -> FAIL"; exit 1; }
  sleep 0.1
done
memra_port_owned admit-mem-burst-gate "$PORT" "$SPID" || exit 2
echo "  ready (pid $SPID); releasing the burst"
python3 tools/admit-mem-burst-client.py "$BASE" amb "$BURST" "$OUT/burst.jsonl" > "$OUT/client.log" 2>&1

alive=0; kill -0 "$SPID" 2>/dev/null && alive=1
health=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$BASE/health" 2>/dev/null); health=${health:-000}
if kill -0 "$SPID" 2>/dev/null && [ "$(readlink /proc/$SPID/cwd 2>/dev/null)" = "$PWD" ]; then kill -TERM "$SPID"; fi
wait "$SPID" 2>/dev/null; echo "$?" > "$OUT/exit"; SPID=0
gpu_apps "$OUT/compute-apps-after.csv"

python3 - "$OUT" "$alive" "$health" <<'PY' > "$OUT/reading.txt"
import json, re, sys
out, alive, health = sys.argv[1], sys.argv[2], sys.argv[3]
rows = [json.loads(l) for l in open(f"{out}/burst.jsonl")]
log = open(f"{out}/server.log", encoding="utf-8", errors="replace").read().splitlines()
st = {}
for r in rows:
    st[r["status"]] = st.get(r["status"], 0) + 1
oom = sum("CUDA_ERROR_OUT_OF_MEMORY" in l for l in log)
crash = {k: sum(k in l for l in log) for k in ("panicked", "argmax sentinel", "[worker] FATAL", "[worker] respawn",
                                               "spec verify refused")}
kv = lambda l: dict(re.findall(r"(\w+)=(\S+)", l))
admits = [kv(l) for l in log if "[admit-mem] id=" in l and "verdict=admit " in l]
refuse = sum(1 for l in log if "[admit-mem] id=" in l and "verdict=refuse " in l)
bad = sum(1 for a in admits if not (a.get("est_bytes", "").isdigit() and a.get("device_free", "").isdigit()
                                    and int(a["est_bytes"]) <= int(a["device_free"]) and "pending_prime" in a))
n429 = [r for r in rows if r["status"] == 429]
ra = all(str(r.get("retry_after") or "").isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in n429)
other = sum(v for k, v in st.items() if k not in (200, 429))
def v(ok): return "PASS" if ok else "FAIL"
print(f"AMB no prefill OOM: oom_lines={oom} status={st} -> {v(oom == 0 and st.get(503, 0) == 0)}")
print(f"AMB typed refusals: served={st.get(200, 0)} r429={len(n429)} refuse_lines={refuse} retry_after_in_1_60={ra} "
      f"other_non200={other} -> {v(st.get(200, 0) >= 1 and other == 0 and ra and len(n429) == refuse)}")
print(f"AMB booked admits: admit_lines={len(admits)} est_over_booked_free={bad} served={st.get(200, 0)} "
      f"-> {v(len(admits) >= 1 and bad == 0 and len(admits) >= st.get(200, 0))}")
print(f"AMB boot census: alive={alive} health={health} " + " ".join(f"{k.replace(' ', '_')}={n}" for k, n in crash.items())
      + f" -> {v(alive == '1' and health == '200' and not any(crash.values()))}")
PY
while IFS= read -r line; do verdict "$line"; done < "$OUT/reading.txt"

echo "== admit-mem-burst-gate: $PASSN PASS, $FAILN FAIL =="
if [ "$FAILN" = 0 ] && [ "$PASSN" -gt 0 ]; then echo "ADMIT-MEM BURST GATE: ALL GREEN"; exit 0; fi
echo "ADMIT-MEM BURST GATE: RED"; exit 1
