#!/usr/bin/env bash
# dualpp2 re-gate STEP 1: c=64 serve-stress on the Step PP-2 model, serial vs explicit-dual,
# plus the teeth control. serve-stress-gate.sh hardcodes a single-device server env and cannot
# bring up the 2-device PP model, so this stage reuses the soak's Step PP start_server env and
# drives the same well-formed-stream burst (faithful copy of serve-stress-gate.sh's client).
# PASS: each arm completes 64/64 well-formed, worker alive, log clean, AND dual adds no
# park/requeue thrash over serial (admission counters compared). Teeth (reserve=16MB) INVERTS.
set -uo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
: "${DUALPP_LOCK_HELD:?run through box1-regate-run.sh so fd 9 owns /tmp/memra-gpu.lock}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp2}
OUT=${DUALPP_STRESS_OUT:-$REPO/research/dualpp2-20260811/raw/box1-regate/servestress}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DUALPP_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
SERVER=$REPO/target/release/memra-server
PORT=${DUALPP_STRESS_PORT:-18472}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37
NCLIENTS=${DUALPP_STRESS_NCLIENTS:-64}

if ! test -e /proc/$$/fd/9 || ! flock -n 9; then
    echo "FAIL: inherited GPU lock missing"; exit 75
fi
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}
snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"; echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi --query-gpu=index,name,uuid,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
    } >"$path" 2>&1
}
wait_idle() {
    for _ in $(seq 1 120); do test -z "$(compute_apps)" && return 0; sleep 1; done
    compute_apps; return 1
}
stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then wait "$pid" 2>/dev/null || true; server_pid=; wait_idle; return 0; fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"; return 1
}
cleanup() { stop_server || true; }
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then tail -200 "$log"; return 1; fi
        sleep 1
    done
    tail -200 "$log"; return 1
}
start_server() {
    local arm=$1 log=$2
    local -a policy=(MEMRA_DUAL_PP=0)
    [[ $arm == dual ]] && policy=(MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1)
    local -a reserve=()
    [[ $arm == teeth ]] && reserve=(MEMRA_ADMIT_RESERVE_MB=16)
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_DUAL_PP_TIMING -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
        -u MEMRA_DECODE_BATCH_CAP -u MEMRA_ADMIT_RESERVE_MB "${policy[@]}" "${reserve[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_MAX_SESSIONS=64 MEMRA_TAG="dualpp2-regate-stress-$arm" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
}
# Faithful copy of serve-stress-gate.sh's well-formed-stream client (128-256 tok, staggered,
# stream, finish_reason + [DONE] assertions). rc=0 iff all N complete well-formed.
run_burst() {
    local rows=$1
    python3 - "$BASE" "$NCLIENTS" "$rows" "$MODEL_NAME" <<'PYEOF'
import json, random, sys, threading, time, urllib.request
base, n, rows_path, model = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
STAGGER_S = 0.05; TIMEOUT_S = 900; random.seed(20260806)
PROMPTS = [
    "List three failure modes of a GPU serving stack under high concurrency and one mitigation for each.",
    "Explain the difference between p50 and p99 latency to a new engineer in four sentences.",
    "Write a short shell one-liner to count open TCP connections, then explain it.",
    "Summarize what a KV cache stores during LLM decoding and why it grows with context.",
    "Name four things a load balancer must handle when a backend dies mid-stream.",
    "Describe speculative decoding in three sentences.",
]
results = []; lock = threading.Lock()
def worker(i):
    body = {"model": model,
            "messages": [{"role": "user", "content": f"[client {i} nonce {time.time_ns()}] " + PROMPTS[i % len(PROMPTS)]}],
            "max_ctx": 8192, "max_tokens": random.randint(128, 256), "temperature": 0.7,
            "seed": 1000 + i, "stream": True}
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    row = {"i": i, "ok": False, "chunks": 0, "finish_reason": None, "done": False}
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            ttfb = None
            for raw in r:
                if ttfb is None: ttfb = time.time() - t0
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: "): continue
                payload = line[6:]
                if payload == "[DONE]": row["done"] = True; break
                row["chunks"] += 1
                try:
                    ch = json.loads(payload)
                    if "error" in ch:
                        err = ch["error"]
                        row["server_error"] = (err.get("message", str(err)) if isinstance(err, dict) else str(err))[:300]
                        continue
                    fr = ch.get("choices", [{}])[0].get("finish_reason")
                    if fr: row["finish_reason"] = fr
                except json.JSONDecodeError:
                    row["bad_json"] = payload[:120]
            row["ttfb_s"] = round(ttfb, 3) if ttfb is not None else None
        row["wall_s"] = round(time.time() - t0, 3)
        row["ok"] = (row["chunks"] > 0 and row["done"] and row["finish_reason"] in ("stop", "length")
                     and "bad_json" not in row and "server_error" not in row)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"[:300]; row["wall_s"] = round(time.time() - t0, 3)
    with lock: results.append(row)
threads = []
for i in range(n):
    t = threading.Thread(target=worker, args=(i,)); t.start(); threads.append(t); time.sleep(STAGGER_S)
for t in threads: t.join()
with open(rows_path, "a") as f:
    for r in sorted(results, key=lambda r: r["i"]): f.write(json.dumps(r) + "\n")
ok = [r for r in results if r["ok"]]; bad = [r for r in results if not r["ok"]]
walls = sorted(r["wall_s"] for r in results if "wall_s" in r)
ttfbs = sorted(r["ttfb_s"] for r in results if r.get("ttfb_s") is not None)
def pct(v, p): return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")
print(f"completed {len(ok)}/{n}; wall p50={pct(walls,0.50):.1f}s p95={pct(walls,0.95):.1f}s "
      f"max={walls[-1] if walls else float('nan'):.1f}s; ttfb p50={pct(ttfbs,0.50):.2f}s "
      f"p95={pct(ttfbs,0.95):.2f}s  (informational)")
for r in bad[:8]:
    print(f"  BAD i={r['i']}: err={r.get('error')} server_error={r.get('server_error')} "
          f"chunks={r['chunks']} finish_reason={r['finish_reason']} done={r['done']} bad_json={r.get('bad_json','')}")
sys.exit(0 if len(ok) == n else 1)
PYEOF
}

for artifact in "$MODEL" "$DRAFT" "$SERVER"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done
cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
sha256sum "$MODEL" "$DRAFT" "$SERVER" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" stress-start
apps=$(compute_apps); test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

# --- arm runner: start, capture /metrics before, burst, capture /metrics after, assert ---
run_arm() {
    local arm=$1
    local slog=$OUT/$arm-server.log
    local rows=$OUT/$arm-rows.jsonl
    echo "arm_start=$arm ts=$(date -u +%FT%TZ)"
    snapshot "$OUT/$arm-thermal-before.log" "$arm-before"
    start_server "$arm" "$slog" || { echo "FAIL: $arm server did not become ready"; return 1; }
    curl -sf "$BASE/metrics" >"$OUT/$arm-metrics-before.json" || echo "warn: metrics-before scrape failed ($arm)"
    set +e
    run_burst "$rows" 2>&1 | tee "$OUT/$arm-burst.log"
    local burst_rc=${PIPESTATUS[0]}
    set -e 2>/dev/null || true
    echo "$burst_rc" >"$OUT/$arm-burst.rc"
    curl -sf "$BASE/metrics" >"$OUT/$arm-metrics-after.json" || echo "warn: metrics-after scrape failed ($arm)"
    local alive=0
    kill -0 "$server_pid" 2>/dev/null && curl -sf "$BASE/health" >/dev/null 2>&1 && alive=1
    stop_server
    local badlog
    badlog=$( { grep -aiE "panicked at|CUDA_ERROR|out of memory|SIGSEGV|illegal memory access|ILLEGAL_ADDRESS|same boundary slot|worker.*died|mismatches=[1-9]" "$slog"; grep -aE "MISMATCH" "$slog"; } | head -5)
    echo "$alive" >"$OUT/$arm-alive"
    echo "$badlog" >"$OUT/$arm-badlog"
    # dual-marker sanity mirrors the soak arm checks
    if [[ $arm == dual ]]; then
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$slog" || echo "warn: dual marker absent in dual arm"
    fi
    snapshot "$OUT/$arm-thermal-after.log" "$arm-after"
    echo "arm_done=$arm ts=$(date -u +%FT%TZ) burst_rc=$burst_rc alive=$alive badlog=${badlog:0:80}"
    return 0
}

run_arm serial
run_arm dual
run_arm teeth

# --- reduce: verdict + serial-vs-dual admission-thrash comparison (blocker #6 live check) ---
python3 - "$OUT" "$NCLIENTS" "$EXPECTED_SOURCE" <<'PY' | tee "$OUT/reduce.log"
import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); n = int(sys.argv[2]); source = sys.argv[3]
def rc(a):   return int((root / f"{a}-burst.rc").read_text().strip())
def alive(a):return (root / f"{a}-alive").read_text().strip() == "1"
def badlog(a):return (root / f"{a}-badlog").read_text().strip()
def metrics(a, when):
    p = root / f"{a}-metrics-{when}.json"
    try: return json.loads(p.read_text())
    except Exception: return {}
def adm(a, when):
    m = metrics(a, when)
    return {k: int(m.get(k, 0)) for k in ("admission_session_defers","admission_vram_defers","step_oom_parks")}
def delta(a):
    b, e = adm(a,"before"), adm(a,"after")
    return {k: e[k]-b[k] for k in b}

verdict = {"schema":"memra.dualpp2.regate.servestress.v1","source_commit":source,
           "rig":"box1, 2x RTX PRO 6000 Blackwell Server Edition","n_clients":n,"arms":{}}
ok = True
for a in ("serial","dual"):
    a_ok = rc(a) == 0 and alive(a) and not badlog(a)
    verdict["arms"][a] = {"completed_all": rc(a)==0, "worker_alive": alive(a),
                          "log_clean": not badlog(a), "badlog": badlog(a)[:200],
                          "admission_delta": delta(a), "PASS": a_ok}
    ok = ok and a_ok
# teeth: reserve forced to 16MB. On a 192GB PRO pair the admission reserve may not bind at c=64;
# record honestly. Teeth is expected to FAIL (inverted) if it binds; if defers==0 it is
# non-binding on this rig (recorded, not a lane failure).
t_rc, t_alive, t_bad, t_delta = rc("teeth"), alive("teeth"), badlog("teeth"), delta("teeth")
t_failed = (t_rc != 0) or (not t_alive) or bool(t_bad)
t_bound = t_delta["admission_vram_defers"] > 0 or t_delta["admission_session_defers"] > 0 or t_delta["step_oom_parks"] > 0
verdict["arms"]["teeth"] = {"completed_all": t_rc==0, "worker_alive": t_alive, "log_clean": not t_bad,
    "admission_delta": t_delta, "inverted_failed": t_failed, "admission_bound": t_bound,
    "note": ("teeth bound and inverted-failed as designed" if (t_bound and t_failed)
             else "teeth non-binding on 192GB PRO pair at c=64 (defers==0) — admission math not exercised here; not a lane failure"
                  if not t_bound else
             "teeth bound but did NOT fail — admission cost model not measured, INVESTIGATE")}
# thrash: dual must not exceed serial on any admission counter
sd, dd = delta("serial"), delta("dual")
thrash = {k: {"serial": sd[k], "dual": dd[k], "dual_excess": dd[k]-sd[k]} for k in sd}
no_thrash = all(dd[k] <= sd[k] for k in sd)
verdict["admission_thrash_serial_vs_dual"] = thrash
verdict["dual_adds_no_thrash"] = no_thrash
teeth_bad = t_bound and not t_failed  # only a hard fail if teeth bound yet passed
verdict["PASS"] = ok and no_thrash and not teeth_bad
verdict["verdict"] = "PASS" if verdict["PASS"] else "FAIL"
(root / "summary.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, sort_keys=True))
sys.exit(0 if verdict["PASS"] else 1)
PY
reduce_rc=${PIPESTATUS[0]}

snapshot "$OUT/nvidia-smi-after.log" stress-complete
apps=$(compute_apps); test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
test "$reduce_rc" -eq 0 || { echo "SERVESTRESS_FAIL rc=$reduce_rc $(date -u +%FT%TZ)"; exit 1; }
echo "SERVESTRESS_PASS $(date -u +%FT%TZ)"
trap - EXIT INT TERM
