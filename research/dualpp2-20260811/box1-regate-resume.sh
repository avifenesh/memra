#!/usr/bin/env bash
# dualpp2 RE-GATE CONTINUATION after the vacuous-canary false-fail.
# Root cause (fixed at source): the servestress reducer + soak assert_clean matched benign
# peer-probe telemetry `mismatches=0` via a case-insensitive `MISMATCH` token. Corrected greps
# (case-sensitive MISMATCH sentinel + `mismatches=[1-9]` for real nonzero counts + narrowed
# `illegal`) are canary-verified: 0 on the clean captured log, 7/7 on injected faults, 0 on
# benign telemetry alone.
#
# Correctness (STAGE_CORRECTNESS_PASS 19:12) + servestress raw data are already GREEN and captured:
# serial/dual/teeth each 64/64 completed, worker alive, dual_adds_no_thrash=True (admission
# counters 0/0/0 serial==dual = blocker #6 live PASS), dual-pp marker present. We do NOT re-run
# those (proven; re-running burns time + spot risk). This driver:
#   1. Re-reduces the CAPTURED servestress logs with the fixed grep -> true PASS summary.
#   2. Runs the SOAK (the only stage that never ran) with the fixed assert_clean.
# Reuses the fresh release binaries from the aborted run (HEAD 64a86925 clean, GPU idle, verified).
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp2}
ROOT=$REPO/research/dualpp2-20260811/raw/box1-regate
SS_OUT=$ROOT/servestress
NCLIENTS=64

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
git diff --quiet && git diff --cached --quiet

exec 9>/tmp/memra-gpu.lock
flock -w 3600 9 || { echo "FAIL: GPU lock timeout"; exit 75; }
export DUALPP_LOCK_HELD=1
echo "DUALPP2_RESUME_LOCK_ACQUIRED $(date -u +%FT%TZ) pid=$$"

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
test -z "$apps" || { echo "$apps"; echo "FAIL: not GPU-idle"; exit 1; }

# --- STAGE A: re-reduce captured servestress with the FIXED grep ---
echo "RESUME_REDUCE_START $(date -u +%FT%TZ)"
# Recompute per-arm badlog with the corrected pattern, overwrite the *-badlog files, then
# re-run the same reducer the servestress script uses (it reads the badlog files + metrics json).
for arm in serial dual teeth; do
    slog="$SS_OUT/$arm-server.log"
    { grep -aiE "panicked at|CUDA_ERROR|out of memory|SIGSEGV|illegal memory access|ILLEGAL_ADDRESS|same boundary slot|worker.*died|mismatches=[1-9]" "$slog" || true
      grep -aE "MISMATCH" "$slog" || true; } | head -5 > "$SS_OUT/$arm-badlog"
    echo "  $arm badlog lines: $(wc -l < "$SS_OUT/$arm-badlog")"
done
mv "$SS_OUT/summary.json" "$SS_OUT/summary.json.false-fail" 2>/dev/null || true
python3 - "$SS_OUT" "$NCLIENTS" "$EXPECTED_SOURCE" <<'PY' | tee "$SS_OUT/reduce-fixed.log"
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
           "rig":"box1, 2x RTX PRO 6000 Blackwell Server Edition","n_clients":n,
           "note_reduction":"re-reduced with canary-fixed bad-log grep (mismatches=0 no longer false-matches MISMATCH)","arms":{}}
ok = True
for a in ("serial","dual"):
    a_ok = rc(a) == 0 and alive(a) and not badlog(a)
    verdict["arms"][a] = {"completed_all": rc(a)==0, "worker_alive": alive(a),
                          "log_clean": not badlog(a), "badlog": badlog(a)[:200],
                          "admission_delta": delta(a), "PASS": a_ok}
    ok = ok and a_ok
t_rc, t_alive, t_bad, t_delta = rc("teeth"), alive("teeth"), badlog("teeth"), delta("teeth")
t_failed = (t_rc != 0) or (not t_alive) or bool(t_bad)
t_bound = t_delta["admission_vram_defers"] > 0 or t_delta["admission_session_defers"] > 0 or t_delta["step_oom_parks"] > 0
verdict["arms"]["teeth"] = {"completed_all": t_rc==0, "worker_alive": t_alive, "log_clean": not t_bad,
    "admission_delta": t_delta, "inverted_failed": t_failed, "admission_bound": t_bound,
    "note": ("teeth bound and inverted-failed as designed" if (t_bound and t_failed)
             else "teeth non-binding on 192GB PRO pair at c=64 (defers==0) — admission math not exercised here; not a lane failure"
                  if not t_bound else
             "teeth bound but did NOT fail — admission cost model not measured, INVESTIGATE")}
sd, dd = delta("serial"), delta("dual")
thrash = {k: {"serial": sd[k], "dual": dd[k], "dual_excess": dd[k]-sd[k]} for k in sd}
no_thrash = all(dd[k] <= sd[k] for k in sd)
verdict["admission_thrash_serial_vs_dual"] = thrash
verdict["dual_adds_no_thrash"] = no_thrash
teeth_bad = t_bound and not t_failed
verdict["PASS"] = ok and no_thrash and not teeth_bad
verdict["verdict"] = "PASS" if verdict["PASS"] else "FAIL"
(root / "summary.json").write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n")
print(json.dumps(verdict, sort_keys=True))
sys.exit(0 if verdict["PASS"] else 1)
PY
echo "RESUME_SERVESTRESS_PASS $(date -u +%FT%TZ)"

# --- STAGE B: SOAK with the fixed assert_clean ---
echo "RESUME_SOAK_START $(date -u +%FT%TZ)"
EXPECTED_SOURCE="$EXPECTED_SOURCE" DUALPP_REPO="$REPO" \
    DUALPP_SOAK_OUT="$ROOT/soak" \
    bash "$REPO/research/dualpp2-20260811/box1-soak-fixed.sh"
echo "RESUME_SOAK_PASS $(date -u +%FT%TZ)"

apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "DUALPP2_REGATE_ALL_PASS $(date -u +%FT%TZ) source=$EXPECTED_SOURCE (correctness+servestress+soak; servestress re-reduced canary-fixed)"
echo "NOTE: PROGRESS step 4 (default flip + N=5 A/B) NOT run — owner-gated orchestrator promotion."
