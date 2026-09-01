#!/usr/bin/env bash
# Cross-device alternating-slot soak plus c=1..17 one-hash matrix.
set -euo pipefail

: "${EXPECTED_SOURCE:?set EXPECTED_SOURCE to the staged lane commit}"
: "${DUALPP_LOCK_HELD:?run through box1-run.sh so fd 9 owns /tmp/memra-gpu.lock}"
REPO=${DUALPP_REPO:-/home/ubuntu/memra-cx-dualpp1}
OUT=${DUALPP_SOAK_OUT:-$REPO/research/dualpp1-20260811/raw/box1/soak}
MODEL_ROOT=${DUALPP_MODEL_ROOT:-/home/ubuntu/step37/models/step-3.7-flash}
MODEL=${DUALPP_MODEL:-$MODEL_ROOT/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${DUALPP_DRAFT:-$MODEL_ROOT/Step3.7-flash-mtp-Q8_0.gguf}
GOLDEN=${DUALPP_GOLDEN:-/home/ubuntu/darktrain2/golden-response.bin}
EXPECTED_GOLDEN=21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de
SERVER=$REPO/target/release/memra-server
QOS=$REPO/research/p0iso-20260810/qos_probe.py
PORT=${DUALPP_SOAK_PORT:-18458}
BASE=http://127.0.0.1:$PORT
MODEL_NAME=step37
MIXED_WIDTHS=(2 9 16 3 15 4 14 5 13 6 12 7 11 8 10 17 2 16 5 12 7)

if ! test -e /proc/$$/fd/9 || ! flock -n 9; then
    echo "FAIL: inherited GPU lock missing"
    exit 75
fi
test ! -e "$OUT" || { echo "FAIL: output already exists: $OUT"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

server_pid=
sampler_pid=

compute_apps() {
    nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory,process_name \
        --format=csv,noheader,nounits 2>/dev/null
}

snapshot() {
    local path=$1 label=$2
    {
        echo "label=$label"
        echo "ts=$(date -u +%FT%TZ)"
        nvidia-smi \
            --query-gpu=index,name,uuid,driver_version,pstate,temperature.gpu,clocks.sm,power.draw,power.limit,memory.total,memory.used,memory.free \
            --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
            --format=csv,noheader
    } >"$path" 2>&1
}

wait_idle() {
    for _ in $(seq 1 120); do
        test -z "$(compute_apps)" && return 0
        sleep 1
    done
    compute_apps
    return 1
}

stop_sampler() {
    local pid=${sampler_pid:-}
    test -n "$pid" || return 0
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sampler_pid=
}

stop_server() {
    local pid=${server_pid:-}
    test -n "$pid" || return 0
    kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 120); do
        if ! kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
            server_pid=
            wait_idle
            return 0
        fi
        sleep 1
    done
    echo "FAIL: server $pid did not stop"
    return 1
}

cleanup() {
    stop_server || true
    stop_sampler
}
trap cleanup EXIT INT TERM

wait_ready() {
    local log=$1
    for _ in $(seq 1 900); do
        curl -sf "$BASE/readyz" >/dev/null 2>&1 && return 0
        if ! kill -0 "$server_pid" 2>/dev/null; then
            tail -200 "$log"
            return 1
        fi
        sleep 1
    done
    tail -200 "$log"
    return 1
}

assert_clean() {
    local log=$1
    if grep -Ein \
        'CUDA_ERROR|out of memory|MISMATCH|panicked at|worker.*died|server fatal|illegal|sentinel|same boundary slot' \
        "$log"; then
        return 1
    fi
}

start_server() {
    local arm=$1 label=$2 log=$3
    local -a policy=(MEMRA_DUAL_PP=0)
    if [[ $arm == dual ]]; then
        policy=(MEMRA_DUAL_PP=1 MEMRA_PP_OVERLAP=1)
    fi
    env -u MEMRA_DUAL_PP -u MEMRA_PP_OVERLAP -u MEMRA_PP_HOST_BOUNCE \
        -u MEMRA_DUAL_PP_TIMING -u MEMRA_SPEC_K -u MEMRA_SERVE_BATCH -u MEMRA_BG_JOB \
        -u MEMRA_DECODE_BATCH_CAP "${policy[@]}" \
        CUDA_VISIBLE_DEVICES=0,1 MEMRA_MODELS="$MODEL_NAME=$MODEL+$DRAFT" \
        MEMRA_COMPAT=openai MEMRA_ADDR="127.0.0.1:$PORT" MEMRA_SERVE_SPEC=0 \
        MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_CTX=262144 \
        MEMRA_MOE_GROUPED=1 MEMRA_PREFILL_TICK=2048 MEMRA_PREFIX_CACHE_MB=0 \
        MEMRA_MAX_SESSIONS=64 MEMRA_LANE_MAX_JUDGE=64 MEMRA_LANE_MAX_HARVEST=64 \
        MEMRA_SLO_P99_MS=1000000 MEMRA_TAG="dualpp1-soak-$label" \
        "$SERVER" >"$log" 2>&1 &
    server_pid=$!
    wait_ready "$log"
}

run_qos() {
    local arm=$1 label=$2 width=$3
    local dir=$OUT/$label
    mkdir -p "$dir"
    "$QOS" --base "$BASE" --model "$MODEL_NAME" --label "$label" \
        --requests "$width" --max-tokens 64 --golden "$GOLDEN" \
        --lanes interactive,judge,harvest \
        --rows "$dir/qos-rows.jsonl" --summary "$dir/qos-summary.json" \
        2>&1 | tee "$dir/qos.log"
    grep -q '"exactness": "match"' "$dir/qos-summary.json"
    grep -q "\"golden_matches\": $width" "$dir/qos-summary.json"
    echo "soak_point=$label arm=$arm c=$width result=PASS"
}

for artifact in "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS"; do
    test -f "$artifact" || { echo "FAIL: missing $artifact"; exit 1; }
done

cd "$REPO"
test "$(git rev-parse HEAD)" = "$EXPECTED_SOURCE"
test "$(sha256sum "$GOLDEN" | awk '{print $1}')" = "$EXPECTED_GOLDEN"
sha256sum "$MODEL" "$DRAFT" "$GOLDEN" "$SERVER" "$QOS" >"$OUT/SHA256SUMS"
snapshot "$OUT/nvidia-smi-before.log" soak-start
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: box1 not GPU-idle"; exit 1; }

nvidia-smi \
    --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader,nounits -lms 250 >"$OUT/gpu.csv" 2>&1 &
sampler_pid=$!

# Ten fresh processes, alternating arms. Boots 1/2 are the complete c=1..17 one-hash matrix;
# every later boot runs 21 rotated mixed widths. That yields N=101 points per arm and 101 live
# dual points across five independent CUDA contexts, satisfying the x100 collision-soak bar.
for boot in $(seq 1 10); do
    if (( boot % 2 == 1 )); then arm=serial; else arm=dual; fi
    boot_label=$(printf 'boot-%02d-%s' "$boot" "$arm")
    server_log=$OUT/$boot_label-server.log
    echo "boot_start=$boot_label ts=$(date -u +%FT%TZ)"
    snapshot "$OUT/$boot_label-thermal-before.log" "$boot_label-before"
    start_server "$arm" "$boot_label" "$server_log"
    curl -sf "$BASE/metrics" >"$OUT/$boot_label-metrics-before.json"

    if (( boot <= 2 )); then
        for width in $(seq 1 17); do
            label=$(printf '%s-matrix-c%02d' "$boot_label" "$width")
            run_qos "$arm" "$label" "$width"
        done
    else
        rotate=$(( (boot - 3) % ${#MIXED_WIDTHS[@]} ))
        for point in $(seq 0 20); do
            index=$(( (point + rotate) % ${#MIXED_WIDTHS[@]} ))
            width=${MIXED_WIDTHS[$index]}
            label=$(printf '%s-soak-p%02d-c%02d' "$boot_label" "$((point + 1))" "$width")
            run_qos "$arm" "$label" "$width"
        done
    fi

    curl -sf "$BASE/metrics" >"$OUT/$boot_label-metrics-after.json"
    stop_server
    assert_clean "$server_log"
    if [[ $arm == dual ]]; then
        grep -q 'decode wave cap 8; scheduler tick cap 16 (dual PP, default-off arm)' "$server_log"
        grep -q '\[dual-pp\] dual-active PP-2 decode engaged' "$server_log"
    else
        grep -q 'decode wave cap 8; scheduler tick cap 8' "$server_log"
        if grep -q '\[dual-pp\]' "$server_log"; then
            echo "FAIL: dual marker present in serial arm $boot_label"
            exit 1
        fi
    fi
    snapshot "$OUT/$boot_label-thermal-after.log" "$boot_label-after"
    echo "boot_done=$boot_label ts=$(date -u +%FT%TZ)"
done

stop_sampler

python3 - "$OUT" "$EXPECTED_GOLDEN" "$EXPECTED_SOURCE" <<'PY' | tee "$OUT/reduce.log"
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
source = sys.argv[3]
paths = sorted(root.glob("*/qos-summary.json"))
assert len(paths) == 202, len(paths)
rows = [json.loads(path.read_text()) for path in paths]
assert all(row["exactness"] == "match" for row in rows)
assert all(row["expected_sha256"] == expected for row in rows)
assert all(row["golden_matches"] == row["requests"] for row in rows)
assert all(row["hash_counts"] == {expected: row["requests"]} for row in rows)
assert all(set(row["lanes"]) <= {"interactive", "judge", "harvest"} for row in rows)

by_arm = {"serial": [], "dual": []}
for row in rows:
    arm = "dual" if "-dual-" in row["label"] else "serial"
    by_arm[arm].append(row)
assert len(by_arm["serial"]) == len(by_arm["dual"]) == 101

matrix = [row for row in rows if "-matrix-" in row["label"]]
assert len(matrix) == 34
for arm in ("serial", "dual"):
    widths = sorted(row["requests"] for row in matrix if f"-{arm}-" in row["label"])
    assert widths == list(range(1, 18)), (arm, widths)

slot_boots = []
for boot in (2, 4, 6, 8, 10):
    path = root / f"boot-{boot:02d}-dual-metrics-after.json"
    metrics = json.loads(path.read_text())["dual_pp"]
    pairs = int(metrics["slot_pairs"])
    uses = [int(v) for v in metrics["slot_uses"]]
    collisions = int(metrics["slot_collisions"])
    overlaps = int(metrics["overlaps"])
    assert pairs > 0, (path, metrics)
    assert uses == [pairs, pairs], (path, metrics)
    assert collisions == 0, (path, metrics)
    assert overlaps > 0, (path, metrics)
    slot_boots.append({
        "boot": boot,
        "slot_pairs": pairs,
        "slot_uses": uses,
        "slot_collisions": collisions,
        "overlaps": overlaps,
    })

temperatures = []
clocks = []
with (root / "gpu.csv").open(newline="", errors="replace") as stream:
    for row in csv.reader(stream):
        if len(row) < 7:
            continue
        try:
            temperatures.append(float(row[3].strip()))
            clocks.append(float(row[6].strip()))
        except ValueError:
            pass
assert temperatures and clocks

summary = {
    "schema": "memra.dualpp1.slot-soak.v1",
    "source_commit": source,
    "rig": "box1, 2x RTX PRO 6000 Blackwell Server Edition",
    "protocol": "10 alternating fresh boots; c1..17 matrix then rotated mixed widths; one inherited GPU lock hold",
    "arms": {
        arm: {
            "boots": 5,
            "N_points": len(points),
            "requests": sum(point["requests"] for point in points),
            "golden_matches": sum(point["golden_matches"] for point in points),
        }
        for arm, points in by_arm.items()
    },
    "one_hash_matrix": {
        "widths": list(range(1, 18)),
        "arms": ["serial", "dual"],
        "points": 34,
        "expected_sha256": expected,
        "verdict": "PASS",
    },
    "dual_slot_boots": slot_boots,
    "slot_totals": {
        "pairs": sum(row["slot_pairs"] for row in slot_boots),
        "slot_0_uses": sum(row["slot_uses"][0] for row in slot_boots),
        "slot_1_uses": sum(row["slot_uses"][1] for row in slot_boots),
        "collisions": sum(row["slot_collisions"] for row in slot_boots),
    },
    "thermal_regime": {
        "artificial_cooldown": False,
        "sample_interval_ms": 250,
        "samples": len(temperatures),
        "temperature_c_min": min(temperatures),
        "temperature_c_max": max(temperatures),
        "sm_clock_mhz_min": min(clocks),
        "sm_clock_mhz_max": max(clocks),
    },
    "verdict": "PASS",
}
(root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(json.dumps(summary, sort_keys=True))
PY

snapshot "$OUT/nvidia-smi-after.log" soak-complete
apps=$(compute_apps)
test -z "$apps" || { echo "$apps"; echo "FAIL: GPU processes remained"; exit 1; }
echo "SLOT_SOAK_PASS $(date -u +%FT%TZ)"
trap - EXIT INT TERM
