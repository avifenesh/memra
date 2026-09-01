#!/usr/bin/env bash
# Local RTX 5090 branch-point/candidate decode-batch A/B.
#
# Each outer rep is one fresh process. The binary performs one discarded warmup and one
# timed rep per B; outer reps alternate arm order under a single GPU lock and thermal sample.
set -euo pipefail

: "${MODEL:?set the 9B GGUF under test}"
: "${OUT:?set the raw-output directory}"
: "${BASE_SOURCE_DIR:?set the detached branch-point worktree}"
: "${BASE_TARGET_DIR:?set the branch-point-only Cargo target directory}"

REPO=${REPO:-/home/avifenesh/projects/wt-cx-fadecoderow}
BASE_SOURCE_COMMIT=${BASE_SOURCE_COMMIT:-ba3e70c9af455320dc661ab023e5c653539bc447}
CANDIDATE_CODE_COMMIT=${CANDIDATE_CODE_COMMIT:-3845bda8358a6fe5883095250d3d8e6df84fda2a}
BASE_BIN=${BASE_BIN:-$BASE_TARGET_DIR/release/decode-batch-bench}
CANDIDATE_BIN=${CANDIDATE_BIN:-$REPO/target/release/decode-batch-bench}
EXPECTED_BASE_BIN=${EXPECTED_BASE_BIN:-308e8194e5b15d8ea1dd025cb619a5eb28d436765c53d3edef2b8d3f83aaac36}
EXPECTED_CANDIDATE_BIN=${EXPECTED_CANDIDATE_BIN:-3db8c7e9bc9e8a9da745484358443dbf9bd88e1d73dc029a848854d00d6dde7a}
STEPS=${STEPS:-128}
N=${N:-5}

mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1

SAMPLER_PID=0
cleanup() {
    if (( SAMPLER_PID > 0 )); then
        kill "$SAMPLER_PID" 2>/dev/null || true
        wait "$SAMPLER_PID" 2>/dev/null || true
        SAMPLER_PID=0
    fi
}
trap cleanup EXIT INT TERM

binary_hash() {
    sha256sum "$1" | awk '{print $1}'
}

snapshot() {
    local path=$1
    {
        date -u +ts=%FT%TZ
        nvidia-smi \
          --query-gpu=timestamp,name,uuid,driver_version,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,memory.total,utilization.gpu \
          --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
          --format=csv,noheader
    } 2>&1 | tee "$path"
}

run_arm() {
    local arm=$1 rep=$2 bin=$3
    local log="$OUT/$arm-r$rep.log" rc=0
    snapshot "$OUT/$arm-r$rep-before.log"
    set +e
    env -u MEMRA_SWA_RING -u MEMRA_SPEC_K -u MEMRA_BG_JOB \
      CUDA_VISIBLE_DEVICES=0 MEMRA_FAST=1 \
      timeout 1200 nice -n 15 ionice -c3 "$bin" "$MODEL" \
        --steps "$STEPS" --reps 1 --batches 1,2,4,8 --ctx 512 \
      2>&1 | tee "$log" >/dev/null
    rc=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$rc" | tee "$OUT/$arm-r$rep.exit"
    (( rc == 0 ))
    [[ $(grep -cE '^B=(1|2|4|8): aggregate ' "$log") -eq 4 ]]
    snapshot "$OUT/$arm-r$rep-after.log"
    grep -E '^B=(1|2|4|8): aggregate ' "$log"
}

cd "$REPO"
[[ $(git -C "$BASE_SOURCE_DIR" rev-parse HEAD) == "$BASE_SOURCE_COMMIT" ]]
git diff --quiet "$CANDIDATE_CODE_COMMIT" -- crates/memra-engine
[[ $(binary_hash "$BASE_BIN") == "$EXPECTED_BASE_BIN" ]]
[[ $(binary_hash "$CANDIDATE_BIN") == "$EXPECTED_CANDIDATE_BIN" ]]
[[ $EXPECTED_BASE_BIN != "$EXPECTED_CANDIDATE_BIN" ]]
{
    printf 'baseline_source=%s\n' "$BASE_SOURCE_COMMIT"
    printf 'candidate_code_source=%s\n' "$CANDIDATE_CODE_COMMIT"
    printf 'candidate_lane_head=%s\n' "$(git rev-parse HEAD)"
    printf 'baseline_binary=%s sha256=%s\n' "$BASE_BIN" "$EXPECTED_BASE_BIN"
    printf 'candidate_binary=%s sha256=%s\n' "$CANDIDATE_BIN" "$EXPECTED_CANDIDATE_BIN"
    stat -c 'artifact=%n bytes=%s mtime=%y' "$MODEL" "$BASE_BIN" "$CANDIDATE_BIN"
    printf 'protocol=N=%s outer reps; alternating arm order; steps=%s; inner warmup=1; inner timed reps=1; B=1,2,4,8; ctx=512; MEMRA_FAST=1; nice=15; ionice=idle\n' "$N" "$STEPS"
} | tee "$OUT/provenance.txt"

exec 9>/tmp/gpu5090.lock
flock -w 7200 9
apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader,nounits 2>/dev/null)
[[ -z $apps ]] || { printf '%s\n' "$apps"; exit 1; }
snapshot "$OUT/window-before.log"
nvidia-smi \
  --query-gpu=timestamp,name,uuid,pstate,temperature.gpu,power.draw,power.limit,clocks.sm,memory.used,utilization.gpu \
  --format=csv,noheader,nounits -lms 500 > "$OUT/thermal.csv" 2>&1 &
SAMPLER_PID=$!

# Equal discarded process warmups put model loading and both binaries into the same regime.
run_arm warm-base 0 "$BASE_BIN"
run_arm warm-candidate 0 "$CANDIDATE_BIN"

for (( rep=1; rep<=N; rep++ )); do
    if (( rep % 2 == 1 )); then
        run_arm baseline "$rep" "$BASE_BIN"
        run_arm candidate "$rep" "$CANDIDATE_BIN"
    else
        run_arm candidate "$rep" "$CANDIDATE_BIN"
        run_arm baseline "$rep" "$BASE_BIN"
    fi
done

cleanup
snapshot "$OUT/window-after.log"

python3 - "$OUT" "$N" "$BASE_SOURCE_COMMIT" "$CANDIDATE_CODE_COMMIT" \
  "$EXPECTED_BASE_BIN" "$EXPECTED_CANDIDATE_BIN" <<'PY' | tee "$OUT/summary.json"
import csv
import json
import pathlib
import re
import statistics
import sys

out = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
sources = {"baseline": sys.argv[3], "candidate": sys.argv[4]}
binaries = {"baseline": sys.argv[5], "candidate": sys.argv[6]}
row = re.compile(r"^B=(1|2|4|8): aggregate ([0-9.]+) tok/s")

def read_rep(arm, rep):
    values = {}
    for line in (out / f"{arm}-r{rep}.log").read_text(errors="replace").splitlines():
        match = row.match(line)
        if match:
            values[int(match.group(1))] = float(match.group(2))
    assert sorted(values) == [1, 2, 4, 8], (arm, rep, values)
    return values

reps = {arm: [read_rep(arm, rep) for rep in range(1, n + 1)]
        for arm in ("baseline", "candidate")}
widths = {}
for b in (1, 2, 4, 8):
    base = [sample[b] for sample in reps["baseline"]]
    cand = [sample[b] for sample in reps["candidate"]]
    base_median = statistics.median(base)
    cand_median = statistics.median(cand)
    paired = [(c / a - 1.0) * 100.0 for a, c in zip(base, cand)]
    widths[str(b)] = {
        "N": n,
        "baseline_tok_s": base,
        "candidate_tok_s": cand,
        "baseline_median_tok_s": base_median,
        "candidate_median_tok_s": cand_median,
        "delta_pct": (cand_median / base_median - 1.0) * 100.0,
        "baseline_spread_tok_s": [min(base), max(base)],
        "candidate_spread_tok_s": [min(cand), max(cand)],
        "paired_delta_pct": paired,
        "paired_wins": sum(delta > 0.0 for delta in paired),
    }

temps = []
clocks = []
powers = []
with (out / "thermal.csv").open(newline="", errors="replace") as handle:
    for values in csv.reader(handle):
        if len(values) < 8:
            continue
        try:
            temps.append(float(values[4].strip()))
            powers.append(float(values[5].strip()))
            clocks.append(float(values[7].strip()))
        except ValueError:
            pass

summary = {
    "schema": "memra.fadecoderow.local-ab.v1",
    "rig": "NVIDIA GeForce RTX 5090 Laptop GPU",
    "protocol": {
        "outer_reps": n,
        "arm_order": "alternating baseline/candidate by round",
        "steps_per_inner_rep": 128,
        "discarded_inner_warmups": 1,
        "timed_inner_reps": 1,
        "batches": [1, 2, 4, 8],
        "ctx": 512,
        "memra_fast": 1,
        "nice": 15,
        "ionice": "idle",
        "gpu_lock": "/tmp/gpu5090.lock",
        "artificial_cooldown": False,
    },
    "provenance": {"sources": sources, "binary_sha256": binaries},
    "widths": widths,
    "thermal": {
        "samples": len(temps),
        "temperature_c": [min(temps), max(temps)] if temps else [],
        "sm_clock_mhz": [min(clocks), max(clocks)] if clocks else [],
        "power_w": [min(powers), max(powers)] if powers else [],
    },
}
print(json.dumps(summary, indent=2, sort_keys=True))
PY
