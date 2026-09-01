#!/usr/bin/env bash
# Keep the four-card HY3 lane busy after own-output corpus generation completes.

set -euo pipefail

root=${1:-/workspace/hy3-stage/masked-mtp}
artifact=${2:-/workspace/hy3-nvfp4}
rank_bin=${3:-/workspace/hy3-stage/target-quad-gu-row2/release/frspec-rank}
kernel_check=${4:-/workspace/hy3-stage/target-quad-gu-row2/release/kernel-check}

for required in \
  "$root/corpus-exact-c8.pid" \
  "$root/dataset-heldout.jsonl" \
  "$root/gen_mask_corpus.py" \
  "$rank_bin" \
  "$kernel_check"; do
  [[ -e $required ]] || {
    echo "missing required path: $required" >&2
    exit 66
  }
done

wait_pid_file() {
  local pid_file=$1
  local pid
  pid=$(<"$pid_file")
  while kill -0 "$pid" 2>/dev/null; do
    sleep 5
  done
}

extract_completion_text() {
  local input=$1
  local output=$2
  jq -r 'select(.status == "ok") | .text' "$input" > "$output"
}

coverage_value() {
  local pattern=$1
  local log=$2
  grep -E "$pattern" "$log" | tail -n 1 | sed -E 's/.*covers ([0-9.]+)%.*/\1/'
}

require_floor() {
  local label=$1
  local value=$2
  local floor=$3
  python3 - "$label" "$value" "$floor" <<'PY'
import sys

label, value, floor = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
if value < floor:
    raise SystemExit(f"{label} coverage {value:.4f}% < required {floor:.4f}%")
print(f"{label} coverage PASS: {value:.4f}% >= {floor:.4f}%")
PY
}

coverage_meets() {
  local value=$1
  local floor=$2
  python3 - "$value" "$floor" <<'PY'
import sys

raise SystemExit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)
PY
}

summary_meets() {
  local summary=$1
  local target=$2
  [[ -s $summary ]] && jq -e --argjson target "$target" \
    '.target_met and .completion_tokens >= $target' "$summary" >/dev/null
}

ensure_corpus() {
  local dataset=$1
  local output=$2
  local summary=$3
  local pid_file=$4
  local log=$5
  local target=$6
  if summary_meets "$summary" "$target"; then
    return
  fi
  if [[ ! -s $pid_file ]] || ! kill -0 "$(<"$pid_file")" 2>/dev/null; then
    nohup python3 gen_mask_corpus.py \
      --dataset "$dataset" \
      --output "$output" \
      --summary "$summary" \
      --endpoint http://127.0.0.1:18087 \
      --target-tokens "$target" \
      --max-tokens 512 \
      --concurrency 8 \
      > "$log" 2>&1 &
    echo $! > "$pid_file"
  fi
  wait_pid_file "$pid_file"
  summary_meets "$summary" "$target"
}

cd "$root"
echo "WAIT train corpus $(date -Is)"
wait_pid_file "$root/corpus-exact-c8.pid"
echo "TRAIN corpus complete $(date -Is)"
jq -e '.target_met and .completion_tokens >= 140000' \
  hy3-exact-own-output.summary.json >/dev/null

extract_completion_text hy3-exact-own-output.jsonl hy3-exact-own-output.txt

# Keep the GPU server occupied with a disjoint prompt slice while CPU tokenization/ranking runs.
ensure_corpus \
  dataset-heldout.jsonl \
  hy3-heldout-own-output.jsonl \
  hy3-heldout-own-output.summary.json \
  corpus-heldout.pid \
  corpus-heldout.log \
  20000 &
heldout_wait_pid=$!

"$rank_bin" \
  "$artifact" \
  hy3-own-output-ranks-32768.gguf \
  32768 \
  hy3-exact-own-output.txt \
  > rank-train.log 2>&1

train_tokens=$(grep -E 'tokens counted' rank-train.log | tail -n 1 | awk '{print $2}')
[[ $train_tokens =~ ^[0-9]+$ ]] || {
  echo "could not parse re-tokenized train count" >&2
  exit 65
}
((train_tokens >= 131072)) || {
  echo "re-tokenized train corpus $train_tokens < 131072" >&2
  exit 65
}
train_coverage=$(coverage_value 'top 32768 covers' rank-train.log)
require_floor train "$train_coverage" 99.0

wait "$heldout_wait_pid"
extract_completion_text hy3-heldout-own-output.jsonl hy3-heldout-own-output.txt
"$rank_bin" \
  "$artifact" \
  hy3-heldout-self-ranks.gguf \
  32768 \
  --coverage-ranks hy3-own-output-ranks-32768.gguf.txt \
  hy3-heldout-own-output.txt \
  > rank-heldout.log 2>&1
heldout_coverage=$(coverage_value 'supplied map .* covers' rank-heldout.log)
rank_train_log=rank-train.log
rank_heldout_log=rank-heldout.log
selected_width=
selected_rank=

if coverage_meets "$heldout_coverage" 98.5; then
  selected_width=32768
  selected_rank="$root/hy3-own-output-ranks-32768.gguf.txt"
else
  [[ -s dataset-train-extra.jsonl ]] || {
    echo "heldout coverage failed and dataset-train-extra.jsonl is absent" >&2
    exit 66
  }
  echo "32K heldout coverage ${heldout_coverage}%: expanding only with disjoint served HY3 output"
  for target in 60000 120000; do
    ensure_corpus \
      dataset-train-extra.jsonl \
      hy3-extra-own-output.jsonl \
      hy3-extra-own-output.summary.json \
      corpus-extra.pid \
      "corpus-extra-${target}.log" \
      "$target"
    extract_completion_text hy3-extra-own-output.jsonl hy3-extra-own-output.txt
    cat hy3-exact-own-output.txt hy3-extra-own-output.txt > hy3-combined-own-output.txt

    for width in 32768 49152 65536 73728 81920 98304; do
      rank_train_log="rank-train-extra-${target}-w${width}.log"
      "$rank_bin" \
        "$artifact" \
        "hy3-own-output-ranks-${width}.gguf" \
        "$width" \
        hy3-combined-own-output.txt \
        > "$rank_train_log" 2>&1
      train_tokens=$(grep -E 'tokens counted' "$rank_train_log" | tail -n 1 | awk '{print $2}')
      [[ $train_tokens =~ ^[0-9]+$ ]] || {
        echo "could not parse expanded re-tokenized train count" >&2
        exit 65
      }
      ((train_tokens >= 131072)) || {
        echo "expanded re-tokenized train corpus $train_tokens < 131072" >&2
        exit 65
      }
      train_coverage=$(coverage_value "top ${width} covers" "$rank_train_log")
      require_floor train "$train_coverage" 99.0

      rank_heldout_log="rank-heldout-extra-${target}-w${width}.log"
      "$rank_bin" \
        "$artifact" \
        "hy3-heldout-self-${width}.gguf" \
        "$width" \
        --coverage-ranks "hy3-own-output-ranks-${width}.gguf.txt" \
        hy3-heldout-own-output.txt \
        > "$rank_heldout_log" 2>&1
      heldout_coverage=$(coverage_value 'supplied map .* covers' "$rank_heldout_log")
      echo "width=$width extra_tokens=$target heldout_coverage=${heldout_coverage}%"
      if coverage_meets "$heldout_coverage" 98.5; then
        selected_width=$width
        selected_rank="$root/hy3-own-output-ranks-${width}.gguf.txt"
        break
      fi
    done
    [[ -z $selected_width ]] || break
    echo "no tested width passed after $target extra served tokens; continuing"
  done
fi
[[ -n $selected_width && -s $selected_rank ]] || {
  echo "no model-specific masked-head width passed the heldout gate" >&2
  exit 65
}
require_floor "heldout w${selected_width}" "$heldout_coverage" 98.5
printf '%s\n' "$selected_width" > selected-rank-width.txt
printf '%s\n' "$selected_rank" > selected-rank-path.txt
cp "$rank_train_log" rank-train-final.log
cp "$rank_heldout_log" rank-heldout-final.log

manifest_paths=(
  hy3-exact-own-output.jsonl
  hy3-exact-own-output.summary.json
  hy3-exact-own-output.txt
  hy3-heldout-own-output.jsonl
  hy3-heldout-own-output.summary.json
  hy3-heldout-own-output.txt
  selected-rank-width.txt
  selected-rank-path.txt
  rank-train-final.log
  rank-heldout-final.log
)
shopt -s nullglob
for rank_artifact in hy3-own-output-ranks-*.gguf hy3-own-output-ranks-*.gguf.txt; do
  manifest_paths+=("$rank_artifact")
done
for optional in \
  hy3-extra-own-output.jsonl \
  hy3-extra-own-output.summary.json \
  hy3-extra-own-output.txt \
  hy3-combined-own-output.txt; do
  [[ ! -e $optional ]] || manifest_paths+=("$optional")
done
sha256sum "${manifest_paths[@]}" > masked-mtp-corpus.sha256

# Stop the corpus server and take the GPU lock for the kernel and same-artifact stage sweeps.
server_pid=$(<server-exact-corpus-c8.pid)
case "$(tr '\0' ' ' < "/proc/$server_pid/cmdline")" in
  *target-q8-consistent/release/memra-server*) ;;
  *)
    echo "refusing to stop unexpected server pid $server_pid" >&2
    exit 70
    ;;
esac
kill -TERM "$server_pid"
for _ in $(seq 1 30); do
  kill -0 "$server_pid" 2>/dev/null || break
  sleep 1
done
if kill -0 "$server_pid" 2>/dev/null; then
  kill -KILL "$server_pid"
fi

exec 9>/tmp/memra-gpu.lock
flock -n 9 || {
  echo "GPU lock remained busy after corpus server stop" >&2
  exit 75
}

env CUDA_VISIBLE_DEVICES=0 "$kernel_check" > kernel-check-quad.log 2>&1

prompt='Write a Rust function that parses a decimal u64 without allocating, then explain its overflow check.'
declare -A binaries=(
  [current]=/workspace/hy3-stage/target-q8-consistent/release/run-spec
  [dual]=/workspace/hy3-stage/target-dual-gu/release/run-spec
  [row2]=/workspace/hy3-stage/target-dual-row2/release/run-spec
  [quad]=/workspace/hy3-stage/target-quad-gu-row2/release/run-spec
)

run_stage() {
  local label=$1
  local q8=$2
  local log="stage-${label}-q8${q8}.log"
  env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_CUDA_ARCH=120a \
    MEMRA_PARALLEL=auto \
    MEMRA_PARALLEL_DEVICES=0,1,2,3 \
    MEMRA_PARALLEL_EP_DEVICE_ROUTER=1 \
    MEMRA_PARALLEL_EP_Q8_ACT="$q8" \
    MEMRA_ST_REPACK_DISK=1 \
    MEMRA_FRSPEC_TRIM="$selected_rank" \
    MEMRA_FRSPEC_TRIM_NVFP4=1 \
    MEMRA_CHAT=1 \
    MEMRA_NGEN=64 \
    MEMRA_PROMPT="$prompt" \
    timeout 7200 "${binaries[$label]}" "$artifact" > "$log" 2>&1
}

for label in current dual row2 quad; do
  run_stage "$label" 0
done
for label in quad row2 dual current; do
  run_stage "$label" 1
done

sha256sum \
  kernel-check-quad.log \
  stage-*-q8*.log \
  > masked-stage-sweeps.sha256
touch stage-sweeps.done
echo "CONTINUATION DONE $(date -Is)"
