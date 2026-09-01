#!/usr/bin/env bash
# Grouped ON vs OFF Step prefill, N=5 interleaved per prompt shape.
set -uo pipefail

REPO=${REPO:-"$HOME/memra-cx-grouped"}
MODEL=${MODEL:-"$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf"}
PROMPT4096=${PROMPT4096:-"$REPO/research/step-sku-20260807/prompt-pp4096.txt"}
RAW=${RAW:-"$REPO/research/grouped-serve-20260810/raw/box1/prefill"}
TS=${TS:-$(date -u +%Y%m%dT%H%M%SZ)}
SUMMARY="$RAW/prefill-summary-$TS.log"
RESULTS="$RAW/prefill-results-$TS.tsv"
SAMPLES="$RAW/prefill-samples-$TS.tsv"
P512="$RAW/prompt-pp512-$TS.txt"
P2048="$RAW/prompt-pp2048-$TS.txt"
P4096="$RAW/prompt-pp4096-$TS.txt"
N=${N:-5}

mkdir -p "$RAW"
head -c 2800 "$PROMPT4096" >"$P512"
head -c 11200 "$PROMPT4096" >"$P2048"
cp -- "$PROMPT4096" "$P4096"
cd "$REPO" || exit 1

snapshot() {
  echo "snapshot $(date -u +%FT%TZ)"
  nvidia-smi \
    --query-gpu=index,name,temperature.gpu,clocks.sm,power.draw,memory.used,utilization.gpu \
    --format=csv,noheader || true
  nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader || true
}

require_idle() {
  local apps
  apps=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader 2>/dev/null || true)
  if [[ -n "$apps" ]]; then
    echo "GPU NOT IDLE"
    printf '%s\n' "$apps"
    return 76
  fi
}

run_arm() {
  local shape=$1
  local prompt=$2
  local arm=$3
  local rep=$4
  local log="$RAW/$shape-$arm-r$rep-$TS.log"
  local grouped=0
  [[ "$arm" == grouped ]] && grouped=1
  echo "########## $shape rep=$rep arm=$arm grouped=$grouped ##########"
  snapshot
  env \
    -u MEMRA_MOE_STATS -u MEMRA_MOE_GATE \
    -u MEMRA_PRIME_CHUNK -u MEMRA_PRIME_PP -u MEMRA_PRIME_PIPE \
    CUDA_VISIBLE_DEVICES=0,1 \
    MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_MOE_GROUPED="$grouped" \
    timeout 3600 ./target/release/concat-prime-probe "$MODEL" ppprime \
    --prompt-a "@$prompt" --reps 1 --warmup 1 >"$log" 2>&1
  local arm_rc=$?
  if ((arm_rc == 0)); then
    grep -E 'ppprime MEDIAN:' "$log" | tail -1
  else
    tail -120 "$log"
  fi
  echo "$shape rep=$rep arm=$arm exit=$arm_rc"
  snapshot
  require_idle || return $?
  return "$arm_rc"
}

summarize_arm() {
  local shape=$1
  local arm=$2
  local times=()
  local tokens=""
  local log line parsed tok seconds rep
  for log in "$RAW"/"$shape"-"$arm"-r*-"$TS".log; do
    [[ -f "$log" ]] || continue
    line=$(grep -E 'ppprime MEDIAN:' "$log" | tail -1)
    [[ -n "$line" ]] || continue
    parsed=$(awk '
      /ppprime MEDIAN:/ {
        for (i = 1; i <= NF; i++) {
          if ($i == "MEDIAN:") tok = $(i + 1)
          if ($i == "in") {
            sec = $(i + 1)
            sub(/s$/, "", sec)
          }
        }
      }
      END { print tok, sec }
    ' <<<"$line")
    read -r tok seconds <<<"$parsed"
    [[ -n "$tok" && -n "$seconds" ]] || continue
    tokens=${tokens:-$tok}
    [[ "$tokens" == "$tok" ]] || return 1
    rep=${log##*-r}
    rep=${rep%%-*}
    times+=("$seconds")
    printf '%s\t%s\t%s\t%s\t%s\t%.3f\n' \
      "$shape" "$tokens" "$arm" "$rep" "$seconds" \
      "$(awk -v tok="$tokens" -v sec="$seconds" 'BEGIN { print tok / sec }')" \
      >>"$SAMPLES"
  done
  if (( ${#times[@]} != N )); then
    echo "$shape $arm expected N=$N, found ${#times[@]}"
    return 1
  fi
  local sorted=()
  mapfile -t sorted < <(printf '%s\n' "${times[@]}" | sort -n)
  local median_s=${sorted[$((N / 2))]}
  local median_rate
  median_rate=$(awk -v tok="$tokens" -v sec="$median_s" \
    'BEGIN { printf "%.1f", tok / sec }')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$shape" "$tokens" "$arm" "$N" "$median_s" "$median_rate" >>"$RESULTS"
}

report_comparison() {
  local shape=$1
  local off grouped delta
  off=$(awk -F '\t' -v shape="$shape" \
    '$1 == shape && $3 == "off" { print $6 }' "$RESULTS")
  grouped=$(awk -F '\t' -v shape="$shape" \
    '$1 == shape && $3 == "grouped" { print $6 }' "$RESULTS")
  [[ -n "$off" && -n "$grouped" ]] || return 1
  delta=$(awk -v grouped="$grouped" -v off="$off" \
    'BEGIN { printf "%+.1f%%", 100.0 * (grouped / off - 1.0) }')
  echo "$shape grouped=$grouped tok/s off=$off tok/s delta=$delta"
}

main() {
  echo "=== grouped-serve prefill $TS commit=$(git rev-parse HEAD)"
  echo "binary=$(sha256sum target/release/concat-prime-probe | awk '{print $1}')"
  echo "model=$MODEL bytes=$(stat -c %s "$MODEL")"
  echo "protocol=N=$N interleaved, order alternated, one warmup per independent process"
  local prompt
  for prompt in "$P512" "$P2048" "$P4096"; do
    echo "prompt=$prompt bytes=$(wc -c <"$prompt") sha256=$(sha256sum "$prompt" | awk '{print $1}')"
  done
  local rc=0

  (
    flock -w 21600 9 || {
      echo "LOCK TIMEOUT"
      exit 75
    }
    echo "lock acquired $(date -u +%FT%TZ)"
    snapshot
    require_idle || exit $?
    local cell shape prompt rep arm order
    for cell in "pp512:$P512" "pp2048:$P2048" "pp4096:$P4096"; do
      shape=${cell%%:*}
      prompt=${cell#*:}
      for ((rep = 1; rep <= N; rep++)); do
        if ((rep % 2 == 1)); then
          order="off grouped"
        else
          order="grouped off"
        fi
        for arm in $order; do
          run_arm "$shape" "$prompt" "$arm" "$rep" || rc=1
        done
      done
    done
    snapshot
    require_idle || rc=1
    echo "lock released $(date -u +%FT%TZ)"
    exit "$rc"
  ) 9>/tmp/memra-gpu.lock
  local perf_rc=$?

  printf 'shape\ttokens\tarm\trep\tseconds\ttok_s\n' >"$SAMPLES"
  printf 'shape\ttokens\tarm\tn\tmedian_s\tmedian_tok_s\n' >"$RESULTS"
  local shape arm
  for shape in pp512 pp2048 pp4096; do
    for arm in off grouped; do
      summarize_arm "$shape" "$arm" || perf_rc=1
    done
  done
  echo "=== samples ==="
  cat "$SAMPLES"
  echo "=== medians ==="
  cat "$RESULTS"
  echo "=== comparisons ==="
  for shape in pp512 pp2048 pp4096; do
    report_comparison "$shape" || perf_rc=1
  done
  echo "=== grouped-serve prefill rc=$perf_rc"
  echo "=== done $(date -u +%FT%TZ)"
  return "$perf_rc"
}

main > >(tee "$SUMMARY") 2>&1
