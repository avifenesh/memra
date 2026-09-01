#!/usr/bin/env bash
# Interleaved, load-isolated HY3 decode A/B. Greedy is an identity/perf instrument only.

set -euo pipefail

if (( $# != 7 )); then
  echo "usage: $0 LABEL_A RUN_SPEC_A LABEL_B RUN_SPEC_B ARTIFACT OUT_DIR Q8_0_OR_1" >&2
  exit 64
fi

label_a=$1
bin_a=$2
label_b=$3
bin_b=$4
artifact=$5
out_dir=$6
q8=$7
reps=${REPS:-1}

[[ $label_a =~ ^[a-zA-Z0-9._-]+$ && $label_b =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "labels must be filename-safe" >&2
  exit 64
}
[[ $q8 == 0 || $q8 == 1 ]] || {
  echo "Q8_0_OR_1 must be 0 or 1" >&2
  exit 64
}
[[ $reps =~ ^[1-9][0-9]*$ ]] || {
  echo "REPS must be a positive integer" >&2
  exit 64
}
[[ -x $bin_a && -x $bin_b && -d $artifact ]] || {
  echo "missing executable or artifact directory" >&2
  exit 66
}

mkdir -p "$out_dir"
exec 9>/tmp/memra-gpu.lock
flock -n 9 || {
  echo "GPU lock busy" >&2
  exit 75
}

prompt='Write a Rust function that parses a decimal u64 without allocating, then explain its overflow check.'
summary=$out_dir/summary.tsv
printf 'sequence\tarm\trep\ttok_s\tprime_s\ttoken_sha256\tlog\n' > "$summary"

run_arm() {
  local arm=$1
  local rep=$2
  local binary label log line tok_s prime_s token_sha
  if [[ $arm == A ]]; then
    binary=$bin_a
    label=$label_a
  else
    binary=$bin_b
    label=$label_b
  fi
  log=$out_dir/$(printf '%02d-%s-r%02d.log' "$sequence" "$label" "$rep")
  env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    MEMRA_CUDA_ARCH=120a \
    MEMRA_PARALLEL=auto \
    MEMRA_PARALLEL_DEVICES=0,1,2,3 \
    MEMRA_PARALLEL_EP_DEVICE_ROUTER=1 \
    MEMRA_PARALLEL_EP_Q8_ACT="$q8" \
    MEMRA_ST_REPACK_DISK=1 \
    MEMRA_CHAT=1 \
    MEMRA_NGEN=64 \
    MEMRA_GEN_ONLY=1 \
    MEMRA_PROMPT="$prompt" \
    timeout 7200 "$binary" "$artifact" > "$log" 2>&1
  line=$(grep -E '^\[generate\]' "$log" | tail -n 1)
  tok_s=$(sed -E 's/.* = ([0-9.]+) tok\/s.*/\1/' <<<"$line")
  prime_s=$(sed -E "s/.*prime ([0-9.]+)s.*/\\1/" <<<"$line")
  token_sha=$(grep -E '^  tokens:' "$log" | sha256sum | cut -d' ' -f1)
  printf '%d\t%s\t%d\t%s\t%s\t%s\t%s\n' \
    "$sequence" "$label" "$rep" "$tok_s" "$prime_s" "$token_sha" "$log" >> "$summary"
  ((sequence += 1))
}

sequence=1
for ((rep = 1; rep <= reps; rep += 1)); do
  if ((rep % 2 == 1)); then
    run_arm A "$rep"
    run_arm B "$rep"
  else
    run_arm B "$rep"
    run_arm A "$rep"
  fi
done

python3 - "$summary" "$label_a" "$label_b" <<'PY'
import csv
import statistics
import sys

path, label_a, label_b = sys.argv[1:]
rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
groups = {
    label: [float(row["tok_s"]) for row in rows if row["arm"] == label]
    for label in (label_a, label_b)
}
hashes = {row["token_sha256"] for row in rows}
if len(hashes) != 1:
    raise SystemExit(f"token identity FAIL: {sorted(hashes)}")
for label, values in groups.items():
    print(
        f"{label}: n={len(values)} median={statistics.median(values):.3f} "
        f"range={min(values):.3f}..{max(values):.3f}"
    )
base = statistics.median(groups[label_a])
candidate = statistics.median(groups[label_b])
print(f"delta {label_b}/{label_a}: {(candidate / base - 1.0) * 100.0:+.3f}%")
print("token identity PASS")
PY
