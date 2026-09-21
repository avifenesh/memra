#!/usr/bin/env bash
# CPU-only accounting cells for the CPU expert companion's speculative prefetch (memra#586).
# Builds tools/memra_cpu_expert_prefetch_test.cpp (the production translation unit included, the
# shm test's pattern) and runs four process-level cells against a source fixture and a
# byte-identical mirror on a SECOND filesystem (the companion's mirror map refuses a mirror on the
# source's device), both opened O_DIRECT by the companion:
#   barrier  three mirrored projections, every alternate half held after its primary half landed:
#            the signed in-flight counter must still carry all three, the cap must refuse a
#            fourth, and the drain must reach exactly zero and admit the retry
#   failure  one mirrored projection whose alternate half fails with EIO, a sentinel queued
#            behind it on a single worker: the charge is released once, the annex never publishes
#            the failed buffer, the retry lands
#   parity   the non-mirrored buffered control: one job per projection, unchanged by the repair
#   submit-throw  one valid projection then an unopenable fd: the loop throws after the first
#            charge and annex claim were taken; the call returns -1 and both are released
# The counter is read inside the translation unit (prefetch_inflight()), never through the public
# stats function's clamp. No GPU, no timing: pass/fail only.
# usage: tools/test_cpu_expert_prefetch.sh [LOG_DIR] [SOURCE_DIR] [MIRROR_DIR]
#   SOURCE_DIR defaults to <repo>/target (a real filesystem); MIRROR_DIR to /dev/shm (tmpfs).
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
source_root=${2:-$repo_dir/target}
mirror_root=${3:-/dev/shm}
mkdir -p -- "$source_root"
source_dir=$(mktemp -d "$source_root/memra-prefetch-test.XXXXXX")
mirror_dir=$(mktemp -d "$mirror_root/memra-prefetch-test.XXXXXX")
log_dir=${1:-$source_dir/logs}
mkdir -p -- "$log_dir"
test_bin=$source_dir/memra-cpu-expert-prefetch-test
source_fixture=$source_dir/source.bin
mirror_fixture=$mirror_dir/mirror.bin
mirror_map=$source_dir/mirror.tsv

cleanup() {
  rm -rf -- "$source_dir" "$mirror_dir"
}
trap cleanup EXIT

if [[ $(stat -c %d -- "$source_dir") == "$(stat -c %d -- "$mirror_dir")" ]]; then
  printf 'source dir %s and mirror dir %s are on one device; the mirror map needs two\n' \
    "$source_dir" "$mirror_dir" >&2
  exit 1
fi
for dir in "$source_dir" "$mirror_dir"; do
  dd if=/dev/zero of="$dir/odirect-probe" bs=4096 count=2 status=none
  if ! dd if="$dir/odirect-probe" of=/dev/null bs=4096 iflag=direct status=none 2>/dev/null; then
    printf 'O_DIRECT reads are unsupported under %s; pick another directory\n' "$dir" >&2
    exit 1
  fi
  rm -f -- "$dir/odirect-probe"
done

test_cxxflags=(-std=c++17 -O2 -march=native -fopenmp -Wall -Wextra -Wpedantic -Werror)
"${CXX:-c++}" "${test_cxxflags[@]}" \
  "$script_dir/memra_cpu_expert_prefetch_test.cpp" -o "$test_bin"

# Every cell runs even after a failure, so a red sitting records all three verdicts; the
# summary line and the exit status carry the count.
failures=0
run_case() {
  local label=$1
  shift
  if "$@" >"$log_dir/$label.log" 2>&1; then
    printf 'ok: %s\n' "$label"
  else
    sed -n '1,120p' "$log_dir/$label.log" >&2
    printf 'FAIL: cell %s exited non-zero\n' "$label"
    failures=$((failures + 1))
  fi
}

assert_log() {
  local label=$1
  local expected=$2
  if grep -F -- "$expected" "$log_dir/$label.log" >/dev/null; then
    printf 'ok: %s has %s\n' "$label" "$expected"
  else
    printf 'FAIL: missing %q in %s\n' "$expected" "$log_dir/$label.log"
    failures=$((failures + 1))
  fi
}

# Companion knobs the cells do not test, held small and explicit.
export MEMRA_CPU_EXPERT_CACHE_GB=0.001
export MEMRA_CPU_EXPERT_RESERVE_GB=0
export MEMRA_CPU_EXPERT_PREFETCH_ANNEX_GB=0.125
unset MEMRA_CPU_EXPERT_CACHE_SHM MEMRA_CPU_EXPERT_IO_CPUSET MEMRA_CPU_EXPERT_PIPELINE

"$test_bin" make-fixture "$source_fixture" >"$log_dir/fixture-source.log" 2>&1
"$test_bin" make-fixture "$mirror_fixture" >"$log_dir/fixture-mirror.log" 2>&1
cmp -- "$source_fixture" "$mirror_fixture"
"$test_bin" write-map "$source_fixture" "$mirror_fixture" "$mirror_map" >"$log_dir/mirror-map.log" 2>&1
grep -F -- MAP_OK "$log_dir/mirror-map.log" >/dev/null

run_case barrier env MEMRA_CPU_EXPERT_IO=direct "MEMRA_CPU_EXPERT_MIRROR_MAP=$mirror_map" \
  MEMRA_CPU_EXPERT_IO_THREADS=3 MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=3 \
  "$test_bin" barrier "$source_fixture" "$mirror_fixture"
assert_log barrier 'barrier: primaries complete, alternates held: inflight_signed=3 expected=3'
assert_log barrier 'extra projection admitted=0 expected=0'
assert_log barrier 'barrier: drained after the retry landed: inflight_signed=0 expected=0'
assert_log barrier 'BARRIER_OK'

run_case failure env MEMRA_CPU_EXPERT_IO=direct "MEMRA_CPU_EXPERT_MIRROR_MAP=$mirror_map" \
  MEMRA_CPU_EXPERT_IO_THREADS=1 MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=8 \
  "$test_bin" failure "$source_fixture" "$mirror_fixture"
assert_log failure 'failure: alternate half failed with EIO, sentinel read entered: inflight_signed=1 expected=1'
assert_log failure 'failure: drained after the retry landed: inflight_signed=0 expected=0'
assert_log failure 'FAILURE_PATH_OK'

run_case parity env MEMRA_CPU_EXPERT_IO_THREADS=3 MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=8 \
  "$test_bin" parity "$source_fixture"
assert_log parity 'parity: three single-job reads held: inflight_signed=3 expected=3'
assert_log parity 'parity: drained: inflight_signed=0 expected=0'
assert_log parity 'PARITY_OK'

# submit-throw: the submit side of the same invariant (review round on #612). The second
# projection's fd is not open, so the submit loop throws after the first projection took its
# charge and annex claim and before any job reached the pool; both must be released.
run_case submit-throw env MEMRA_CPU_EXPERT_IO_THREADS=3 MEMRA_CPU_EXPERT_PREFETCH_MAX_INFLIGHT=8 \
  "$test_bin" submit-throw "$source_fixture"
assert_log submit-throw 'submit-throw: after the failed call: inflight_signed=0 expected=0'
assert_log submit-throw 'submit-throw: after the retry landed: inflight_signed=0 expected=0'
assert_log submit-throw 'submit-throw: PASS'

if (( failures > 0 )); then
  printf 'cpu expert prefetch accounting tests: %d FAILURE(S)\n' "$failures"
  exit 1
fi
printf 'cpu expert prefetch accounting tests: ALL GREEN\n'
