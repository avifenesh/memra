#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
test_tmp=$(mktemp -d "${TMPDIR:-/tmp}/memra-shmown-test.XXXXXX")
log_dir=${1:-$test_tmp/logs}
mkdir -p -- "$log_dir"
test_bin=$test_tmp/memra-cpu-expert-shm-test
fixture=$test_tmp/source.bin
suffix=$$
mode_name=/memra-shmown-mode-$suffix
foreign_name=/memra-shmown-foreign-$suffix
warm_name=/memra-shmown-warm-$suffix
past_name=/memra-shmown-past-$suffix
overflow_name=/memra-shmown-overflow-$suffix
checksum_name=/memra-shmown-checksum-$suffix

uid=$(id -u)
gid=$(id -g)
user_name=$(id -un)
subuid=$(awk -F: -v user="$user_name" '$1 == user { print $2; exit }' /etc/subuid)
subgid=$(awk -F: -v user="$user_name" '$1 == user { print $2; exit }' /etc/subgid)

foreign_unshare() {
  unshare --user \
    --map-users "0:$uid:1" --map-users "1:$subuid:1" \
    --map-groups "0:$gid:1" --map-groups "1:$subgid:1" \
    --setuid 0 --setgid 0 "$@"
}

cleanup() {
  for name in "$mode_name" "$warm_name" "$past_name" "$overflow_name" "$checksum_name"; do
    "$test_bin" unlink "$name" >/dev/null 2>&1 || true
  done
  foreign_unshare unlink -- "/dev/shm/${foreign_name#/}" >/dev/null 2>&1 || true
  rm -rf -- "$test_tmp"
}
trap cleanup EXIT

if [[ -z $subuid || -z $subgid ]]; then
  printf 'subuid/subgid mapping is required for the foreign-owner refusal test\n' >&2
  exit 1
fi

test_cxxflags=(-std=c++17 -march=native -fopenmp -Wall -Wextra -Wpedantic -Werror)
if [[ ${MEMRA_SHM_TEST_SANITIZE:-0} == 1 ]]; then
  test_cxxflags+=(-O1 -g "-fsanitize=address,undefined" -fno-omit-frame-pointer)
else
  test_cxxflags+=(-O2)
fi
"${CXX:-c++}" "${test_cxxflags[@]}" \
  "$script_dir/memra_cpu_expert_shm_test.cpp" -o "$test_bin"

run_case() {
  local label=$1
  shift
  if ! "$@" >"$log_dir/$label.log" 2>&1; then
    sed -n '1,240p' "$log_dir/$label.log" >&2
    return 1
  fi
}

assert_log() {
  local label=$1
  local expected=$2
  if ! rg -F -- "$expected" "$log_dir/$label.log" >/dev/null; then
    printf 'missing %q in %s\n' "$expected" "$log_dir/$label.log" >&2
    sed -n '1,240p' "$log_dir/$label.log" >&2
    exit 1
  fi
}

export MEMRA_CPU_EXPERT_CACHE_SHM=1
export MEMRA_CPU_EXPERT_CACHE_GB=0.001
export MEMRA_CPU_EXPERT_RESERVE_GB=0

"$test_bin" precreate "$mode_name" 0644
export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$mode_name
run_case mode-refusal "$test_bin" cache "$fixture" private
assert_log mode-refusal "shm cache REFUSED existing $mode_name"
assert_log mode-refusal 'mode=0644'
assert_log mode-refusal 'PRIVATE_FALLBACK_OK'
"$test_bin" unlink "$mode_name"

"$test_bin" precreate "$foreign_name" 0666
foreign_unshare chown 1:1 "/dev/shm/${foreign_name#/}"
export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$foreign_name
if ! {
  stat -c 'precreated owner=%u:%g mode=%a' "/dev/shm/${foreign_name#/}"
  "$test_bin" cache "$fixture" private
} >"$log_dir/foreign-refusal.log" 2>&1; then
  sed -n '1,240p' "$log_dir/foreign-refusal.log" >&2
  exit 1
fi
assert_log foreign-refusal "shm cache REFUSED existing $foreign_name"
assert_log foreign-refusal "uid=$subuid mode=0666"
assert_log foreign-refusal 'PRIVATE_FALLBACK_OK'
foreign_unshare unlink -- "/dev/shm/${foreign_name#/}"

export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$warm_name
if ! (umask 0777; "$test_bin" cache "$fixture" seed) \
    >"$log_dir/warm-seed.log" 2>&1; then
  sed -n '1,240p' "$log_dir/warm-seed.log" >&2
  exit 1
fi
stat -c 'self-created owner=%u:%g mode=%a' "/dev/shm/${warm_name#/}" \
  >>"$log_dir/warm-seed.log"
run_case warm-hit "$test_bin" cache "$fixture" warm
assert_log warm-seed 'SEED_OK'
assert_log warm-seed "self-created owner=$uid:$gid mode=600"
assert_log warm-hit 'shm cache: reopened clean, warm entries=1'
assert_log warm-hit 'WARM_HIT_OK'
"$test_bin" unlink "$warm_name"

export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$past_name
run_case past-seed "$test_bin" cache "$fixture" seed
"$test_bin" mutate "$past_name" past-end
run_case past-refusal "$test_bin" cache "$fixture" miss
assert_log past-refusal 'shm cache REFUSED persisted entry 0: range exceeds segment_bytes'
assert_log past-refusal 'MISS_REREAD_OK'
"$test_bin" unlink "$past_name"

export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$overflow_name
run_case overflow-seed "$test_bin" cache "$fixture" seed
"$test_bin" mutate "$overflow_name" overflow
run_case overflow-refusal "$test_bin" cache "$fixture" miss
assert_log overflow-refusal 'shm cache REFUSED persisted entry 0: shm_offset + pool_bytes overflows'
assert_log overflow-refusal 'MISS_REREAD_OK'
"$test_bin" unlink "$overflow_name"

export MEMRA_CPU_EXPERT_CACHE_SHM_NAME=$checksum_name
run_case checksum-seed "$test_bin" cache "$fixture" seed
"$test_bin" mutate "$checksum_name" checksum
run_case checksum-refusal "$test_bin" cache "$fixture" miss
assert_log checksum-refusal 'shm cache REFUSED persisted entry 0: sampled checksum mismatch'
assert_log checksum-refusal 'MISS_REREAD_OK'
"$test_bin" unlink "$checksum_name"

printf 'cpu expert shm fail-closed tests: ALL GREEN\n'
