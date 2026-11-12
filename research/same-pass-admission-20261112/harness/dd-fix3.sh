#!/usr/bin/env bash
# Dev desk (memra#1255 review round 1): the head's code (fix3). tree-fix3 = hardlinked copy of
# tree-fix2 with worker.rs written by rsync (rename, so tree-fix2 keeps its file); the sm_120a
# binary; the full CPU checks; red arms of the review's tests. usage: dd-fix3.sh
#   (worker.rs already copied to $D/worker-fix3.rs)
set -u
D=$HOME/.cache/memra-1255
echo "$$" >> "$D/pids.txt"
cd "$D" || exit 1
grep -q FIX2-DONE "$D/fix2.out" || { echo "fix2 run did not finish"; exit 1; }
if [ ! -d tree-fix3 ]; then
  cp -al tree-fix2 tree-fix3 || exit 1
  cp -a target-fix2-120a target-fix3-120a
fi
rsync -c "$D/worker-fix3.rs" "$D/tree-fix3/crates/memra-server/src/worker.rs"
sha256sum tree-fix2/crates/memra-server/src/worker.rs tree-fix3/crates/memra-server/src/worker.rs
bash "$D/dd-build.sh" fix3
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-fix2-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/fix3
mkdir -p "$L"
cd "$D/tree-fix3" || exit 1
fail=0
step() {
  local name=$1; shift
  echo "== $name: $(date -Is)"
  nice -n 15 ionice -c2 -n7 "$@" > "$L/check-$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc"
  grep -E "^test result|^error|FAILED|panicked" "$L/check-$name.log" | sort | uniq -c | sort -rn | head -n 8
  [ $rc = 0 ] || fail=1
}
step fmt cargo fmt --all -- --check
step clippy cargo clippy --release --all-targets -j 14 -- -D warnings
step test-server cargo test --release -p memra-server -j 14
step flags bash tools/check-flags.sh
step numeric python3 tools/check-numeric-scope.py
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker-fix3.keep"
T="an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict"
red() {  # name, python replacement (old, new) as two args
  local name=$1 old=$2 new=$3
  python3 - "$W" "$old" "$new" <<'EOF'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert s.count(old) == 1, old
open(p, "w").write(s.replace(old, new))
EOF
  diff -u "$D/worker-fix3.keep" "$W" > "$L/$name.patch"
  # shellcheck disable=SC2086
  nice -n 15 cargo test --release -p memra-server -j 14 -- $T > "$L/$name.log" 2>&1
  echo "$name rc=$? $(grep -E '^test .*(ok|FAILED)$|^test result' "$L/$name.log" | tr '\n' ' ')" | tee -a "$L/red-summary.txt"
  rsync -c "$D/worker-fix3.keep" "$W"
}
: > "$L/red-summary.txt"
red red-stack "    (mtp_spec && !mtp_prime_started) || !owns_route_state" "    let _ = owns_route_state;
    mtp_spec && !mtp_prime_started"
red red-unbooked "    unbuilt_booked > 0 && !fits_booked && fits_unbooked" "    let _ = fits_unbooked;
    unbuilt_booked > 0 && !fits_booked"
red red-hold "                        memory_hold.extend(model_devices(&engine, &loaded[&model_key]));" "                        let _ = &memory_hold;"
cmp "$W" "$D/worker-fix3.rs" && echo restored
# shellcheck disable=SC2086
nice -n 15 cargo test --release -p memra-server -j 14 -- $T > "$L/green.log" 2>&1
echo "green rc=$? $(grep -E '^test result' "$L/green.log")" | tee -a "$L/red-summary.txt"
echo "FIX3-DONE fail=$fail $(date -Is)"
