#!/usr/bin/env bash
# Dev desk (memra#1255 review round 3): the head's code (fix6) on main 11b368f7d7. tree-fix6 =
# hardlinked copy of tree-fix5 with worker.rs written by rsync (rename, so tree-fix5 keeps its
# file); the sm_120a binary; the full CPU checks; one red arm per pinned literal of round 2 and 3.
# usage: dd-fix6.sh   (worker.rs already copied to $D/worker-fix6.rs)
set -u
D=$HOME/.cache/memra-1255
echo "$$" >> "$D/pids.txt"
cd "$D" || exit 1
if [ ! -d tree-fix6 ]; then
  cp -al tree-fix5 tree-fix6 || exit 1
  cp -a target-fix5-120a target-fix6-120a
fi
rsync -c "$D/worker-fix6.rs" "$D/tree-fix6/crates/memra-server/src/worker.rs"
sha256sum tree-fix5/crates/memra-server/src/worker.rs tree-fix6/crates/memra-server/src/worker.rs
bash "$D/dd-build.sh" fix6
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-fix2-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/fix6
mkdir -p "$L"
cd "$D/tree-fix6" || exit 1
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
cp "$W" "$D/worker-fix6.keep"
T="an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session an_unbuilt_session_is_booked_once_and_left_out_of_the_reading_without_the_booking every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict reclaim_booking"
red() {  # name, python replacement (old, new) as two args
  local name=$1 old=$2 new=$3
  python3 - "$W" "$old" "$new" <<'EOF'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
assert s.count(old) == 1, old
open(p, "w").write(s.replace(old, new))
EOF
  diff -u "$D/worker-fix6.keep" "$W" > "$L/$name.patch"
  # shellcheck disable=SC2086
  nice -n 15 cargo test --release -p memra-server -j 14 -- $T > "$L/$name.log" 2>&1
  echo "$name rc=$? $(grep -E '^test .*(ok|FAILED)$|^test result' "$L/$name.log" | tr '\n' ' ')" | tee -a "$L/red-summary.txt"
  rsync -c "$D/worker-fix6.keep" "$W"
}
: > "$L/red-summary.txt"
red red-skip "    !skip_unbuilt || owns_route_state" "    let _ = (skip_unbuilt, owns_route_state);
    true"
red red-share "        .saturating_sub(pending_prime.saturating_sub(pending_prime_built))" "        .saturating_sub(0 * pending_prime.saturating_sub(pending_prime_built))"
red red-guard "                    if lane == crate::lanes::Lane::Interactive {
                        memory_hold.extend(model_devices(&engine, &loaded[&model_key]));" "                    if lane == lane {
                        memory_hold.extend(model_devices(&engine, &loaded[&model_key]));"
red red-armed "                        &prime_coop,
                        false,
                    )
                } else {
                    0
                };" "                        &prime_coop,
                        true,
                    )
                } else {
                    0
                };"
red red-built "                        &prime_coop,
                        true,
                    )
                } else {
                    pending_prime
                };" "                        &prime_coop,
                        false,
                    )
                } else {
                    pending_prime
                };"
red red-filter "        if !prime_owed_counts(skip_unbuilt, session_owns_route_state(s)) {" "        if !prime_owed_counts(skip_unbuilt, true) {"
red red-walker "                    s.mtp_prime.is_some(),
                    session_owns_route_state(s),
                )," "                    s.mtp_prime.is_some(),
                    true,
                ),"
cmp "$W" "$D/worker-fix6.rs" && echo restored
# shellcheck disable=SC2086
nice -n 15 cargo test --release -p memra-server -j 14 -- $T > "$L/green.log" 2>&1
echo "green rc=$? $(grep -E '^test result' "$L/green.log")" | tee -a "$L/red-summary.txt"
echo "FIX6-DONE fail=$fail $(date -Is)"
