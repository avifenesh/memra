#!/usr/bin/env bash
# Dev desk (memra#1255): CPU red arms of the two worker tests. Each arm patches one line of
# tree-fix's worker.rs, runs the two tests (expected to FAIL), then restores the file and runs
# them again (expected to PASS). usage: dd-red.sh
set -u
D=$HOME/.cache/memra-1255
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-fix-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
echo "$$" >> "$D/pids.txt"
cd "$D/tree-fix" || exit 1
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker.rs.orig"
OUT=$D/logs/red
mkdir -p "$OUT"
T1=an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it
T2=every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict
run() {
  local name=$1
  diff -u "$D/worker.rs.orig" "$W" > "$OUT/$name.patch"
  nice -n 15 ionice -c2 -n7 cargo test --release -p memra-server -j 14 -- "$T1" "$T2" > "$OUT/$name.log" 2>&1
  echo "$name rc=$? $(grep -E '^test result|^test .*(ok|FAILED)$' "$OUT/$name.log" | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
}
: > "$OUT/summary.txt"
# red-booking: the booking books nothing (main's reading).
python3 - "$W" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        .filter(|(built, _, _)| !built)\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "        .filter(|(built, _, _)| !built && false)\n"))
EOF
run red-booking
cp "$D/worker.rs.orig" "$W"
# red-settle: no settle wait; the gate decides on the booked reading at once.
python3 - "$W" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    unbuilt_booked > 0 && !fits_full\n"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "    let _ = (unbuilt_booked, fits_full);\n    false\n"))
EOF
run red-settle
cp "$D/worker.rs.orig" "$W"
cmp "$W" "$D/worker.rs.orig" && run green
echo "RED-DONE $(date -Is)" | tee -a "$OUT/summary.txt"
