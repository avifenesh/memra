#!/usr/bin/env bash
# Dev desk, CPU only (memra#1307): the two new engine tests, the engine's capture-headroom and
# unified_memory tests, clippy (engine and server, all targets), fmt; then the red arms, each with
# the tests kept and one production line restored to the raw driver reading:
#   R1 pooled_verify_headroom reads e.ctx().mem_get_info() again
#      -> pooled_dspark_verify_headroom_reads_the_allocatable_free_memory must fail
#   R2 the draft-state bracket end reads e.ctx().mem_get_info() again (main's raw reading)
#   R3 the draft-state bracket end reads allocatable_mem_info (MemAvailable on unified memory)
#   R4 one failed capture's appetite floor takes the delta's entry reading (cap_meas0), not the raw one
#      -> each: draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one must fail
# Every cargo pinned to CPUs 14-31 (memra's share of the dev desk).
set -u
D=$HOME/.cache/memra-1307
echo "$$" >> "$D/pids.txt"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/${1:-checks}
mkdir -p "$L"
S=$L/summary.txt
T=$D/tree
SPEC=$T/crates/memra-engine/src/spec.rs
cd "$T" || exit 1
sha256sum "$SPEC" crates/memra-engine/src/unified_memory.rs crates/memra-engine/src/dsv4_gpu.rs | sed "s|$HOME|~|" > "$S"
NEW="pooled_dspark_verify_headroom_reads_the_allocatable_free_memory draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one"
one() {  # one LABEL TEST
  # A red arm whose edit did not apply would run the head's source and pass for the wrong reason.
  case $1 in R*) cmp -s "$L/spec.keep" "$SPEC" && { echo "$1 ARM-NOT-APPLIED (spec.rs unchanged)" | tee -a "$S"; return; } ;; esac
  taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 -- --exact "spec::capture_headroom_tests::$2" > "$L/$1-$2.log" 2>&1
  echo "$1 $2 rc=$? $(grep -E '^test result' "$L/$1-$2.log" | tail -n 1)" | tee -a "$S"
  grep -A4 "panicked at" "$L/$1-$2.log" | head -n 6 | cut -c1-300 >> "$S"
  grep -E '^error' "$L/$1-$2.log" | head -n 3 >> "$S"
}
for t in $NEW; do one green "$t"; done
taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 -- capture_headroom_tests unified_memory env_audit > "$L/engine-scoped.log" 2>&1
echo "engine capture_headroom_tests + unified_memory + env_audit rc=$? $(grep -E '^test result' "$L/engine-scoped.log" | tail -n 1)" | tee -a "$S"
grep -E "^test .*FAILED$" "$L/engine-scoped.log" | head -n 5 >> "$S"
cp "$SPEC" "$L/spec.keep"
python3 - "$SPEC" R1 <<'EOF'
import sys
p, arm = sys.argv[1], sys.argv[2]
s = open(p).read()
old = "    pooled_verify_headroom_bytes(pooled_verify_reading(e))\n}"
new = ("    match e.ctx().mem_get_info() {\n        Ok((free, _total)) => free\n"
       "            .saturating_add(e.pool_cached_bytes())\n"
       "            .saturating_sub(CAPTURE_HEADROOM_FLOOR),\n        Err(_) => 0,\n    }\n}")
assert s.count(old) == 1, "R1 anchor"
open(p, "w").write(s.replace(old, new))
EOF
one R1 pooled_dspark_verify_headroom_reads_the_allocatable_free_memory
cp "$L/spec.keep" "$SPEC"
python3 - "$SPEC" R2 <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = ("            && let Ok((raw1, _)) = e.ctx().mem_get_info()\n"
       "            && let Some(f1) = crate::unified_memory::delta_free_bytes(e.ctx(), raw1)\n")
new = "            && let Ok((f1, _)) = e.ctx().mem_get_info()\n"
assert s.count(old) == 1, "R2 anchor"
open(p, "w").write(s.replace(old, new))
EOF
one R2 draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one
cp "$L/spec.keep" "$SPEC"
python3 - "$SPEC" R3 <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = ("            && let Ok((raw1, _)) = e.ctx().mem_get_info()\n"
       "            && let Some(f1) = crate::unified_memory::delta_free_bytes(e.ctx(), raw1)\n")
new = "            && let Ok((f1, _)) = crate::unified_memory::allocatable_mem_info(e.ctx())\n"
assert s.count(old) == 1, "R3 anchor"
open(p, "w").write(s.replace(old, new))
EOF
one R3 draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one
cp "$L/spec.keep" "$SPEC"
python3 - "$SPEC" R4 <<'EOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = "capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);"
assert s.count(old) == 5, "R4 anchor"  # four production sites and the test's own literal; the first is production
open(p, "w").write(s.replace(old, "capture_oom_entry_eff = capture_oom_entry_eff.max(cap_meas0);", 1))
EOF
one R4 draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one
cp "$L/spec.keep" "$SPEC"
cmp -s "$L/spec.keep" "$SPEC" && echo "spec.rs restored" | tee -a "$S"
taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-engine -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$?" | tee -a "$S"
grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 5 >> "$S"
taskset -c 14-31 nice -n 15 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" | tee -a "$S"
echo "CHECKS-DONE $(date -Is)" | tee -a "$S"
