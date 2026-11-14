#!/usr/bin/env bash
# Dev desk, CPU only (memra#1307): the new engine tests, the engine's capture-headroom and
# unified_memory tests, clippy (engine and server, all targets), fmt; then the red arms, each with
# the tests kept and one production line changed, each of which must fail its test:
#   R1 pooled_verify_headroom reads e.ctx().mem_get_info() again            -> width test
#   R2 the draft-state bracket end reads e.ctx().mem_get_info() again (main) -> bracket test
#   R3 the bracket end reads allocatable_mem_info (MemAvailable on unified)  -> bracket test
#   R4 one failed capture's appetite floor takes the delta's entry reading   -> bracket test
#   R5 delta_free_from's unified branch reads MemAvailable                   -> delta table test
#   R6 the bracket end asks the device "unified?" again                      -> bracket test
#   R7 Engine::unified_memory asks the device on every call                  -> delta table test
# A red arm whose edit did not apply is reported and not run: it would test the head's source and
# pass for the wrong reason. Every cargo pinned to CPUs 14-31 (memra's share of the dev desk).
set -u
D=$HOME/.cache/memra-1307
echo "$$" >> "$D/pids.txt"
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
L=$D/logs/${1:-checks}
mkdir -p "$L/keep"
S=$L/summary.txt
T=$D/tree
E=$T/crates/memra-engine/src
FILES="spec.rs unified_memory.rs lib.rs"
cd "$T" || exit 1
(cd "$E" && sha256sum $FILES dsv4_gpu.rs) > "$S"
WIDTH=spec::capture_headroom_tests::pooled_dspark_verify_headroom_reads_the_allocatable_free_memory
BRACKET=spec::capture_headroom_tests::draft_state_delta_reads_free_plus_reclaimable_and_the_oom_floor_the_raw_one
DELTA=unified_memory::tests::delta_reading_is_the_driver_free_on_a_discrete_card_and_free_plus_reclaimable_on_unified
for f in $FILES; do cp "$E/$f" "$L/keep/$f"; done
restore() { for f in $FILES; do cmp -s "$L/keep/$f" "$E/$f" || cp "$L/keep/$f" "$E/$f"; done; }
changed() { for f in $FILES; do cmp -s "$L/keep/$f" "$E/$f" || return 0; done; return 1; }
one() {  # one LABEL TEST
  local name=${2##*::}
  case $1 in R*) changed || { echo "$1 ARM-NOT-APPLIED (source unchanged)" | tee -a "$S"; return; } ;; esac
  taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 -- --exact "$2" > "$L/$1-$name.log" 2>&1
  echo "$1 $name rc=$? $(grep -E '^test result' "$L/$1-$name.log" | tail -n 1)" | tee -a "$S"
  grep -A4 "panicked at" "$L/$1-$name.log" | head -n 6 | cut -c1-300 >> "$S"
  grep -E '^error' "$L/$1-$name.log" | head -n 3 >> "$S"
}
arm() {  # arm LABEL FILE OLD NEW TEST: replace the one OLD in FILE with NEW, run TEST, restore
  python3 - "$E/$2" "$1" "$3" "$4" <<'EOF'
import sys
p, label, old, new = sys.argv[1:5]
old, new = old.encode().decode("unicode_escape"), new.encode().decode("unicode_escape")
s = open(p).read()
n = s.count(old)
if n < 1:
    sys.exit(f"{label}: anchor not found")
open(p, "w").write(s.replace(old, new, 1))
EOF
  one "$1" "$5"
  restore
}
for t in "$WIDTH" "$BRACKET" "$DELTA"; do one green "$t"; done
taskset -c 14-31 nice -n 15 cargo test --release -p memra-engine --lib -j 14 -- capture_headroom_tests unified_memory env_audit > "$L/engine-scoped.log" 2>&1
echo "engine capture_headroom_tests + unified_memory + env_audit rc=$? $(grep -E '^test result' "$L/engine-scoped.log" | tail -n 1)" | tee -a "$S"
grep -E "^test .*FAILED$" "$L/engine-scoped.log" | head -n 5 >> "$S"
arm R1 spec.rs '    pooled_verify_headroom_bytes(pooled_verify_reading(e))\n}' \
  '    match e.ctx().mem_get_info() {\n        Ok((free, _total)) => free\n            .saturating_add(e.pool_cached_bytes())\n            .saturating_sub(CAPTURE_HEADROOM_FLOOR),\n        Err(_) => 0,\n    }\n}' "$WIDTH"
END='            && let Ok((raw1, _)) = e.ctx().mem_get_info()\n            && let Some(f1) = crate::unified_memory::delta_free_bytes(e.unified_memory(), raw1)\n'
arm R2 spec.rs "$END" '            && let Ok((f1, _)) = e.ctx().mem_get_info()\n' "$BRACKET"
arm R3 spec.rs "$END" '            && let Ok((f1, _)) = crate::unified_memory::allocatable_mem_info(e.ctx())\n' "$BRACKET"
# The first of the five occurrences is production (four failure sites and the test's own literal).
arm R4 spec.rs 'capture_oom_entry_eff = capture_oom_entry_eff.max(cap_eff0);' \
  'capture_oom_entry_eff = capture_oom_entry_eff.max(cap_meas0);' "$BRACKET"
arm R5 unified_memory.rs '        meminfo.and_then(parse_free_plus_reclaimable_bytes)\n' \
  '        meminfo.and_then(parse_mem_available_bytes)\n' "$DELTA"
arm R6 spec.rs '            && let Some(f1) = crate::unified_memory::delta_free_bytes(e.unified_memory(), raw1)\n        {' \
  '            && let Some(f1) = crate::unified_memory::delta_free_bytes(\n                crate::unified_memory::is_unified(e.ctx()),\n                raw1,\n            )\n        {' "$BRACKET"
arm R7 lib.rs '    pub fn unified_memory(&self) -> bool {\n        self.unified_memory\n    }' \
  '    pub fn unified_memory(&self) -> bool {\n        crate::unified_memory::is_unified(self.ctx())\n    }' "$DELTA"
changed && echo "SOURCE NOT RESTORED" | tee -a "$S" || echo "source restored" | tee -a "$S"
taskset -c 14-31 nice -n 15 cargo clippy --release -p memra-engine -p memra-server --all-targets -j 14 -- -D warnings > "$L/clippy.log" 2>&1
echo "clippy rc=$?" | tee -a "$S"
grep -E '^(error|warning)' "$L/clippy.log" | sort | uniq -c | head -n 5 >> "$S"
taskset -c 14-31 nice -n 15 cargo fmt --all -- --check > "$L/fmt.log" 2>&1; echo "fmt rc=$?" | tee -a "$S"
echo "CHECKS-DONE $(date -Is)" | tee -a "$S"
