#!/usr/bin/env bash
# CPU build host, memra#1221 part 2 "fails on main": main 11b368f7d7's production code with the head's
# text-only census tests appended (main-files/census_*.txt), run in the main tree (src-main,
# target-bt, after dd_build5090.sh). Every census must FAIL here. -> red-census.log
set -u
R=$HOME/.cache/memra-1221b
export PATH="$HOME/.local/bin:$PATH"
waitfor --grep 'BUILD-DONE|BUILD-FAIL' "$R/build5090.log" --timeout 120m --quiet || exit 1
export PATH="$HOME/.cargo/bin:$PATH" MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a CARGO_TARGET_DIR=$R/target-bt
T=$R/src-main/crates
python3 - "$T" "$R/main-files" <<'PY'
import sys
t, m = sys.argv[1], sys.argv[2]
def ins(path, src_text, marker, block):
    s = src_text
    i = s.index(marker) + len(marker)
    open(path, "w").write(s[:i] + "\n" + block + "\n" + s[i:])
w = open(f"{m}/worker.main.rs").read()
ins(f"{t}/memra-server/src/worker.rs", w, "\nmod tests {\n", open(f"{m}/census_worker.rs.txt").read())
p = open(f"{m}/crates/memra-kv/src/plane.rs").read()
ins(f"{t}/memra-kv/src/plane.rs", p, "\nmod tests {\n    use super::*;\n", open(f"{m}/census_plane.rs.txt").read())
e = open(f"{m}/crates/memra-engine/src/spec.rs").read()
open(f"{t}/memra-engine/src/spec.rs", "w").write(e + "\n#[cfg(test)]\nmod lend_floor_tests {\n    use super::*;\n\n" + open(f"{m}/census_spec.rs.txt").read() + "\n}\n")
print("main + census tests written")
PY
cd "$R/src-main" || exit 1
taskset -c 20-31 nice -n 15 ionice -c2 -n7 cargo test --release --locked -j 12 --no-fail-fast -p memra-kv -p memra-engine -p memra-server --lib -- \
  every_write_below_a_lend_checks_it scratch_length_setters_check_the_lend_before_writing \
  spec_boundary_publish_lends_on_the_mtp_path every_caller_of_a_write_below_a_lend_is_pinned > "$R/red-census.log" 2>&1
echo "rc=$?" >> "$R/red-census.log"
