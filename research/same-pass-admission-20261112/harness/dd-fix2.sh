#!/usr/bin/env bash
# Dev desk (memra#1255): the head's code (fix2 = fix plus the settle wait's head-of-line hold).
# 1. tree-fix2 = hardlinked copy of tree-fix with the new worker.rs written by rsync (rename, so
#    tree-fix keeps its file); 2. sm_120a and sm_121a binaries; 3. full CPU checks; 4. red arm of
#    the hold assertion. usage: dd-fix2.sh   (worker.rs already copied to $D/worker-fix2.rs)
set -u
D=$HOME/.cache/memra-1255
echo "$$" >> "$D/pids.txt"
cd "$D" || exit 1
if [ ! -d tree-fix2 ]; then
  cp -al tree-fix tree-fix2 || exit 1
  cp -a target-fix-120a target-fix2-120a
  cp -a target-fix-check target-fix2-check
fi
rsync -c "$D/worker-fix2.rs" "$D/tree-fix2/crates/memra-server/src/worker.rs"
cmp -s tree-fix/crates/memra-server/src/worker.rs worker.rs.orig || echo "WARN tree-fix worker.rs moved"
sha256sum tree-fix/crates/memra-server/src/worker.rs tree-fix2/crates/memra-server/src/worker.rs
bash "$D/dd-build.sh" fix2
ARCH=121a bash "$D/dd-build.sh" fix2
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target-fix2-check
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
mkdir -p "$D/logs/fix2"
cd "$D/tree-fix2" || exit 1
fail=0
step() {
  local name=$1; shift
  echo "== $name: $(date -Is)"
  nice -n 15 ionice -c2 -n7 "$@" > "$D/logs/fix2/check-$name.log" 2>&1
  local rc=$?
  echo "== $name rc=$rc"
  grep -E "^test result|^error|FAILED|panicked" "$D/logs/fix2/check-$name.log" | sort | uniq -c | sort -rn | head -n 8
  [ $rc = 0 ] || fail=1
}
step fmt cargo fmt --all -- --check
step clippy cargo clippy --release --all-targets -j 14 -- -D warnings
step test-server cargo test --release -p memra-server -j 14
step flags bash tools/check-flags.sh
step numeric python3 tools/check-numeric-scope.py
# red arm: the settle branch without the hold must fail the order test.
W=crates/memra-server/src/worker.rs
cp "$W" "$D/worker-fix2.keep"
python3 - "$W" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
old = "                        memory_hold.extend(model_devices(&engine, &loaded[&model_key]));\n                    }\n                    requeue.push_back(req);\n                    continue;\n"
assert s.count(old) == 1
s = s.replace(old, "                        let _ = &memory_hold;\n                    }\n                    requeue.push_back(req);\n                    continue;\n")
open(p, "w").write(s)
EOF
diff -u "$D/worker-fix2.keep" "$W" > "$D/logs/fix2/red-hold.patch"
nice -n 15 cargo test --release -p memra-server -j 14 -- every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict > "$D/logs/fix2/red-hold.log" 2>&1
echo "red-hold rc=$? $(grep -E '^test result' "$D/logs/fix2/red-hold.log")"
rsync -c "$D/worker-fix2.keep" "$W"
cmp "$W" "$D/worker-fix2.rs" && echo restored
echo "FIX2-DONE fail=$fail $(date -Is)"
