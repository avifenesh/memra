#!/usr/bin/env bash
# Dev desk (memra#1233): the fence-bound and tail-rule unit tests, green on the tree and red with
# the mechanism removed.
#   green    the synced tree as is
#   red-fn   hit_suffix_fence_yields always answers false (the bound removed)
#   red-wire the tick loop no longer consults the bound (whole-prompt hits and the deadline
#            order still stand the fence down)
#   red-reclaim the kv-flex shed keys on the entry count again, so a shed that prompt-end tails
#            alone cover reads as a failed one
#   red-rule tail_victim strips a proven tail before any entry (strict tails-first: the rule that
#            a proven tail outranks a never-used oldest entry removed)
# Each arm runs `cargo test -p memra-server --lib the_hit_suffix_fence prompt_end_tails_evict`
# in a copy of the tree.
set -u
D=$HOME/.cache/memra-1233
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
OUT=$D/red-fence; mkdir -p "$OUT"
W=crates/memra-server/src/worker.rs
for arm in green red-fn red-wire red-rule red-reclaim; do
  rm -rf "$D/tree-red"; cp -a "$D/tree" "$D/tree-red"; cd "$D/tree-red" || exit 1
  case $arm in
    red-fn) python3 - "$W" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    !deferred.is_zero() && deferred >= allowance\n}"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "    let _ = (deferred, allowance);\n    false\n}"))
PY
    ;;
    red-wire) python3 - "$W" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
pat = re.compile(r"let fence_yields = cached_hit_waiting_for_ttft\s+\|\| prefill_deadline_mode\s+\|\| hit_suffix_fence_yields\(.*?\);", re.S)
assert len(pat.findall(s)) == 1
open(p, "w").write(pat.sub("let fence_yields = cached_hit_waiting_for_ttft || prefill_deadline_mode;", s))
PY
    ;;
    red-reclaim) python3 - "$W" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "        if freed == 0 {\n            let unleased = px.evictable_bytes();"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "        if n == 0 {\n            let unleased = px.evictable_bytes();"))
PY
    ;;
    red-rule) python3 - "$W" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "            if !oldest_was_hit {\n                return None;\n            }"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, "            let _ = oldest_was_hit;"))
PY
    ;;
  esac
  git diff --no-index --stat "$D/tree/$W" "$W" > "$OUT/$arm.diffstat" 2>&1
  diff -u "$D/tree/$W" "$W" > "$OUT/$arm.patch"
  { date -Is; sha256sum "$W"; } > "$OUT/$arm.log"
  nice -n 15 cargo test --release -p memra-server --lib -j 14 -- the_hit_suffix_fence prompt_end_tails_evict a_reclaim_that_tails >> "$OUT/$arm.log" 2>&1
  echo "$arm rc=$? $(grep -E '^test result' "$OUT/$arm.log" | tail -n 1)" | tee -a "$OUT/summary.txt"
done
rm -rf "$D/tree-red"
echo "RED-FENCE-DONE $(date -Is)" | tee -a "$OUT/summary.txt"
