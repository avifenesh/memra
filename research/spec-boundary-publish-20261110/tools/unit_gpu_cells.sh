#!/usr/bin/env bash
# Runs ON a GB10 host under the caller's lease: the server's GPU unit cells of the spec boundary
# publication on the arms' test binaries (bins/<DIR>/server-lib-test).
#   spec_boundary_publish_is_the_copy_program     BASE_TESTS (the base's production code with
#       the head's tests, tools/red_worker.py) and HEAD; the `[specpub cell]` lines carry each
#       published entry's round-trip digest on a fixed seed and must be the same lines
#   b1_snapshot_and_restore_are_the_copy_program  HEAD: the plain snapshot shares the allocator
#   an_unwritten_plane_is_charged_to_the_insert_budget_reading  HEAD: the budget's free reading
#       moves when a plane is allocated, not when it is written
#   spec_boundary_publish_drops_a_plane_whose_copy_failed  HEAD: must pass. With RED (the head
#       with the publisher ignoring a failed copy, tools/red_unwritten.py) it must fail.
# MEMRA_B1_MODEL adds the 27B's own geometry. usage: unit_gpu_cells.sh BASE_TESTS HEAD [RED]
set -u
D=$HOME/.cache/memra-specpub
B=$D/bins/$1; T=$D/bins/$2; RED=${3:+$D/bins/$3}
OUT=$D/runs/unit-cells-$2; mkdir -p "$OUT"
export MEMRA_B1_MODEL=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
cd "$D/src/crates/memra-server" || exit 1
rc=0
run() { # run LABEL BINARY TEST WANT_RC
  "$2" "worker::tests::$3" --exact --ignored --nocapture > "$OUT/$1.log" 2>&1
  local r=$?; echo "$1 rc=$r (want $4) $(grep -E '^test result' "$OUT/$1.log")" | tee -a "$OUT/verdicts.txt"; [ $r = "$4" ] || rc=1
}
: > "$OUT/verdicts.txt"
sha256sum "$B/server-lib-test" "$T/server-lib-test" ${RED:+"$RED/server-lib-test"} | sed "s|$HOME|~|" >> "$OUT/verdicts.txt"
run specpub-base "$B/server-lib-test" spec_boundary_publish_is_the_copy_program 0
run specpub-head "$T/server-lib-test" spec_boundary_publish_is_the_copy_program 0
run b1-head "$T/server-lib-test" b1_snapshot_and_restore_are_the_copy_program 0
run budget-head "$T/server-lib-test" an_unwritten_plane_is_charged_to_the_insert_budget_reading 0
run fault-head "$T/server-lib-test" spec_boundary_publish_drops_a_plane_whose_copy_failed 0
if [ -n "$RED" ]; then
  run fault-red "$RED/server-lib-test" spec_boundary_publish_drops_a_plane_whose_copy_failed 101
  grep -m1 -h 'an entry was published although\|the draft plane was published' "$OUT/fault-red.log" | cut -c1-200 >> "$OUT/verdicts.txt" || { echo "fault-red: no publication assertion in its log" >> "$OUT/verdicts.txt"; rc=1; }
fi
grep -h '^\[specpub budget cell\]\|^\[b1 cell\]\|^\[specpub fault cell\]' "$OUT/budget-head.log" "$OUT/b1-head.log" "$OUT/fault-head.log" | sed "s|$HOME|~|" >> "$OUT/verdicts.txt"
grep '^\[specpub cell\]' "$OUT/specpub-base.log" > "$OUT/digests-base.txt"
grep '^\[specpub cell\]' "$OUT/specpub-head.log" > "$OUT/digests-head.txt"
n=$(wc -l < "$OUT/digests-head.txt")
if [ "$n" -gt 0 ] && cmp -s "$OUT/digests-base.txt" "$OUT/digests-head.txt"; then
  echo "digests: $n entries, base == head" | tee -a "$OUT/verdicts.txt"
else
  echo "digests: DIFFER or missing ($n head lines)" | tee -a "$OUT/verdicts.txt"; rc=1
fi
exit $rc
